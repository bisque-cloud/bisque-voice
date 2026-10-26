//! Which hardware runs the graph.
//!
//! ONNX Runtime executes a model by handing each node to an *execution
//! provider*. Providers are consulted in registration order and whatever none
//! of them claims falls through to the built-in CPU provider, which is always
//! present and always last. Registering an accelerator is therefore a
//! *preference*, never a commitment: a graph can end up fully, partially, or
//! not at all offloaded, and nothing in the public API tells you which.
//!
//! This module owns that preference. It parses `--device`, applies the right
//! provider chain for the platform the binary was built for, and reports what
//! was actually registered rather than what was asked for.
//!
//! # Wiring
//!
//! One line per session-creation site, immediately before `commit_from_file`:
//!
//! ```ignore
//! let mut builder = Session::builder()?;
//! // ...threads, optimization level, etc...
//! builder = device::configure_cached(builder, device, &model_path)?;
//! let session = builder.commit_from_file(&model_path)?;
//! ```
//!
//! Use [`configure`] instead of [`configure_cached`] when there is no single
//! file backing the session; the only thing lost is the CoreML compiled-model
//! cache (see [`configure_cached`]).
//!
//! # Providers by platform
//!
//! | Platform | Chain | Compiled in by default |
//! |---|---|---|
//! | macOS (any arch) | CoreML (ANE + GPU + CPU) → CPU | yes |
//! | Windows | DirectML → CPU | yes |
//! | Linux x86_64 | CUDA → CPU | **no** — `--features cuda` |
//! | Linux aarch64 | CPU only | n/a |
//! | any | XNNPACK → CPU | **no** — `--features xnnpack` |
//!
//! `Cargo.toml` gates the matching `ort` features on the same `cfg`
//! expressions used here, so CoreML is not a build input on Linux and DirectML
//! is not one on macOS. The two lists must stay in sync; if they drift, this
//! module fails to compile, which is the intended failure mode.
//!
//! ## Why CUDA and XNNPACK are opt-in
//!
//! `ort`'s `download-binaries` fetches a prebuilt ONNX Runtime chosen by the
//! enabled feature set, from a fixed table of published builds
//! (`ort-sys/build/download/dist.tsv`). Two facts follow from that table:
//!
//! * **CoreML and DirectML are free.** The macOS build already published is
//!   `aarch64-apple-darwin+coreml` and the Windows one is
//!   `x86_64-pc-windows-msvc+directml` — those are what this crate has been
//!   downloading all along with no EP features enabled. Turning the features
//!   on selects the *same* archive; it only makes the provider reachable from
//!   Rust. No new download, no new system dependency (both are OS components).
//! * **CUDA and XNNPACK are not.** There is no published build with XNNPACK
//!   for *any* target, so enabling `ort/xnnpack` fails the build outright
//!   unless you point `ort` at your own ONNX Runtime. And the only CUDA build
//!   is `x86_64-unknown-linux-gnu+cuda13,tensorrt,nvrtx` — a much larger
//!   archive, with nothing at all for `aarch64-unknown-linux-gnu`, which is in
//!   this crate's release matrix. Defaulting CUDA on would break that target's
//!   build and bloat the other, so it is a feature the person who has a CUDA
//!   box turns on.
//!
//! # Acceleration is not automatically a win
//!
//! Do not assume registering a provider makes synthesis faster. A partially
//! offloaded graph pays a host↔device copy at every boundary between the
//! accelerator's subgraph and the CPU's, and enough boundaries will cost more
//! than the offload saves. The specific trap here is int4 weight-only
//! quantization: `MatMulNBits`, which is most of a q4 Qwen3 export, is not
//! supported by CoreML or DirectML, so those providers claim the cheap ops
//! around it, leave the expensive ones on CPU, and add copies in between. The
//! fp32/fp16 Kokoro graph is a far better candidate than a q4 LLM.
//!
//! This is why [`Device::Cpu`] exists and why it is absolute: when the
//! accelerator is slower or wrong, `--device cpu` is the way out, and it must
//! never quietly register something anyway.
//!
//! # Honesty of the report
//!
//! ONNX Runtime's public API exposes no way to ask a live session which
//! provider each node was assigned to — only the profiling JSON
//! (`SessionBuilder::with_profiling`) carries that, after a run. So
//! [`describe`] reports **registration**, which is a strictly weaker claim
//! than execution, and labels it as such. What it will never do is claim a
//! provider that failed to register: registration is attempted one provider at
//! a time and each outcome is recorded.

//! # Builds without ONNX Runtime
//!
//! On a build compiled without the `onnx` feature (the studio-only Intel-Mac
//! release), [`Device`], [`parse`], [`plan`] and [`describe`] survive — the
//! CLI still takes `--device`, and the ggml engines simply ignore it — while
//! everything that touches `ort` (`configure*`, `commit_file`, the provider
//! chain) is compiled out. `compiled_in()` is then empty, which keeps every
//! report honest: "gpu behaves as cpu" because there is no runtime to
//! register anything with.

// This is a small platform-abstraction surface deliberately wider than any one
// call site: synthesis uses `configure_cached`, `doctor` uses `describe`,
// `compiled_in` and `accelerator_available`, the CLI parser uses `parse`, and
// the result JSON uses `last_registration`. Which of those a given build
// happens to reach is not a signal that the rest are unused.
#![allow(dead_code)]

use std::fmt;
#[cfg(feature = "onnx")]
use std::path::Path;
#[cfg(all(target_os = "macos", feature = "onnx"))]
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{bail, Result};
#[cfg(feature = "onnx")]
use ort::session::builder::SessionBuilder;
#[cfg(feature = "onnx")]
use ort::session::Session;

#[cfg(all(target_os = "macos", feature = "onnx"))]
use ort::ep::coreml;

/// Which hardware the user asked for.
///
/// This is a policy, not a provider name: the provider that implements "gpu"
/// is chosen by platform (CoreML, DirectML, CUDA). Provider names are accepted
/// as input aliases for convenience, but they select the platform's provider,
/// not the named one — `--device cuda` on a Mac means "accelerate", and gets
/// CoreML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// Register the platform's accelerator if one is compiled in, and fall
    /// back to CPU for anything it cannot run. The default.
    #[default]
    Auto,
    /// CPU only. Registers no accelerator at all — the escape hatch for when
    /// the accelerator is slower, or wrong.
    Cpu,
    /// Same chain as [`Device::Auto`]. The difference is intent, not
    /// mechanism: `Auto` is a default the user did not choose, `Gpu` is a
    /// request they did, so a failure to register is worth a louder complaint.
    Gpu,
}

impl Device {
    /// The `--device` spelling of this choice.
    pub fn as_str(self) -> &'static str {
        match self {
            Device::Auto => "auto",
            Device::Cpu => "cpu",
            Device::Gpu => "gpu",
        }
    }

    /// Whether this choice permits registering an accelerator at all.
    pub(crate) fn accelerates(self) -> bool {
        !matches!(self, Device::Cpu)
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Device {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        parse(s)
    }
}

/// Parse a `--device` value.
///
/// Accepts `auto`, `cpu`, `gpu`, and the provider names as aliases for `gpu`
/// so that a script written on one platform runs on another. Case and
/// surrounding whitespace are ignored.
pub fn parse(s: &str) -> Result<Device> {
    match s.trim().to_ascii_lowercase().as_str() {
        "" | "auto" | "default" => Ok(Device::Auto),
        "cpu" => Ok(Device::Cpu),
        // `gpu` and its aliases all mean the same thing: use whatever this
        // platform's accelerator is. They are not a way to pick a provider.
        "gpu" | "accel" | "accelerated" | "coreml" | "ane" | "metal" | "directml" | "dml"
        | "cuda" | "gpu0" => Ok(Device::Gpu),
        other => bail!(
            "unknown --device `{other}`: expected `auto` (the default), `cpu`, or `gpu`.\n\
             `gpu` selects this platform's accelerator — {}.",
            platform_accelerator_phrase()
        ),
    }
}

/// What "gpu" resolves to on the platform this binary was built for, phrased
/// for an error message.
fn platform_accelerator_phrase() -> String {
    let compiled = compiled_in();
    if compiled.is_empty() {
        "this build has none compiled in, so `gpu` behaves as `cpu`".to_string()
    } else {
        format!("here that is {}", compiled.join(", "))
    }
}

/// One provider chain, as it was actually applied to one session.
#[derive(Debug, Clone, Default)]
pub struct Registration {
    /// What was requested.
    pub device: Device,
    /// Providers whose registration call returned `Ok`, in registration order.
    /// The implicit CPU provider is not listed; it is always present.
    pub registered: Vec<&'static str>,
    /// Providers that were tried and refused, with the runtime's reason.
    pub failed: Vec<(&'static str, String)>,
}

impl Registration {
    /// A one-line summary of what this session is actually set up to use.
    ///
    /// Best-effort by construction: it reports which providers *registered*,
    /// not which ones ended up executing nodes (see the module docs). It never
    /// names a provider that failed to register.
    pub fn summary(&self) -> String {
        let mut s = if self.registered.is_empty() {
            "CPU".to_string()
        } else {
            let names: Vec<&str> = self.registered.iter().map(|p| pretty(p)).collect();
            format!("{} → CPU", names.join(" → "))
        };
        if !self.failed.is_empty() {
            let notes: Vec<String> = self
                .failed
                .iter()
                .map(|(p, why)| format!("{} unavailable: {}", pretty(p), first_line(why)))
                .collect();
            s.push_str(&format!(" ({})", notes.join("; ")));
        }
        s
    }
}

/// The last chain [`configure`] applied, if any.
///
/// A process synthesizes with one device choice, so a single slot is enough;
/// several sessions built with the same choice overwrite each other with the
/// same answer. It is read by [`describe`].
static LAST: Mutex<Option<Registration>> = Mutex::new(None);

/// The most recent chain applied by [`configure`] in this process.
pub fn last_registration() -> Option<Registration> {
    LAST.lock().ok().and_then(|g| g.clone())
}

/// Apply the platform's provider chain for `device` to `builder`.
///
/// Never fails because a provider is missing: an accelerator that refuses to
/// register is reported on stderr and the session continues on CPU. A user
/// without CUDA, or on a Mac where CoreML declines the graph, still
/// synthesizes.
#[cfg(feature = "onnx")]
pub fn configure(builder: SessionBuilder, device: Device) -> Result<SessionBuilder> {
    configure_inner(builder, device, None)
}

/// [`configure`], plus a persistent cache for the compiled CoreML model.
///
/// CoreML does not execute an ONNX graph; it compiles one into its own format
/// first, and by default it redoes that on every session load. For a CLI that
/// is invoked per utterance, that compile is paid on every single run and can
/// easily dominate the synthesis it was meant to speed up. Pointing CoreML at
/// a cache directory makes it a one-time cost.
///
/// The cache is keyed on the identity of `model_path` — its path, byte length
/// and mtime — because CoreML's own cache key is derived from graph structure
/// and metadata, so a model whose *weights* changed but whose graph did not
/// would otherwise silently reuse the previously compiled artifact. That is
/// the shape of the garbled-audio class of bug, and it is not worth the disk
/// saved. Any failure to prepare the directory disables caching and is
/// otherwise ignored.
///
/// A no-op on platforms other than macOS.
#[cfg(feature = "onnx")]
pub fn configure_cached(
    builder: SessionBuilder,
    device: Device,
    model_path: &Path,
) -> Result<SessionBuilder> {
    configure_inner(builder, device, Some(model_path))
}

#[cfg(feature = "onnx")]
fn configure_inner(
    #[allow(unused_mut)] mut builder: SessionBuilder,
    device: Device,
    #[allow(unused_variables)] model_path: Option<&Path>,
) -> Result<SessionBuilder> {
    // Both `mut`s are load-bearing on a platform with an accelerator compiled
    // in and dead on one without (Linux, no `--features cuda`), which is a
    // supported build rather than a mistake.
    #[allow(unused_mut)]
    let mut record = Registration {
        device,
        ..Default::default()
    };

    if device.accelerates() {
        #[cfg(target_os = "macos")]
        {
            // MEASURED, not assumed. On the int4 Qwen3 packs:
            //   MLProgram      rejects the graph outright — Apple documents
            //                  that MLProgram forbids *unbounded* dynamic
            //                  dimensions (RangeDim needs an upper bound), and
            //                  these exports have them. We fall back to CPU.
            //   NeuralNetwork  accepts it and runs ~2.8x SLOWER than CPU
            //                  (65s vs 23s), because int4 is not native to the
            //                  ANE and gets emulated, plus host/device copies.
            // Failing fast and falling back beats silently running 3x slower,
            // so MLProgram stays. The fix for accelerators is an fp16 pack with
            // bounded shapes, not a format flag.
            // MLProgram over the default NeuralNetwork format: it supports
            // more operators and native fp16, which is what gets work onto the
            // Neural Engine instead of bouncing it back to CPU. It wants
            // macOS 12+ (2021); anything older fails registration and lands on
            // CPU, which is the same place it would have landed anyway.
            let mut ep = ort::ep::CoreML::default()
                .with_model_format(coreml::ModelFormat::MLProgram)
                .with_compute_units(coreml::ComputeUnits::All);
            if let Some(dir) = model_path.and_then(coreml_cache_dir) {
                ep = ep.with_model_cache_dir(dir.to_string_lossy());
            }
            try_register(&mut builder, &ep, &mut record);
        }

        #[cfg(target_os = "windows")]
        {
            let ep = ort::ep::DirectML::default();
            if try_register(&mut builder, &ep, &mut record) {
                // DirectML cannot run with ONNX Runtime's memory-pattern
                // optimization; leaving it on is a documented way to get wrong
                // results out of DML. Non-fatal like everything else here —
                // recover the builder and carry on without DML's benefit
                // rather than refusing to synthesize.
                builder = match builder.with_memory_pattern(false) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!(
                            "bisque-voice: could not disable memory pattern for DirectML ({e}); \
                             continuing on CPU"
                        );
                        record.registered.retain(|p| *p != "DmlExecutionProvider");
                        record.failed.push(("DmlExecutionProvider", e.to_string()));
                        e.recover()
                    }
                };
            }
        }

        #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
        {
            let ep = ort::ep::CUDA::default();
            try_register(&mut builder, &ep, &mut record);
        }

        // XNNPACK sits ahead of the plain CPU provider rather than replacing
        // it: it claims the dense kernels (Conv, Gemm, MatMul) and leaves the
        // rest to MLAS. It has no build published with `download-binaries`, so
        // it only exists behind `--features xnnpack` against a custom runtime.
        #[cfg(feature = "xnnpack")]
        {
            let ep = ort::ep::XNNPACK::default();
            try_register(&mut builder, &ep, &mut record);
        }
    }

    if device == Device::Gpu && record.registered.is_empty() {
        // `auto` staying on CPU is unremarkable. `--device gpu` staying on CPU
        // is the user's request going unmet, so say so.
        eprintln!(
            "bisque-voice: --device gpu requested but no accelerator registered; running on CPU."
        );
    }

    // The CPU provider is deliberately *not* registered explicitly. It is
    // already ONNX Runtime's implicit fallback, and `ort::ep::CPU`'s
    // registration additionally toggles the CPU arena allocator — defaulting
    // to *disabling* it, which is a performance regression we would be
    // introducing by accident.

    if let Ok(mut slot) = LAST.lock() {
        *slot = Some(record);
    }
    Ok(builder)
}

/// Register one provider, recording the outcome. Returns whether it took.
///
/// Unused on a platform with nothing compiled in, which is a legitimate
/// configuration (Linux without `--features cuda`).
#[allow(dead_code)]
#[cfg(feature = "onnx")]
fn try_register<E: ort::ep::ExecutionProvider>(
    builder: &mut SessionBuilder,
    ep: &E,
    record: &mut Registration,
) -> bool {
    match ep.register(builder) {
        Ok(()) => {
            record.registered.push(ep.name());
            true
        }
        Err(e) => {
            eprintln!(
                "bisque-voice: {} is unavailable, continuing on CPU ({})",
                pretty(ep.name()),
                first_line(&e.to_string())
            );
            record.failed.push((ep.name(), e.to_string()));
            false
        }
    }
}

/// A directory for CoreML's compiled-model cache, keyed to this exact model
/// file. `None` if it cannot be created — caching is an optimization, not a
/// requirement.
#[cfg(all(target_os = "macos", feature = "onnx"))]
fn coreml_cache_dir(model_path: &Path) -> Option<PathBuf> {
    use sha2::{Digest, Sha256};

    let meta = std::fs::metadata(model_path).ok()?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);

    let mut hasher = Sha256::new();
    hasher.update(model_path.to_string_lossy().as_bytes());
    hasher.update(meta.len().to_le_bytes());
    hasher.update(mtime.to_le_bytes());
    let key = format!("{:x}", hasher.finalize());

    let dir = dirs::cache_dir()?
        .join("bisque-voice")
        .join("coreml")
        .join(&key[..16]);
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// The accelerators compiled into *this* binary, in the order they are tried.
///
/// One list, gated exactly like `Cargo.toml`'s per-target `ort` features and
/// exactly like the chain in [`configure`]. Empty means the build can only use
/// the CPU provider.
const ACCELERATORS: &[&str] = &[
    // Every entry additionally requires the `onnx` feature: providers are
    // ONNX Runtime constructs, and a build without that runtime (the
    // studio-only Intel-Mac release) has none, whatever the OS. `cuda` and
    // `xnnpack` imply `onnx` in Cargo.toml, so their entries need no extra
    // clause.
    #[cfg(all(target_os = "macos", feature = "onnx"))]
    "CoreML",
    #[cfg(all(target_os = "windows", feature = "onnx"))]
    "DirectML",
    #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
    "CUDA",
    #[cfg(feature = "xnnpack")]
    "XNNPACK",
];

/// The accelerators compiled into *this* binary, in the order they would be
/// tried. Empty means the build can only use the CPU provider.
///
/// This is a build-time fact — it says nothing about whether the hardware or
/// the OS component is present at runtime. For that, see
/// [`accelerator_available`].
pub fn compiled_in() -> &'static [&'static str] {
    ACCELERATORS
}

/// The provider chain `device` would produce in this build, ending in CPU.
///
/// A plan, not a result — see [`describe`] for the distinction.
pub fn plan(device: Device) -> Vec<&'static str> {
    let mut v = if device.accelerates() {
        compiled_in().to_vec()
    } else {
        Vec::new()
    };
    v.push("CPU");
    v
}

/// Whether ONNX Runtime says this platform's accelerator is present in the
/// runtime it linked against.
///
/// `None` when nothing is compiled in, or when the query itself fails. A
/// `Some(true)` still does not mean any node will run on it — the provider can
/// be present and decline the whole graph.
pub fn accelerator_available() -> Option<bool> {
    #[cfg(feature = "onnx")]
    #[allow(unused_imports)]
    use ort::ep::ExecutionProvider;

    #[cfg(all(target_os = "macos", feature = "onnx"))]
    {
        return ort::ep::CoreML::default().is_available().ok();
    }
    #[cfg(all(target_os = "windows", feature = "onnx"))]
    {
        return ort::ep::DirectML::default().is_available().ok();
    }
    #[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
    {
        return ort::ep::CUDA::default().is_available().ok();
    }
    #[allow(unreachable_code)]
    {
        None
    }
}

/// What is actually running, for `doctor` output and the result JSON.
///
/// If a session has already been built with this same `device`, this reports
/// the providers that really registered — including saying "CPU" when the
/// accelerator refused. If no session has been built yet (the `doctor` case),
/// it reports the chain that *would* be registered and says so, because a
/// prediction dressed as a measurement is the thing this function exists to
/// avoid.
///
/// Either way the claim is registration, not execution; ONNX Runtime does not
/// expose per-node provider assignment outside of a profiling run.
pub fn describe(device: Device) -> String {
    if let Some(record) = last_registration() {
        if record.device == device {
            return record.summary();
        }
    }
    format!(
        "{} (would register {}; no session built yet, so this is the plan, not a measurement)",
        device,
        plan(device).join(" → ")
    )
}

/// Long provider ids are ONNX Runtime's; these are what a person reads.
fn pretty(provider: &str) -> &str {
    match provider {
        "CoreMLExecutionProvider" => "CoreML",
        "DmlExecutionProvider" => "DirectML",
        "CUDAExecutionProvider" => "CUDA",
        "XnnpackExecutionProvider" => "XNNPACK",
        "CPUExecutionProvider" => "CPU",
        other => other,
    }
}

/// ONNX Runtime errors are frequently multi-line with a stack of context; a
/// one-line summary is what belongs in a one-line summary.
fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or(s).trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_three_canonical_spellings() {
        assert_eq!(parse("auto").unwrap(), Device::Auto);
        assert_eq!(parse("cpu").unwrap(), Device::Cpu);
        assert_eq!(parse("gpu").unwrap(), Device::Gpu);
    }

    #[test]
    fn parsing_ignores_case_and_surrounding_space() {
        assert_eq!(parse("  CPU ").unwrap(), Device::Cpu);
        assert_eq!(parse("Auto").unwrap(), Device::Auto);
        assert_eq!(parse("\tGpU\n").unwrap(), Device::Gpu);
    }

    #[test]
    fn an_empty_value_is_auto() {
        assert_eq!(parse("").unwrap(), Device::Auto);
        assert_eq!(parse("   ").unwrap(), Device::Auto);
    }

    #[test]
    fn provider_names_are_aliases_for_gpu_on_every_platform() {
        // A script written on a Mac has to keep working on Windows, so these
        // select the platform's accelerator rather than the named one.
        for alias in ["coreml", "directml", "dml", "cuda", "metal", "ane"] {
            assert_eq!(parse(alias).unwrap(), Device::Gpu, "alias {alias}");
        }
    }

    #[test]
    fn an_unknown_device_is_rejected_with_the_valid_choices() {
        let err = parse("tpu").unwrap_err().to_string();
        assert!(err.contains("tpu"), "{err}");
        assert!(err.contains("cpu") && err.contains("gpu"), "{err}");
    }

    #[test]
    fn from_str_and_display_round_trip() {
        for device in [Device::Auto, Device::Cpu, Device::Gpu] {
            let s = device.to_string();
            assert_eq!(s.parse::<Device>().unwrap(), device);
        }
    }

    #[test]
    fn the_default_is_auto() {
        assert_eq!(Device::default(), Device::Auto);
    }

    #[test]
    fn every_plan_ends_on_cpu() {
        for device in [Device::Auto, Device::Cpu, Device::Gpu] {
            assert_eq!(*plan(device).last().unwrap(), "CPU");
        }
    }

    #[test]
    fn the_cpu_plan_is_only_cpu() {
        // The escape hatch has to actually be an escape hatch.
        assert_eq!(plan(Device::Cpu), vec!["CPU"]);
    }

    /// The one test that touches the `LAST` slot, so its steps are ordered
    /// rather than racing each other across the test harness's threads.
    ///
    /// This is also the only end-to-end check available without model files —
    /// and it is a real one: registering an execution provider mutates session
    /// *options*, which needs no graph, so this exercises the actual ONNX
    /// Runtime this binary linked against rather than a stand-in.
    #[cfg(feature = "onnx")]
    #[test]
    fn configure_registers_for_real_and_describe_reports_what_happened() {
        use ort::session::Session;

        // Before anything is built, `describe` must not present a prediction
        // as an observation.
        let before = describe(Device::Auto);
        assert!(before.contains("would register"), "{before}");
        assert!(before.contains("not a measurement"), "{before}");

        let builder = Session::builder().expect("ONNX Runtime session builder");
        let builder = configure(builder, Device::Auto).expect("configure must never be fatal");
        drop(builder);

        let record = last_registration().expect("configure records what it applied");
        assert_eq!(record.device, Device::Auto);
        assert_eq!(describe(Device::Auto), record.summary());

        // Every compiled-in provider must be ACCOUNTED FOR — either it
        // registered, or it declined and we recorded why. What must never
        // happen is a provider silently vanishing, which is the failure this
        // module exists to expose.
        //
        // Deliberately not asserting that registration SUCCEEDS: a CI runner
        // has no GPU, so DirectML is genuinely unavailable there and falling
        // back to CPU is the correct behavior, not a defect. Requiring success
        // made this test assert the presence of hardware rather than the
        // correctness of the code.
        assert_eq!(
            record.registered.len() + record.failed.len(),
            compiled_in().len(),
            "registered {:?}, failed {:?}, compiled in {:?}",
            record.registered,
            record.failed,
            compiled_in()
        );
        for (name, why) in &record.failed {
            assert!(!why.is_empty(), "{name} declined without saying why");
        }
        #[cfg(target_os = "macos")]
        {
            assert_eq!(record.registered, ["CoreMLExecutionProvider"]);
            assert_eq!(record.summary(), "CoreML → CPU");
        }

        // `--device cpu` is absolute: it registers nothing at all, whatever is
        // compiled in.
        let builder = Session::builder().expect("ONNX Runtime session builder");
        let builder = configure(builder, Device::Cpu).expect("configure must never be fatal");
        drop(builder);
        let record = last_registration().expect("configure records what it applied");
        assert_eq!(record.device, Device::Cpu);
        assert!(record.registered.is_empty(), "{:?}", record.registered);
        assert!(record.failed.is_empty(), "{:?}", record.failed);
        assert_eq!(record.summary(), "CPU");
        assert_eq!(describe(Device::Cpu), "CPU");
    }

    #[test]
    fn the_runtime_we_linked_actually_carries_the_platform_accelerator() {
        // Distinct from `compiled_in`, which is only a build-time fact. This
        // asks ONNX Runtime itself.
        if compiled_in().is_empty() {
            assert_eq!(accelerator_available(), None);
        } else {
            assert_eq!(accelerator_available(), Some(true));
        }
    }

    #[cfg(all(target_os = "macos", feature = "onnx"))]
    #[test]
    fn the_coreml_cache_is_keyed_to_the_exact_model_file() {
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.onnx");
        let b = dir.path().join("b.onnx");
        std::fs::File::create(&a).unwrap().write_all(b"aa").unwrap();
        std::fs::File::create(&b).unwrap().write_all(b"bb").unwrap();

        let ca = coreml_cache_dir(&a).expect("cache dir");
        let cb = coreml_cache_dir(&b).expect("cache dir");
        assert!(ca.is_dir() && cb.is_dir());
        // Two different models must never share a compiled artifact.
        assert_ne!(ca, cb);
        // The same model must reuse one, or the cache buys nothing.
        assert_eq!(ca, coreml_cache_dir(&a).unwrap());

        // A model whose bytes changed under the same path gets a new key,
        // because CoreML's own key is graph-shaped and would not notice.
        std::fs::File::create(&a)
            .unwrap()
            .write_all(b"aaa-different")
            .unwrap();
        let ca2 = coreml_cache_dir(&a).expect("cache dir");
        assert_ne!(ca, ca2);

        for d in [ca, cb, ca2] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    #[test]
    fn describe_never_claims_an_accelerator_for_cpu() {
        let d = describe(Device::Cpu);
        for accelerator in ["CoreML", "DirectML", "CUDA", "XNNPACK"] {
            assert!(!d.contains(accelerator), "`{d}` mentions {accelerator}");
        }
    }

    #[test]
    fn a_summary_with_nothing_registered_says_cpu() {
        let record = Registration {
            device: Device::Auto,
            registered: Vec::new(),
            failed: Vec::new(),
        };
        assert_eq!(record.summary(), "CPU");
    }

    #[test]
    fn a_summary_never_claims_a_provider_that_failed() {
        // The whole point of recording outcomes: a failed CoreML registration
        // must not read as a CoreML session.
        let record = Registration {
            device: Device::Gpu,
            registered: Vec::new(),
            failed: vec![(
                "CoreMLExecutionProvider",
                "CoreML is not supported in this build\n  at some_file.cc:12".to_string(),
            )],
        };
        let s = record.summary();
        assert!(s.starts_with("CPU"), "{s}");
        assert!(s.contains("CoreML unavailable"), "{s}");
        // The multi-line runtime error is collapsed to its first line.
        assert!(!s.contains("some_file.cc"), "{s}");
    }

    #[test]
    fn a_summary_lists_registered_providers_ahead_of_cpu() {
        let record = Registration {
            device: Device::Auto,
            registered: vec!["CoreMLExecutionProvider"],
            failed: Vec::new(),
        };
        assert_eq!(record.summary(), "CoreML → CPU");
    }

    #[test]
    fn compiled_in_matches_the_platform_this_was_built_for() {
        let compiled = compiled_in();
        if cfg!(not(feature = "onnx")) {
            // No ONNX Runtime, no providers — whatever the OS. This is the
            // studio-only build's whole claim about acceleration.
            assert!(compiled.is_empty(), "{compiled:?}");
        } else if cfg!(target_os = "macos") {
            assert_eq!(compiled, ["CoreML"]);
        } else if cfg!(target_os = "windows") {
            assert_eq!(compiled, ["DirectML"]);
        } else if cfg!(all(
            target_os = "linux",
            not(feature = "cuda"),
            not(feature = "xnnpack")
        )) {
            assert!(compiled.is_empty(), "{compiled:?}");
        }
    }

    #[test]
    fn provider_ids_render_as_readable_names() {
        assert_eq!(pretty("CoreMLExecutionProvider"), "CoreML");
        assert_eq!(pretty("DmlExecutionProvider"), "DirectML");
        assert_eq!(pretty("CUDAExecutionProvider"), "CUDA");
        // An id this module has not seen passes through rather than being lost.
        assert_eq!(pretty("RocmExecutionProvider"), "RocmExecutionProvider");
    }
}

/// Build, configure and load a session, falling back to CPU if the accelerator
/// cannot run this graph.
///
/// Registration succeeding is not the same as the graph *running*. CoreML
/// registers fine and then fails to initialize Qwen3's int4 `talker_cache` —
/// a graph-level rejection that surfaces at `commit`, long after the provider
/// said yes. Treating that as fatal made `--device auto` break every
/// synthesis on a machine whose only sin was having a Neural Engine.
///
/// So an accelerator is an optimization that is allowed to decline. A failure
/// here is reported once, on stderr, and retried on CPU. `Device::Cpu` never
/// retries — there is nothing to fall back to, and hiding that error would be
/// hiding a real one.
#[cfg(feature = "onnx")]
pub fn commit_file(threads: Option<usize>, device: Device, path: &Path) -> Result<Session> {
    fn build(threads: Option<usize>) -> Result<SessionBuilder> {
        let mut b = Session::builder().map_err(|e| anyhow::anyhow!("ONNX Runtime: {e}"))?;
        if let Some(n) = threads {
            b = b
                .with_intra_threads(n)
                .map_err(|e| anyhow::anyhow!("setting intra-op threads: {e}"))?;
        }
        Ok(b)
    }

    let first = (|| -> Result<Session> {
        let mut b = configure_cached(build(threads)?, device, path)?;
        b.commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", path.display()))
    })();

    match first {
        Ok(s) => Ok(s),
        Err(e) if device != Device::Cpu => {
            eprintln!(
                "bisque-voice: {} could not run {} ({e}); falling back to CPU",
                describe(device),
                path.file_name().unwrap_or_default().to_string_lossy(),
            );
            let mut b = configure_cached(build(threads)?, Device::Cpu, path)?;
            b.commit_from_file(path)
                .map_err(|e| anyhow::anyhow!("loading {} on CPU: {e}", path.display()))
        }
        Err(e) => Err(e),
    }
}
