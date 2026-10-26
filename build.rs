//! Bakes each pack's checksum into the binary, and builds the vendored
//! qwentts.cpp that the `qwen3-clone` engines run on.
//!
//! The release workflow sets `BISQUE_VOICE_PACK_SHA256_<ID>` after building and
//! uploading that pack. A local build leaves them empty, and `fetch.rs` then
//! refuses to download anything — an unverified model is how the garbled-audio
//! incident happened, so "no checksum" fails loudly instead of trusting the
//! network.
//!
//! Adding an engine or an aligner adds its id here. Keep these lists in step
//! with `engine::REGISTRY` and `align::REGISTRY`; `engine::pack_sha256` and
//! `align::pack_sha256` map ids back to these values.
//!
//! Ids are uppercased with `-` replaced by `_`, so the aligner `qwen3-fp32`
//! reads `BISQUE_VOICE_PACK_SHA256_QWEN3_FP32`.

use std::path::{Path, PathBuf};

const ENGINES: &[&str] = &[
    "KOKORO",
    "QWEN3_CLONE_06B",
    "QWEN3_CLONE_17B",
    "QWEN3_VOICES_06B",
    "QWEN3_VOICES_17B",
];
const ALIGNERS: &[&str] = &["QWEN3_FP32", "QWEN3_Q4"];
/// Per-language G2P data packs; see `g2p::pack::REGISTRY` and its
/// `pack_sha256`.
const G2P: &[&str] = &["G2P_ESPEAK", "G2P_ZH", "G2P_JA"];

fn main() {
    for pack in ENGINES.iter().chain(ALIGNERS).chain(G2P) {
        let var = format!("BISQUE_VOICE_PACK_SHA256_{pack}");
        println!("cargo:rerun-if-env-changed={var}");
        let sha = std::env::var(&var).unwrap_or_default();
        println!("cargo:rustc-env={var}={sha}");
    }

    // Order is the link order. Static archives resolve left to right, so every
    // ggml dependent must be named before the ggml archives `build_qwentts`
    // emits — see `build_vibevoice`.
    build_vibevoice();
    build_qwentts();
}

/// Compile `vendor/vibevoice.cpp` against the ggml `build_qwentts` already
/// built, and generate its bindings.
///
/// **It must not build a ggml of its own.** Both projects vendor one, and every
/// `ggml_*` symbol is defined in both archives — 534 in `ggml-base` alone. A
/// static link resolves that collision by search order rather than by error, so
/// the failure would not be a broken build, it would be one engine silently
/// running against the other's ggml. Our fork carries a
/// `VIBEVOICE_GGML_EXTERNAL` option for exactly this: it skips the
/// `add_subdirectory(third_party/ggml)` and compiles against the headers we
/// point it at, leaving the symbols for the final link to settle.
///
/// This runs *before* `build_qwentts` so its `-l` lands ahead of the ggml
/// archives it depends on. It needs no build output from there — only the ggml
/// headers, which are in the source tree. The two trees are pinned to the same
/// ggml commit (the submodule in our vibevoice fork points at it), so the
/// headers compiled against and the archives linked against always agree.
fn build_vibevoice() {
    let manifest = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let vendor = manifest.join("vendor/vibevoice.cpp");
    let header = vendor.join("include/vibevoice.h");
    if !header.exists() {
        panic!(
            "vendor/vibevoice.cpp is empty — the vendored engines were not fetched.\n\
             Run: sh vendor/fetch.sh (or, where vendor/ holds submodules, \
             git submodule update --init --recursive)"
        );
    }
    println!("cargo:rerun-if-changed={}", header.display());

    let ggml_include = manifest.join("vendor/qwentts.cpp/ggml/include");

    // No backend toggles here on purpose. Those are ggml's, and ggml is not
    // built by this project — `build_qwentts` owns which backends exist, and
    // this links whatever it produced.
    let mut cfg = cmake::Config::new(&vendor);
    // Its own output tree. `cmake::Config` defaults to `OUT_DIR`, which
    // `build_qwentts` is also using — two CMake projects generating into one
    // directory clobber each other's cache and archives, and the symptom is a
    // missing `libvibevoice.a` rather than an error from either build.
    cfg.out_dir(PathBuf::from(env("OUT_DIR")).join("vibevoice"))
        .define("BUILD_SHARED_LIBS", "OFF")
        // Our fork's option. Without it this builds a second ggml.
        .define("VIBEVOICE_GGML_EXTERNAL", "ON")
        .define("VIBEVOICE_GGML_INCLUDE_DIR", &ggml_include)
        // Only the library — upstream's CLI and quantize tools are not part of
        // this binary.
        .define("VIBEVOICE_BUILD_EXAMPLES", "OFF")
        .define("VIBEVOICE_BUILD_TESTS", "OFF")
        .build_target("vibevoice")
        .profile("Release");

    let dst = cfg.build();
    let build = dst.join("build");

    // The invariant this whole arrangement exists for, checked rather than
    // trusted. If `VIBEVOICE_GGML_EXTERNAL` ever stops taking effect — an
    // option renamed upstream, a fork resynced past our patch — the build
    // would keep succeeding and quietly link two ggmls, resolving 534 shared
    // symbols by search order. Fail here instead, where the reason is legible.
    if let Ok(entries) = std::fs::read_dir(build.join("third_party/ggml/src")) {
        let stray: Vec<_> = entries
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("libggml"))
            .collect();
        if !stray.is_empty() {
            panic!(
                "vibevoice.cpp built its own ggml ({}). VIBEVOICE_GGML_EXTERNAL \
                 did not take effect, and linking this would silently run one \
                 engine against the other's ggml. Check that the vendored \
                 submodule is our fork's `bisque` branch.",
                stray.join(", ")
            );
        }
    }

    println!("cargo:rustc-link-search=native={}", build.display());
    println!(
        "cargo:rustc-link-search=native={}",
        build.join("Release").display()
    );
    // Ahead of the ggml archives `build_qwentts` names, since this depends on
    // them: static archives resolve left to right.
    println!("cargo:rustc-link-lib=static=vibevoice");

    generate_vibevoice_bindings(&header);
}

/// Generate the FFI declarations from `vibevoice.h`, restricted to the `vv_*`
/// surface for the same reason the qwentts binding is.
fn generate_vibevoice_bindings(header: &Path) {
    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .allowlist_function("vv_.*")
        .allowlist_type("vv_.*")
        .allowlist_var("VV_.*")
        .derive_default(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("generating bindings for vibevoice.h");

    bindings
        .write_to_file(PathBuf::from(env("OUT_DIR")).join("vibevoice_ffi.rs"))
        .expect("writing vibevoice_ffi.rs");
}

/// Compile `vendor/qwentts.cpp` into this binary and generate its bindings.
///
/// Everything is linked statically, ggml included: `bisque-voice` is installed by
/// unpacking a tarball onto a stranger's machine, so a dylib beside the binary
/// is one more thing to locate at runtime, sign, and get wrong.
fn build_qwentts() {
    let vendor = PathBuf::from(env("CARGO_MANIFEST_DIR")).join("vendor/qwentts.cpp");
    let header = vendor.join("src/qwen.h");
    if !header.exists() {
        panic!(
            "vendor/qwentts.cpp is empty — the vendored engines were not fetched.\n\
             Run: sh vendor/fetch.sh (or, where vendor/ holds submodules, \
             git submodule update --init --recursive)"
        );
    }
    println!("cargo:rerun-if-changed={}", header.display());

    // The TARGET's OS and arch, not the host's: `cfg!` in a build script is
    // evaluated for the machine running the build, which silently picks the
    // wrong backends the moment anyone runs `cargo build --target`. The Intel
    // macOS leg made this real — same OS as the arm64 leg, different arch,
    // different backend set.
    let target_os = env("CARGO_CFG_TARGET_OS");
    let target_arch = env("CARGO_CFG_TARGET_ARCH");
    // Metal is Apple-Silicon-only here. ggml's Metal kernels gate their real
    // work on `MTLGPUFamilyApple7`+ (simdgroup reduction/matmul), and no
    // Intel-Mac GPU is in any Apple family — on the machines the x86_64 build
    // serves (Intel iGPUs, macOS 13) the backend would initialize, claim the
    // graph, and then limp or fail per-op. CPU plus Accelerate BLAS is the
    // honest fast path there.
    let metal = target_os == "macos" && target_arch == "aarch64";

    let mut cfg = cmake::Config::new(&vendor);
    cfg.define("BUILD_SHARED_LIBS", "OFF")
        // Only the library is needed; upstream's four executables and the HTTP
        // server are not part of this binary.
        .build_target("qwen-core")
        .profile("Release");

    if target_os == "macos" {
        cfg.define("GGML_BLAS", "ON");
        // Same reason as the Linux arm below, and it bites here too: ggml turns
        // OpenMP on wherever the compiler supports it, and Apple clang does once
        // a Homebrew `libomp` for the target arch is on the machine. Nothing
        // links it, so the build fails with undefined `__kmpc_*` / `omp_*` —
        // and only on machines that happen to have it, which makes the failure
        // look like a broken checkout rather than a toolchain difference. Off
        // explicitly, so every Mac produces the same binary.
        cfg.define("GGML_OPENMP", "OFF");
        if metal {
            // Embed the Metal shader source in the archive. Without this, ggml
            // looks for `ggml-metal.metal` next to the executable at runtime
            // and silently falls back to CPU when a relocated binary cannot
            // find it.
            cfg.define("GGML_METAL", "ON")
                .define("GGML_METAL_EMBED_LIBRARY", "ON");
        } else {
            // ggml defaults GGML_METAL to ON for every Apple target, so the
            // Intel build must say no explicitly.
            cfg.define("GGML_METAL", "OFF");
        }
    } else if target_os == "linux" {
        // ggml's CMake turns OpenMP on wherever the compiler supports it, and
        // gcc does — but nothing here links libgomp, so the final link fails
        // with undefined GOMP_* symbols. Linking libgomp.so instead would hand
        // the installed binary a runtime dependency the tarball cannot
        // guarantee, which is exactly what the static-linking rule above
        // exists to prevent. With OpenMP off, ggml uses its own thread pool.
        cfg.define("GGML_OPENMP", "OFF");
    }

    let dst = cfg.build();
    // `build_target` skips the install step, so the archives stay in the CMake
    // build tree rather than landing in `<dst>/lib`.
    let build = dst.join("build");
    for dir in [
        "",
        "ggml/src",
        "ggml/src/ggml-metal",
        "ggml/src/ggml-blas",
        "ggml/src/ggml-cpu",
    ] {
        let base = if dir.is_empty() {
            build.clone()
        } else {
            build.join(dir)
        };
        println!("cargo:rustc-link-search=native={}", base.display());
        // MSVC drives a multi-config generator, which puts each archive in a
        // per-configuration subdirectory rather than beside the project files.
        println!("cargo:rustc-link-search=native={}", base.join("Release").display());
    }

    // Order matters for static archives: dependents before dependencies.
    // `yyjson` and `httplib` belong to upstream's `tts-server` target, not to
    // `qwen-core`, so they are deliberately absent here.
    println!("cargo:rustc-link-lib=static=qwen-core");
    for lib in ["ggml", "ggml-cpu", "ggml-base"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }

    if target_os == "macos" {
        if metal {
            println!("cargo:rustc-link-lib=static=ggml-metal");
        }
        println!("cargo:rustc-link-lib=static=ggml-blas");
        if metal {
            for framework in ["Metal", "MetalKit"] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
        }
        for framework in ["Foundation", "Accelerate", "CoreFoundation"] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        // MSVC links its own C++ runtime from the objects' directives; naming
        // one here is not just unnecessary, it is a library that does not exist
        // on that toolchain.
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    generate_bindings(&header);
}

/// Generate the FFI declarations from `qwen.h` itself.
///
/// Restricted to the `qt_*` surface so the 398-line public header does not drag
/// in every system type it happens to include.
fn generate_bindings(header: &Path) {
    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .allowlist_function("qt_.*")
        .allowlist_type("qt_.*")
        .allowlist_var("QT_ABI_.*")
        .derive_default(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("generating bindings for qwen.h");

    bindings
        .write_to_file(PathBuf::from(env("OUT_DIR")).join("qwentts_ffi.rs"))
        .expect("writing qwentts_ffi.rs");
}

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("{key} is not set"))
}
