//! Forced alignment — measured word timings, whatever produced the audio.
//!
//! A TTS engine either exposes text↔audio alignment or it doesn't. Kokoro does
//! (a duration tensor per phoneme); Qwen3-TTS does not, and its decode trace
//! carries no text position at all. Rather than let that be a permanent quality
//! ceiling for the larger engine, `Qwen3-ForcedAligner-0.6B` measures the
//! synthesized waveform against the transcript and returns a real `[start,
//! end]` per word.
//!
//! It aligns *any* audio against *any* transcript, so it lives here rather than
//! inside any one engine — it is a second, independent path to the same
//! guarantee, and a future engine gets it for free.
//!
//! Three properties this module holds, mirroring `engine.rs` for the same
//! reasons:
//!
//! - **Aligners are a registry, and there is no default.** Two ONNX exports of
//!   the same checkpoint exist at very different sizes with very different
//!   reported parity. Both are defensible choices; the registry states what is
//!   known about each and stops. Nothing here ranks them.
//! - **None installed is not an error.** Alignment is an upgrade, not a
//!   requirement. With no aligner present the engine's own timings are used and
//!   `exact` reports honestly what they are — the tool does not commit someone
//!   to a multi-gigabyte download on their behalf.
//! - **Fail closed, never approximate silently.** Once an aligner *is* chosen,
//!   every step that could disagree with the pack — the feature extractor's
//!   parameters, the special token ids, the number of timestamps the graph
//!   returned — is checked and fails loudly. A forced aligner that quietly
//!   emits plausible-looking wrong times is worse than none.
//!
//! The registry lives here; the implementation — feature extraction, prompt,
//! forward pass — lives in `align/run.rs` behind the `onnx` feature, because
//! it needs ONNX Runtime and this registry must survive on builds that have
//! none (the studio-only Intel-Mac build lists the aligners and says why it
//! cannot run them).

#[cfg(not(feature = "onnx"))]
use std::path::Path;

use anyhow::{bail, Result};
use serde::Serialize;

use crate::fetch::{Models, Pack};
#[cfg(not(feature = "onnx"))]
use crate::reconcile::Word;

#[cfg(feature = "onnx")]
mod run;
#[cfg(feature = "onnx")]
pub use run::Aligner;

/// What a build without ONNX Runtime compiles instead of `run::Aligner`.
/// Unreachable through `resolve`, which refuses every aligner on such a build
/// — but the call site in `main.rs` still has to link against something, and
/// an honest sentence beats an `unreachable!`.
#[cfg(not(feature = "onnx"))]
pub struct Aligner;

#[cfg(not(feature = "onnx"))]
pub struct Measured {
    pub words: Vec<Word>,
    pub tokens: Vec<crate::quality::TokenSpan>,
}

#[cfg(not(feature = "onnx"))]
impl Aligner {
    pub fn open(
        _pack: &Path,
        _threads: Option<usize>,
        _device: crate::device::Device,
    ) -> Result<Self> {
        bail!(
            "this build carries no forced aligner — the aligners run on ONNX \
             Runtime, which is not part of it. Word timings come from the engine \
             itself."
        )
    }

    pub fn measure(&mut self, _audio_24k: &[f32], _text: &str) -> Result<Measured> {
        bail!("this build carries no forced aligner")
    }
}

// ── the aligner registry ────────────────────────────────────────────────────

/// Both aligners are ONNX exports of one checkpoint, so a build compiled
/// without the `onnx` feature (the Intel-Mac release: ONNX Runtime publishes
/// no `x86_64-apple-darwin` build) can run neither. They stay listed, with
/// this as the reason — the same pattern `engine::KOKORO_UNSUPPORTED` uses.
#[cfg(not(feature = "onnx"))]
const ALIGNER_UNSUPPORTED: Option<&crate::fetch::Unsupported> =
    Some(&crate::fetch::Unsupported {
        code: "platform-unsupported",
        message: "not available on Intel Macs — the forced aligners run on \
                  the same runtime Kokoro does (ONNX Runtime), which dropped \
                  this platform",
    });
#[cfg(feature = "onnx")]
const ALIGNER_UNSUPPORTED: Option<&crate::fetch::Unsupported> = None;

/// Everything a person choosing needs, and everything `fetch` needs to install
/// it. Deliberately the same shape as `EngineSpec`: an aligner pack downloads,
/// verifies and installs by exactly the same path.
#[derive(Debug, Clone, Serialize)]
pub struct AlignerSpec {
    pub id: &'static str,
    #[serde(rename = "displayName")]
    pub display_name: &'static str,
    /// One line stating what is known about this export. Not a recommendation:
    /// see the module doc.
    pub summary: &'static str,
    /// Immutable pack name. Bump it when contents change; never edit in place,
    /// since installed clients resolve a pack by this name and verify its hash.
    #[serde(rename = "packVersion")]
    pub pack_version: &'static str,
    #[serde(rename = "downloadBytes")]
    pub download_bytes: u64,
    /// Why THIS BUILD cannot run this aligner, or `None` when it can. See
    /// `EngineSpec::unsupported` — same field, same reason, same JSON shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unsupported: Option<&'static crate::fetch::Unsupported>,
}

/// The full-precision export.
const QWEN3_FP32: AlignerSpec = AlignerSpec {
    id: "qwen3-fp32",
    display_name: "Qwen3 forced aligner (fp32)",
    summary: "Full-precision graph, 3.67 GB. Validated against PyTorch: \
              max_abs_diff 9.06e-05.",
    pack_version: "qwen3-aligner-fp32-1",
    download_bytes: 3_670_000_000,
    unsupported: ALIGNER_UNSUPPORTED,
};

/// The 4-bit export.
const QWEN3_Q4: AlignerSpec = AlignerSpec {
    id: "qwen3-q4",
    display_name: "Qwen3 forced aligner (q4)",
    summary: "4-bit quantized graph, 1.05 GB. Smoke-tested against PyTorch: \
              max_abs_diff 7.80.",
    pack_version: "qwen3-aligner-q4-1",
    download_bytes: 1_050_000_000,
    unsupported: ALIGNER_UNSUPPORTED,
};

/// Every aligner that can be installed today.
///
/// Order is a stable listing order and is NOT a ranking. The two entries are
/// exports of the same checkpoint; the numbers in their summaries are raw-logit
/// deviations, while this path consumes only `argmax` over those logits — so a
/// larger deviation does not straightforwardly mean worse timestamps, and the
/// measurement that would settle it (how often the two graphs pick a different
/// timestamp id over real narration) has not been made. "Unmeasured" is the
/// honest label for `qwen3-q4`, not "worse".
pub const REGISTRY: &[AlignerSpec] = &[QWEN3_FP32, QWEN3_Q4];

/// `--align none` — use the engine's own timings even though an aligner is
/// installed. Reserved rather than parsed as an id; no registry entry may
/// claim it.
pub const NONE: &str = "none";

pub fn spec(id: &str) -> Option<&'static AlignerSpec> {
    REGISTRY.iter().find(|a| a.id == id)
}

/// The compiled-in SHA-256 for an aligner's pack, or empty in a dev build.
/// `build.rs` reads `BISQUE_VOICE_PACK_SHA256_<ID>` for each registry entry, with
/// the id uppercased and `-` replaced by `_`.
pub fn pack_sha256(id: &str) -> &'static str {
    match id {
        "qwen3-fp32" => env!("BISQUE_VOICE_PACK_SHA256_QWEN3_FP32"),
        "qwen3-q4" => env!("BISQUE_VOICE_PACK_SHA256_QWEN3_Q4"),
        _ => "",
    }
}

impl Pack for AlignerSpec {
    fn id(&self) -> &'static str {
        self.id
    }
    fn display_name(&self) -> &'static str {
        self.display_name
    }
    fn pack_version(&self) -> &'static str {
        self.pack_version
    }
    fn download_bytes(&self) -> u64 {
        self.download_bytes
    }
    /// Both entries are exports of the same 0.6B checkpoint; what differs is
    /// the numeric precision, which the summary and the download size already
    /// say.
    fn parameters(&self) -> &'static str {
        "0.6B"
    }
    fn expected_sha256(&self) -> &'static str {
        pack_sha256(self.id)
    }
    fn unsupported(&self) -> Option<&'static crate::fetch::Unsupported> {
        self.unsupported
    }
}

#[derive(Debug, Serialize)]
pub struct AlignerStatus {
    #[serde(flatten)]
    pub spec: AlignerSpec,
    pub installed: bool,
    /// See `EngineStatus::installable`.
    pub installable: bool,
    pub path: String,
}

pub fn statuses(models: &Models) -> Vec<AlignerStatus> {
    REGISTRY
        .iter()
        .map(|spec| AlignerStatus {
            spec: spec.clone(),
            installed: models.is_installed(spec),
            installable: crate::fetch::is_installable(spec),
            path: models.pack_dir(spec).display().to_string(),
        })
        .collect()
}

pub fn installed(models: &Models) -> Vec<&'static AlignerSpec> {
    REGISTRY
        .iter()
        // Same rule as `engine::installed`: a pack on disk that this build
        // cannot run must not resolve.
        .filter(|s| s.unsupported.is_none() && models.is_installed(*s))
        .collect()
}

/// Decide which aligner to run, without ever choosing *for* the user.
///
/// The one deliberate difference from `engine::resolve`: **none installed is
/// `Ok(None)`, not an error.** An engine is required to synthesize anything; an
/// aligner only sharpens timings that already exist, so its absence degrades to
/// the engine's own timings (reported as approximate) rather than failing.
pub fn resolve(models: &Models, explicit: Option<&str>) -> Result<Option<&'static AlignerSpec>> {
    if let Some(id) = explicit {
        if id == NONE {
            return Ok(None);
        }
        let spec = spec(id).ok_or_else(|| unknown_aligner(id))?;
        // Platform before install state, same as `engine::resolve`: the user
        // asked for measured timings this build cannot produce, and the
        // honest answer is why — plus the fallback they still have.
        if let Some(unsupported) = spec.unsupported {
            bail!(
                "aligner {id:?} is {}. Word timings come from the engine \
                 itself (reported as approximate).",
                unsupported.message
            );
        }
        if !models.is_installed(spec) {
            bail!("aligner {id:?} is not installed yet — run: bisque-voice install {id}");
        }
        return Ok(Some(spec));
    }

    let installed = installed(models);
    match installed.len() {
        // Not an error: this is the documented fallback path.
        0 => Ok(None),
        // Exactly one installed is not a default — it is the choice already made.
        1 => Ok(Some(installed[0])),
        _ => bail!(
            "several aligners are installed ({}). Say which to use with --align, \
             or --align none to use the engine's own timings.",
            installed
                .iter()
                .map(|s| s.id)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn unknown_aligner(id: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "unknown aligner {id:?}. Known aligners: {} (or {NONE:?} for the engine's own timings)",
        REGISTRY.iter().map(|a| a.id).collect::<Vec<_>>().join(", ")
    )
}
#[cfg(test)]
mod tests {
    use super::*;

    // ── registry ────────────────────────────────────────────────────────────

    #[test]
    fn every_registry_entry_is_addressable_and_complete() {
        assert!(!REGISTRY.is_empty());
        for a in REGISTRY {
            assert!(spec(a.id).is_some(), "{} not resolvable by id", a.id);
            assert!(!a.summary.is_empty(), "{} has no summary", a.id);
            assert!(a.download_bytes > 0, "{} has no download size", a.id);
            assert!(
                !a.pack_version.is_empty(),
                "{} has no pack version",
                a.id
            );
        }
    }

    #[test]
    fn aligner_ids_and_pack_versions_are_unique() {
        for mut v in [
            REGISTRY.iter().map(|a| a.id).collect::<Vec<_>>(),
            REGISTRY.iter().map(|a| a.pack_version).collect::<Vec<_>>(),
        ] {
            v.sort_unstable();
            let before = v.len();
            v.dedup();
            assert_eq!(v.len(), before, "duplicate entry in the aligner registry");
        }
    }

    /// `none` is how a user asks for the engine's own timings; a registry entry
    /// claiming that id would make the escape hatch unreachable.
    #[test]
    fn no_aligner_claims_the_none_sentinel() {
        assert!(spec(NONE).is_none());
    }

    /// The registry describes; it does not steer. Both entries are exports of
    /// one checkpoint and nobody has measured which picks better timestamps.
    #[test]
    fn summaries_do_not_recommend() {
        for a in REGISTRY {
            let lower = a.summary.to_lowercase();
            for banned in [
                "recommend",
                "preferred",
                "best",
                "better",
                "worse",
                "should",
                "instead",
            ] {
                assert!(
                    !lower.contains(banned),
                    "{} steers with {banned:?}: {}",
                    a.id,
                    a.summary
                );
            }
        }
    }

    #[test]
    fn unknown_aligner_error_lists_the_real_ones() {
        let msg = unknown_aligner("qwen3").to_string();
        for a in REGISTRY {
            assert!(msg.contains(a.id), "error should name {}", a.id);
        }
    }

    /// Points a `Models` at an empty temporary directory rather than setting
    /// `BISQUE_VOICE_MODEL_DIR`: the process environment is shared by every test
    /// thread, so an override installed here would decide what unrelated tests
    /// see.
    #[test]
    fn nothing_installed_falls_back_rather_than_failing() {
        let dir = tempfile::tempdir().unwrap();
        let models = Models::from_vars(Some(dir.path().to_path_buf()), None).unwrap();
        assert!(
            matches!(resolve(&models, None), Ok(None)),
            "no aligner installed must degrade to the engine's timings, not error"
        );
        assert!(matches!(resolve(&models, Some(NONE)), Ok(None)));
        assert!(resolve(&models, Some("nonsense")).is_err());
    }

    /// Per-engine platform support: a build without ONNX Runtime (the
    /// Intel-Mac release) lists both aligners with the platform reason,
    /// offers neither, and refuses `--align <id>` with that same reason —
    /// while degrading gracefully to engine timings when none is named.
    #[test]
    fn aligners_report_why_when_this_build_cannot_run_them() {
        if cfg!(feature = "onnx") {
            for a in REGISTRY {
                assert!(a.unsupported.is_none(), "{}", a.id);
            }
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let models = Models::from_vars(Some(dir.path().to_path_buf()), None).unwrap();
        for a in REGISTRY {
            let unsupported = a.unsupported.expect("must say why");
            assert_eq!(unsupported.code, "platform-unsupported");
            assert!(
                !crate::fetch::is_installable(a),
                "{} must not be offered",
                a.id
            );
            let err = resolve(&models, Some(a.id)).unwrap_err().to_string();
            assert!(err.contains(unsupported.message), "{err}");
        }
        // No aligner named is still the documented fallback, not an error.
        assert!(matches!(resolve(&models, None), Ok(None)));
    }
}
