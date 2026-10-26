//! The engine registry.
//!
//! `bisque-voice` is a host for local speech models, not a wrapper around one.
//! Which model to run is **the user's call**, made once at onboarding from the
//! real trade-offs — download size, voice count, which languages — and more
//! than one can be installed at a time. New models get added here and old ones
//! keep working; nothing in the codebase may assume a particular engine is
//! "the" engine.
//!
//! Two consequences worth stating, because they are easy to erode:
//!
//! - **There is no default.** With nothing installed, synthesis fails and tells
//!   the caller to choose. Silently installing one would be this tool making
//!   the decision on the user's behalf.
//! - **The word-timing guarantee lives above the trait.** `Engine::narrate`
//!   returns audio and timings; the caller reconciles them onto source tokens
//!   and fails closed. An engine cannot opt out of that by being new.

use std::path::PathBuf;

use anyhow::{bail, Result};
use serde::Serialize;

use crate::fetch::Models;
use crate::kokoro_voices::VoiceSpec;
use crate::reconcile::Word;

/// A language an engine can speak. BCP-47 plus a name for onboarding to print.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Language {
    pub tag: &'static str,
    pub name: &'static str,
}

const fn lang(tag: &'static str, name: &'static str) -> Language {
    Language { tag, name }
}

/// Everything onboarding needs to describe a choice, and everything `fetch`
/// needs to install it.
#[derive(Debug, Clone, Serialize)]
pub struct EngineSpec {
    pub id: &'static str,
    #[serde(rename = "displayName")]
    pub display_name: &'static str,
    /// One line, written for a person choosing — not a model card.
    pub summary: &'static str,
    /// Immutable pack name. Bump it when contents change; never edit in place,
    /// since installed clients resolve a pack by this name and verify its hash.
    #[serde(rename = "packVersion")]
    pub pack_version: &'static str,
    #[serde(rename = "downloadBytes")]
    pub download_bytes: u64,
    pub languages: &'static [Language],
    #[serde(rename = "voiceCount")]
    pub voice_count: usize,
    /// Parameter count, for the size/quality trade-off onboarding presents.
    pub parameters: &'static str,
    /// Why THIS BUILD cannot run this engine, or `None` when it can. Platform
    /// support is per-engine: a build compiled without the runtime an engine
    /// needs still lists it — an entry vanishing between releases reads as
    /// the engine being gone — but says why it cannot install. Serialized
    /// into `engines --json` so onboarding can relay the reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unsupported: Option<&'static crate::fetch::Unsupported>,
    /// The forced aligner this engine needs for trustworthy output — part of
    /// the engine's contract, not tribal knowledge for clients. The qwentts
    /// engines emit frames, not word boundaries, and can open with stochastic
    /// non-speech the synthesis path only trims when measured timings exist;
    /// without the companion, cues drift. `install <engine>` resolves it
    /// automatically, and this field is serialized so clients can display
    /// the total download honestly.
    #[serde(rename = "companionAligner", skip_serializing_if = "Option::is_none")]
    pub companion_aligner: Option<&'static str>,
    /// Per-voice quality metadata, for engines whose voices ship in the pack
    /// (`kokoro_voices`). Empty for the cloning engines — their voices are
    /// created by the user. Serialized recommended-first with `listed: false`
    /// voices excluded entirely; the metadata informs choice and never gates
    /// synthesis (see `VOICE-CURATION.md`).
    #[serde(
        serialize_with = "serialize_listed_voices",
        skip_serializing_if = "has_no_listed_voices"
    )]
    pub voices: &'static [VoiceSpec],
}

/// `engines --json` emits only listed voices — an unlisted voice appears in
/// no enumeration, though it still synthesizes when named explicitly. The
/// table is already ordered recommended-first, so a filter preserves that.
fn serialize_listed_voices<S: serde::Serializer>(
    voices: &&'static [VoiceSpec],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(voices.iter().filter(|v| v.listed))
}

fn has_no_listed_voices(voices: &&'static [VoiceSpec]) -> bool {
    !voices.iter().any(|v| v.listed)
}

/// Kokoro runs on ONNX Runtime, which publishes no build for Intel macOS
/// (1.28 dropped `x86_64-apple-darwin`) — the reason the `kokoro` Cargo
/// feature exists at all. A build compiled without it keeps the registry
/// entry and reports this instead, so `engines` can say why and `install`
/// can refuse with the same sentence.
#[cfg(not(feature = "kokoro"))]
const KOKORO_UNSUPPORTED: Option<&crate::fetch::Unsupported> =
    Some(&crate::fetch::Unsupported {
        code: "platform-unsupported",
        message: "not available on Intel Macs — its runtime (ONNX Runtime) \
                  dropped this platform",
    });
#[cfg(feature = "kokoro")]
const KOKORO_UNSUPPORTED: Option<&crate::fetch::Unsupported> = None;

/// Kokoro-82M. Small, fast, and multilingual.
///
/// The weights always spoke nine languages and the pack always held 55 voices;
/// what was missing was the G2P to feed them, which is per-language and lives
/// in `g2p`. Each language's pronunciation data is a separate small download
/// (`g2p::pack`) fetched the first time a voice of that language is used, so
/// nobody pays for Japanese to narrate English.
const KOKORO: EngineSpec = EngineSpec {
    id: "kokoro",
    display_name: "Kokoro 82M",
    summary: "Small and fast. Nine languages, exact word timings. Runs \
              comfortably on a laptop CPU.",
    // fp32. The fp16 export's vocoder overflows to NaN for some (voice, text)
    // pairs — deterministic silent slides in real output.
    // fp32 is the reference precision, synthesizes every known-failing pair,
    // and measured the same speed on CPU; the only cost is download size.
    pack_version: "kokoro-v1.0-fp32-1",
    download_bytes: 337_000_000,
    languages: &[
        lang("en-US", "English (US)"),
        lang("en-GB", "English (UK)"),
        lang("es", "Spanish"),
        lang("fr-FR", "French"),
        lang("hi", "Hindi"),
        lang("it", "Italian"),
        lang("pt-BR", "Portuguese (Brazil)"),
        lang("zh-CN", "Chinese"),
        lang("ja-JP", "Japanese"),
    ],
    voice_count: 55,
    parameters: "82M",
    unsupported: KOKORO_UNSUPPORTED,
    // Kokoro's duration predictor gives exact timings natively.
    companion_aligner: None,
    voices: crate::kokoro_voices::VOICES,
};

/// What both cloning engines speak. Shared because they are the same
/// checkpoint at two sizes, and a language listed for one but not the other
/// would be a typo rather than a fact.
///
/// This list is what was actually synthesized and transcribed back, not what
/// the model card claims.
const CLONE_LANGUAGES: &[Language] = &[
    lang("en-US", "English"),
    lang("es", "Spanish"),
    lang("de-DE", "German"),
    lang("fr-FR", "French"),
    lang("it", "Italian"),
    lang("pt", "Portuguese"),
    lang("ru-RU", "Russian"),
    lang("zh-CN", "Chinese"),
    lang("ja-JP", "Japanese"),
    lang("ko-KR", "Korean"),
];

/// Qwen3-TTS Base, on qwentts.cpp. The engines that clone a voice.
///
/// These exist for the one thing Kokoro cannot do at any download size: speak
/// in a voice given as a recording. Cloning a voice is a separate, once-per-
/// voice step (`bisque-voice clone`) that costs a few seconds and produces ~13 KB
/// of latents; narration then reuses those and never touches the clip again.
///
/// **Ten languages, measured.** Every one was synthesized and transcribed back:
/// in the model's own voice the result is verbatim or near-verbatim in all ten,
/// and nothing runs away to the frame cap — the failure that sank the ONNX
/// Qwen3 path does not reproduce here.
///
/// **Cloning across languages costs accuracy.** A voice cloned from an English
/// reference and then asked to speak another language picks up occasional
/// word-level errors that the same engine's default voice does not make —
/// clearest in Japanese, Korean and Chinese, absent in Italian and Russian.
/// Cloning and narrating in the *same* language is verbatim. Kokoro remains the
/// engine for plain multilingual narration; it has exact timings and is a
/// seventh of the download. Word timings here are approximate — install a
/// forced aligner to measure them.
///
/// The two sizes are a real choice rather than a big/small pair. Both share the
/// codec, which dominates synthesis, so 0.6B is only ~10% faster end to end
/// (RTF 1.01 vs 1.09 measured); what it actually buys is 1.1 GB of download and
/// ~0.8 GB of resident memory.
const QWEN3_CLONE_06B: EngineSpec = EngineSpec {
    id: "qwen3-clone-06b",
    display_name: "Qwen3-TTS 0.6B (voice cloning)",
    summary: "Clones a voice from a short recording. Ten languages, approximate \
              word timings.",
    pack_version: "qwen3-clone-06b-q8-1",
    download_bytes: 1_284_000_000,
    languages: CLONE_LANGUAGES,
    // The model's own voice. Cloned voices are created by the user and are
    // unlimited, so they cannot be counted here.
    voice_count: 1,
    parameters: "0.6B",
    // ggml via the vendored qwentts.cpp — builds and runs on every platform
    // the binary ships for, Intel Macs included.
    unsupported: None,
    companion_aligner: Some("qwen3-q4"),
    voices: &[],
};

/// The larger talker. Same codec, same cloning, more model.
const QWEN3_CLONE_17B: EngineSpec = EngineSpec {
    id: "qwen3-clone-17b",
    display_name: "Qwen3-TTS 1.7B (voice cloning)",
    summary: "Clones a voice from a short recording. Ten languages. Larger \
              model, 2.4 GB, ~5.7 GB while running.",
    pack_version: "qwen3-clone-17b-q8-1",
    download_bytes: 2_371_000_000,
    languages: CLONE_LANGUAGES,
    voice_count: 1,
    parameters: "1.7B",
    unsupported: None,
    companion_aligner: Some("qwen3-q4"),
    voices: &[],
};

/// Qwen3-TTS CustomVoice, on qwentts.cpp. The engines with named speakers.
///
/// Same checkpoint family and the same codec as the cloning engines, but a
/// different talker: `customvoice` carries a nine-speaker table baked into
/// the GGUF, and the mode split is upstream's — `base` clones from a
/// reference clip, `customvoice` speaks as a named speaker, and the two are
/// different weights. That is why these are **separate engine ids rather
/// than a second pack on the `qwen3-clone-*` engines**: an engine id owns
/// exactly one immutable `pack_version` (the `fetch::Pack` contract —
/// `expected_sha256` is compiled in per id, `is_installable` derives from
/// it, and `Models::pack_dir` resolves by it), so a second checkpoint under
/// the same id would have nowhere to put its pack or its checksum.
///
/// The language list is shared with the cloning engines: the base-checkpoint
/// sibling measured verbatim-or-near-verbatim in all ten in its own voice,
/// and the named speakers *are* the model's own voices —
/// there is no cross-lingual cloning penalty here, because nothing is cloned.
/// `eric` and `dylan` are Chinese dialect voices: when the text's language is
/// Chinese (or unstated), the model swaps in the dialect's own language id.
const QWEN3_VOICES_06B: EngineSpec = EngineSpec {
    id: "qwen3-voices-06b",
    display_name: "Qwen3-TTS 0.6B (named voices)",
    summary: "Nine ready-made voices, no recording needed. Ten languages, \
              approximate word timings.",
    pack_version: "qwen3-voices-06b-q8-1",
    download_bytes: 1_260_000_000,
    languages: CLONE_LANGUAGES,
    voice_count: 9,
    parameters: "0.6B",
    unsupported: None,
    companion_aligner: Some("qwen3-q4"),
    voices: crate::qwen3_voices::VOICES,
};

/// The larger CustomVoice talker. Same speakers, same codec, more model.
const QWEN3_VOICES_17B: EngineSpec = EngineSpec {
    id: "qwen3-voices-17b",
    display_name: "Qwen3-TTS 1.7B (named voices)",
    summary: "Nine ready-made voices, no recording needed. Ten languages. \
              Larger model, 2.3 GB, ~5.7 GB while running.",
    pack_version: "qwen3-voices-17b-q8-1",
    download_bytes: 2_334_000_000,
    languages: CLONE_LANGUAGES,
    voice_count: 9,
    parameters: "1.7B",
    unsupported: None,
    companion_aligner: Some("qwen3-q4"),
    voices: crate::qwen3_voices::VOICES,
};

/// Every engine that can be installed today. Adding one is an entry here plus
/// an `Engine` implementation — deliberately the whole cost of supporting a
/// new model.
///
/// Order is presentation order for onboarding, and is NOT a recommendation:
/// the tool does not have an opinion about which model the user should run.
pub const REGISTRY: &[EngineSpec] = &[
    KOKORO,
    QWEN3_CLONE_06B,
    QWEN3_CLONE_17B,
    QWEN3_VOICES_06B,
    QWEN3_VOICES_17B,
];

/// Whether this engine clones voices, i.e. whether `bisque-voice clone` and the
/// voice library apply to it. Asked as a capability rather than by matching on
/// ids at each call site, so adding a third cloning engine is one line here.
pub fn clones_voices(id: &str) -> bool {
    matches!(id, "qwen3-clone-06b" | "qwen3-clone-17b")
}

/// Whether this engine's voices are the named speakers baked into its
/// CustomVoice talker. Mutually exclusive with `clones_voices` — upstream's
/// checkpoints are per-mode, and the qwentts params reject a speaker and a
/// reference together.
pub fn named_speakers(id: &str) -> bool {
    matches!(id, "qwen3-voices-06b" | "qwen3-voices-17b")
}

/// The talker GGUF inside a qwentts-backed engine's pack. The codec is shared
/// and named by `qwentts::CODEC_FILE`.
pub fn talker_file(id: &str) -> Option<&'static str> {
    match id {
        "qwen3-clone-06b" => Some("qwen-talker-0.6b-base-Q8_0.gguf"),
        "qwen3-clone-17b" => Some("qwen-talker-1.7b-base-Q8_0.gguf"),
        "qwen3-voices-06b" => Some("qwen-talker-0.6b-customvoice-Q8_0.gguf"),
        "qwen3-voices-17b" => Some("qwen-talker-1.7b-customvoice-Q8_0.gguf"),
        _ => None,
    }
}

pub fn spec(id: &str) -> Option<&'static EngineSpec> {
    REGISTRY.iter().find(|e| e.id == id)
}

/// An engine pack downloads and installs exactly like any other pack; see
/// `fetch::Pack` for why that is one implementation and not two.
impl crate::fetch::Pack for EngineSpec {
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
    fn parameters(&self) -> &'static str {
        self.parameters
    }
    fn expected_sha256(&self) -> &'static str {
        pack_sha256(self.id)
    }
    fn unsupported(&self) -> Option<&'static crate::fetch::Unsupported> {
        self.unsupported
    }
}

/// The compiled-in SHA-256 for an engine's pack, or empty in a dev build.
/// `build.rs` reads `BISQUE_VOICE_PACK_SHA256_<ID>` for each registry entry.
pub fn pack_sha256(id: &str) -> &'static str {
    match id {
        "kokoro" => env!("BISQUE_VOICE_PACK_SHA256_KOKORO"),
        "qwen3-clone-06b" => env!("BISQUE_VOICE_PACK_SHA256_QWEN3_CLONE_06B"),
        "qwen3-clone-17b" => env!("BISQUE_VOICE_PACK_SHA256_QWEN3_CLONE_17B"),
        "qwen3-voices-06b" => env!("BISQUE_VOICE_PACK_SHA256_QWEN3_VOICES_06B"),
        "qwen3-voices-17b" => env!("BISQUE_VOICE_PACK_SHA256_QWEN3_VOICES_17B"),
        _ => "",
    }
}

#[derive(Debug, Serialize)]
pub struct EngineStatus {
    #[serde(flatten)]
    pub spec: EngineSpec,
    pub installed: bool,
    /// Whether this build can fetch this engine at all — see
    /// `fetch::is_installable`. Onboarding must not offer an engine whose
    /// pack this release never published.
    pub installable: bool,
    pub path: String,
}

pub fn statuses(models: &Models) -> Vec<EngineStatus> {
    REGISTRY
        .iter()
        .map(|spec| EngineStatus {
            spec: spec.clone(),
            installed: models.is_installed(spec),
            installable: crate::fetch::is_installable(spec),
            path: models.pack_dir(spec).display().to_string(),
        })
        .collect()
}

pub fn installed(models: &Models) -> Vec<&'static EngineSpec> {
    REGISTRY
        .iter()
        // A pack sitting on disk that this build cannot run (installed by a
        // build that could) must not resolve, or `open` would fail later with
        // a message about implementations instead of the real, platform-level
        // reason.
        .filter(|s| s.unsupported.is_none() && models.is_installed(*s))
        .collect()
}

/// A voice reference is `engine:voice` ("kokoro:af_heart"), or a bare voice id
/// when only one installed engine could own it.
pub fn split_voice(voice: &str) -> (Option<&str>, &str) {
    match voice.split_once(':') {
        Some((engine, rest)) if spec(engine).is_some() => (Some(engine), rest),
        _ => (None, voice),
    }
}

/// Decide which engine to run, without ever choosing *for* the user.
pub fn resolve(
    models: &Models,
    explicit: Option<&str>,
    voice: &str,
) -> Result<&'static EngineSpec> {
    if let Some(id) = explicit {
        let spec = spec(id).ok_or_else(|| unknown_engine(id))?;
        // Platform before install state: asking for an engine this build
        // cannot run gets the reason, not an install command that would
        // refuse with the same reason one step later.
        if let Some(unsupported) = spec.unsupported {
            bail!("engine {id:?} is {}", unsupported.message);
        }
        if !models.is_installed(spec) {
            // "install it" is only useful advice if this build could. An engine
            // in the registry that this release did not ship needs a different
            // sentence, not a command that will refuse.
            if !crate::fetch::is_installable(spec) {
                bail!(
                    "engine {id:?} is in the registry but this build did not \
                     ship it, so there is nothing to install. Run \
                     `bisque-voice engines` to see what this build can install."
                );
            }
            bail!("engine {id:?} is not installed yet — run: bisque-voice install {id}");
        }
        return Ok(spec);
    }
    if let (Some(id), _) = split_voice(voice) {
        return resolve(models, Some(id), voice);
    }

    let installed = installed(models);
    match installed.len() {
        0 => bail!(
            "no speech engine is installed yet.\n\
             Choose one (or more) with `bisque-voice install <engine>`; \
             `bisque-voice engines` lists them with sizes and languages."
        ),
        // Exactly one installed is not a default — it is the choice already made.
        1 => Ok(installed[0]),
        _ => bail!(
            "several engines are installed ({}). Say which to use with --engine, \
             or qualify the voice as <engine>:<voice>.",
            installed
                .iter()
                .map(|s| s.id)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn unknown_engine(id: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "unknown engine {id:?}. Known engines: {}",
        REGISTRY.iter().map(|e| e.id).collect::<Vec<_>>().join(", ")
    )
}

/// What an engine hands back. `words` are the engine's own timings, one per
/// token it recognized; the caller reconciles them onto source whitespace
/// tokens and fails closed, so no engine can weaken that guarantee.
pub struct Narration {
    pub samples: Vec<f32>,
    pub words: Vec<Word>,
    /// Forward passes; >1 means the text was chunked to fit the model.
    pub chunks: usize,
}

pub trait Engine {
    /// Text in, audio and timings out. Everything below this line —
    /// phonemization, chunking, forced alignment — is the engine's business.
    /// `language` is the spoken language, when the caller knows it. An
    /// end-to-end multilingual model conditions its prefill on this, and
    /// leaving it unset is not neutral — it selects a "no language stated"
    /// branch that empirically makes non-English generation run away without
    /// ever ending the utterance. Engines that speak one language ignore it.
    ///
    /// `gate` judges each synthesized chunk (`quality.rs`). An engine that
    /// samples its speech retakes a chunk the gate rejects; a deterministic
    /// one ignores it, since a retake would return the same audio.
    fn narrate(
        &mut self,
        text: &str,
        voice: &str,
        speed: f32,
        language: Option<&str>,
        gate: &mut dyn crate::quality::Gate,
    ) -> Result<Narration>;

    /// `(surface text, what the engine will actually say)` per token, for
    /// building a cross-platform pronunciation fixture. Empty for engines with
    /// no inspectable intermediate — an end-to-end model has nothing to show.
    ///
    /// Takes the voice because pronunciation is not a property of the text
    /// alone: Kokoro's voice id carries its language, and the same characters
    /// go through a different G2P for `af_heart` than for `zf_xiaobei`. Takes
    /// `&mut self` because that G2P may have to be loaded to answer.
    fn debug_tokens(&mut self, _text: &str, _voice: &str) -> Vec<(String, String)> {
        Vec::new()
    }

    /// Words this engine could have got wrong, with what it will say for each.
    ///
    /// Returns `Unavailable` for an engine with no G2P to interrogate — the
    /// same reason `debug_tokens` is empty there. The variant is what keeps
    /// "cannot tell" from being printed as "nothing wrong"; returning an empty
    /// list here would say the text is clean, which this engine cannot know.
    fn pronunciation_report(&mut self, _text: &str, _voice: &str) -> Result<crate::check::Report> {
        Ok(crate::check::Report::Unavailable(
            "this engine has no letters-to-sounds stage to inspect".to_string(),
        ))
    }
}

/// Build the engine named by `spec` against an installed pack.
pub fn open(
    spec: &EngineSpec,
    pack: PathBuf,
    threads: Option<usize>,
    device: crate::device::Device,
    models_root: &std::path::Path,
) -> Result<Box<dyn Engine>> {
    // The Studio engines take neither knob; without the Kokoro arm below the
    // parameters would otherwise read as unused.
    #[cfg(not(feature = "kokoro"))]
    let _ = (threads, device);
    match spec.id {
        #[cfg(feature = "kokoro")]
        "kokoro" => Ok(Box::new(crate::model::KokoroEngine::load(
            &pack,
            threads,
            device,
            models_root,
        )?)),
        // One implementation for both qwentts modes: the struct carries which
        // conditioning its talker takes (a cloned voice's latents, or a named
        // speaker from the checkpoint's own table).
        id if clones_voices(id) || named_speakers(id) => {
            let talker = talker_file(id)
                .ok_or_else(|| anyhow::anyhow!("engine {id:?} declares no talker weights"))?;
            Ok(Box::new(crate::qwentts::QwenCloneEngine::load(
                &pack,
                id,
                models_root,
                talker,
            )?))
        }
        other => bail!("engine {other:?} has no implementation in this build"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registry_entry_is_addressable_and_complete() {
        assert!(!REGISTRY.is_empty());
        for e in REGISTRY {
            assert!(spec(e.id).is_some(), "{} not resolvable by id", e.id);
            assert!(!e.languages.is_empty(), "{} lists no languages", e.id);
            assert!(e.voice_count > 0, "{} lists no voices", e.id);
            assert!(e.download_bytes > 0, "{} has no download size", e.id);
            assert!(!e.summary.is_empty(), "{} has no summary", e.id);
        }
    }

    /// Per-engine platform support: a Kokoro-less build (the Intel-Mac
    /// release) still lists kokoro but reports it unavailable with a
    /// machine-readable code and the human reason, and refuses to resolve it.
    /// A default build reports nothing of the kind — behavior is unchanged.
    #[test]
    fn kokoro_reports_why_when_this_build_cannot_run_it() {
        let kokoro = spec("kokoro").expect("kokoro stays in the registry");
        if cfg!(feature = "kokoro") {
            assert!(kokoro.unsupported.is_none());
        } else {
            let unsupported = kokoro
                .unsupported
                .expect("a Kokoro-less build must say why");
            assert_eq!(unsupported.code, "platform-unsupported");
            assert!(unsupported.message.contains("Intel Macs"), "{}", unsupported.message);
            assert!(
                !crate::fetch::is_installable(kokoro),
                "an engine the build cannot run must not be offered"
            );
            // `--engine kokoro` gets the reason, not an install suggestion.
            let dir = tempfile::tempdir().unwrap();
            let models =
                Models::from_vars(Some(dir.path().to_path_buf()), None).unwrap();
            let err = resolve(&models, Some("kokoro"), "")
                .expect_err("resolving an unsupported engine must fail")
                .to_string();
            assert!(err.contains(unsupported.message), "{err}");
        }
        // The Studio engines are the reason an Intel binary exists at all.
        for id in [
            "qwen3-clone-06b",
            "qwen3-clone-17b",
            "qwen3-voices-06b",
            "qwen3-voices-17b",
        ] {
            assert!(spec(id).unwrap().unsupported.is_none(), "{id}");
        }
    }

    #[test]
    fn engine_ids_are_unique() {
        let mut ids: Vec<&str> = REGISTRY.iter().map(|e| e.id).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), before, "duplicate engine id in the registry");
    }

    #[test]
    fn pack_versions_are_unique_so_installs_cannot_collide() {
        let mut packs: Vec<&str> = REGISTRY.iter().map(|e| e.pack_version).collect();
        packs.sort_unstable();
        let before = packs.len();
        packs.dedup();
        assert_eq!(packs.len(), before, "two engines share a pack version");
    }

    #[test]
    fn a_qualified_voice_names_its_engine() {
        assert_eq!(split_voice("kokoro:af_heart"), (Some("kokoro"), "af_heart"));
        assert_eq!(split_voice("af_heart"), (None, "af_heart"));
        // An unknown prefix is not an engine — treat the whole thing as a voice
        // rather than guessing.
        assert_eq!(split_voice("nonsense:x"), (None, "nonsense:x"));
    }

    /// The `engines --json` voice contract: kokoro's entry carries the graded
    /// voice list, recommended-first with the default leading, each voice with
    /// its grade — and the unlisted voice is not in it, in any form.
    #[test]
    fn engines_json_lists_graded_voices_recommended_first_without_unlisted() {
        let kokoro = spec("kokoro").expect("kokoro is registered");
        let json = serde_json::to_value(kokoro).expect("serialize");
        let voices = json["voices"].as_array().expect("kokoro serializes voices");
        assert_eq!(voices.len(), 27, "28 graded voices minus the unlisted one");
        assert_eq!(voices[0]["id"], "af_heart");
        assert_eq!(voices[0]["grade"], "A");
        assert_eq!(voices[0]["recommended"], true);
        assert_eq!(voices[0]["note"], "default");
        let first_other = voices
            .iter()
            .position(|v| v["recommended"] == false)
            .expect("non-recommended voices are listed too");
        assert!(
            voices[first_other..]
                .iter()
                .all(|v| v["recommended"] == false),
            "recommended voices must all come first"
        );
        for v in voices {
            assert!(v["grade"].is_string(), "every voice carries its grade");
        }
        // Not merely filtered from `voices` — absent from the whole document.
        assert!(
            !serde_json::to_string(kokoro).unwrap().contains("af_nicole"),
            "an unlisted voice appears in no listing"
        );
        // The cloning engines have no shipped voices to grade, and say nothing.
        let studio = serde_json::to_value(spec("qwen3-clone-17b").unwrap()).unwrap();
        assert!(studio.get("voices").is_none());
    }

    /// The named-speaker engines carry the nine CustomVoice speakers in the
    /// same serialized shape as Kokoro's list — every voice a string grade
    /// ("—": upstream publishes none) and a recommended flag, all nine listed.
    /// `voice_count` must agree with the list, since onboarding prints one and
    /// enumerates the other.
    #[test]
    fn named_speaker_engines_list_all_nine_speakers() {
        for id in ["qwen3-voices-06b", "qwen3-voices-17b"] {
            let spec = spec(id).unwrap();
            let json = serde_json::to_value(spec).expect("serialize");
            let voices = json["voices"].as_array().expect("voices serialize");
            assert_eq!(voices.len(), 9, "{id}");
            assert_eq!(voices.len(), spec.voice_count, "{id}");
            for v in voices {
                assert_eq!(v["grade"], "—", "no upstream grade exists");
                assert_eq!(v["recommended"], true);
                assert!(v["id"].is_string() && v["displayName"].is_string());
            }
            // The dialect voices say what they are, in the accent column.
            let accent_of = |name: &str| {
                voices
                    .iter()
                    .find(|v| v["id"] == name)
                    .unwrap_or_else(|| panic!("{name} missing from {id}"))["accent"]
                    .as_str()
                    .unwrap()
                    .to_string()
            };
            assert!(accent_of("eric").contains("Sichuan"));
            assert!(accent_of("dylan").contains("Beijing"));
        }
    }

    /// The two qwentts capabilities are exclusive per engine — upstream's
    /// checkpoints are per-mode, and the params reject a speaker and a clone
    /// reference together — and each such engine declares its talker GGUF.
    #[test]
    fn cloning_and_named_speakers_are_exclusive_capabilities() {
        for e in REGISTRY {
            assert!(
                !(clones_voices(e.id) && named_speakers(e.id)),
                "{} claims both qwentts modes",
                e.id
            );
            if clones_voices(e.id) || named_speakers(e.id) {
                let talker = talker_file(e.id)
                    .unwrap_or_else(|| panic!("{} declares no talker", e.id));
                let expected_mode =
                    if named_speakers(e.id) { "customvoice" } else { "base" };
                assert!(
                    talker.contains(expected_mode),
                    "{}: talker {talker} is not the {expected_mode} checkpoint",
                    e.id
                );
            } else {
                assert!(talker_file(e.id).is_none(), "{}", e.id);
            }
        }
        assert!(named_speakers("qwen3-voices-06b"));
        assert!(named_speakers("qwen3-voices-17b"));
        assert!(!named_speakers("qwen3-clone-17b"));
        assert!(!named_speakers("kokoro"));
    }

    /// The never-block half of the policy: an unlisted voice resolves exactly
    /// like the default one. Resolution only picks the engine — the voice
    /// itself is looked up in the pack's `voices/` directory at synthesis
    /// time, never against the metadata table.
    #[test]
    fn an_unlisted_voice_still_resolves_to_its_engine() {
        let dir = tempfile::tempdir().unwrap();
        let models = Models::from_vars(Some(dir.path().to_path_buf()), None).unwrap();
        let kokoro = spec("kokoro").unwrap();
        // Fake-install the pack the way `fetch` marks completion.
        let pack = models.pack_dir(kokoro);
        std::fs::create_dir_all(&pack).unwrap();
        std::fs::File::create(pack.join(".complete")).unwrap();

        for voice in ["kokoro:af_nicole", "kokoro:af_heart"] {
            // In a Kokoro-less build resolution refuses for the platform
            // reason — equally for every voice, which is the point.
            let resolved = resolve(&models, None, voice);
            if cfg!(feature = "kokoro") {
                assert_eq!(resolved.expect(voice).id, "kokoro");
            } else {
                resolved.expect_err("unsupported on this build");
            }
        }
        assert_eq!(
            split_voice("kokoro:af_nicole"),
            (Some("kokoro"), "af_nicole")
        );
    }

    #[test]
    fn unknown_engine_error_lists_the_real_ones() {
        let msg = unknown_engine("qwen").to_string();
        assert!(msg.contains("qwen"));
        for e in REGISTRY {
            assert!(msg.contains(e.id), "error should name {}", e.id);
        }
    }

    /// An engine that cannot inspect anything must say so, not report clean.
    ///
    /// This is the whole point of `check::Report`. When the report was a bare
    /// `Vec<Finding>`, this default returned an empty list and `--check`
    /// printed "nothing flagged" for text it had never looked at — a clean
    /// bill of health from an engine with no letters-to-sounds stage.
    #[test]
    fn an_engine_with_no_g2p_reports_unavailable_and_never_clean() {
        struct Opaque;
        impl Engine for Opaque {
            fn narrate(
                &mut self,
                _text: &str,
                _voice: &str,
                _speed: f32,
                _language: Option<&str>,
                _gate: &mut dyn crate::quality::Gate,
            ) -> Result<Narration> {
                unreachable!("the report must not synthesize")
            }
        }

        let report = Opaque.pronunciation_report("ggml", "any:voice").unwrap();
        assert!(
            matches!(report, crate::check::Report::Unavailable(_)),
            "an opaque engine must report Unavailable, got {report:?}"
        );
        assert_ne!(
            report,
            crate::check::Report::Checked(Vec::new()),
            "\"cannot tell\" must not be equal to \"nothing wrong\""
        );
    }
}
