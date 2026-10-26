//! `bisque-voice` — local narration and voice cloning, on any platform.
//!
//! The one part of `/present` that cannot be done in the cloud: on-device
//! synthesis is what makes unlimited free narration free. Everything else —
//! authoring, hosting, rendering, the watch page — is already portable.
//!
//! Contract (mirrors the server's narration boundary, so the local and server
//! paths agree):
//!
//! ```text
//! bisque-voice --text "…" --voice af_heart --speed 1.0 --out narration.mp3
//!   → writes 24 kHz mono audio: MP3, or WAV when --out ends in .wav
//!   → prints one JSON object on stdout:
//!     {"words":[{"word","start","end"}],"durationMs":…,"exact":…,
//!      "contentType":…,"size":…,"hash":"sha256:…","voice":…,"speed":…,"chunks":…}
//! ```
//!
//! `contentType`, `size` and `hash` are shaped to drop verbatim into a
//! `presentation.json` v2 audio asset, and `durationMs` into
//! `slides[].audioDurationMs` — see `Output` for why each is typed as it is.
//!
//! `words` is 1:1 with the source's whitespace tokens, always. Cue markers
//! index that array, so the guarantee is load-bearing and fails closed.

// Platform support is per-engine. The Kokoro engine (`model`), its G2P stack
// (`g2p`) and its duration-tensor walk (`timing`) exist only behind the
// `kokoro` feature: they run on ONNX Runtime, which publishes no build for
// Intel macOS, so the Intel release compiles without them and ships the
// ggml-backed Studio engines alone. The registries still list everything —
// see `fetch::Unsupported` for how a missing engine reports itself.
mod check;
mod pronounce;
mod align;
mod audio;
mod device;
mod engine;
mod fetch;
mod host_guard;
#[cfg(feature = "kokoro")]
mod g2p;
// NOT feature-gated: a Kokoro-less build still lists kokoro in `engines`,
// and the listing carries the voice metadata either way.
mod kokoro_voices;
#[cfg(feature = "kokoro")]
mod model;
mod profile;
mod quality;
mod qwen3_voices;
mod qwen_common;
mod qwentts;
mod reconcile;
#[cfg(feature = "kokoro")]
mod timing;
mod verify;
mod vibevoice;
mod voices;

use std::io::Write;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;

/// Calibration against the macOS MLX engine.
///
/// With phonemes, voice pack and speed held identical, the two runtimes
/// disagree on duration by a uniform ~1.30x. Kokoro's `speed` input divides
/// `pred_dur`, so one constant reconciles them. 1.0 is the ONNX reference pace;
/// `--match-macos` selects the shipping macOS pace instead.
const MACOS_PACE: f32 = 1.30;

#[derive(Parser)]
#[command(
    name = "bisque-voice",
    version,
    about = "Local narration and voice cloning, with word timings"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Text to synthesize. Use `-` to read stdin.
    #[arg(long)]
    text: Option<String>,

    /// Which installed engine to use. Optional when only one is installed, or
    /// when the voice is qualified as `<engine>:<voice>`.
    #[arg(long)]
    engine: Option<String>,

    /// Which installed forced aligner should measure the word timings, or
    /// `none` for the engine's own. Optional: with exactly one installed that
    /// is the choice already made, and with none installed the engine's
    /// timings are used and reported as approximate.
    #[arg(long, value_name = "ALIGNER")]
    align: Option<String>,

    /// Voice id, optionally qualified: `af_heart` or `kokoro:af_heart`.
    /// Deliberately has no default — the voice belongs to whichever engine the
    /// user installed, and this tool does not pick one for them.
    #[arg(long)]
    voice: Option<String>,

    /// Language of the text, e.g. `english`, `chinese`, `russian`. Optional
    /// for a single-language engine; on a multilingual one, stating it selects
    /// the model's language-conditioned prefill. `bisque-voice engines` lists
    /// what each engine speaks.
    #[arg(long)]
    language: Option<String>,

    /// Speech rate. Higher is faster.
    #[arg(long, default_value_t = 1.0)]
    speed: f32,

    /// Speak at the same pace as the Bisque macOS app.
    #[arg(long)]
    match_macos: bool,

    /// Where to write the WAV.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Which hardware to run on: `auto` (accelerator when present), `cpu`, or
    /// `gpu`. `auto` is not a guess about your model — it is using the machine
    /// you already have.
    #[arg(long, default_value = "auto")]
    device: device::Device,

    /// ONNX Runtime intra-op threads. Defaults to the runtime's own choice.
    #[arg(long)]
    threads: Option<usize>,

    /// Dump the phoneme stream and pre-reconciliation timings to stderr. This
    /// is what a macOS↔cross-platform parity fixture is built from.
    #[arg(long)]
    debug_tokens: bool,

    /// Deprecated alias for the `pronunciation-report` subcommand: print every
    /// word with what this voice will say for it, then stop without
    /// synthesizing anything. Needs no `--out`.
    #[arg(long)]
    check: bool,

    /// With `--check`, emit the machine-readable envelope on stdout instead
    /// of the table. Same contents either way — the flag changes the format,
    /// never the word list.
    #[arg(long, requires = "check")]
    json: bool,
}

#[derive(Subcommand)]
enum Command {
    /// List every engine with its size, voices and languages.
    ///
    /// This is what an onboarding flow renders so the user can choose. Use
    /// `--json` to consume it programmatically.
    Engines {
        #[arg(long)]
        json: bool,
    },
    /// List every forced aligner with its size and reported parity.
    ///
    /// An aligner measures word timings against the synthesized waveform
    /// instead of inferring them, so it is an upgrade to any engine — but it is
    /// a large download, and there is no default. Use `--json` to consume it
    /// programmatically.
    Aligners {
        #[arg(long)]
        json: bool,
    },
    /// Install one or more engines or aligners. Installing several is normal —
    /// pick every model you want available.
    Install {
        /// Engine or aligner ids, e.g. `kokoro` or `qwen3-q4`.
        #[arg(required = true, value_name = "ID")]
        ids: Vec<String>,
    },
    /// Remove an installed engine's or aligner's model files.
    Uninstall {
        #[arg(required = true, value_name = "ID")]
        ids: Vec<String>,
    },
    /// Report where models live and which are installed.
    Doctor,
    /// Clone a voice from a recording, once, so narration can use it by name.
    ///
    /// This runs the speaker and codec encoders over the clip and stores ~13 KB
    /// of latents; the clip is never needed again. Give it 10–30 seconds of
    /// clean speech from one person, and the transcript of exactly what is said
    /// in it — the clone is conditioned on that text, so an approximate
    /// transcript produces an approximate voice.
    Clone {
        /// Name to narrate with later, e.g. `--voice qwen3-clone-17b:narrator`.
        #[arg(long, value_name = "NAME")]
        name: String,
        /// The reference recording. WAV; any rate, mono or stereo.
        #[arg(long, value_name = "PATH")]
        reference: PathBuf,
        /// What the recording says, verbatim.
        #[arg(long, value_name = "TEXT", conflicts_with = "transcript_file")]
        transcript: Option<String>,
        /// A file containing what the recording says, verbatim.
        #[arg(long, value_name = "PATH")]
        transcript_file: Option<PathBuf>,
        /// Which cloning engine to clone with. Optional when exactly one is
        /// installed. A voice belongs to the engine that made it.
        #[arg(long)]
        engine: Option<String>,
        /// Opaque provenance for the reference recording (e.g. an upload
        /// timestamp). Stored in the voice manifest and returned by
        /// `list --json` as `sourceId`, so a caller can tell whether a
        /// voice was cloned from the recording it currently cares about
        /// without keeping state of its own.
        #[arg(long, value_name = "ID")]
        source_id: Option<String>,
    },
    /// Per-word pronunciation facts, without synthesizing anything.
    ///
    /// For every word: the phonemes the voice will say, the same phonemes
    /// respelled in ordinary letters, whether the lexicon holds the word, and
    /// how many distinct-sounding readings it holds. No filtering and no risk
    /// scores — judging which words are wrong takes knowing how they should
    /// sound, which belongs to whoever reads the report. `--check` is the
    /// deprecated alias.
    PronunciationReport {
        /// Text to inspect. Use `-` to read stdin.
        #[arg(long)]
        text: String,
        /// Voice id, e.g. `kokoro:af_heart`. Required — the voice id chooses
        /// the lexicon.
        #[arg(long)]
        voice: String,
        /// Which installed engine to use. Optional when only one is installed,
        /// or when the voice is qualified as `<engine>:<voice>`.
        #[arg(long)]
        engine: Option<String>,
        /// Emit the machine-readable envelope on stdout instead of the table.
        #[arg(long)]
        json: bool,
    },
    /// List cloned voices.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Delete a cloned voice. The original recording is not touched.
    Forget {
        #[arg(required = true, value_name = "NAME")]
        names: Vec<String>,
    },
}

/// One synthesized slide, in the shape the publish path consumes.
///
/// The three asset fields are deliberately named and typed to drop verbatim
/// into a `presentation.json` v2 audio asset (`{contentType, size, hash}`) and
/// into `slides[].audioDurationMs`. Anything that needs reshaping here would
/// need reshaping in every consumer forever, so it is shaped once, at source.
#[derive(Serialize)]
struct Output {
    words: Vec<reconcile::Word>,
    /// Integer milliseconds. Every other producer writes an integer and the
    /// manifest reader parses it with `as_u64()`, which rejects a float
    /// outright — a fractional millisecond would read back as absent.
    #[serde(rename = "durationMs")]
    duration_ms: u64,
    /// False when timings were distributed proportionally because the engine
    /// expanded the text (e.g. "$4.2M" → "four point two million dollars").
    /// The word *count* is exact either way.
    exact: bool,
    /// What the publish path needs to declare about the object it uploads.
    #[serde(rename = "contentType")]
    content_type: String,
    /// Encoded byte count — the manifest's `size`.
    size: usize,
    /// Algorithm-prefixed digest, `sha256:<64 lowercase hex>`. The publish
    /// endpoint hard-rejects a bare hex digest, so the prefix is part of the
    /// value rather than something each caller has to remember to prepend.
    hash: String,
    voice: String,
    speed: f32,
    /// Number of forward passes; >1 means the narration was chunked.
    chunks: usize,
    /// Which forced aligner measured `words`, or absent when the timings came
    /// from the engine itself. Present so a consumer can tell a measured
    /// timing from an inferred one *and* know which graph measured it —
    /// `exact` alone says the mapping was exact, not how it was obtained.
    #[serde(skip_serializing_if = "Option::is_none")]
    aligner: Option<String>,
    /// Kokoro mispronunciations a listener has reported, found in this
    /// narration, each with the marker that fixes it (`verify.rs`). Empty when
    /// none matched; absent when nothing was checked: another engine, or a
    /// Kokoro voice with no pronunciation report.
    #[serde(
        rename = "pronunciationWarnings",
        skip_serializing_if = "Option::is_none"
    )]
    pronunciation_warnings: Option<Vec<verify::Warning>>,
}

/// The `pronunciation-report --json` envelope.
///
/// `inspectable: false` carries a `reason` and an empty `words` array — the
/// machine-readable form of `check::Report::Unavailable`, kept distinct so
/// "nothing was inspected" can never parse as "every word is fine".
#[derive(Serialize)]
struct PronunciationReportOut<'a> {
    engine: &'a str,
    voice: &'a str,
    inspectable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'a str>,
    words: &'a [check::Finding],
}

/// The `pronunciation-report` subcommand: resolve the engine, run the
/// letters-to-sounds stage, emit the facts. Never synthesizes.
fn run_pronunciation_report(
    models: &fetch::Models,
    text: &str,
    voice: &str,
    engine_id: Option<&str>,
    json: bool,
    progress: &mut impl Write,
) -> Result<()> {
    let text = text_arg(text)?;
    let spec = engine::resolve(models, engine_id, voice)?;
    let pack = models.ensure_pack(spec, progress)?;
    let mut backend = engine::open(spec, pack, None, device::Device::Auto, models.root())?;
    let report = backend.pronunciation_report(&text, voice)?;
    if !json {
        return print_pronunciation_facts(&report, progress);
    }
    emit_pronunciation_json(&report, spec.id, voice)
}

/// Emit the `--json` envelope on stdout. Shared by `pronunciation-report`
/// and the deprecated `--check --json`, so the two cannot drift.
fn emit_pronunciation_json(report: &check::Report, engine_id: &str, voice: &str) -> Result<()> {
    let (_, bare_voice) = engine::split_voice(voice);
    let out = match report {
        check::Report::Checked(words) => PronunciationReportOut {
            engine: engine_id,
            voice: bare_voice,
            inspectable: true,
            reason: None,
            words,
        },
        check::Report::Unavailable(reason) => PronunciationReportOut {
            engine: engine_id,
            voice: bare_voice,
            inspectable: false,
            reason: Some(reason),
            words: &[],
        },
    };
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

/// Print every word with what the voice will say for it, in ordinary letters.
///
/// The point is legibility: `ˌiɡmˈil` means nothing to a reader, `ee-GMEEL` is
/// wrong on sight. An engine with no G2P reports nothing, and says so, because
/// "no findings" and "cannot tell" must not look the same.
///
/// Every word prints, in every mode — `--json` changes the format, never the
/// contents. No filter belongs here, fact-based or otherwise. A filter on
/// lexicon facts drops real defects: `A` in "I would pick A" is in the
/// lexicon with one reading, read as the article, and wrong for that
/// sentence. A filter on spelling was tried and could not work — `Kokoro`,
/// `ElevenLabs` and `companies` are shaped alike and exactly one is
/// mispronounced. The consumer is the authoring agent (or a developer
/// debugging), and it wants every word. The command states what the engine
/// will say; whether that is correct is the reader's judgment, so no output
/// line claims the text is fine.
fn print_pronunciation_facts(report: &check::Report, progress: &mut impl Write) -> Result<()> {
    let findings = match report {
        check::Report::Unavailable(reason) => {
            writeln!(
                progress,
                "bisque-voice: checked nothing — {reason}. This says nothing \
                 about the text; only Kokoro voices can be inspected, see the \
                 README's Pronunciation section."
            )?;
            return Ok(());
        }
        check::Report::Checked(findings) => findings,
    };
    if findings.is_empty() {
        writeln!(progress, "bisque-voice: the text has no words to report.")?;
        return Ok(());
    }
    let word_width = findings
        .iter()
        .map(|f| f.word.chars().count())
        .max()
        .unwrap_or(0);
    let say_width = findings
        .iter()
        .map(|f| f.respelled.chars().count())
        .max()
        .unwrap_or(0);
    for f in findings {
        let facts = if !f.in_lexicon {
            "not in lexicon, guessed from spelling".to_string()
        } else if f.readings > 1 {
            format!("{} readings, one was chosen", f.readings)
        } else {
            "in lexicon".to_string()
        };
        writeln!(
            progress,
            "  {:word_width$}  {:say_width$}  [{}]  {}",
            f.word, f.respelled, facts, f.phonemes,
        )?;
    }
    writeln!(
        progress,
        "\n{} word(s). Judge each respelling against how the word should \
         sound; fix a wrong one with [word](/ipa/) or [word](respelling), \
         never by respelling the word itself.",
        findings.len()
    )?;
    Ok(())
}

/// A `--text` value, with `-` meaning stdin. Refuses empty text.
fn text_arg(text: &str) -> Result<String> {
    let text = if text == "-" {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        buf
    } else {
        text.to_string()
    };
    let text = text.trim().to_string();
    if text.is_empty() {
        bail!("--text is empty");
    }
    Ok(text)
}

fn main() {
    if let Err(err) = run() {
        eprintln!("bisque-voice: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let mut progress = std::io::stderr();
    // The environment is consulted exactly here; everything below is handed the
    // resolved locations rather than re-reading process state.
    let models = fetch::Models::from_env()?;

    match &cli.command {
        Some(Command::Engines { json }) => return list_engines(&models, *json),
        Some(Command::Aligners { json }) => return list_aligners(&models, *json),
        Some(Command::Install { ids }) => {
            for id in ids {
                let spec = installable(id)?;
                let dir = models.ensure_pack(spec, &mut progress)?;
                println!("{} {}", spec.id(), dir.display());

                // Installing an engine installs what the engine needs to be
                // usable — a companion aligner is part of its contract, not
                // a second decision. Explicitly installing the aligner id
                // alone still works, and a build that cannot run aligners
                // (Intel Macs) states why and moves on rather than failing
                // the engine it just installed.
                let companion = engine::spec(id).and_then(|e| e.companion_aligner);
                if let Some(aligner_id) = companion {
                    if ids.iter().any(|other| other == aligner_id) {
                        continue; // being installed by name in this same run
                    }
                    match installable(aligner_id) {
                        Ok(aligner) => {
                            if !models.pack_dir(aligner).exists() {
                                writeln!(
                                    progress,
                                    "bisque-voice: {id} narrates with measured word timings \
                                     via its companion aligner — installing {aligner_id} \
                                     ({}):",
                                    align::spec(aligner_id)
                                        .map(|a| format!("{} MB", a.download_bytes / 1_000_000))
                                        .unwrap_or_else(|| "size unknown".into())
                                )
                                .ok();
                            }
                            let dir = models.ensure_pack(aligner, &mut progress)?;
                            println!("{aligner_id} {}", dir.display());
                        }
                        Err(why) => {
                            writeln!(
                                progress,
                                "bisque-voice: note — {id}'s companion aligner \
                                 {aligner_id} cannot install here ({why}); word \
                                 timings will be approximate."
                            )
                            .ok();
                        }
                    }
                }
            }
            return Ok(());
        }
        Some(Command::Uninstall { ids }) => {
            for id in ids {
                let spec = installable(id)?;
                let dir = models.pack_dir(spec);
                if dir.exists() {
                    std::fs::remove_dir_all(&dir)
                        .with_context(|| format!("removing {}", dir.display()))?;
                    println!("removed {}", dir.display());
                } else {
                    println!("{} was not installed", spec.id());
                }
            }
            return Ok(());
        }
        Some(Command::Doctor) => {
            println!("models dir : {}", models.root().display());
            println!("engines");
            for status in engine::statuses(&models) {
                print_doctor_row(
                    status.spec.id,
                    status.installed,
                    status.spec.pack_version,
                    engine::pack_sha256(status.spec.id),
                    status.spec.unsupported,
                );
            }
            // Aligners are reported whether or not any is installed: "none" is
            // a supported state, and seeing it is how a user learns the timings
            // they are getting are the engine's own.
            println!("aligners");
            for status in align::statuses(&models) {
                print_doctor_row(
                    status.spec.id,
                    status.installed,
                    status.spec.pack_version,
                    align::pack_sha256(status.spec.id),
                    status.spec.unsupported,
                );
            }
            if align::installed(&models).is_empty() {
                println!("  none installed — word timings come from the engine itself");
            }
            // Pronunciation data, one pack per language. Reported the same way
            // and for the same reason: a voice whose language pack is missing
            // fails at narration time, and this is where that is visible first.
            // The packs feed Kokoro's G2P and nothing else, so a build without
            // that engine has no languages section — the Studio engines carry
            // their multilingual pipeline inside their own packs.
            #[cfg(feature = "kokoro")]
            {
                println!("languages");
                for pack in g2p::pack::REGISTRY {
                    print_doctor_row(
                        pack.id,
                        models.is_installed(pack),
                        pack.pack_version,
                        g2p::pack::pack_sha256(pack.id),
                        None,
                    );
                    println!("               {}", pack.summary);
                }
            }
            let library = voices::VoiceLibrary::new(models.root());
            let cloned = library.list();
            println!("cloned voices");
            if cloned.is_empty() {
                println!("  none — create one with `bisque-voice clone`");
            } else {
                for v in &cloned {
                    println!("  {:<12} {}", v.name, v.engine);
                }
            }
            return Ok(());
        }
        Some(Command::Clone {
            name,
            reference,
            transcript,
            transcript_file,
            engine,
            source_id,
        }) => {
            return clone_voice(
                &models,
                name,
                reference,
                transcript.as_deref(),
                transcript_file.as_deref(),
                engine.as_deref(),
                source_id.as_deref(),
                &mut progress,
            )
        }
        Some(Command::PronunciationReport {
            text,
            voice,
            engine,
            json,
        }) => {
            return run_pronunciation_report(
                &models,
                text,
                voice,
                engine.as_deref(),
                *json,
                &mut progress,
            )
        }
        Some(Command::List { json }) => return list_voices(&models, *json),
        Some(Command::Forget { names }) => {
            let library = voices::VoiceLibrary::new(models.root());
            for name in names {
                if library.remove(name)? {
                    println!("forgot {name}");
                } else {
                    println!("{name} was not a cloned voice");
                }
            }
            return Ok(());
        }
        None => {}
    }

    let text = match cli.text.as_deref() {
        Some(t) => text_arg(t)?,
        None => bail!("--text is required (use `-` to read stdin)"),
    };
    let text = text.as_str();
    // Author pronunciation overrides split the narration in two: `text` keeps
    // the markers and goes to synthesis, because each engine honours them
    // differently; `display` is the words a reader sees and is what every
    // word-counting or word-reporting consumer gets. Feeding the aligner the
    // raw string would have it try to place `[live](liv)` against audio saying
    // "liv", and reconcile would then report the marker as the word.
    let display = pronounce::display_text(text);
    let display = display.as_str();
    // `--check` writes nothing, so it must not demand somewhere to write it.
    let out_path = match cli.out.clone() {
        Some(path) => Some(path),
        None if cli.check => None,
        None => bail!("--out is required"),
    };

    // Which engine — never guessed. With none installed this fails and tells
    // the caller to choose.
    let voice_ref = cli.voice.clone().unwrap_or_default();
    // `--check` inspects the voice's own letters-to-sounds stage, so a voice is
    // as required here as it is for narration: Kokoro reads the same characters
    // differently for `af_heart` than for `jf_alpha`, and the lexicon it
    // consults is chosen by the voice id. Refuse before anything is loaded —
    // `--engine kokoro` alone used to reach the G2P with an empty voice, fail
    // to resolve a language, and report the text as clean.
    if cli.check && voice_ref.is_empty() {
        bail!(
            "--check needs a voice: the voice id chooses the lexicon, so the \
             report changes with it. Pass --voice <engine>:<voice>, e.g. \
             --voice kokoro:af_heart."
        );
    }
    let spec = engine::resolve(&models, cli.engine.as_deref(), &voice_ref)?;

    // Which aligner — resolved and installed *before* synthesis. A typo'd id or
    // a download failure should cost nothing; discovering it after an
    // autoregressive decode has run would throw that work away.
    let aligner = align::resolve(&models, cli.align.as_deref())?;
    let aligner_pack = match aligner {
        Some(spec) => Some(models.ensure_pack(spec, &mut progress)?),
        None => None,
    };

    // An engine that declares a companion aligner produces timings that
    // drift — and leading non-speech that cannot be trimmed — without one.
    // Running it unaligned still works, but silently would hide the exact
    // failure users report as "the presentation is out of sync". `--align
    // none` is the deliberate opt-out and stays quiet.
    if let Some(companion) = spec.companion_aligner {
        if aligner.is_none() && cli.align.as_deref() != Some("none") {
            writeln!(
                progress,
                "bisque-voice: warning — {} has approximate word timings, \
                 may start with untrimmed non-speech, and cannot have a garbled \
                 take caught and retaken without a forced aligner. Install its \
                 companion:  bisque-voice install {companion}",
                spec.id
            )
            .ok();
        }
    }

    let pack = models.ensure_pack(spec, &mut progress)?;
    let mut backend = engine::open(spec, pack, cli.threads, cli.device, models.root())?;

    if cli.check {
        writeln!(
            progress,
            "bisque-voice: --check is deprecated; use `bisque-voice \
             pronunciation-report` (add --json for the machine-readable form)."
        )?;
        let report = backend.pronunciation_report(text, &voice_ref)?;
        if cli.json {
            return emit_pronunciation_json(&report, spec.id, &voice_ref);
        }
        return print_pronunciation_facts(&report, &mut progress);
    }

    let voice = if voice_ref.is_empty() {
        bail!(
            "--voice is required. `bisque-voice engines` lists the installed \
             engines; voices may be qualified as <engine>:<voice>."
        );
    } else {
        voice_ref
    };

    let speed = if cli.match_macos {
        cli.speed * MACOS_PACE
    } else {
        cli.speed
    };

    // One loaded aligner serves both the retake gate, which measures each
    // chunk as it is synthesized, and the narration's own timings below.
    let mut loaded_aligner = match &aligner_pack {
        Some(pack) => Some(align::Aligner::open(pack, cli.threads, cli.device)?),
        None => None,
    };
    let narration = match loaded_aligner.as_mut() {
        Some(aligner) => backend.narrate(
            text,
            &voice,
            speed,
            cli.language.as_deref(),
            &mut AlignerGate(aligner),
        )?,
        None => backend.narrate(
            text,
            &voice,
            speed,
            cli.language.as_deref(),
            &mut quality::Blind,
        )?,
    };
    let samples = narration.samples;
    let chunks = narration.chunks;

    // Fail closed on silence, before an aligner or an encoder touches it.
    // This caught the fp16 Kokoro pack returning an all-NaN waveform — which
    // encodes to exact zeros — with every other signal healthy: plausible
    // duration, full word timings, valid hash. The root
    // cause is fixed (the shipped pack is fp32, and Kokoro now rejects
    // non-finite model output at the source), but the floor stays: it is
    // engine-agnostic, and silence is an error here, never a success.
    let peak = audio::peak_dbfs(&samples);
    if peak < audio::SILENCE_FLOOR_DBFS {
        bail!(
            "the engine produced silence for voice {voice:?} on this text (peak \
             {peak:.1} dBFS; anything below {} dBFS is treated as silent). This \
             is a known defect for some (voice, text) pairs. Try a different \
             voice, or reword the text slightly; the failure is deterministic, \
             so retrying unchanged will not help.",
            audio::SILENCE_FLOOR_DBFS
        );
    }

    // Forced alignment measures the waveform that was actually produced, so
    // when an aligner is present its timings REPLACE the engine's rather than
    // being blended with them. A failure here is fatal: the caller asked for
    // measured timings, and quietly handing back inferred ones would be the
    // kind of silent downgrade this tool exists not to do.
    let (timings, aligner_id, samples) = match (aligner, loaded_aligner.as_mut()) {
        (Some(spec), Some(loaded)) => {
            writeln!(progress, "bisque-voice: aligning with {}…", spec.id).ok();
            let mut words = loaded
                .measure(&samples, display)
                .with_context(|| format!("forced alignment with {}", spec.id))?
                .words;
            // Measured timings make junk outside the speech identifiable —
            // the clone engines stochastically emit dead air or a
            // speech-loud phantom fragment before the first real word,
            // which no amplitude heuristic can tell from narration. Cut to
            // the aligned span and shift the timings with it, so the audio
            // starts where the first word does.
            let before = samples.len();
            let samples = audio::trim_to_aligned_words(samples, &mut words);
            if samples.len() < before {
                let cut = (before - samples.len()) as f64 / f64::from(audio::SAMPLE_RATE);
                writeln!(
                    progress,
                    "bisque-voice: trimmed {cut:.2}s outside the aligned speech"
                )
                .ok();
            }
            (words, Some(spec.id.to_string()), samples)
        }
        _ => (narration.words, None, samples),
    };

    if cli.debug_tokens {
        for (surface, phonemes) in backend.debug_tokens(text, &voice) {
            writeln!(progress, "token {surface:?} -> {phonemes:?}").ok();
        }
        writeln!(
            progress,
            "{} words: {}",
            aligner_id.as_deref().unwrap_or("engine"),
            timings.len()
        )?;
        for w in &timings {
            writeln!(progress, "  {:?} {:.3}..{:.3}", w.word, w.start, w.end)?;
        }
    }

    let reconciled = reconcile::reconcile(display, &timings)
        .context("reconciling word timings onto source tokens")?;

    // An aligner returns one timing per source whitespace token carrying that
    // token's own surface string, so reconciliation is exact by construction.
    // If it somehow isn't, the two sides have diverged and `exact: true` would
    // be a lie — so say so rather than shipping the flag.
    if aligner_id.is_some() && !reconciled.exact {
        bail!(
            "the aligner returned {} timings that did not map onto the {} source words. \
             Refusing to report measured timings that were not measured.",
            timings.len(),
            reconcile::source_words(display).len()
        );
    }

    // The reported Kokoro mispronunciations, checked against what the voice
    // was just given. A match warns with its fix and never stops the
    // narration: the audio is still written, as it was before this check.
    // A voice whose language has no report is not checked, and says so by
    // leaving the field out rather than reporting an empty list.
    let pronunciation_warnings = match spec.id {
        "kokoro" => match backend.pronunciation_report(text, &voice)? {
            check::Report::Checked(findings) => Some(verify::warnings(&findings, text)),
            check::Report::Unavailable(_) => None,
        },
        _ => None,
    };
    for w in pronunciation_warnings.iter().flatten() {
        writeln!(progress, "bisque-voice: warning — {}", w.message).ok();
    }

    // MP3 unless `--out` ends in .wav. `ManifestAudioAsset.contentType` is
    // literally "audio/mpeg", so MP3 is what the publish path can accept.
    // Past the `--check` early return, so this is always present.
    let out_path = out_path.expect("--out was required for a run that writes audio");
    let format = audio::Format::for_path(&out_path);
    let encoded = audio::encode(&samples, format)?;
    std::fs::write(&out_path, &encoded)
        .with_context(|| format!("writing {}", out_path.display()))?;

    let output = Output {
        words: reconciled.words,
        duration_ms: audio::duration_ms(&samples).round() as u64,
        exact: reconciled.exact,
        content_type: format.content_type().to_string(),
        size: encoded.len(),
        hash: sha256_prefixed(&encoded),
        voice,
        speed,
        chunks,
        aligner: aligner_id,
        pronunciation_warnings,
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

/// The retake gate as this binary runs it: each Studio chunk is aligned
/// against its own text the moment it is synthesized, and the gate reads the
/// spans (`quality.rs`). The text is the chunk as spoken, pronunciation
/// markers already reduced, which is what the audio says.
struct AlignerGate<'a>(&'a mut align::Aligner);

impl quality::Gate for AlignerGate<'_> {
    fn judge(&mut self, text: &str, pcm: &[f32]) -> Result<Option<quality::Reading>> {
        let measured = self
            .0
            .measure(pcm, text)
            .context("measuring a synthesized chunk for the retake gate")?;
        let mut reading = quality::read(&measured.tokens);
        let take_ms = (pcm.len() as u64 * 1000 / u64::from(audio::SAMPLE_RATE)) as i64;
        reading.dropped_tail = quality::dropped_tail(&measured.tokens, take_ms);
        // Each is retaken whatever the opening holds, so it needs no
        // second alignment.
        if reading.is_collapse() || reading.is_unsure() || reading.dropped_tail > 0 {
            return Ok(Some(reading));
        }
        // A gap after the first word: align again from where that word was
        // said to end, and see whether it is found later (`quality.rs`).
        if let Some(cut_ms) = quality::lead_cut_ms(&measured.tokens) {
            let cut = (cut_ms as usize * audio::SAMPLE_RATE as usize / 1000).min(pcm.len());
            let recut: Vec<quality::TokenSpan> = self
                .0
                .measure(&pcm[cut..], text)
                .context("measuring a synthesized chunk's opening for the retake gate")?
                .tokens
                .into_iter()
                .map(|t| quality::TokenSpan {
                    start_ms: t.start_ms + cut_ms,
                    end_ms: t.end_ms + cut_ms,
                    ..t
                })
                .collect();
            reading.absorbed_lead = quality::lead_absorbed(&measured.tokens, &recut);
        }
        Ok(Some(reading))
    }
}

/// Resolve an id against BOTH registries.
///
/// `install` and `uninstall` take engines and aligners alike — they are one
/// download to a user, and making them two commands would only be a
/// vocabulary lesson. An id in neither registry lists both sets, since a
/// person who typed `qwen3-q4` at the wrong prompt should see where it lives.
fn installable(id: &str) -> Result<&'static dyn fetch::Pack> {
    if let Some(spec) = engine::spec(id) {
        return Ok(spec);
    }
    if let Some(spec) = align::spec(id) {
        return Ok(spec);
    }
    // Per-language pronunciation data. Installed by the same command as an
    // engine because it is the same kind of thing to a user — something the
    // tool needs before it can do what was asked — and the same kind of thing
    // to `fetch`: one hash-verified archive. Kokoro-only: the packs feed its
    // G2P, so a build without that engine has none to offer.
    #[cfg(feature = "kokoro")]
    if let Some(spec) = g2p::pack::by_id(id) {
        return Ok(spec);
    }
    #[cfg(feature = "kokoro")]
    let languages = g2p::pack::REGISTRY
        .iter()
        .map(|p| p.id)
        .collect::<Vec<_>>()
        .join(", ");
    #[cfg(not(feature = "kokoro"))]
    let languages = "none in this build — language packs feed the Kokoro engine";
    bail!(
        "unknown engine, aligner or language pack {id:?}.\n  \
         engines  : {}\n  aligners : {}\n  languages: {}",
        engine::REGISTRY
            .iter()
            .map(|e| e.id)
            .collect::<Vec<_>>()
            .join(", "),
        align::REGISTRY
            .iter()
            .map(|a| a.id)
            .collect::<Vec<_>>()
            .join(", "),
        languages,
    )
}

fn print_doctor_row(
    id: &str,
    installed: bool,
    pack_version: &str,
    sha: &str,
    unsupported: Option<&'static fetch::Unsupported>,
) {
    println!(
        "  {:<12} {:<14} {:<30} {}",
        id,
        if installed {
            "installed"
        } else {
            "not installed"
        },
        pack_version,
        // The platform verdict outranks the checksum: the release workflow
        // pins every pack for every leg, and a pinned checksum for an engine
        // this build cannot run would read as installable.
        if let Some(unsupported) = unsupported {
            unsupported.message
        } else if sha.is_empty() {
            // Empty means this build pinned no checksum for this pack, which
            // is true both of a dev build and of a release that did not ship
            // this engine. Say what is observable rather than guess which.
            "not in this build (no pinned checksum)"
        } else {
            sha
        },
    );
}

/// `sha256:<64 lowercase hex>` of the encoded audio. The publish path declares
/// this up front so the server can verify what actually landed in storage
/// rather than trusting the bucket.
///
/// The `sha256:` prefix is emitted here, not left to callers: the publish
/// endpoint rejects a bare digest, so every consumer that got a bare one would
/// have to prepend it — and the one that forgets fails at upload time with a
/// validation error far from the cause.
fn sha256_prefixed(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let hex: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    format!("sha256:{hex}")
}


/// Clone a voice from a reference recording.
///
/// The expensive work — the speaker encoder and the codec encoder — runs once,
/// here, and everything after it reuses ~13 KB of latents. That is why this is
/// its own command rather than a flag on synthesis: paying it per narration
/// would make cloning cost more than the narration it is for.
#[allow(clippy::too_many_arguments)]
fn clone_voice(
    models: &fetch::Models,
    name: &str,
    reference: &std::path::Path,
    transcript: Option<&str>,
    transcript_file: Option<&std::path::Path>,
    engine_id: Option<&str>,
    source_id: Option<&str>,
    progress: &mut dyn Write,
) -> Result<()> {
    voices::VoiceLibrary::validate_name(name)?;

    let transcript = match (transcript, transcript_file) {
        (Some(t), _) => t.to_string(),
        (None, Some(path)) => std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?,
        (None, None) => bail!(
            "--transcript (or --transcript-file) is required: cloning conditions \
             on what the reference clip actually says."
        ),
    };

    let spec = resolve_cloning_engine(models, engine_id)?;
    let pack = models.ensure_pack(spec, progress)?;
    let talker = engine::talker_file(spec.id)
        .ok_or_else(|| anyhow::anyhow!("engine {:?} declares no talker weights", spec.id))?;

    // Decode before loading the model: a wrong path or an unreadable clip
    // should not cost a multi-second model load first.
    let bytes = std::fs::read(reference)
        .with_context(|| format!("reading {}", reference.display()))?;
    let raw = audio::decode_wav_to_mono_24k(&bytes)
        .with_context(|| format!("decoding {}", reference.display()))?;

    // Hygiene is part of this command's contract, not the caller's job: a
    // clip that opens with a click and a pause teaches the clone to start
    // narrations the same way. Trim to the speech, and refuse takes with
    // no speech in them at all.
    let audio = audio::trim_recording(&raw);
    if audio.is_empty() {
        bail!(
            "the reference clip contains no audible speech — check the \
             microphone that recorded it, and record again."
        );
    }
    let cut = (raw.len() - audio.len()) as f64 / f64::from(audio::SAMPLE_RATE);
    if cut >= 0.1 {
        writeln!(
            progress,
            "bisque-voice: trimmed {cut:.2}s of non-speech from the reference"
        )
        .ok();
    }
    let seconds = audio.len() as f64 / f64::from(audio::SAMPLE_RATE);
    if seconds < 3.0 {
        bail!(
            "the reference clip is only {seconds:.1}s of speech. Cloning \
             needs enough to characterise a voice — aim for 10–30 seconds."
        );
    }
    writeln!(
        progress,
        "bisque-voice: cloning {name:?} from {:.1}s of audio with {}…",
        seconds, spec.id
    )
    .ok();

    let mut backend =
        qwentts::QwenCloneEngine::load(&pack, spec.id, models.root(), talker)?;
    let voice = backend.extract_voice(
        name,
        &audio,
        &transcript,
        Some(reference.display().to_string()),
        source_id.map(str::to_string),
    )?;
    let dir = backend.library().save(&voice)?;

    println!("{} {}", name, dir.display());
    writeln!(
        progress,
        "bisque-voice: narrate with it using --voice {}:{name}",
        spec.id
    )
    .ok();
    Ok(())
}

/// Which cloning engine to clone with — same rule as `engine::resolve`: never
/// choose on the user's behalf when the choice is real.
fn resolve_cloning_engine(
    models: &fetch::Models,
    explicit: Option<&str>,
) -> Result<&'static engine::EngineSpec> {
    if let Some(id) = explicit {
        let spec = engine::spec(id)
            .ok_or_else(|| anyhow::anyhow!("unknown engine {id:?}"))?;
        if !engine::clones_voices(spec.id) {
            bail!(
                "engine {id:?} cannot clone voices. Cloning engines: {}",
                cloning_engine_ids()
            );
        }
        return Ok(spec);
    }
    let installed: Vec<&'static engine::EngineSpec> = engine::installed(models)
        .into_iter()
        .filter(|s| engine::clones_voices(s.id))
        .collect();
    match installed.len() {
        0 => bail!(
            "no voice-cloning engine is installed. Install one:\n  \
             bisque-voice install <engine>\nCloning engines: {}",
            cloning_engine_ids()
        ),
        1 => Ok(installed[0]),
        _ => bail!(
            "several cloning engines are installed ({}). Say which with --engine.",
            installed.iter().map(|s| s.id).collect::<Vec<_>>().join(", ")
        ),
    }
}

fn cloning_engine_ids() -> String {
    engine::REGISTRY
        .iter()
        .filter(|s| engine::clones_voices(s.id))
        .map(|s| s.id)
        .collect::<Vec<_>>()
        .join(", ")
}

/// List cloned voices. Shows which engine each belongs to, because that is not
/// cosmetic — a voice only loads into the engine that extracted it.
fn list_voices(models: &fetch::Models, json: bool) -> Result<()> {
    let library = voices::VoiceLibrary::new(models.root());
    let all = library.list();
    if json {
        println!("{}", serde_json::to_string_pretty(&all)?);
        return Ok(());
    }
    if all.is_empty() {
        println!("No cloned voices yet.");
        println!();
        println!(
            "Create one from 10–30 seconds of clean speech:\n  \
             bisque-voice clone --name narrator --reference clip.wav \\\n    \
             --transcript \"exactly what the clip says\""
        );
        return Ok(());
    }
    for v in &all {
        println!("{}:{}", v.engine, v.name);
        println!(
            "    {} frames of reference · {}-dim speaker embedding",
            v.ref_t, v.spk_dim
        );
        if let Some(src) = &v.source_clip {
            println!("    from {src}");
        }
    }
    println!();
    println!("narrate with:  --voice <engine>:<name>");
    Ok(())
}

/// Render the engine catalogue. This is the onboarding surface: the user picks
/// which models to install from real trade-offs — size, voices, languages —
/// and can pick more than one. Nothing here recommends a choice.
fn list_engines(models: &fetch::Models, json: bool) -> Result<()> {
    let statuses = engine::statuses(models);
    if json {
        println!("{}", serde_json::to_string_pretty(&statuses)?);
        return Ok(());
    }
    for s in &statuses {
        let langs: Vec<&str> = s.spec.languages.iter().map(|l| l.name).collect();
        println!(
            "{}  {}{}",
            s.spec.id,
            s.spec.display_name,
            match (s.installed, s.installable) {
                // Still listed, not hidden: the registry is what this tool
                // knows how to run, and an entry vanishing between releases
                // reads as the engine being gone rather than not yet shipped.
                // "This platform can never run it" and "this release did not
                // ship it" are different facts, so they get different labels —
                // and the platform case says why, right below. It also beats
                // `[installed]`: a pack put on disk by a build that could run
                // it does not make this build able to.
                _ if s.spec.unsupported.is_some() => "  [not available on this platform]",
                (true, _) => "  [installed]",
                (false, false) => "  [not in this build]",
                (false, true) => "",
            }
        );
        if let Some(unsupported) = s.spec.unsupported {
            println!("    {}", unsupported.message);
        }
        println!("    {}", s.spec.summary);
        println!(
            "    {} params · ~{} MB download · {} voices",
            s.spec.parameters,
            s.spec.download_bytes / 1_000_000,
            s.spec.voice_count
        );
        println!("    languages: {}", langs.join(", "));
        // Voice quality is graded upstream and the difference is audible, so
        // the catalogue says which voices to reach for — recommended first,
        // grades attached. Informational only: any voice passed via --voice
        // synthesizes regardless (see VOICE-CURATION.md).
        let listed: Vec<&kokoro_voices::VoiceSpec> =
            s.spec.voices.iter().filter(|v| v.listed).collect();
        if !listed.is_empty() {
            let line = |voices: &[&kokoro_voices::VoiceSpec]| {
                voices
                    .iter()
                    .map(|v| format!("{} ({})", v.id, v.grade))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let (recommended, rest): (Vec<_>, Vec<_>) =
                listed.into_iter().partition(|v| v.recommended);
            println!("    recommended voices: {}", line(&recommended));
            println!(
                "    other voices (audible quality issues — graded below C+ \
                 upstream): {}",
                line(&rest)
            );
        }
        println!();
    }
    println!("install one or more:  bisque-voice install <engine> [<engine>…]");
    Ok(())
}

/// Render the aligner catalogue.
///
/// Same job as `list_engines`, same rule: state what is known about each and
/// stop. The two entries are exports of one checkpoint, their published
/// deviation figures are over raw logits, and this path reads only the `argmax`
/// of those logits — so the numbers do not translate into a ranking, and
/// nothing here pretends they do.
fn list_aligners(models: &fetch::Models, json: bool) -> Result<()> {
    let statuses = align::statuses(models);
    if json {
        println!("{}", serde_json::to_string_pretty(&statuses)?);
        return Ok(());
    }
    for s in &statuses {
        println!(
            "{}  {}{}",
            s.spec.id,
            s.spec.display_name,
            if s.spec.unsupported.is_some() {
                // Beats `[installed]` for the same reason as in the engine
                // catalogue: a pack on disk does not make this build able.
                "  [not available on this platform]"
            } else if s.installed {
                "  [installed]"
            } else {
                ""
            }
        );
        if let Some(unsupported) = s.spec.unsupported {
            println!("    {}", unsupported.message);
        }
        println!("    {}", s.spec.summary);
        println!("    ~{} MB download", s.spec.download_bytes / 1_000_000);
        println!();
    }
    println!(
        "Both figures are deviations in raw logits against PyTorch. This path uses only the\n\
         argmax over those logits, and how often the two graphs pick a different timestamp has\n\
         not been measured — so qwen3-q4 is unmeasured here, not known to be worse."
    );
    println!();
    println!("install one:          bisque-voice install <aligner>");
    println!(
        "With none installed, word timings come from the engine itself and are reported as\n\
         approximate. There is no default."
    );
    Ok(())
}

#[cfg(test)]
mod output_contract_tests {
    use super::*;

    /// The publish endpoint matches `/^sha256:[a-f0-9]{64}$/i`. This is the
    /// contract that makes the digest a verbatim drop-in for a manifest
    /// asset's `hash`, so it is asserted rather than assumed.
    #[test]
    fn digest_is_algorithm_prefixed_lowercase_hex() {
        let h = sha256_prefixed(b"");
        assert_eq!(
            h,
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let hex = h.strip_prefix("sha256:").expect("sha256: prefix");
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_digit() || c.is_ascii_lowercase()));
    }

    /// A float would round-trip to "absent" through the manifest reader's
    /// `as_u64()`, so the serialized form must carry no decimal point. 24012
    /// samples at 24 kHz is 1000.5 ms — a value that only survives as an
    /// integer if it is deliberately rounded.
    #[test]
    fn duration_serializes_as_an_integer() {
        let out = Output {
            words: vec![],
            duration_ms: audio::duration_ms(&vec![0.0; 24_012]).round() as u64,
            exact: true,
            content_type: "audio/mpeg".to_string(),
            size: 1,
            hash: sha256_prefixed(b""),
            voice: "af_heart".to_string(),
            speed: 1.0,
            chunks: 1,
            aligner: None,
            pronunciation_warnings: None,
        };
        let json = serde_json::to_string(&out).expect("serialize");
        assert!(json.contains("\"durationMs\":1001"), "got {json}");
    }
}

#[cfg(test)]
mod pronunciation_table_tests {
    use super::*;

    fn finding(word: &str, phonemes: &str, in_lexicon: bool, readings: u32) -> check::Finding {
        check::Finding {
            word: word.to_string(),
            phonemes: phonemes.to_string(),
            respelled: check::respell(phonemes),
            in_lexicon,
            readings,
        }
    }

    fn printed(report: &check::Report) -> String {
        let mut buf = Vec::new();
        print_pronunciation_facts(report, &mut buf).expect("print");
        String::from_utf8(buf).expect("utf8")
    }

    /// One table row per input word, always. This is the test that makes
    /// reintroducing any filter fail loudly. Two filters existed and both
    /// dropped real defects: the spelling heuristic hid `Kokoro` (which the
    /// engine mispronounces) because it has vowels and no digits, and a
    /// lexicon-fact filter would hide `A` in "I would pick A" (in the lexicon,
    /// one reading, read as the article, wrong for that sentence). The mix
    /// below covers every fact combination a filter could cut on.
    #[test]
    fn the_table_prints_one_row_per_word_with_no_filter() {
        let words = vec![
            finding("Kokoro", "kəkˈɔɹO", false, 1), // out of lexicon, vowels, no digits
            finding("ggml", "ʤˌiʤˌiˌɛmˈɛl", false, 1), // out of lexicon, vowelless
            finding("read", "ɹˈid", true, 2),       // heteronym
            finding("servers", "sˈɜɹvəɹz", true, 1), // in lexicon, one reading
            finding("A", "ɐ", true, 1), // in lexicon, one reading, still wrong in context
        ];
        let report = check::Report::Checked(words.clone());
        let out = printed(&report);
        let rows = out.lines().filter(|l| l.starts_with("  ")).count();
        assert_eq!(rows, words.len(), "a filter dropped a word:\n{out}");
        for f in &words {
            assert!(
                out.lines()
                    .any(|l| l.starts_with("  ") && l.contains(&f.word)),
                "{} has no row in:\n{out}",
                f.word
            );
        }
        assert!(out.contains("5 word(s)."), "wrong count line:\n{out}");
    }

    /// The command reports what the engine will say and judges nothing, so no
    /// output line may claim the text is fine. "nothing flagged. Every word is
    /// in the lexicon with a single reading." shipped in 0.1.17 and was false
    /// for a sentence containing `Kokoro`.
    #[test]
    fn no_output_line_claims_the_text_is_fine() {
        for report in [
            check::Report::Checked(vec![finding("servers", "sˈɜɹvəɹz", true, 1)]),
            check::Report::Checked(vec![]),
            check::Report::Unavailable("no letters-to-sounds stage".to_string()),
        ] {
            let out = printed(&report).to_lowercase();
            for claim in [
                "nothing flagged",
                "every word is in the lexicon",
                "fine",
                "clean",
            ] {
                assert!(!out.contains(claim), "{claim:?} found in:\n{out}");
            }
        }
    }

    /// "Nothing was inspected" and "the text had no words" must never print
    /// alike, and neither is a pass.
    #[test]
    fn an_unavailable_report_says_it_checked_nothing() {
        let report = check::Report::Unavailable(
            "this engine has no letters-to-sounds stage to inspect".to_string(),
        );
        let out = printed(&report);
        assert!(out.contains("checked nothing"), "got:\n{out}");
        assert!(!out.contains("no words to report"), "got:\n{out}");
    }

    /// The JSON envelope carries the same complete word list as the table:
    /// every word, whatever the lexicon says about it.
    #[test]
    fn the_json_envelope_carries_every_word_unfiltered() {
        let words = vec![
            finding("servers", "sˈɜɹvəɹz", true, 1),
            finding("Kokoro", "kəkˈɔɹO", false, 1),
        ];
        let out = PronunciationReportOut {
            engine: "kokoro",
            voice: "af_heart",
            inspectable: true,
            reason: None,
            words: &words,
        };
        let json = serde_json::to_string(&out).expect("serialize");
        assert!(json.contains("\"servers\""), "got {json}");
        assert!(
            json.contains("\"inLexicon\":true") && json.contains("\"inLexicon\":false"),
            "got {json}"
        );
    }
}
