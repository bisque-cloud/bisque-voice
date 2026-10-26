//! Kokoro synthesis on ONNX Runtime.
//!
//! The model pack ships the stock `onnx-community/Kokoro-82M-v1.0-ONNX` export
//! with one addition: the duration predictor's output is promoted to a graph
//! output (`tools/promote-duration.ts`). That tensor is Kokoro's `pred_dur` —
//! per-phoneme frame counts — and it is what makes native word timings possible
//! off macOS. Verified by `frame_sum * 600 == waveform.len()`, exactly.

use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use ort::session::Session;
use ort::value::Tensor;

use crate::g2p::Token;
use crate::timing;

/// The name of the promoted tensor inside the graph.
const DURATION_OUTPUT: &str = "/encoder/Clip_output_0";
/// Kokoro's positional limit, minus the `<bos>`/`<eos>` pads.
pub const MAX_PHONEMES: usize = 508;
/// One duration frame is 600 samples at 24 kHz.
pub const SAMPLES_PER_FRAME: usize = 600;

pub struct Kokoro {
    session: Session,
    vocab: std::collections::HashMap<char, i64>,
    voices_dir: std::path::PathBuf,
    /// Whether an accelerator actually registered for this session. Read when
    /// a run fails, so the error can say "the accelerator did this" only when
    /// one is in play.
    accelerated: bool,
}

/// The device Kokoro's session is actually built with, given what was asked.
///
/// `--device auto` is a promise to use the machine well, not a promise to use
/// CoreML — and CoreML cannot run this graph. It registers, compiles, and then
/// fails at run time for every language (`E5RT ... PropagateInputTensorShapes`),
/// which is past the point where `device::commit_file`'s
/// commit-time fallback can catch it. So on macOS, `auto` resolves to CPU here,
/// where the engine is known-good and fast for an 82M model.
///
/// An explicit `--device gpu` (or `coreml`) is a request the user made and
/// stays honored: the session is built with CoreML and the run-time failure is
/// surfaced with a hint, not pre-empted.
pub fn effective_device(requested: crate::device::Device) -> crate::device::Device {
    resolve_auto(requested, cfg!(target_os = "macos"))
}

/// The platform-independent core of [`effective_device`], split out so the
/// mapping is testable on every platform.
fn resolve_auto(
    requested: crate::device::Device,
    coreml_platform: bool,
) -> crate::device::Device {
    use crate::device::Device;
    match requested {
        Device::Auto if coreml_platform => Device::Cpu,
        other => other,
    }
}

pub struct Synthesis {
    pub samples: Vec<f32>,
    /// Per-input-token frame counts, including the `<bos>`/`<eos>` pads.
    pub frames: Vec<f32>,
}

impl Kokoro {
    pub fn load(pack: &Path, threads: Option<usize>, device: crate::device::Device) -> Result<Self> {
        let requested = device;
        let device = effective_device(requested);
        if device != requested {
            eprintln!(
                "bisque-voice: Kokoro runs on CPU on this platform — CoreML accepts \
                 the graph and then fails it at run time, for every language. \
                 Pass --device gpu to force the accelerator."
            );
        }
        let model_path = pack.join("kokoro.onnx");
        // ort's typed errors aren't Send + Sync, so they can't ride `?` into
        // anyhow; flatten them at each call site.
        let session = crate::device::commit_file(threads, device, &model_path)?;
        // What actually registered — `commit_file` may itself have fallen back
        // to CPU, so ask the record rather than trusting `device`.
        let accelerated = crate::device::last_registration()
            .map(|r| !r.registered.is_empty())
            .unwrap_or(false);

        if !session
            .outputs()
            .iter()
            .any(|o| o.name() == DURATION_OUTPUT)
        {
            bail!(
                "this model has no {DURATION_OUTPUT} output, so it cannot produce word timings.\n\
                 The pack must be built with tools/promote-duration.ts."
            );
        }

        let vocab_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(pack.join("tokenizer.json"))?)
                .context("parsing tokenizer.json")?;
        let vocab = vocab_json
            .pointer("/model/vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("tokenizer.json has no /model/vocab"))?
            .iter()
            .filter_map(|(k, v)| Some((k.chars().next()?, v.as_i64()?)))
            .collect();

        Ok(Kokoro {
            session,
            vocab,
            voices_dir: pack.join("voices"),
            accelerated,
        })
    }

    /// Phonemes → model ids, one id per character.
    ///
    /// Deliberately not lossy. Dropping an unmapped character would shorten the
    /// duration tensor relative to the token walk, and every word after it —
    /// including, reliably, the last one — would fall off the end of the
    /// timings with no error.
    fn encode(&self, phonemes: &str) -> Result<Vec<i64>> {
        let mut ids = Vec::with_capacity(phonemes.chars().count() + 2);
        ids.push(0);
        for c in phonemes.chars() {
            match self.vocab.get(&c) {
                Some(id) => ids.push(*id),
                None => bail!(
                    "phoneme {c:?} (U+{:04X}) is not in Kokoro's vocabulary; \
                     synthesizing it would silently desynchronize every later cue",
                    c as u32
                ),
            }
        }
        ids.push(0);
        Ok(ids)
    }

    /// Drop phonemes Kokoro's vocabulary does not contain, **before** the
    /// duration walk sees them.
    ///
    /// Misaki legitimately emits characters the model cannot encode: Spanish
    /// keeps `¿`, Chinese keeps the full-width `，`, and neither is in the
    /// vocabulary. Kokoro's own pipeline drops them silently —
    /// `filter(lambda i: i is not None, map(vocab.get, phonemes))` in
    /// `KModel.forward` — so dropping them is what the model was trained to
    /// receive, not a liberty.
    ///
    /// What is *not* safe is dropping them inside `encode`, which is why that
    /// function refuses instead. By then the tokens have been counted, and a
    /// character removed from the phoneme string without being removed from the
    /// token that owns it makes the duration tensor shorter than the walk
    /// expects — every later cue slides, with no error anywhere. Filtering here
    /// keeps the tokens and the model input describing the same thing.
    pub fn retain_known(&self, tokens: &mut Vec<crate::g2p::Token>) {
        for token in tokens.iter_mut() {
            if let Some(phonemes) = token.phonemes.as_mut() {
                if phonemes.chars().any(|c| !self.vocab.contains_key(&c)) {
                    *phonemes = phonemes
                        .chars()
                        .filter(|c| self.vocab.contains_key(c))
                        .collect();
                }
            }
        }
    }

    fn voice_row(&self, voice: &str, phoneme_count: usize) -> Result<Vec<f32>> {
        let path = self.voices_dir.join(format!("{voice}.bin"));
        let bytes = std::fs::read(&path)
            .with_context(|| format!("unknown voice {voice:?} (looked for {})", path.display()))?;
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let rows = floats.len() / 256;
        if rows == 0 {
            bail!("voice pack {voice} is empty");
        }
        // Kokoro's packs are indexed by utterance length.
        let row = phoneme_count.min(rows - 1);
        Ok(floats[row * 256..row * 256 + 256].to_vec())
    }

    /// One forward pass over a phoneme string that fits the model's limit.
    pub fn synthesize_chunk(
        &mut self,
        phonemes: &str,
        voice: &str,
        speed: f32,
    ) -> Result<Synthesis> {
        let ids = self.encode(phonemes)?;
        let phoneme_count = ids.len().saturating_sub(2);
        if phoneme_count > MAX_PHONEMES {
            bail!("chunk of {phoneme_count} phonemes exceeds Kokoro's limit of {MAX_PHONEMES}");
        }
        let style = self.voice_row(voice, phoneme_count)?;

        let n = ids.len();
        let input_ids = Tensor::from_array(([1_usize, n], ids))
            .map_err(|e| anyhow!("building input_ids: {e}"))?;
        let style_t = Tensor::from_array(([1_usize, 256_usize], style))
            .map_err(|e| anyhow!("building style: {e}"))?;
        let speed_t = Tensor::from_array(([1_usize], vec![speed]))
            .map_err(|e| anyhow!("building speed: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "style" => style_t,
                "speed" => speed_t,
            ])
            .map_err(|e| {
                if self.accelerated {
                    // Registration succeeding is not the graph running: CoreML
                    // says yes to this graph and then fails inside the run,
                    // which is too late for `commit_file`'s fallback. Name the
                    // way out rather than leaving a runtime backtrace as the
                    // whole answer.
                    anyhow!(
                        "synthesis: {e}\nThe accelerator failed while running \
                         the graph; retry with --device cpu."
                    )
                } else {
                    anyhow!("synthesis: {e}")
                }
            })?;

        let samples = extract_floats(&outputs["waveform"], "waveform")?;
        let frames = extract_floats(&outputs[DURATION_OUTPUT], DURATION_OUTPUT)?;
        // This was the "silent synthesis" defect: the fp16
        // export's vocoder overflows for some (voice, phoneme-sequence) pairs
        // and returns a waveform of 100% NaN while the duration branch stays
        // finite — so duration, word timings and hash all look healthy. NaN
        // then masquerades as digital silence downstream (`f32::max` ignores
        // NaN, `NaN as i16` is 0). The shipped pack is fp32 now, which does not
        // do this; if a pinned fp16 pack ever produces it again, name the real
        // failure at the boundary where it happens instead of shipping zeros.
        if let Some(bad) = samples.iter().find(|s| !s.is_finite()) {
            bail!(
                "the model produced a non-finite sample ({bad}) for voice {voice:?} \
                 — this is the fp16 pack's numeric overflow. \
                 Install the current model pack, which is fp32."
            );
        }
        // The invariant that makes these timings trustworthy: the duration
        // predictor's frames account for exactly the audio produced. If this
        // ever stops holding, `pred_dur` is not what we think it is and every
        // cue would drift silently — so fail closed instead.
        let predicted = frames.iter().sum::<f32>().round() as usize * SAMPLES_PER_FRAME;
        if predicted != samples.len() {
            bail!(
                "duration/waveform mismatch: {} frames predict {predicted} samples but the model \
                 produced {}. Refusing to emit timings that would desynchronize every cue.",
                frames.len(),
                samples.len()
            );
        }
        Ok(Synthesis { samples, frames })
    }
}

/// Split tokens into runs that fit `MAX_PHONEMES`, preferring sentence ends and
/// falling back to any whitespace boundary.
///
/// Not optional: ordinary slide narration exceeds Kokoro's limit, and the macOS
/// engine simply errors out on it.
pub fn chunk_tokens(tokens: &[Token], limit: usize) -> Vec<Vec<Token>> {
    let phoneme_len = |t: &Token| {
        t.phonemes
            .as_deref()
            .map(|p| p.chars().count())
            .unwrap_or(0)
            + t.whitespace.chars().count().min(1)
    };

    let mut chunks: Vec<Vec<Token>> = Vec::new();
    let mut current: Vec<Token> = Vec::new();
    let mut used = 0usize;
    // The last index in `current` that ended a sentence, so an over-long run can
    // be cut at a natural pause rather than mid-clause.
    let mut last_sentence_end: Option<usize> = None;

    for token in tokens {
        let len = phoneme_len(token);
        if used + len > limit && !current.is_empty() {
            match last_sentence_end {
                Some(idx) if idx + 1 < current.len() => {
                    let rest = current.split_off(idx + 1);
                    chunks.push(std::mem::replace(&mut current, rest));
                }
                _ => chunks.push(std::mem::take(&mut current)),
            }
            used = current.iter().map(phoneme_len).sum();
            last_sentence_end = None;
        }
        current.push(token.clone());
        used += len;
        if token.text.ends_with(['.', '!', '?', '…']) {
            last_sentence_end = Some(current.len() - 1);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    // The phoneme string handed to the model is trimmed, so a chunk's final
    // token must not claim a trailing space. Leaving it there makes the
    // timestamp walk consume one frame too many and silently drop the last
    // word of every chunk but the last.
    for chunk in chunks.iter_mut() {
        if let Some(last) = chunk.last_mut() {
            last.whitespace.clear();
        }
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str, phonemes: &str, ws: &str) -> Token {
        Token {
            text: text.into(),
            phonemes: Some(phonemes.into()),
            whitespace: ws.into(),
            ..Token::default()
        }
    }

    #[test]
    fn short_input_is_a_single_chunk() {
        let tokens = vec![tok("hi", "hˈI", " "), tok("there", "ðˈɛɹ", "")];
        let chunks = chunk_tokens(&tokens, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 2);
    }

    #[test]
    fn long_input_splits_and_keeps_every_token() {
        let tokens: Vec<Token> = (0..50)
            .map(|i| tok(&format!("w{i}"), "abcde", " "))
            .collect();
        let chunks = chunk_tokens(&tokens, 60);
        assert!(chunks.len() > 1, "should have split");
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total, tokens.len(), "no token may be lost at a boundary");
        for c in &chunks {
            let len: usize = c
                .iter()
                .map(|t| t.phonemes.as_deref().unwrap().chars().count() + 1)
                .sum();
            assert!(
                len <= 60 || c.len() == 1,
                "chunk of {len} exceeds the limit"
            );
        }
    }

    #[test]
    fn splits_prefer_a_sentence_boundary() {
        let tokens = vec![
            tok("one", "wˈʌn", " "),
            tok("two.", "tˈu.", " "),
            tok("three", "θɹˈi", " "),
            tok("four", "fˈɔɹ", ""),
        ];
        // Room for roughly the first two tokens.
        let chunks = chunk_tokens(&tokens, 11);
        assert_eq!(
            chunks[0].last().unwrap().text,
            "two.",
            "cut after the period"
        );
    }

    #[test]
    fn a_single_oversized_token_still_emits_rather_than_looping() {
        let tokens = vec![tok("supercalifragilistic", &"a".repeat(200), "")];
        let chunks = chunk_tokens(&tokens, 50);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 1);
    }

    #[test]
    fn every_chunk_ends_without_a_trailing_space() {
        // The model is fed a trimmed phoneme string, so a chunk whose last
        // token still claims a space makes the timestamp walk overrun and drop
        // that word.
        let tokens: Vec<Token> = (0..20)
            .map(|i| tok(&format!("w{i}"), "abcde", " "))
            .collect();
        let chunks = chunk_tokens(&tokens, 30);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert_eq!(
                c.last().unwrap().whitespace,
                "",
                "chunk must not end in whitespace"
            );
        }
    }

    /// The filter has to run on the tokens, not on the joined string, or the
    /// duration walk and the model input stop describing the same thing.
    ///
    /// Spanish keeps `¿` and Chinese keeps the full-width `，`; neither is in
    /// Kokoro's vocabulary, and before this the engine refused to narrate them
    /// at all.
    #[test]
    fn unencodable_phonemes_are_dropped_from_the_token_that_owns_them() {
        let vocab: std::collections::HashMap<char, i64> =
            "kˈomo ".chars().map(|c| (c, 1)).collect();
        let kokoro_vocab = vocab;

        let mut tokens = vec![
            tok("¿cómo", "¿kˈomo", " "),
            tok("estás?", "kom", ""),
        ];
        // Stand-in for Kokoro::retain_known, which needs a loaded session.
        for token in tokens.iter_mut() {
            if let Some(p) = token.phonemes.as_mut() {
                *p = p.chars().filter(|c| kokoro_vocab.contains_key(c)).collect();
            }
        }

        assert_eq!(
            tokens[0].phonemes.as_deref(),
            Some("kˈomo"),
            "the inverted question mark is gone"
        );
        // The token still exists and still owns its run, so the walk stays 1:1.
        assert_eq!(tokens.len(), 2);
        assert!(tokens.iter().all(|t| t.phonemes.is_some()));
        assert_eq!(
            crate::g2p::phoneme_string(&tokens),
            "kˈomo kom",
            "what the model is fed is exactly what the tokens describe"
        );
    }

    /// The default invocation must work on the headline platform. CoreML fails
    /// this graph at run time for every language, so `auto` — the default —
    /// resolves to CPU on macOS.
    #[test]
    fn auto_resolves_to_cpu_on_a_coreml_platform() {
        use crate::device::Device;
        assert_eq!(resolve_auto(Device::Auto, true), Device::Cpu);
        // Elsewhere, `auto` keeps its meaning: accelerate when possible.
        assert_eq!(resolve_auto(Device::Auto, false), Device::Auto);
    }

    /// Explicit choices are the user's and pass through untouched — `gpu` on
    /// macOS still builds a CoreML session (whose run-time failure is then
    /// surfaced, not hidden), and `cpu` stays `cpu` everywhere.
    #[test]
    fn explicit_devices_are_honored_on_every_platform() {
        use crate::device::Device;
        for coreml_platform in [true, false] {
            assert_eq!(resolve_auto(Device::Cpu, coreml_platform), Device::Cpu);
            assert_eq!(resolve_auto(Device::Gpu, coreml_platform), Device::Gpu);
        }
    }

    #[test]
    fn effective_device_matches_the_platform_this_was_built_for() {
        use crate::device::Device;
        if cfg!(target_os = "macos") {
            assert_eq!(effective_device(Device::Auto), Device::Cpu);
        } else {
            assert_eq!(effective_device(Device::Auto), Device::Auto);
        }
        assert_eq!(effective_device(Device::Gpu), Device::Gpu);
        assert_eq!(effective_device(Device::Cpu), Device::Cpu);
    }

    /// Regression for the silent-synthesis defect: on the
    /// fp16 pack, these exact (voice, text) pairs deterministically returned a
    /// waveform of 100% NaN — reported as digital silence with healthy
    /// duration, timings and hash. The fp32 pack synthesizes them; this pins
    /// that, and the non-finite check in `synthesize_chunk` would name the
    /// real failure if it ever recurred.
    ///
    /// Needs a real model, so it is gated: set `BISQUE_VOICE_TEST_KOKORO_PACK`
    /// to an installed Kokoro pack directory (the one `bisque-voice install
    /// kokoro` creates under `~/.bisque/models/<pack_version>/`). Unset, the
    /// test passes vacuously — CI for the release binary runs it against the
    /// pack it just built.
    #[test]
    fn previously_silent_pairs_synthesize_audible_speech() {
        let Ok(pack) = std::env::var("BISQUE_VOICE_TEST_KOKORO_PACK") else {
            eprintln!(
                "previously_silent_pairs_synthesize_audible_speech: skipped \
                 (BISQUE_VOICE_TEST_KOKORO_PACK is not set)"
            );
            return;
        };
        let pack = std::path::PathBuf::from(pack);
        let models_root = pack.parent().expect("pack dir has a parent").to_path_buf();
        let mut engine = KokoroEngine::load(&pack, None, crate::device::Device::Cpu, &models_root)
            .expect("loading the Kokoro pack");

        // The minimal repro pair, and the (voice, text) pair that shipped a
        // silent slide in a real presentation build. Both are English so the
        // pack alone suffices — no per-language G2P download in a test.
        let cases: &[(&str, &str)] = &[
            ("af_heart", "Test word1 word2."),
            (
                "bf_emma",
                "Every presentation starts as a rough idea and a handful of \
                 bullet points. The work that follows is mechanical.",
            ),
        ];
        for (voice, text) in cases {
            let narration = crate::engine::Engine::narrate(&mut engine, text, voice, 1.0, None, &mut crate::quality::Blind)
                .unwrap_or_else(|e| panic!("{voice} on {text:?} failed: {e}"));
            assert!(
                narration.samples.iter().all(|s| s.is_finite()),
                "{voice} produced non-finite samples"
            );
            let peak = crate::audio::peak_dbfs(&narration.samples);
            assert!(
                peak > crate::audio::SILENCE_FLOOR_DBFS,
                "{voice} on {text:?} is silent again (peak {peak} dBFS)"
            );
        }
    }

    /// Regression for the October 5 Hacker News episode, whose narration
    /// failed on all three attempts: a sentence ending in a pronunciation
    /// marker of two or more shown words made bisque-voice 0.1.34 report
    /// "timestamp walk produced 2 timings for 3 tokens". Gated on a real pack,
    /// like the test above.
    #[test]
    fn a_sentence_ending_in_a_multi_word_marker_narrates_with_one_timing_per_word() {
        let Ok(pack) = std::env::var("BISQUE_VOICE_TEST_KOKORO_PACK") else {
            eprintln!(
                "a_sentence_ending_in_a_multi_word_marker_narrates_with_one_timing_per_word: \
                 skipped (BISQUE_VOICE_TEST_KOKORO_PACK is not set)"
            );
            return;
        };
        let pack = std::path::PathBuf::from(pack);
        let models_root = pack.parent().expect("pack dir has a parent").to_path_buf();
        let mut engine = KokoroEngine::load(&pack, None, crate::device::Device::Cpu, &models_root)
            .expect("loading the Kokoro pack");
        for (text, words) in [
            ("see [New York](new york).", 3),
            ("see [New York City](new-york).", 4),
            ("see [New York](new york), then.", 4),
            (
                "The administrators noticed it on the evening of Friday, \
                 [October 2](October second).",
                11,
            ),
        ] {
            let narration =
                crate::engine::Engine::narrate(&mut engine, text, "af_heart", 1.0, None, &mut crate::quality::Blind)
                    .unwrap_or_else(|e| panic!("{text:?} failed: {e}"));
            assert_eq!(narration.words.len(), words, "{text:?}");
            let audio = narration.samples.len() as f64 / crate::audio::SAMPLE_RATE as f64;
            assert!(narration.words.windows(2).all(|w| w[0].end <= w[1].start + 1e-9));
            assert!(narration.words.last().unwrap().end <= audio + 1e-6, "{text:?}");
        }
    }

    #[test]
    fn tokens_without_phonemes_do_not_consume_the_budget() {
        let tokens = vec![
            Token {
                text: "$".into(),
                phonemes: None,
                whitespace: "".into(),
                ..Default::default()
            },
            tok("five", "fˈIv", ""),
        ];
        assert_eq!(chunk_tokens(&tokens, 10).len(), 1);
    }
}

/// Read a tensor as `f32` whether the graph stores it as float or half.
///
/// The shipping pack is the fp32 export (the fp16 export's vocoder overflows
/// to NaN for some voice/text pairs), so both tensors come
/// back as `f32`; an fp16 pack returns `f16`. Accept either so the engine
/// works against any variant a build pins.
fn extract_floats(value: &ort::value::DynValue, what: &str) -> Result<Vec<f32>> {
    if let Ok((_, data)) = value.try_extract_tensor::<f32>() {
        return Ok(data.to_vec());
    }
    match value.try_extract_tensor::<half::f16>() {
        Ok((_, data)) => Ok(data.iter().map(|v| v.to_f32()).collect()),
        Err(e) => Err(anyhow!("reading {what}: {e}")),
    }
}

/// The Kokoro engine as the registry sees it: G2P, chunking, synthesis and
/// timestamping behind the one `narrate` call every engine implements.
pub struct KokoroEngine {
    kokoro: Kokoro,
    /// One G2P per language, built on first use. Which one runs is decided by
    /// the voice id, not by a flag — see `g2p::router`.
    g2p: crate::g2p::router::Router,
    /// This machine's one voice-model slot. Last, so `Drop` frees the session
    /// before the next waiter is let in.
    _host_guard: crate::host_guard::HostGuard,
}

impl KokoroEngine {
    pub fn load(
        pack: &Path,
        threads: Option<usize>,
        device: crate::device::Device,
        models_root: &Path,
    ) -> Result<Self> {
        // No G2P is built here. Each language's pronunciation data is a
        // separate download, and the largest — UniDic, for Japanese — is 204 MB
        // on disk; loading it for someone narrating English would be paying the
        // whole cost of the feature to not use it.
        //
        // Before the session is built: wait for any other voice model on this
        // machine. A running Kokoro holds about 1.8 GB, and three at once was
        // part of what panicked the mini.
        let host_guard = crate::host_guard::admit("kokoro", &mut std::io::stderr())?;
        Ok(KokoroEngine {
            kokoro: Kokoro::load(pack, threads, device)?,
            g2p: crate::g2p::router::Router::new(pack, models_root),
            _host_guard: host_guard,
        })
    }
}

impl crate::engine::Engine for KokoroEngine {
    /// The phoneme stream — Kokoro has an inspectable G2P stage, which is
    /// exactly what makes a pronunciation regression findable.
    fn debug_tokens(&mut self, text: &str, voice: &str) -> Vec<(String, String)> {
        self.g2p
            .phonemize(voice, text)
            .unwrap_or_default()
            .into_iter()
            .map(|t| (t.text, t.phonemes.unwrap_or_default()))
            .collect()
    }

    fn pronunciation_report(&mut self, text: &str, voice: &str) -> Result<crate::check::Report> {
        let (_, voice) = crate::engine::split_voice(voice);
        // Deliberately `?` and not `unwrap_or_default()`. Swallowing the error
        // here reported an unreadable voice, a missing data pack and a failed
        // lexicon load as an empty finding list, which printed as "nothing
        // flagged" — a clean bill of health for text nothing had looked at.
        self.g2p.report(voice, text)
    }

    fn narrate(
        &mut self,
        text: &str,
        voice: &str,
        speed: f32,
        // Kokoro's language comes from the voice id, not from the caller: every
        // voice speaks exactly one language, and `af_heart` saying Japanese
        // would be wrong however the caller labelled the text.
        _language: Option<&str>,
        // Kokoro is deterministic: the same text and voice return the same
        // samples, so a retake could never fix what the gate rejected.
        _gate: &mut dyn crate::quality::Gate,
    ) -> Result<crate::engine::Narration> {
        let (_, voice) = crate::engine::split_voice(voice);
        let mut tokens = self.g2p.phonemize(voice, text)?;
        // Before anything counts these tokens: strip what the model cannot
        // encode, so the token list and the model input stay the same shape.
        self.kokoro.retain_known(&mut tokens);
        if let Some(missing) = tokens.iter().find(|t| t.phonemes.is_none()) {
            bail!(
                "no pronunciation for {:?}; refusing to narrate a word as silence",
                missing.text
            );
        }

        let chunks = chunk_tokens(&tokens, MAX_PHONEMES);
        let mut samples: Vec<f32> = Vec::new();
        let mut words: Vec<crate::reconcile::Word> = Vec::new();

        for chunk in &chunks {
            // Deliberately not trimmed. A chunk can legitimately begin with a
            // token that has no phonemes (a bare ">" or "$"), whose separating
            // space is a real model token; trimming it shortens the duration
            // tensor by one and drops the chunk's last word.
            let phonemes = crate::g2p::phoneme_string(chunk);
            let synth = self.kokoro.synthesize_chunk(&phonemes, voice, speed)?;

            // Offset by the audio already emitted, so a chunked narration and a
            // single-pass one are indistinguishable.
            let offset = samples.len() as f64 / crate::audio::SAMPLE_RATE as f64;
            let mut chunk_words = timing::word_timings(chunk, &synth.frames);
            for w in chunk_words.iter_mut() {
                w.start += offset;
                w.end += offset;
            }
            // One timing per token, or the walk ran off the end of the duration
            // tensor and the reconciler would quietly paper over it.
            if chunk_words.len() != chunk.len() {
                bail!(
                    "timestamp walk produced {} timings for {} tokens; \
                     the phoneme stream and the duration tensor are out of step",
                    chunk_words.len(),
                    chunk.len()
                );
            }
            words.extend(chunk_words);
            samples.extend_from_slice(&synth.samples);
        }

        Ok(crate::engine::Narration {
            samples,
            words,
            chunks: chunks.len(),
        })
    }
}
