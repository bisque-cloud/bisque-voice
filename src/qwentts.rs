//! Qwen3-TTS on the vendored qwentts.cpp — voice cloning and named speakers.
//!
//! One struct serves both qwentts-backed engine families, because they are the
//! same runtime around different talker checkpoints: the `qwen3-clone-*`
//! engines load the `base` talker and condition on a cloned voice's latents,
//! the `qwen3-voices-*` engines load the `customvoice` talker and condition on
//! a named speaker from the table baked into its GGUF. Upstream's modes are
//! per-checkpoint and mutually exclusive (`--ref-*` is "Base only",
//! `--speaker` is "CustomVoice only"), which is why the conditioning is a
//! branch on the engine's declared capability rather than a free mix.
//!
//! Kokoro narrates with exact word timings behind a 337 MB download. These
//! decode one 12.5 Hz frame at a time behind GGUF pairs measured in gigabytes,
//! and what that buys is what Kokoro can never do: speak in a voice supplied
//! as a recording, or in one of Qwen's own nine studio voices.
//!
//! Two things follow from measurement and are worth
//! stating where the code is, because both look like bugs otherwise.
//!
//! **Word timings are approximate, by design.** An autoregressive codec model
//! emits audio frames, not word boundaries. Like the ONNX Qwen path, `narrate`
//! returns one empty span per chunk and lets `reconcile()` distribute it, which
//! still yields exactly one timing per source token and reports `exact: false`.
//! Anyone wanting measured timings installs a forced aligner, which measures the
//! waveform this produced and replaces these outright.
//!
//! **The first synthesis of a process is much slower than the rest.** On Metal
//! the shader pipelines compile on first use — measured at 25.2 s wall against
//! 8.2 s of actual work. It is a fixed cost per process, not per narration, and
//! benchmarking a cold run reads as a 3x regression that is not there.

use std::ffi::{CStr, CString};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

use crate::engine::{Engine, Narration};
use crate::reconcile::Word;
use crate::voices::{ClonedVoice, VoiceManifest};

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code
)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/qwentts_ffi.rs"));
}

/// Chunk budget, in `cost_of` units, shared with the ONNX Qwen path.
///
/// The talker stops emitting end-of-utterance past a certain amount of text and
/// decodes to its frame cap instead, so this is a correctness bound rather than
/// a memory one — see the chunking section of `qwen.rs`.
const CHUNK_BUDGET: usize = 300;

/// What the reference clip must be resampled to before the speaker encoder
/// sees it. The codec runs at 24 kHz and the ABI takes the buffer as-is.
pub const REF_SAMPLE_RATE: u32 = 24_000;

pub struct QwenCloneEngine {
    ctx: *mut ffi::qt_context,
    /// The engine id this context was opened as. A cloned voice records the
    /// same string, and the two must agree or the speaker embedding is the
    /// wrong width for this talker.
    engine_id: String,
    /// True when this context holds a CustomVoice talker, whose voices are
    /// the named speakers in its own table rather than cloned latents.
    named_speakers: bool,
    library: crate::voices::VoiceLibrary,
    /// This machine's one heavy-model slot. Last, so `Drop` frees the context
    /// before the next waiter is let in.
    _host_guard: crate::host_guard::HostGuard,
}

// The handle owns its backend and `qt_synthesize` is documented thread safe,
// but nothing here shares one across threads; `Send` is what lets the engine be
// boxed into the same `Box<dyn Engine>` the rest of the CLI passes around.
unsafe impl Send for QwenCloneEngine {}

impl QwenCloneEngine {
    /// Open a pack. The pack holds both GGUFs; the talker carries the LM, the
    /// code predictor and the speaker encoder, and the codec is the shared
    /// 12.5 Hz tokenizer.
    pub fn load(
        pack: &Path,
        engine_id: &str,
        models_root: &Path,
        talker_file: &str,
    ) -> Result<Self> {
        let talker = pack.join(talker_file);
        let codec = pack.join(CODEC_FILE);
        for (label, path) in [("talker", &talker), ("codec", &codec)] {
            if !path.is_file() {
                bail!(
                    "the {engine_id} pack is missing its {label} weights ({}). \
                     Reinstall it: bisque-voice install {engine_id}",
                    path.display()
                );
            }
        }

        // Before any weights are mapped: wait for any other Qwen run on this
        // machine, then refuse a 1.7B load on a disk too full to swap.
        let host_guard = crate::host_guard::admit(engine_id, &mut std::io::stderr())?;

        let talker_c = cstring(&talker.to_string_lossy())?;
        let codec_c = cstring(&codec.to_string_lossy())?;

        // SAFETY: `qt_init_default_params` fills every field including
        // `abi_version`, so the two paths are the only things set by hand. The
        // C side copies both strings during `qt_init`, so they only need to
        // outlive the call.
        let ctx = unsafe {
            let mut params = std::mem::zeroed::<ffi::qt_init_params>();
            ffi::qt_init_default_params(&mut params);
            params.talker_path = talker_c.as_ptr();
            params.codec_path = codec_c.as_ptr();
            ffi::qt_init(&params)
        };
        if ctx.is_null() {
            bail!(
                "loading the {engine_id} model failed: {}",
                last_error().unwrap_or_else(|| "no diagnostic from qwentts".into())
            );
        }

        Ok(Self {
            ctx,
            engine_id: engine_id.to_string(),
            named_speakers: crate::engine::named_speakers(engine_id),
            library: crate::voices::VoiceLibrary::new(models_root),
            _host_guard: host_guard,
        })
    }

    /// The loaded talker's own speaker table. Nine names for a CustomVoice
    /// checkpoint, empty for base — the model, not the registry, is the
    /// authority on what it can speak as.
    fn speaker_names(&self) -> Vec<String> {
        // SAFETY: `qt_speaker_name` returns a pointer into the context's own
        // table, valid while the context lives; copied out immediately.
        unsafe {
            let n = ffi::qt_n_speakers(self.ctx).max(0);
            (0..n)
                .filter_map(|i| {
                    let p = ffi::qt_speaker_name(self.ctx, i);
                    (!p.is_null()).then(|| CStr::from_ptr(p).to_string_lossy().into_owned())
                })
                .collect()
        }
    }

    /// Run the speaker encoder and the codec encoder over a reference clip.
    ///
    /// This is the expensive half of cloning and it happens once. `audio` is
    /// mono f32 at 24 kHz; the caller is responsible for having resampled it,
    /// since the ABI takes the buffer verbatim.
    pub fn extract_voice(
        &mut self,
        name: &str,
        audio: &[f32],
        transcript: &str,
        source_clip: Option<String>,
        source_id: Option<String>,
    ) -> Result<ClonedVoice> {
        if audio.is_empty() {
            bail!("the reference clip decoded to no audio");
        }
        if transcript.trim().is_empty() {
            bail!(
                "a transcript of the reference clip is required — ICL cloning \
                 conditions on what the clip actually says, and guessing it \
                 degrades the voice."
            );
        }

        // SAFETY: `out` is zero-initialised as the header requires, and is
        // freed through `qt_voice_ref_free` on every path below.
        let mut out = unsafe { std::mem::zeroed::<ffi::qt_voice_ref>() };
        let status = unsafe {
            ffi::qt_extract_voice_ref(
                self.ctx,
                audio.as_ptr(),
                audio.len() as std::os::raw::c_int,
                &mut out,
            )
        };
        if status != ffi::qt_status_QT_STATUS_OK {
            unsafe { ffi::qt_voice_ref_free(&mut out) };
            bail!(
                "extracting the voice reference failed: {}",
                status_text(status)
            );
        }

        let spk_dim = out.ref_spk_dim.max(0) as usize;
        let codebooks = out.num_codebooks.max(0) as usize;
        let ref_t = out.ref_T.max(0) as usize;
        if out.ref_spk_emb.is_null() || out.ref_codes.is_null() || spk_dim == 0 || ref_t == 0 {
            unsafe { ffi::qt_voice_ref_free(&mut out) };
            bail!("the voice reference came back empty — the clip may be too short");
        }

        // SAFETY: both pointers are non-null and the header documents their
        // lengths as the dimensions returned alongside them. Copied out before
        // the C buffers are released.
        let (spk, codes) = unsafe {
            (
                std::slice::from_raw_parts(out.ref_spk_emb, spk_dim).to_vec(),
                std::slice::from_raw_parts(out.ref_codes, codebooks * ref_t).to_vec(),
            )
        };
        unsafe { ffi::qt_voice_ref_free(&mut out) };

        Ok(ClonedVoice {
            manifest: VoiceManifest {
                name: name.to_string(),
                engine: self.engine_id.clone(),
                spk_dim,
                num_codebooks: codebooks,
                ref_t,
                transcript: transcript.trim().to_string(),
                source_clip,
                source_id,
            },
            spk,
            codes,
        })
    }

    pub fn library(&self) -> &crate::voices::VoiceLibrary {
        &self.library
    }

    /// Synthesize one chunk — in the default voice, a cloned one, or a named
    /// speaker. `voice` and `speaker` are mutually exclusive by construction:
    /// `narrate` sets whichever its engine's mode takes, and the C header
    /// rejects both together anyway.
    fn synthesize(
        &mut self,
        text: &str,
        language: Option<&str>,
        voice: Option<&ClonedVoice>,
        speaker: Option<&str>,
    ) -> Result<Vec<f32>> {
        let text_c = cstring(text)?;
        let lang_c = language
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(cstring)
            .transpose()?;
        let speaker_c = speaker.map(cstring).transpose()?;
        let ref_text_c = voice.map(|v| cstring(&v.manifest.transcript)).transpose()?;

        let mut audio = unsafe { std::mem::zeroed::<ffi::qt_audio>() };
        let status = unsafe {
            let mut params = std::mem::zeroed::<ffi::qt_tts_params>();
            ffi::qt_tts_default_params(&mut params);
            params.text = text_c.as_ptr();
            // NULL selects the model's "auto" branch, which infers the language
            // from the text. Stating it is not cosmetic on this family of
            // models — see the `Engine::narrate` contract.
            params.lang = lang_c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
            // CustomVoice speaker conditioning: the name selects a row of the
            // talker's own speaker table (resolved case-insensitively by the
            // C side, which also swaps in a dialect speaker's language id
            // when the stated language is Chinese or unstated).
            params.speaker = speaker_c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());

            if let (Some(v), Some(ref_text)) = (voice, ref_text_c.as_ref()) {
                // ICL clone mode B: embedding + codes + transcript. The three
                // travel together; the header rejects any partial set.
                params.ref_spk_emb = v.spk.as_ptr();
                params.ref_spk_dim = v.spk.len() as std::os::raw::c_int;
                params.ref_codes = v.codes.as_ptr();
                params.ref_T = v.manifest.ref_t as std::os::raw::c_int;
                params.ref_text = ref_text.as_ptr();
            }
            ffi::qt_synthesize(self.ctx, &params, &mut audio)
        };

        if status != ffi::qt_status_QT_STATUS_OK {
            unsafe { ffi::qt_audio_free(&mut audio) };
            bail!("synthesis failed: {}", status_text(status));
        }
        let n = audio.n_samples.max(0) as usize;
        if audio.samples.is_null() || n == 0 {
            unsafe { ffi::qt_audio_free(&mut audio) };
            bail!("synthesis returned no audio for {text:?}");
        }
        // SAFETY: non-null with `n_samples` valid floats, per `struct qt_audio`.
        let samples = unsafe { std::slice::from_raw_parts(audio.samples, n).to_vec() };
        // Read before freeing — `qt_audio_free` resets the struct to empty.
        let rate = audio.sample_rate as u32;
        unsafe { ffi::qt_audio_free(&mut audio) };

        // The whole pipeline is fixed at the codec's rate; anything else would
        // play back at the wrong pitch with nothing downstream saying so.
        if rate != REF_SAMPLE_RATE {
            bail!(
                "the model returned {rate} Hz audio but this pipeline is fixed \
                 at {REF_SAMPLE_RATE} Hz"
            );
        }
        Ok(samples)
    }
}

impl Drop for QwenCloneEngine {
    fn drop(&mut self) {
        // SAFETY: `ctx` came from `qt_init` and is freed exactly once; `qt_free`
        // is documented safe on NULL.
        unsafe { ffi::qt_free(self.ctx) };
    }
}

impl Engine for QwenCloneEngine {
    fn narrate(
        &mut self,
        text: &str,
        voice: &str,
        speed: f32,
        language: Option<&str>,
        gate: &mut dyn crate::quality::Gate,
    ) -> Result<Narration> {
        // Resolve the voice before any decoding: a typo should not cost a
        // multi-second autoregressive run that is then thrown away.
        let (_, voice_name) = crate::engine::split_voice(voice);
        let (cloned, speaker) = if self.named_speakers {
            // A CustomVoice engine's voices are its own speaker table; the
            // cloned-voice library does not apply, and there is no anonymous
            // default worth offering when nine named voices exist. Validate
            // against the model itself — the wrong name gets the real list.
            let names = self.speaker_names();
            let resolved = names
                .iter()
                .find(|n| n.eq_ignore_ascii_case(voice_name))
                .cloned();
            match resolved {
                Some(name) => (None, Some(name)),
                None => bail!(
                    "{:?} is not a speaker of {}. Its voices: {}",
                    voice_name,
                    self.engine_id,
                    names.join(", ")
                ),
            }
        } else {
            let cloned = match voice_name {
                "" | DEFAULT_VOICE => None,
                name => Some(
                    self.library
                        .load(name, &self.engine_id)
                        .with_context(|| format!("resolving voice {name:?}"))?,
                ),
            };
            (cloned, None)
        };

        // This engine has no phoneme stage, so an author override can only be
        // honoured by changing the text it is given. `spoken_text` reduces each
        // marker to its substitute; the display words never come here, they go
        // to the aligner and the word list in `main`. Rewriting before chunking
        // matters: `chunk_text` splits on sentence boundaries and knows nothing
        // about `[...](...)`, so a marker could straddle two chunks.
        let spoken = crate::pronounce::spoken_text(text);
        let chunks = crate::qwen_common::chunk_text(&spoken, CHUNK_BUDGET);
        if chunks.is_empty() {
            bail!("--text has nothing to narrate");
        }

        let mut parts: Vec<(Vec<f32>, Vec<Word>)> = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            let label = format!("chunk {}/{}", i + 1, chunks.len());
            // This model samples its speech, so a take can come out garbled
            // where the next one is clean. The gate judges each take before
            // it joins the narration, and retakes only this chunk.
            let pcm = crate::quality::take_until_accepted(&label, chunk, gate, || {
                let pcm = self
                    .synthesize(chunk, language, cloned.as_ref(), speaker.as_deref())
                    .with_context(|| format!("synthesizing {label} — {chunk:?}"))?;
                // Retimed per chunk so each chunk's own duration is the offset
                // the chunks after it are shifted by.
                Ok(crate::qwen_common::resample(&pcm, speed))
            })?;
            let duration = pcm.len() as f64 / f64::from(crate::audio::SAMPLE_RATE);
            parts.push((
                pcm,
                vec![Word {
                    word: String::new(),
                    start: 0.0,
                    end: duration,
                }],
            ));
        }

        let (samples, spans) = crate::qwen_common::concat_chunks(parts, crate::audio::SAMPLE_RATE);
        crate::profile::dump(samples.len() as f64 / f64::from(crate::audio::SAMPLE_RATE));

        // One deliberately empty word; see the module doc.
        let words = vec![Word {
            word: String::new(),
            start: spans.first().map_or(0.0, |w| w.start),
            end: spans.last().map_or(0.0, |w| w.end),
        }];

        Ok(Narration {
            samples,
            words,
            chunks: chunks.len(),
        })
    }
}

/// The voice name that means "this model's own voice, no reference clip".
pub const DEFAULT_VOICE: &str = "default";

/// The codec GGUF is shared by every talker size, so its name is fixed.
pub const CODEC_FILE: &str = "qwen-tokenizer-12hz-Q8_0.gguf";

fn cstring(s: &str) -> Result<CString> {
    // An interior NUL cannot be passed to a C string API, and truncating at it
    // would silently narrate less text than the caller supplied.
    CString::new(s).map_err(|_| anyhow!("text contains a NUL byte, which cannot be synthesized"))
}

/// The C side's diagnostic for the last failure, if it left one.
fn last_error() -> Option<String> {
    // SAFETY: returns a static or thread-local buffer owned by the library;
    // copied into an owned String before returning.
    unsafe {
        let p = ffi::qt_last_error();
        if p.is_null() {
            return None;
        }
        let s = CStr::from_ptr(p).to_string_lossy().trim().to_string();
        (!s.is_empty()).then_some(s)
    }
}

/// Render a status code together with whatever the library said about it. The
/// code alone is not actionable, and the message alone loses the distinction
/// between "you asked for something this mode forbids" and "generation failed".
fn status_text(status: ffi::qt_status) -> String {
    let name = match status {
        ffi::qt_status_QT_STATUS_INVALID_PARAMS => "invalid parameters",
        ffi::qt_status_QT_STATUS_MODE_INVALID => "not valid for this model's mode",
        ffi::qt_status_QT_STATUS_GENERATE_FAILED => "generation failed",
        ffi::qt_status_QT_STATUS_OOM => "out of memory",
        ffi::qt_status_QT_STATUS_CANCELLED => "cancelled",
        _ => "unknown error",
    };
    match last_error() {
        Some(detail) => format!("{name} ({detail})"),
        None => name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bindings are generated from the vendored header at build time; if
    /// the submodule moves to a layout this crate does not address, that must
    /// fail loudly here rather than by reading fields at the wrong offsets.
    #[test]
    fn the_vendored_abi_is_the_one_these_bindings_target() {
        assert_eq!(ffi::QT_ABI_VERSION, 4);
        assert_eq!(ffi::QT_ABI_MIN_VERSION, 4);
    }

    #[test]
    fn the_library_reports_the_commit_it_was_built_from() {
        // SAFETY: documented as a static string, valid for the process.
        let version = unsafe { CStr::from_ptr(ffi::qt_version()) }
            .to_string_lossy()
            .to_string();
        assert!(!version.is_empty(), "qt_version returned nothing");
    }

    #[test]
    fn text_with_an_interior_nul_is_refused_rather_than_truncated() {
        assert!(cstring("before\0after").is_err());
    }

    #[test]
    fn every_status_code_renders_something_actionable() {
        for code in [
            ffi::qt_status_QT_STATUS_INVALID_PARAMS,
            ffi::qt_status_QT_STATUS_MODE_INVALID,
            ffi::qt_status_QT_STATUS_GENERATE_FAILED,
            ffi::qt_status_QT_STATUS_OOM,
            ffi::qt_status_QT_STATUS_CANCELLED,
        ] {
            assert!(!status_text(code).is_empty());
        }
    }
}
