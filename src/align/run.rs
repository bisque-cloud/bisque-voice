//! The aligner implementation — the Whisper front end, prompt assembly, the
//! monotonicity repair, and the ONNX forward pass.
//!
//! Split from the registry (`align.rs`) because this half needs ONNX Runtime
//! and the registry must survive on builds that have none — platform support
//! is per-engine, and the studio-only build (`--no-default-features`, the
//! Intel-Mac release) still lists the aligners and says why it cannot run
//! them. Everything here compiles only with the `onnx` feature.

use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use ort::session::Session;
use ort::value::{DynValue, Tensor, TensorElementType};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::reconcile::{source_words, Word};

// ── audio front end ─────────────────────────────────────────────────────────

/// The aligner's Whisper feature extractor, from the pack's
/// `preprocessor_config.json`. These are checked against the pack rather than
/// assumed — one implementation of one front end, and a pack that wants a
/// different one must say so loudly instead of being fed the wrong features.
const MEL_BINS: usize = 128;
const N_FFT: usize = 400;
const HOP: usize = 160;
const ALIGN_RATE: usize = 16_000;
/// `mel_floor` in HF's `spectrogram()` — the clamp applied after the mel
/// projection and before `log10`.
const MEL_FLOOR: f64 = 1e-10;
/// Whisper's dynamic-range clamp: everything more than 8 decades below the
/// loudest mel bin is flattened to that floor.
const LOG_SPEC_RANGE: f64 = 8.0;

/// Resample 24 kHz (what every engine here emits) to the 16 kHz the aligner's
/// feature extractor is defined for.
///
/// Band-limited windowed-sinc rather than the linear interpolation `qwen.rs`
/// uses for `--speed`: this is a *downsample*, so everything above 8 kHz folds
/// back into the passband unless it is filtered out first. Linear interpolation
/// would alias sibilance down onto the vowels and move the very energy the
/// aligner reads to find word boundaries.
///
/// Taps are normalized per output sample, so DC is preserved exactly and the
/// edges (where the kernel is clamped to the signal) need no special case.
pub(crate) fn resample_24k_to_16k(input: &[f32]) -> Vec<f32> {
    /// Input samples per output sample: 24000 / 16000.
    const RATIO: f64 = 1.5;
    /// Cutoff in cycles per *input* sample — the output's Nyquist.
    const CUTOFF: f64 = 0.5 / RATIO;
    /// Sinc zero-crossings kept each side. 16 gives a ~49-tap kernel here,
    /// which is far more stopband than 80 ms timestamp resolution can notice
    /// and still costs nothing next to a 0.6B forward pass.
    const ZEROS: f64 = 16.0;

    if input.len() < 2 {
        return input.to_vec();
    }
    let half = ZEROS / (2.0 * CUTOFF);
    let last = input.len() - 1;
    let out_len = (input.len() as f64 / RATIO).floor() as usize;
    (0..out_len)
        .map(|i| {
            let center = i as f64 * RATIO;
            let first = (center - half).ceil() as isize;
            let stop = (center + half).floor() as isize;
            let mut acc = 0.0f64;
            let mut norm = 0.0f64;
            for k in first..=stop {
                let x = center - k as f64;
                // Hann taper over the kernel's support.
                let w = 0.5 + 0.5 * (std::f64::consts::PI * x / half).cos();
                let tap = w * sinc(2.0 * CUTOFF * x) * 2.0 * CUTOFF;
                let idx = k.clamp(0, last as isize) as usize;
                acc += input[idx] as f64 * tap;
                norm += tap;
            }
            if norm.abs() < 1e-12 {
                0.0
            } else {
                (acc / norm) as f32
            }
        })
        .collect()
}

/// Normalized sinc, `sin(pi x) / (pi x)`.
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Slaney-scale mel, the librosa/`htk=False` variant HF's
/// `WhisperFeatureExtractor` builds its filterbank on: linear below 1 kHz,
/// logarithmic above.
fn hertz_to_mel(freq: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP; // 15.0
    if freq >= MIN_LOG_HZ {
        min_log_mel + (freq / MIN_LOG_HZ).ln() * 27.0 / 6.4f64.ln()
    } else {
        freq / F_SP
    }
}

fn mel_to_hertz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    if mel >= min_log_mel {
        MIN_LOG_HZ * ((mel - min_log_mel) * 6.4f64.ln() / 27.0).exp()
    } else {
        mel * F_SP
    }
}

/// Frequency bins in a `n_fft`-point real spectrum.
const fn n_freqs() -> usize {
    N_FFT / 2 + 1
}

/// The 128 × 201 triangular mel filterbank, row-major `[mel][frequency]`.
///
/// A transcription of HF's `mel_filter_bank(num_frequency_bins=201,
/// num_mel_filters=128, min_frequency=0, max_frequency=8000,
/// sampling_rate=16000, norm="slaney", mel_scale="slaney")`. The Slaney area
/// normalization is not cosmetic: it changes every filter's gain, which shifts
/// the whole log-mel surface the aligner was trained on.
pub(crate) fn mel_filters() -> Vec<f64> {
    let bins = n_freqs();
    let mel_min = hertz_to_mel(0.0);
    let mel_max = hertz_to_mel(ALIGN_RATE as f64 / 2.0);
    // num_mel_filters + 2 band edges: each filter rises from edge i to i+1 and
    // falls to i+2.
    let edges: Vec<f64> = (0..MEL_BINS + 2)
        .map(|i| {
            let m = mel_min + (mel_max - mel_min) * i as f64 / (MEL_BINS + 1) as f64;
            mel_to_hertz(m)
        })
        .collect();
    let fft_freqs: Vec<f64> = (0..bins)
        .map(|i| ALIGN_RATE as f64 / 2.0 * i as f64 / (bins - 1) as f64)
        .collect();

    let mut out = vec![0.0f64; MEL_BINS * bins];
    for m in 0..MEL_BINS {
        let (lo, mid, hi) = (edges[m], edges[m + 1], edges[m + 2]);
        // Slaney norm: each filter integrates to a constant area rather than a
        // constant peak, so wide high-frequency bands are not louder.
        let enorm = 2.0 / (hi - lo);
        for (f, &freq) in fft_freqs.iter().enumerate() {
            let down = (freq - lo) / (mid - lo);
            let up = (hi - freq) / (hi - mid);
            out[m * bins + f] = down.min(up).max(0.0) * enorm;
        }
    }
    out
}

/// A `[MEL_BINS, frames]` log-mel surface, row-major — the layout the graph's
/// `input_features [1, 128, T]` wants.
pub(crate) struct Mel {
    pub frames: usize,
    pub data: Vec<f64>,
}

/// numpy's `pad(mode="reflect")` index mapping: mirror without repeating the
/// edge sample, and keep mirroring for signals shorter than the pad.
fn reflect(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let n = n as isize;
    let period = 2 * n - 2;
    let mut k = i.rem_euclid(period);
    if k >= n {
        k = period - k;
    }
    k as usize
}

/// Whisper log-mel features for 16 kHz mono audio.
///
/// Transcribed from `WhisperFeatureExtractor._np_extract_fbank_features`:
/// centred STFT with reflect padding, periodic Hann, power spectrum, mel
/// projection floored at 1e-10, `log10`, **drop the final frame**, clamp to
/// eight decades below the maximum, then `(x + 4) / 4`. Every one of those
/// steps is load-bearing — the last two put the features in the range the
/// encoder was trained on, and skipping the frame drop shifts every timestamp
/// by 10 ms.
pub(crate) fn log_mel(samples: &[f32], filters: &[f64]) -> Result<Mel> {
    let frames = samples.len() / HOP;
    if frames == 0 {
        bail!(
            "the audio is {} samples at {ALIGN_RATE} Hz — shorter than one \
             {HOP}-sample feature frame, so there is nothing to align against",
            samples.len()
        );
    }
    let bins = n_freqs();
    // Periodic Hann, i.e. numpy's `hanning(N + 1)[:-1]`, which is what
    // `window_function(n_fft, "hann")` builds.
    let window: Vec<f64> = (0..N_FFT)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N_FFT as f64).cos())
        .collect();

    let fft = FftPlanner::<f64>::new().plan_fft_forward(N_FFT);
    let mut buf = vec![Complex::new(0.0f64, 0.0f64); N_FFT];
    let mut power = vec![0.0f64; bins];
    let mut out = vec![0.0f64; MEL_BINS * frames];

    for t in 0..frames {
        // `center=True`: the frame is centred on sample t*HOP, so it starts
        // HOP/2… no — N_FFT/2 before it, in reflect-padded coordinates.
        let start = t as isize * HOP as isize - (N_FFT / 2) as isize;
        for (k, (slot, w)) in buf.iter_mut().zip(&window).enumerate() {
            let v = samples[reflect(start + k as isize, samples.len())] as f64;
            *slot = Complex::new(v * w, 0.0);
        }
        fft.process(&mut buf);
        for (p, c) in power.iter_mut().zip(&buf[..bins]) {
            *p = c.norm_sqr();
        }
        for (m, filt) in filters.chunks_exact(bins).enumerate() {
            let acc: f64 = filt.iter().zip(&power).map(|(a, b)| a * b).sum();
            out[m * frames + t] = acc.max(MEL_FLOOR).log10();
        }
    }

    let max = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let floor = max - LOG_SPEC_RANGE;
    for v in &mut out {
        *v = (v.max(floor) + 4.0) / 4.0;
    }
    Ok(Mel { frames, data: out })
}

/// One stride-2, kernel-3, padding-1 convolution's output length.
fn conv_stride2(len: i64) -> i64 {
    // Python floor division, which for the len == 0 case yields 0 rather than 1.
    (len - 1).div_euclid(2) + 1
}

/// How many audio embedding positions the encoder produces for `mel_frames`
/// frames — and therefore how many times `<|audio_pad|>` is repeated.
///
/// **Derived, not guessed.** The encoder does not run over the whole surface at
/// once: it splits the mel into chunks of `n_window * 2` frames and runs the
/// three stride-2 convolutions independently per chunk, so the total is not
/// `T / 8`. This is a transcription of
/// `Qwen3ASRProcessor._get_audio_token_length` / the model's
/// `_get_feat_extract_output_lengths`, with the per-chunk count computed from
/// `chunk_len` rather than hardcoded to upstream's `13` — 13 is exactly
/// `conv_stride2³(100)`, so the two agree at the shipped `n_window = 50` and
/// this one stays correct if a pack ever ships another window.
///
/// Getting this wrong is not a subtle degradation: the graph scatters audio
/// embeddings into the `<|audio_pad|>` positions, so a miscount either fails
/// the run or silently shifts every word.
pub(crate) fn audio_token_count(mel_frames: usize, n_window: usize) -> Result<usize> {
    if n_window == 0 {
        bail!("the pack's config.json declares audio_config.n_window = 0, which cannot be chunked");
    }
    let chunk = (n_window * 2) as i64;
    let frames = mel_frames as i64;
    let per_chunk = conv_stride2(conv_stride2(conv_stride2(chunk)));
    let tail = conv_stride2(conv_stride2(conv_stride2(frames % chunk)));
    Ok((tail + (frames / chunk) * per_chunk) as usize)
}

// ── the pack's declarations ─────────────────────────────────────────────────

/// The four added tokens the prompt is built from. Read from the pack, never
/// guessed: a pack rebuilt from a different checkpoint must fail here rather
/// than emit a prompt the graph will happily score as nonsense.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Specials {
    pub audio_start: i64,
    pub audio_pad: i64,
    pub audio_end: i64,
    pub timestamp: i64,
}

impl Specials {
    /// Resolve the four ids from `tokenizer_config.json`'s
    /// `added_tokens_decoder`, then cross-check every one that `config.json`
    /// also names. Two files disagreeing means the pack was assembled from
    /// mismatched sources, and that is not something to average out.
    fn load(pack: &Path, config: &serde_json::Value) -> Result<Self> {
        let path = pack.join("tokenizer_config.json");
        let raw: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        )
        .with_context(|| format!("parsing {}", path.display()))?;
        let decoder = raw
            .get("added_tokens_decoder")
            .and_then(|d| d.as_object())
            .ok_or_else(|| {
                anyhow!(
                    "{} has no `added_tokens_decoder`, so the aligner's special token ids cannot \
                     be resolved. This build will not guess them.",
                    path.display()
                )
            })?;

        let find = |content: &str| -> Option<i64> {
            decoder.iter().find_map(|(id, entry)| {
                (entry.get("content")?.as_str()? == content).then(|| id.parse::<i64>().ok())?
            })
        };
        let wanted = [
            "<|audio_start|>",
            "<|audio_pad|>",
            "<|audio_end|>",
            "<timestamp>",
        ];
        let found: Vec<Option<i64>> = wanted.iter().map(|t| find(t)).collect();
        let missing: Vec<&str> = wanted
            .iter()
            .zip(&found)
            .filter(|(_, id)| id.is_none())
            .map(|(t, _)| *t)
            .collect();
        if !missing.is_empty() {
            bail!(
                "{} declares no id for {}. This pack is not a Qwen3 forced-aligner checkpoint; \
                 reinstall it.",
                path.display(),
                missing.join(", ")
            );
        }

        let specials = Specials {
            audio_start: found[0].expect("checked above"),
            audio_pad: found[1].expect("checked above"),
            audio_end: found[2].expect("checked above"),
            timestamp: found[3].expect("checked above"),
        };

        // config.json names the same ids from the model's side. Where it does,
        // the two must agree.
        for (key, tokenizer_id, token) in [
            ("audio_start_token_id", specials.audio_start, "<|audio_start|>"),
            ("audio_token_id", specials.audio_pad, "<|audio_pad|>"),
            ("audio_end_token_id", specials.audio_end, "<|audio_end|>"),
            ("timestamp_token_id", specials.timestamp, "<timestamp>"),
        ] {
            if let Some(from_config) = config.get(key).and_then(|v| v.as_i64()) {
                if from_config != tokenizer_id {
                    bail!(
                        "the pack disagrees with itself about {token}: config.json says {key} = \
                         {from_config}, tokenizer_config.json says {tokenizer_id}. Reinstall it."
                    );
                }
            }
        }
        Ok(specials)
    }
}

/// What `config.json` and `preprocessor_config.json` pin.
#[derive(Debug)]
struct AlignConfig {
    /// Milliseconds per timestamp unit — 80 on this checkpoint.
    segment_ms: f64,
    /// Mel frames per encoder chunk is `n_window * 2`.
    n_window: usize,
}

impl AlignConfig {
    fn load(pack: &Path, config: &serde_json::Value) -> Result<Self> {
        // The shipped checkpoint nests this under `thinker_config`; accept a
        // top-level one too, since the ASR configs are not consistent about it
        // across variants and both spellings mean the same thing.
        let audio = config
            .get("thinker_config")
            .and_then(|t| t.get("audio_config"))
            .or_else(|| config.get("audio_config"))
            .ok_or_else(|| {
                anyhow!(
                    "the pack's config.json has no `audio_config` (looked at \
                     `thinker_config.audio_config` and the top level); it is not a Qwen3-ASR \
                     checkpoint"
                )
            })?;
        let segment_ms = config
            .get("timestamp_segment_time")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| {
                anyhow!(
                    "the pack's config.json has no `timestamp_segment_time`, so a predicted \
                     timestamp cannot be converted to milliseconds"
                )
            })?;
        if !(segment_ms.is_finite() && segment_ms > 0.0) {
            bail!("the pack declares timestamp_segment_time = {segment_ms}, which is not a duration");
        }
        let n_window = audio
            .get("n_window")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                anyhow!(
                    "the pack's config.json has no `audio_config.n_window`, so the number of audio \
                     embedding positions cannot be derived. This build will not guess it."
                )
            })? as usize;
        let mel_bins = audio
            .get("num_mel_bins")
            .and_then(|v| v.as_u64())
            .unwrap_or(MEL_BINS as u64) as usize;
        if mel_bins != MEL_BINS {
            bail!(
                "the pack wants {mel_bins} mel bins; this build extracts {MEL_BINS}. \
                 Feeding it the wrong feature size would produce confident nonsense."
            );
        }
        check_feature_extractor(pack)?;
        Ok(AlignConfig {
            segment_ms,
            n_window,
        })
    }
}

/// The front end above is one specific configuration. Assert the pack asks for
/// that one rather than silently extracting features it was not trained on.
fn check_feature_extractor(pack: &Path) -> Result<()> {
    let path = pack.join("preprocessor_config.json");
    let raw: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))?;
    for (key, want) in [
        ("feature_size", MEL_BINS as f64),
        ("n_fft", N_FFT as f64),
        ("hop_length", HOP as f64),
        ("sampling_rate", ALIGN_RATE as f64),
        // Dither adds noise to the spectrum. Upstream ships 0; anything else
        // would make alignment non-reproducible run to run.
        ("dither", 0.0),
    ] {
        let got = raw.get(key).and_then(|v| v.as_f64()).ok_or_else(|| {
            anyhow!("{} does not declare {key}; refusing to assume it", path.display())
        })?;
        if (got - want).abs() > f64::EPSILON {
            bail!(
                "{} declares {key} = {got}, but this build's Whisper front end is fixed at {want}",
                path.display()
            );
        }
    }
    Ok(())
}

// ── the prompt ──────────────────────────────────────────────────────────────

/// `<|audio_start|>` `<|audio_pad|>`×N `<|audio_end|>`, then each word followed
/// by its two `<timestamp>` slots (start and end).
///
/// Assembled from ids rather than tokenized as one string: the four markers are
/// *added* tokens that live in `tokenizer_config.json`, so the raw BPE the pack
/// ships (`vocab.json` + `merges.txt`) would shred them into ordinary subwords.
/// Upstream joins the words with `"<timestamp><timestamp>"` and no separator,
/// so each word is encoded with no leading space — which is what this does.
pub(crate) fn prompt_ids(sp: &Specials, audio_tokens: usize, words: &[Vec<i64>]) -> Vec<i64> {
    let mut ids = Vec::with_capacity(2 + audio_tokens + words.iter().map(|w| w.len() + 2).sum::<usize>());
    ids.push(sp.audio_start);
    ids.resize(ids.len() + audio_tokens, sp.audio_pad);
    ids.push(sp.audio_end);
    for word in words {
        ids.extend_from_slice(word);
        ids.push(sp.timestamp);
        ids.push(sp.timestamp);
    }
    ids
}

/// Where the timestamps are read back from — upstream's
/// `output_id[input_id == timestamp_token_id]`, two entries per word in order.
pub(crate) fn timestamp_positions(ids: &[i64], timestamp: i64) -> Vec<usize> {
    ids.iter()
        .enumerate()
        .filter(|(_, &id)| id == timestamp)
        .map(|(i, _)| i)
        .collect()
}

// ── monotonicity repair ─────────────────────────────────────────────────────

/// Repair a timestamp sequence that is not non-decreasing.
///
/// An exact port of `Qwen3ForceAlignProcessor.fix_timestamp`. The model
/// predicts each word's two timestamps independently, so a single bad
/// prediction can put a word's end before its start, or a word before its
/// predecessor — which would make cues jump backwards. The repair:
///
/// 1. Take the longest non-decreasing subsequence; those positions are trusted.
/// 2. A run of at most two untrusted values snaps to whichever trusted
///    neighbour is nearer *in index* (ties go left).
/// 3. A longer run is linearly interpolated between the bracketing trusted
///    values; with only one side available, it is copied.
///
/// The two branches are genuinely different rules, not an optimization of one
/// another — a short run keeps a real measured value, a long one spreads the
/// gap evenly — so both are kept verbatim rather than unified.
pub(crate) fn fix_timestamp(data: &[f64]) -> Vec<i64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    // Longest non-decreasing subsequence, O(n²) exactly as upstream. n is two
    // per word, so this is a few hundred at worst.
    let mut dp = vec![1usize; n];
    let mut parent = vec![usize::MAX; n];
    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }
    let max_length = *dp.iter().max().expect("n > 0");
    // `list.index` returns the FIRST maximum; which subsequence is chosen when
    // several tie changes the output, so this must not become an rposition.
    let max_idx = dp.iter().position(|&d| d == max_length).expect("n > 0");

    let mut is_normal = vec![false; n];
    let mut idx = max_idx;
    loop {
        is_normal[idx] = true;
        if parent[idx] == usize::MAX {
            break;
        }
        idx = parent[idx];
    }

    let mut result = data.to_vec();
    let mut i = 0usize;
    while i < n {
        if is_normal[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && !is_normal[j] {
            j += 1;
        }
        let anomaly_count = j - i;

        // The nearest trusted value on each side. Because the run is maximal,
        // these sit at exactly i-1 and j when they exist.
        let left = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
        let right = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);

        if anomaly_count <= 2 {
            for (k, slot) in result.iter_mut().enumerate().take(j).skip(i) {
                *slot = match (left, right) {
                    (None, Some(r)) => r,
                    (Some(l), None) => l,
                    (Some(l), Some(r)) => {
                        // Distance to the left neighbour (i-1) against the
                        // right one (j); ties go left.
                        if (k as isize - (i as isize - 1)) <= (j as isize - k as isize) {
                            l
                        } else {
                            r
                        }
                    }
                    (None, None) => *slot,
                };
            }
        } else {
            match (left, right) {
                (Some(l), Some(r)) => {
                    let step = (r - l) / (anomaly_count + 1) as f64;
                    for (k, slot) in result.iter_mut().enumerate().take(j).skip(i) {
                        *slot = l + step * (k - i + 1) as f64;
                    }
                }
                (Some(l), None) => result[i..j].fill(l),
                (None, Some(r)) => result[i..j].fill(r),
                (None, None) => {}
            }
        }
        i = j;
    }

    // Python's `int()`: truncation toward zero. Every value here is a
    // non-negative millisecond count, so this is a floor.
    result.iter().map(|v| *v as i64).collect()
}

// ── the graph ───────────────────────────────────────────────────────────────

/// The graph's four inputs, in the order upstream's export declares them.
const GRAPH_INPUTS: [&str; 4] = [
    "input_ids",
    "attention_mask",
    "input_features",
    "feature_attention_mask",
];

// ── tokenizing for the model ────────────────────────────────────────────────

/// Strip a source word down to what the aligner was trained on.
///
/// Upstream's `tokenize_space_lang` splits on whitespace and then keeps only
/// letters, digits and the apostrophe (`clean_token`). We must feed the model
/// the same thing: sending `machine.` where it expects `machine` puts the
/// trailing punctuation in its own token position, and the timestamp it
/// predicts there collapses — every sentence- and clause-final word came back
/// with a zero-length span until this matched.
///
/// Returns empty for a word that is entirely punctuation; `model_tokens`
/// decides what to do about that, because the answer depends on the guarantee
/// rather than on the cleaning.
fn clean_token(word: &str) -> String {
    word.chars()
        .filter(|c| *c == '\'' || c.is_alphanumeric())
        .collect()
}

/// Han ideographs — the ranges upstream's `is_cjk_char` lists, verbatim.
///
/// Chinese writes no spaces, so whitespace tokenization hands the model an
/// entire sentence as one token with exactly two `<timestamp>` slots. Upstream
/// splits a Han run per character and that is what the checkpoint was trained
/// to place, so it is what we send.
fn is_han(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF        // CJK Unified Ideographs
        | 0x3400..=0x4DBF      // Extension A
        | 0x20000..=0x2A6DF    // Extension B
        | 0x2A700..=0x2B73F    // Extension C
        | 0x2B740..=0x2B81F    // Extension D
        | 0x2B820..=0x2CEAF    // Extension E
        | 0xF900..=0xFAFF      // Compatibility Ideographs
    )
}

/// Kana that stand as a syllable of their own.
///
/// **This is a deliberate departure from upstream, and the only one.** Its
/// `is_cjk_char` covers Han and nothing else, because its Japanese path routes
/// through a morphological analyzer (`nagisa`) that we do not have and cannot
/// acquire without a dictionary the size of the aligner itself. Without kana
/// here, a fully-kana sentence — which has no spaces and no Han either — falls
/// straight back into the one-token-per-utterance bug that this whole section
/// exists to fix. A kana is a mora, i.e. the same order of unit as a Han
/// syllable, so splitting on it asks the model the same *kind* of question it
/// was trained on rather than a new one.
fn is_kana_lead(c: char) -> bool {
    !is_kana_tail(c)
        && matches!(
            c as u32,
            0x3041..=0x3096        // Hiragana
            | 0x309F               // ゟ (hiragana digraph yori)
            | 0x30A1..=0x30FA      // Katakana
            | 0x30FF               // ヿ (katakana digraph koto)
            | 0xFF66..=0xFF9D      // halfwidth katakana
        )
}

/// Kana marks that belong to the preceding mora rather than carrying one of
/// their own: the small kana of a yōon (`きょ`) or sokuon (`って`), the
/// long-vowel mark (`ー`), the iteration marks, and the standalone or combining
/// voiced-sound marks.
///
/// These are split off only at the cost of creating a token with no acoustic
/// segment to call its own — exactly the zero-length-span failure `clean_token`
/// was written to avoid — so they attach leftwards instead.
///
/// The sokuon `っ` is the one genuine judgement call: its geminate closure sits
/// *between* the two syllables, so either side is arguable, and attaching it
/// left is chosen because it is the side that never leaves a token with nothing
/// of its own to be measured against. The cost is at most one mora of boundary,
/// well inside the checkpoint's 80 ms resolution. `ヵ`/`ヶ` are deliberately
/// absent from this list: small in shape only, they are read as full morae
/// (`三ヶ月`), so they lead.
///
/// The dakuten entries are belt-and-braces — `clean_token` already drops them,
/// exactly as upstream's does, since none are letters or digits — and are listed
/// so this stays correct if the cleaning is ever loosened.
fn is_kana_tail(c: char) -> bool {
    matches!(
        c as u32,
        // small hiragana ぁぃぅぇぉっゃゅょゎ
        0x3041 | 0x3043 | 0x3045 | 0x3047 | 0x3049 | 0x3063 | 0x3083 | 0x3085
        | 0x3087 | 0x308E
        // small katakana ァィゥェォッャュョヮ
        | 0x30A1 | 0x30A3 | 0x30A5 | 0x30A7 | 0x30A9 | 0x30C3 | 0x30E3 | 0x30E5
        | 0x30E7 | 0x30EE
        // combining + standalone dakuten/handakuten, and the ゝゞ iteration marks
        | 0x3099..=0x309E
        // ー ヽ ヾ
        | 0x30FC..=0x30FE
        // halfwidth small kana, halfwidth ｰ, halfwidth ﾞﾟ
        | 0xFF67..=0xFF70 | 0xFF9E | 0xFF9F
    )
}

/// A token as the *model* sees it, and the source whitespace token it came
/// from.
///
/// Two indices are in play here and they are emphatically not the same one: the
/// prompt carries one entry per `ModelToken`, so the graph predicts one
/// `[start, end]` per entry; the caller is promised one span per SOURCE
/// whitespace token. `source` is the bridge, and `fold_onto_source` is the
/// crossing.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ModelToken {
    /// What the model is asked to place. Cleaned, and never empty.
    pub text: String,
    /// Index into `source_words(..)` — the whitespace token this came from.
    pub source: usize,
}

/// Split one already-cleaned source word into the pieces the model is prompted
/// with: Han and kana per syllable, everything else kept as a whole run.
///
/// Upstream's `split_segment_with_chinese` in one line is "flush the buffer on
/// every CJK char"; this adds only the kana handling described above. A Latin
/// run inside a Han sentence (`Qwen3` in `使用Qwen3模型`) stays one token, which
/// is both what upstream does and what the whitespace path already did.
fn split_for_model(cleaned: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut run = String::new();
    for ch in cleaned.chars() {
        if is_han(ch) || is_kana_lead(ch) {
            if !run.is_empty() {
                out.push(std::mem::take(&mut run));
            }
            out.push(ch.to_string());
        } else if is_kana_tail(ch)
            && run.is_empty()
            && out
                .last()
                .and_then(|t| t.chars().next_back())
                .is_some_and(|p| is_han(p) || is_kana_lead(p) || is_kana_tail(p))
        {
            // Attach to the syllable it modifies. Guarded on `run.is_empty()`
            // so a stray mark after Latin text does not reach back across it.
            out.last_mut().expect("checked by the guard above").push(ch);
        } else {
            run.push(ch);
        }
    }
    if !run.is_empty() {
        out.push(run);
    }
    out
}

/// The full prompt tokenization: every source whitespace token, cleaned and
/// split, tagged with the index it came from.
///
/// **Every source word yields at least one token, always.** Upstream drops a
/// token that cleans to nothing; we cannot, because a dropped source word would
/// have no measured span to fold back and the returned list would stop being
/// 1:1 with `source_words`. A word that is entirely punctuation therefore keeps
/// its original text — an empty token position is something the model cannot
/// place at all, whereas `—` is merely something it places badly.
fn model_tokens(words: &[&str]) -> Vec<ModelToken> {
    let mut out = Vec::with_capacity(words.len());
    for (i, w) in words.iter().enumerate() {
        let mut pieces = split_for_model(&clean_token(w));
        if pieces.is_empty() {
            pieces.push((*w).to_string());
        }
        out.extend(pieces.into_iter().map(|text| ModelToken { text, source: i }));
    }
    out
}

/// Fold per-model-token spans back onto source whitespace tokens.
///
/// A source word's span is `[min(start), max(end)]` over its own model tokens.
/// `fix_timestamp` already makes the flat sequence non-decreasing, so in
/// practice this is the first token's start and the last one's end — but min/max
/// does not *depend* on that repair having succeeded, and a fold that silently
/// inverted a word would be worse than one that is merely wide.
///
/// `fixed` is milliseconds, two entries per model token, in prompt order.
fn fold_onto_source(words: &[&str], tokens: &[ModelToken], fixed: &[i64]) -> Result<Vec<Word>> {
    if fixed.len() != tokens.len() * 2 {
        bail!(
            "the repair returned {} timestamps for {} model tokens; expected two each",
            fixed.len(),
            tokens.len()
        );
    }
    let mut spans: Vec<Option<(i64, i64)>> = vec![None; words.len()];
    for (k, token) in tokens.iter().enumerate() {
        let (start, end) = (fixed[k * 2], fixed[k * 2 + 1]);
        let slot = spans.get_mut(token.source).ok_or_else(|| {
            anyhow!(
                "model token {:?} claims source word {} of {}; the tokenization and the source \
                 have diverged",
                token.text,
                token.source,
                words.len()
            )
        })?;
        *slot = Some(match *slot {
            None => (start, end),
            Some((s, e)) => (s.min(start), e.max(end)),
        });
    }
    words
        .iter()
        .zip(spans)
        .enumerate()
        .map(|(i, (w, span))| {
            // Unreachable via `model_tokens`, which emits at least one token per
            // source word — but this is the invariant the 1:1 guarantee rests
            // on, so it is checked rather than assumed.
            let (start, end) = span.ok_or_else(|| {
                anyhow!(
                    "source word {i} ({w:?}) got no model token, so it has no measured span. \
                     Refusing to invent one."
                )
            })?;
            Ok(Word {
                word: (*w).to_string(),
                start: start as f64 / 1000.0,
                end: end as f64 / 1000.0,
            })
        })
        .collect()
}

// ── running the alignment ───────────────────────────────────────────────────

/// What one alignment measured, at both granularities.
pub struct Measured {
    /// One span per source whitespace token — the published contract.
    pub words: Vec<Word>,
    /// One span per token the model was actually asked to place, before the
    /// fold-back. This is the unit a collapse is visible in: a Chinese
    /// sentence is one source word but twenty model tokens, so judging the
    /// folded words would see nothing go wrong in it.
    pub tokens: Vec<crate::quality::TokenSpan>,
}

/// A loaded aligner: the pack read and the graph committed once, so one
/// narration can be measured chunk by chunk for the retake gate and then
/// whole for its timings without loading a gigabyte of weights each time.
pub struct Aligner {
    cfg: AlignConfig,
    specials: Specials,
    tokenizer: tokenizers::Tokenizer,
    session: Session,
}

impl Aligner {
    pub fn open(pack: &Path, threads: Option<usize>, device: crate::device::Device) -> Result<Self> {
        let config_path = pack.join("config.json");
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?,
        )
        .with_context(|| format!("parsing {}", config_path.display()))?;
        let cfg = AlignConfig::load(pack, &config)?;
        let specials = Specials::load(pack, &config)?;
        // Text: the same Qwen byte-level BPE the TTS engine uses, so the
        // pre-tokenizer pattern lives in exactly one place.
        let tokenizer = crate::qwen_common::load_qwen_bpe(
            pack,
            "the aligner pack",
            "bisque-voice install <aligner>",
        )?;
        let session = open_graph(pack, threads, device)?;
        Ok(Self {
            cfg,
            specials,
            tokenizer,
            session,
        })
    }

    /// Measure per-word `[start, end]` for `audio_24k` against `text`.
    ///
    /// The returned `Vec<Word>` is 1:1 with `text`'s whitespace tokens and carries
    /// their exact surface strings, which is what lets `reconcile()` take its EXACT
    /// path — these timings are measured, not distributed.
    ///
    /// **Two tokenizations, deliberately.** What the model is prompted with is not
    /// what comes back out:
    ///
    /// - The *model* sees `model_tokens()`: punctuation stripped, Han and kana split
    ///   per syllable, every other run kept whole. That is the unit the checkpoint
    ///   was trained to place, and for Chinese or Japanese it is the difference
    ///   between timings and no timings — a Chinese sentence carries no spaces, so
    ///   as a single whitespace token it gets exactly two `<timestamp>` slots for
    ///   the whole utterance, which is why it used to come back as one word spanning
    ///   0.00–5.04.
    /// - The *caller* gets one span per source whitespace token, folded back as
    ///   `[min(start), max(end)]` over that token's model tokens.
    ///
    /// The fold-back is not a concession, it **is** the guarantee: cue markers index
    /// source whitespace tokens (see `reconcile`'s module doc, and
    /// `parseTokenStream` in `packages/presentation-format`), so handing back a
    /// finer list would silently re-base every cue in every already-published
    /// presentation. For a space-free Chinese sentence that still means one span for
    /// the caller — but it is now a measured min/max over ~20 per-character
    /// predictions rather than a single two-slot guess, and for mixed text every
    /// Latin word in it now has Han neighbours the model could place separately.
    ///
    /// Making the per-syllable timings *visible* would mean changing what a "word"
    /// is end to end — here, `reconcile::source_words`, `parseTokenStream`, and the
    /// meaning of every stored `marker.wordIndex`. That is a product decision about
    /// a published contract, not something this file may take unilaterally, so the
    /// finer unit stops at `ModelToken` and the boundary is right here.
    pub fn measure(&mut self, audio_24k: &[f32], text: &str) -> Result<Measured> {
        let words = source_words(text);
        if words.is_empty() {
            return Ok(Measured {
                words: Vec::new(),
                tokens: Vec::new(),
            });
        }
        let cfg = &self.cfg;
        let specials = self.specials;

        // Audio: 24 kHz in, 16 kHz mel out, padded to a whole number of encoder
        // chunks. Padding the *waveform* rather than the mel surface means the
        // encoder sees real (silent) audio instead of an out-of-distribution
        // constant, and it makes the real and padded frame counts identical — so
        // there is exactly one possible answer for how many `<|audio_pad|>` tokens
        // the prompt needs, whichever length the graph derives it from. Upstream's
        // own export validation only covered chunk-aligned lengths, so this is also
        // the shape that was actually tested.
        let audio_16k = resample_24k_to_16k(audio_24k);
        let chunk_frames = cfg.n_window * 2;
        if chunk_frames == 0 {
            bail!("the pack declares a zero-frame encoder chunk");
        }
        let real_frames = audio_16k.len() / HOP;
        let padded_frames = real_frames.div_ceil(chunk_frames) * chunk_frames;
        let mut padded = audio_16k;
        padded.resize(padded_frames * HOP, 0.0);

        let mel = log_mel(&padded, &mel_filters())?;
        if mel.frames != padded_frames {
            bail!(
                "feature extraction produced {} frames for a {padded_frames}-frame pad — the front \
                 end and the chunk padding disagree",
                mel.frames
            );
        }
        let audio_tokens = audio_token_count(mel.frames, cfg.n_window)?;
        if audio_tokens == 0 {
            bail!("the audio is too short for the aligner's encoder to produce a single position");
        }

        let tokens = model_tokens(&words);
        let token_ids: Vec<Vec<i64>> = tokens
            .iter()
            .map(|t| {
                let ids = crate::qwen_common::encode_with(&self.tokenizer, &t.text)?;
                if ids.is_empty() {
                    bail!("the pack's tokenizer encoded the token {:?} to nothing", t.text);
                }
                Ok(ids)
            })
            .collect::<Result<_>>()?;

        let ids = prompt_ids(&specials, audio_tokens, &token_ids);
        let positions = timestamp_positions(&ids, specials.timestamp);
        // Two slots per MODEL token, not per source word — for Chinese those two
        // counts differ by a factor of twenty, and reading the wrong one back would
        // shift every span.
        if positions.len() != tokens.len() * 2 {
            bail!(
                "built {} timestamp slots for {} model tokens; the prompt is malformed",
                positions.len(),
                tokens.len()
            );
        }

        let session = &mut self.session;
        let seq = ids.len();

        // Integer width is read off the graph, not assumed. The export metadata
        // documents these as int64, but the shipped q4 graph declares int32 — and a
        // mismatch is a hard `ort` error, not a coercion. Different exports of the
        // same model disagree here, so ask rather than pick.
        let int_ty = |name: &str| -> Result<TensorElementType> {
            session
                .inputs()
                .iter()
                .find(|i| i.name() == name)
                .map(|i| i.dtype().tensor_type())
                .ok_or_else(|| anyhow!("the aligner graph has no `{name}` input"))?
                .ok_or_else(|| anyhow!("the aligner graph's `{name}` is not a tensor"))
        };
        let ints = |name: &str, v: Vec<i64>, shape: Vec<usize>| -> Result<DynValue> {
            Ok(match int_ty(name)? {
                TensorElementType::Int32 => {
                    let v32: Vec<i32> = v.into_iter().map(|x| x as i32).collect();
                    Tensor::from_array((shape, v32))
                        .map_err(|e| anyhow!("building {name}: {e}"))?
                        .into_dyn()
                }
                _ => Tensor::from_array((shape, v))
                    .map_err(|e| anyhow!("building {name}: {e}"))?
                    .into_dyn(),
            })
        };

        let input_ids = ints("input_ids", ids, vec![1, seq])?;
        let attention_mask = ints("attention_mask", vec![1i64; seq], vec![1, seq])?;
        let feature_mask = ints(
            "feature_attention_mask",
            vec![1i64; mel.frames],
            vec![1, mel.frames],
        )?;
        let features: Vec<f32> = mel.data.iter().map(|v| *v as f32).collect();
        let input_features = Tensor::from_array(([1usize, MEL_BINS, mel.frames], features))
            .map_err(|e| anyhow!("building input_features: {e}"))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "input_features" => input_features,
                "feature_attention_mask" => feature_mask,
            ])
            .map_err(|e| anyhow!("forced-alignment forward pass: {e}"))?;
        if outputs.len() != 1 {
            bail!(
                "the aligner graph returned {} outputs; expected only `logits`",
                outputs.len()
            );
        }
        let (picked, sure) = argmax_at(&outputs[0], seq, &positions)?;

        // Predicted units → milliseconds → monotonic repair → fold back onto source
        // whitespace tokens → seconds. The repair runs over the *model* tokens,
        // before the fold: it is the model's own prediction sequence that can go
        // backwards, and repairing after folding would have nothing left to repair
        // with.
        let ms: Vec<f64> = picked.iter().map(|id| *id as f64 * cfg.segment_ms).collect();
        let fixed = fix_timestamp(&ms);
        // The fold checks there are two repaired timestamps per model token, so it
        // runs before anything indexes them.
        let folded = fold_onto_source(&words, &tokens, &fixed)?;
        let spans = tokens
            .iter()
            .enumerate()
            .map(|(k, t)| crate::quality::TokenSpan {
                text: t.text.clone(),
                start_ms: fixed[k * 2],
                end_ms: fixed[k * 2 + 1],
                sure: Some(sure[k * 2].min(sure[k * 2 + 1])),
            })
            .collect();
        Ok(Measured {
            words: folded,
            tokens: spans,
        })
    }
}

fn open_graph(pack: &Path, threads: Option<usize>, device: crate::device::Device) -> Result<Session> {
    let path = pack.join("model.onnx");
    if !path.exists() {
        bail!(
            "the aligner pack is missing {}. Reinstall it: bisque-voice install <aligner>",
            path.display()
        );
    }
    // The aligner ALWAYS runs on CPU, whatever `--device` says. The exported
    // graph carries unbounded dynamic dimensions that E5RT rejects
    // deterministically ("unbounded dimension which is not supported"), so a
    // CoreML attempt never runs a single inference — it compiles for seconds,
    // spews Apple's STL-exception text onto STDOUT (corrupting this tool's
    // JSON output, which callers parse), and falls back to CPU anyway.
    // Kokoro's device handling is unchanged; this is per-graph, not policy.
    let _ = device;
    let session = crate::device::commit_file(threads, crate::device::Device::Cpu, &path)?;

    // Feeding by name means a reordered export still works, but a *missing*
    // input would surface as an opaque runtime error — so check up front.
    let names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
    let missing: Vec<&str> = GRAPH_INPUTS
        .iter()
        .filter(|want| !names.iter().any(|got| got == *want))
        .copied()
        .collect();
    if !missing.is_empty() {
        bail!(
            "{} does not take {} — its inputs are {names:?}. This pack was built from a different \
             export than the aligner expects.",
            path.display(),
            missing.join(", ")
        );
    }
    Ok(session)
}

/// `argmax(logits, -1)` at the timestamp positions only, with the softmax
/// probability the graph gave each pick.
///
/// The full tensor is `[1, text_seq, vocab]` with a ~152k vocabulary; only two
/// rows per word are ever read. The probability is what the retake gate reads
/// as how sure the aligner was (`quality::Reading::least_sure`): a word it
/// could find in the audio gets most of the mass, a word it could not gets
/// spread across many times.
fn argmax_at(value: &DynValue, seq: usize, positions: &[usize]) -> Result<(Vec<i64>, Vec<f32>)> {
    let (shape, data) = match value.try_extract_tensor::<f32>() {
        Ok((shape, data)) => (shape.to_vec(), data.to_vec()),
        Err(_) => match value.try_extract_tensor::<half::f16>() {
            Ok((shape, data)) => (shape.to_vec(), data.iter().map(|v| v.to_f32()).collect()),
            Err(e) => bail!("reading the aligner's logits: {e}"),
        },
    };
    let vocab = match shape.as_slice() {
        [1, s, v] if *s as usize == seq && *v > 0 => *v as usize,
        other => bail!(
            "the aligner returned logits shaped {other:?}; expected [1, {seq}, vocab]. The prompt \
             and the graph disagree about the sequence length."
        ),
    };
    if data.len() != seq * vocab {
        bail!(
            "the aligner's logits hold {} values, not {seq}×{vocab}",
            data.len()
        );
    }
    Ok(positions
        .iter()
        .map(|&p| {
            let row = &data[p * vocab..(p + 1) * vocab];
            // First maximum, matching numpy/torch argmax tie-breaking.
            let (best, top) = row.iter().enumerate().fold(
                (0usize, f32::NEG_INFINITY),
                |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
            );
            // Softmax at the maximum, shifted by it so nothing overflows.
            let mass: f64 = row.iter().map(|&v| f64::from(v - top).exp()).sum();
            (best as i64, (1.0 / mass) as f32)
        })
        .unzip())
}
#[cfg(test)]
mod tests {
    use super::*;

    // ── fix_timestamp ───────────────────────────────────────────────────────

    /// Already non-decreasing: every index is in the longest non-decreasing
    /// subsequence, so there is nothing to repair.
    #[test]
    fn a_monotonic_sequence_is_left_alone() {
        let data = [0.0, 80.0, 80.0, 160.0, 240.0];
        assert_eq!(fix_timestamp(&data), vec![0, 80, 80, 160, 240]);
    }

    /// Hand-worked from the Python, short-run branch (`anomaly_count <= 2`).
    ///
    /// dp = [1,2,2,3,4] → longest run ends at index 4, backtracking to
    /// [0,1,3,4], so index 2 is the lone anomaly. Its distance to the left
    /// normal (index 1) is 1 and to the right (index 3) is 1; ties go left, so
    /// it snaps to 100 rather than 200.
    #[test]
    fn a_short_anomaly_run_snaps_to_the_nearer_neighbour() {
        let data = [0.0, 100.0, 50.0, 200.0, 300.0];
        assert_eq!(fix_timestamp(&data), vec![0, 100, 100, 200, 300]);
    }

    /// Hand-worked from the Python, long-run branch (`anomaly_count > 2`).
    ///
    /// The longest non-decreasing subsequence is [0, 900, 1000, 1100] at
    /// indices [0,1,5,6], leaving a three-long anomaly run at 2..5. It is
    /// interpolated: step = (1000 - 900) / 4 = 25.
    #[test]
    fn a_long_anomaly_run_is_interpolated_between_its_brackets() {
        let data = [0.0, 900.0, 800.0, 700.0, 600.0, 1000.0, 1100.0];
        assert_eq!(
            fix_timestamp(&data),
            vec![0, 900, 925, 950, 975, 1000, 1100]
        );
    }

    /// With no trusted value to the left, the right one is copied rather than
    /// interpolated from nothing. dp = [1,1,2,3]: the subsequence starts at
    /// index 1, so index 0 is the anomaly and has no left bracket.
    #[test]
    fn a_run_with_only_one_bracket_copies_it() {
        let data = [500.0, 0.0, 100.0, 200.0];
        assert_eq!(fix_timestamp(&data), vec![0, 0, 100, 200]);
    }

    #[test]
    fn repair_output_is_non_decreasing_for_adversarial_input() {
        let data = [900.0, 10.0, 800.0, 20.0, 700.0, 30.0, 40.0, 50.0];
        let fixed = fix_timestamp(&data);
        for pair in fixed.windows(2) {
            assert!(pair[0] <= pair[1], "not repaired: {fixed:?}");
        }
    }

    #[test]
    fn empty_input_repairs_to_empty() {
        assert!(fix_timestamp(&[]).is_empty());
    }

    // ── prompt ──────────────────────────────────────────────────────────────

    fn specials() -> Specials {
        Specials {
            audio_start: 151_669,
            audio_pad: 151_676,
            audio_end: 151_670,
            timestamp: 151_705,
        }
    }

    #[test]
    fn the_prompt_is_audio_then_two_timestamps_per_word() {
        let sp = specials();
        let ids = prompt_ids(&sp, 3, &[vec![10, 11], vec![12]]);
        assert_eq!(
            ids,
            vec![
                151_669, // <|audio_start|>
                151_676, 151_676, 151_676, // one <|audio_pad|> per audio position
                151_670, // <|audio_end|>
                10, 11, 151_705, 151_705, // word one, start + end
                12, 151_705, 151_705, // word two
            ]
        );
    }

    #[test]
    fn timestamp_slots_are_two_per_word_and_in_order() {
        let sp = specials();
        let words = vec![vec![10, 11], vec![12], vec![13, 14, 15]];
        let ids = prompt_ids(&sp, 2, &words);
        let positions = timestamp_positions(&ids, sp.timestamp);
        assert_eq!(positions.len(), words.len() * 2);
        assert!(positions.windows(2).all(|w| w[0] < w[1]), "must be in order");
        for &p in &positions {
            assert_eq!(ids[p], sp.timestamp);
        }
        // The audio block must contribute no slots — a `<|audio_pad|>` picked
        // up here would shift every word's timing by one.
        assert!(positions[0] > 2 + 2);
    }

    #[test]
    fn a_wordless_prompt_still_carries_the_audio_block() {
        let sp = specials();
        let ids = prompt_ids(&sp, 1, &[]);
        assert_eq!(ids, vec![151_669, 151_676, 151_670]);
        assert!(timestamp_positions(&ids, sp.timestamp).is_empty());
    }

    // ── tokenizing for the model ────────────────────────────────────────────

    /// `(text, source)` pairs, so the expectations below read as the prompt.
    fn tokens_of(text: &str) -> Vec<(String, usize)> {
        let words = source_words(text);
        model_tokens(&words)
            .into_iter()
            .map(|t| (t.text, t.source))
            .collect()
    }

    fn texts_of(text: &str) -> Vec<String> {
        tokens_of(text).into_iter().map(|(t, _)| t).collect()
    }

    /// The regression guard. English must come out of the new tokenizer exactly
    /// as it came out of the old one: one model token per source word, cleaned,
    /// in order. If this ever changes, every English timing changed with it.
    #[test]
    fn latin_text_is_one_cleaned_token_per_source_word() {
        assert_eq!(
            tokens_of("Revenue grew 3.5x to $4.2M in Q3."),
            vec![
                ("Revenue".into(), 0),
                ("grew".into(), 1),
                ("35x".into(), 2),
                ("to".into(), 3),
                ("42M".into(), 4),
                ("in".into(), 5),
                ("Q3".into(), 6),
            ]
        );
    }

    #[test]
    fn the_apostrophe_survives_cleaning_but_nothing_else_does() {
        assert_eq!(texts_of("don't — \"stop\"!"), vec!["don't", "—", "stop"]);
    }

    /// A word with nothing left after cleaning keeps its original text: it still
    /// needs a token position, or it has no span to fold back and the 1:1
    /// guarantee breaks.
    #[test]
    fn a_punctuation_only_word_keeps_a_token_of_its_own() {
        assert_eq!(tokens_of("a — b"), vec![
            ("a".into(), 0),
            ("—".into(), 1),
            ("b".into(), 2)
        ]);
    }

    /// The bug this section exists for: a Chinese sentence is ONE whitespace
    /// token, so it used to reach the model as one token with two timestamp
    /// slots for the whole utterance. Per character it gets twenty-three.
    #[test]
    fn a_chinese_sentence_is_split_per_character() {
        let text = "这个演示的旁白，来自一个正在这台电脑上运行的模型。";
        assert_eq!(source_words(text).len(), 1, "no spaces — one source token");
        let tokens = tokens_of(text);
        assert_eq!(tokens.len(), 23, "one per Han character, punctuation dropped");
        assert!(tokens.iter().all(|(t, _)| t.chars().count() == 1));
        assert!(
            tokens.iter().all(|(_, source)| *source == 0),
            "every one folds back onto the single source word"
        );
        assert_eq!(tokens[0].0, "这");
        assert_eq!(tokens[22].0, "型");
        assert!(
            !tokens.iter().any(|(t, _)| t == "，" || t == "。"),
            "CJK punctuation is not something the model was asked to place"
        );
    }

    /// Mixed text: Han per character, the Latin run whole, and each piece
    /// tagged with the whitespace token it belongs to.
    #[test]
    fn mixed_cjk_and_latin_splits_the_han_and_keeps_the_latin_whole() {
        assert_eq!(
            tokens_of("使用 Qwen3 模型。"),
            vec![
                ("使".into(), 0),
                ("用".into(), 0),
                ("Qwen3".into(), 1),
                ("模".into(), 2),
                ("型".into(), 2),
            ]
        );
    }

    /// Han and Latin with no space between them are one source word but several
    /// model tokens — the case whitespace tokenization cannot see at all.
    #[test]
    fn a_latin_run_inside_a_han_run_is_its_own_token() {
        assert_eq!(
            tokens_of("使用Qwen3模型"),
            vec![
                ("使".into(), 0),
                ("用".into(), 0),
                ("Qwen3".into(), 0),
                ("模".into(), 0),
                ("型".into(), 0),
            ]
        );
    }

    /// Japanese: kana lead, and the marks that carry no mora of their own ride
    /// along with the syllable they modify rather than becoming token positions
    /// the model cannot place.
    #[test]
    fn japanese_splits_per_mora_and_keeps_modifiers_attached() {
        assert_eq!(
            texts_of("きょうはいい天気です。"),
            vec!["きょ", "う", "は", "い", "い", "天", "気", "で", "す"]
        );
        // Sokuon and the long-vowel mark attach leftwards — `っ` lands on `ちょ`,
        // not on `と`. See `is_kana_tail` for why either side is defensible and
        // why leftwards is the one that never invents a silent token.
        assert_eq!(texts_of("ちょっと"), vec!["ちょっ", "と"]);
        assert_eq!(texts_of("コンピューター"), vec!["コ", "ン", "ピュー", "ター"]);
        // ヶ is small in shape only — 三ヶ月 reads it as a full mora.
        assert_eq!(texts_of("三ヶ月"), vec!["三", "ヶ", "月"]);
    }

    /// Korean writes spaces between eojeol, so whitespace tokenization already
    /// gives it real units. Splitting Hangul per syllable would be a
    /// regression, not a fix.
    #[test]
    fn korean_is_left_to_its_own_spacing() {
        assert_eq!(texts_of("안녕하세요 여러분"), vec!["안녕하세요", "여러분"]);
    }

    // ── folding model tokens back onto source words ─────────────────────────

    fn tok(text: &str, source: usize) -> ModelToken {
        ModelToken {
            text: text.into(),
            source,
        }
    }

    /// One token per word — the Latin case — must be a straight pass-through in
    /// milliseconds-to-seconds, identical to what this returned before the fold
    /// existed.
    #[test]
    fn one_token_per_word_folds_to_itself() {
        let words = source_words("hello world");
        let tokens = vec![tok("hello", 0), tok("world", 1)];
        let folded = fold_onto_source(&words, &tokens, &[0, 480, 480, 1_040]).unwrap();
        assert_eq!(
            folded,
            vec![
                Word { word: "hello".into(), start: 0.0, end: 0.48 },
                Word { word: "world".into(), start: 0.48, end: 1.04 },
            ]
        );
    }

    /// Several tokens per word take the outer bounds, and the returned word
    /// carries the SOURCE surface string — punctuation included — which is what
    /// keeps `reconcile()` on its exact path.
    #[test]
    fn many_tokens_per_word_take_the_outer_bounds() {
        let words = source_words("这个演示。");
        let tokens = vec![tok("这", 0), tok("个", 0), tok("演", 0), tok("示", 0)];
        let folded =
            fold_onto_source(&words, &tokens, &[0, 320, 320, 640, 640, 960, 960, 1_280]).unwrap();
        assert_eq!(folded.len(), 1, "1:1 with source whitespace tokens");
        assert_eq!(folded[0].word, "这个演示。");
        assert_eq!(folded[0].start, 0.0);
        assert_eq!(folded[0].end, 1.28);
    }

    /// min/max, not first/last. `fix_timestamp` normally makes these the same
    /// thing; the fold must not be the place where a failed repair turns into an
    /// inverted span.
    #[test]
    fn the_fold_is_min_max_rather_than_first_and_last() {
        let words = source_words("这个");
        let tokens = vec![tok("这", 0), tok("个", 0)];
        let folded = fold_onto_source(&words, &tokens, &[500, 900, 100, 300]).unwrap();
        assert_eq!(folded[0].start, 0.1);
        assert_eq!(folded[0].end, 0.9);
    }

    #[test]
    fn mixed_text_folds_each_run_onto_its_own_source_word() {
        let words = source_words("使用 Qwen3 模型");
        let tokens = model_tokens(&words);
        // 使 用 Qwen3 模 型
        let fixed = [0, 200, 200, 400, 400, 900, 900, 1_100, 1_100, 1_400];
        let folded = fold_onto_source(&words, &tokens, &fixed).unwrap();
        assert_eq!(folded.len(), 3);
        assert_eq!((folded[0].start, folded[0].end), (0.0, 0.4));
        assert_eq!((folded[1].start, folded[1].end), (0.4, 0.9));
        assert_eq!((folded[2].start, folded[2].end), (0.9, 1.4));
    }

    /// Every source word gets a token by construction, over everything above.
    #[test]
    fn every_source_word_is_covered_by_at_least_one_model_token() {
        for text in [
            "Revenue grew 3.5x to $4.2M in Q3.",
            "这个演示的旁白，来自一个正在这台电脑上运行的模型。",
            "使用 Qwen3 模型。",
            "きょうはいい天気です。",
            "안녕하세요 여러분",
            "a — b",
            "one",
        ] {
            let words = source_words(text);
            let tokens = model_tokens(&words);
            assert!(tokens.len() >= words.len(), "for {text:?}");
            for (i, w) in words.iter().enumerate() {
                assert!(
                    tokens.iter().any(|t| t.source == i),
                    "source word {w:?} of {text:?} has no model token"
                );
            }
            // And the fold therefore always succeeds, 1:1.
            let fixed: Vec<i64> = (0..tokens.len() as i64 * 2).map(|k| k * 80).collect();
            let folded = fold_onto_source(&words, &tokens, &fixed).unwrap();
            assert_eq!(folded.len(), words.len(), "for {text:?}");
            assert_eq!(
                folded.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
                words,
                "the fold must return the SOURCE surface strings"
            );
        }
    }

    /// The invariant `main.rs` bails on: an aligner's output must reconcile
    /// EXACTLY, or measured timings get reported that were never measured.
    ///
    /// It holds because the fold returns the source surface strings untouched —
    /// punctuation included — even though the model never saw them. Returning
    /// the *cleaned* per-syllable tokens instead would strip `，` and `。` out of
    /// the character stream, drop `reconcile()` onto its APPROXIMATE path, and
    /// turn every Chinese narration into a hard failure. That is the concrete
    /// reason the finer unit cannot simply be handed back from here.
    #[test]
    fn what_the_fold_returns_always_reconciles_exactly() {
        for text in [
            "Revenue grew 3.5x to $4.2M in Q3.",
            "这个演示的旁白，来自一个正在这台电脑上运行的模型。",
            "使用 Qwen3 模型。",
            "きょうはいい天気です。",
            "a — b",
        ] {
            let words = source_words(text);
            let tokens = model_tokens(&words);
            let fixed: Vec<i64> = (0..tokens.len() as i64 * 2).map(|k| k * 80).collect();
            let folded = fold_onto_source(&words, &tokens, &fixed).unwrap();
            let r = crate::reconcile::reconcile(text, &folded).unwrap();
            assert!(r.exact, "{text:?} would make main.rs refuse the alignment");
            assert_eq!(r.words, folded, "for {text:?}");
        }
    }

    /// The two hard failures. Both are unreachable through `model_tokens`, and
    /// both would otherwise be a silently wrong timing rather than an error.
    #[test]
    fn a_malformed_fold_fails_loudly_rather_than_guessing() {
        let words = source_words("hello world");
        let tokens = vec![tok("hello", 0), tok("world", 1)];
        assert!(
            fold_onto_source(&words, &tokens, &[0, 480, 480]).is_err(),
            "an odd number of timestamps is not two per token"
        );
        // Source word 1 covered by nothing.
        let orphaned = vec![tok("hello", 0)];
        let err = fold_onto_source(&words, &orphaned, &[0, 480])
            .expect_err("a source word with no span must fail")
            .to_string();
        assert!(err.contains("world"), "{err}");
        // A token pointing past the end of the source.
        let stray = vec![tok("hello", 0), tok("world", 9)];
        assert!(fold_onto_source(&words, &stray, &[0, 480, 480, 960]).is_err());
    }

    // ── what the pack declares ──────────────────────────────────────────────

    /// A pack's JSON side — everything except the multi-gigabyte graph, which
    /// none of these paths touch.
    fn stage_pack(dir: &Path, tokenizer: serde_json::Value, preprocessor: serde_json::Value) {
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::to_vec(&tokenizer).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("preprocessor_config.json"),
            serde_json::to_vec(&preprocessor).unwrap(),
        )
        .unwrap();
    }

    fn real_tokenizer_config() -> serde_json::Value {
        serde_json::json!({
            "added_tokens_decoder": {
                "151669": { "content": "<|audio_start|>" },
                "151670": { "content": "<|audio_end|>" },
                "151676": { "content": "<|audio_pad|>" },
                "151705": { "content": "<timestamp>" },
            }
        })
    }

    fn real_preprocessor_config() -> serde_json::Value {
        serde_json::json!({
            "feature_extractor_type": "WhisperFeatureExtractor",
            "feature_size": 128,
            "n_fft": 400,
            "hop_length": 160,
            "sampling_rate": 16000,
            "dither": 0.0,
        })
    }

    fn real_config() -> serde_json::Value {
        serde_json::json!({
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_token_id": 151676,
            "timestamp_token_id": 151705,
            "timestamp_segment_time": 80,
            "audio_config": { "n_window": 50, "num_mel_bins": 128 },
        })
    }

    #[test]
    fn special_ids_come_from_the_pack_not_from_this_source_file() {
        let dir = tempfile::tempdir().unwrap();
        stage_pack(dir.path(), real_tokenizer_config(), real_preprocessor_config());
        let sp = Specials::load(dir.path(), &real_config()).unwrap();
        assert_eq!(sp, specials());
    }

    #[test]
    fn a_pack_missing_a_special_token_names_it_rather_than_guessing() {
        let dir = tempfile::tempdir().unwrap();
        let mut tok = real_tokenizer_config();
        tok["added_tokens_decoder"]
            .as_object_mut()
            .unwrap()
            .remove("151705");
        stage_pack(dir.path(), tok, real_preprocessor_config());
        let err = Specials::load(dir.path(), &real_config())
            .expect_err("a missing <timestamp> must not be filled in from memory")
            .to_string();
        assert!(err.contains("<timestamp>"), "{err}");
    }

    /// The two files name the same ids from opposite sides. Averaging a
    /// disagreement out would mean scoring a prompt the graph never saw.
    #[test]
    fn a_pack_that_disagrees_with_itself_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        stage_pack(dir.path(), real_tokenizer_config(), real_preprocessor_config());
        let mut config = real_config();
        config["timestamp_token_id"] = serde_json::json!(999);
        let err = Specials::load(dir.path(), &config)
            .expect_err("mismatched ids must fail")
            .to_string();
        assert!(err.contains("<timestamp>") && err.contains("999"), "{err}");
    }

    #[test]
    fn the_config_yields_the_segment_time_and_the_window() {
        let dir = tempfile::tempdir().unwrap();
        stage_pack(dir.path(), real_tokenizer_config(), real_preprocessor_config());
        let cfg = AlignConfig::load(dir.path(), &real_config()).unwrap();
        assert_eq!(cfg.segment_ms, 80.0);
        assert_eq!(cfg.n_window, 50);
    }

    /// This build implements exactly one Whisper front end. A pack asking for
    /// another must say so loudly rather than be fed features it never saw in
    /// training.
    #[test]
    fn a_different_feature_extractor_is_refused() {
        for (key, wrong) in [
            ("n_fft", serde_json::json!(512)),
            ("hop_length", serde_json::json!(256)),
            ("sampling_rate", serde_json::json!(24000)),
            ("feature_size", serde_json::json!(80)),
            ("dither", serde_json::json!(0.1)),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let mut pre = real_preprocessor_config();
            pre[key] = wrong;
            stage_pack(dir.path(), real_tokenizer_config(), pre);
            let err = AlignConfig::load(dir.path(), &real_config())
                .expect_err("a changed front end must fail")
                .to_string();
            assert!(err.contains(key), "error should name {key}: {err}");
        }
    }

    #[test]
    fn a_config_without_n_window_refuses_to_guess_the_pad_count() {
        let dir = tempfile::tempdir().unwrap();
        stage_pack(dir.path(), real_tokenizer_config(), real_preprocessor_config());
        let mut config = real_config();
        config["audio_config"] = serde_json::json!({ "num_mel_bins": 128 });
        let err = AlignConfig::load(dir.path(), &config)
            .expect_err("n_window is not something to assume")
            .to_string();
        assert!(err.contains("n_window"), "{err}");
    }

    // ── audio token count ───────────────────────────────────────────────────

    /// Upstream hardcodes 13 tokens per 100-frame chunk. Deriving it from the
    /// three stride-2 convolutions must reproduce that exactly, or every
    /// `<|audio_pad|>` count is wrong.
    #[test]
    fn a_full_chunk_produces_upstreams_thirteen_positions() {
        assert_eq!(audio_token_count(100, 50).unwrap(), 13);
        assert_eq!(audio_token_count(500, 50).unwrap(), 65);
        assert_eq!(audio_token_count(2000, 50).unwrap(), 260);
    }

    /// The remainder is convolved separately, so a partial chunk is NOT
    /// `frames / 8`: 50 leftover frames give 7 positions, not 6.
    #[test]
    fn a_partial_chunk_is_convolved_on_its_own() {
        assert_eq!(audio_token_count(50, 50).unwrap(), 7);
        assert_eq!(audio_token_count(250, 50).unwrap(), 2 * 13 + 7);
        assert_eq!(audio_token_count(0, 50).unwrap(), 0);
    }

    #[test]
    fn a_zero_window_is_rejected_rather_than_dividing_by_zero() {
        assert!(audio_token_count(100, 0).is_err());
    }

    // ── mel filterbank ──────────────────────────────────────────────────────

    #[test]
    fn the_filterbank_is_128_by_201_and_non_negative() {
        let f = mel_filters();
        assert_eq!(f.len(), MEL_BINS * n_freqs());
        assert_eq!(n_freqs(), 201);
        assert!(f.iter().all(|v| *v >= 0.0), "a triangular filter cannot dip below zero");
    }

    #[test]
    fn every_mel_band_has_energy_and_they_climb_in_frequency() {
        let bins = n_freqs();
        let f = mel_filters();
        let mut previous_peak = 0usize;
        for m in 0..MEL_BINS {
            let row = &f[m * bins..(m + 1) * bins];
            let sum: f64 = row.iter().sum();
            assert!(sum > 0.0, "mel band {m} is empty");
            let peak = row
                .iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0;
            assert!(peak >= previous_peak, "mel band {m} peaks below its predecessor");
            previous_peak = peak;
        }
    }

    /// The Slaney break point: 1 kHz is exactly 15 mels, and the scale is
    /// linear below it. Getting this wrong shifts every filter.
    #[test]
    fn the_mel_scale_is_slaney_not_htk() {
        assert!((hertz_to_mel(1000.0) - 15.0).abs() < 1e-9);
        assert!((hertz_to_mel(500.0) - 7.5).abs() < 1e-9);
        for hz in [0.0, 250.0, 1000.0, 4000.0, 8000.0] {
            assert!((mel_to_hertz(hertz_to_mel(hz)) - hz).abs() < 1e-6, "{hz} Hz");
        }
    }

    #[test]
    fn log_mel_has_one_frame_per_hop_and_whispers_output_range() {
        // 1 s of a 440 Hz tone at 16 kHz.
        let samples: Vec<f32> = (0..ALIGN_RATE)
            .map(|i| (i as f32 / ALIGN_RATE as f32 * 440.0 * std::f32::consts::TAU).sin() * 0.5)
            .collect();
        let mel = log_mel(&samples, &mel_filters()).unwrap();
        assert_eq!(mel.frames, 100, "100 feature frames per second");
        assert_eq!(mel.data.len(), MEL_BINS * mel.frames);
        // Whisper's clamp keeps eight decades and the rescale divides by four,
        // so the surface spans exactly two units however loud the input was.
        let max = mel.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = mel.data.iter().copied().fold(f64::INFINITY, f64::min);
        assert!((max - min - 2.0).abs() < 1e-9, "range was {}", max - min);
    }

    /// A pure tone must light up the mel band that actually covers it. This is
    /// what catches a filterbank built on the wrong scale, an FFT indexed the
    /// wrong way round, or a window applied to the wrong samples — all of which
    /// still produce a plausible-looking surface.
    #[test]
    fn a_pure_tone_peaks_in_the_band_that_covers_it() {
        let tone = 440.0f64;
        let samples: Vec<f32> = (0..ALIGN_RATE)
            .map(|i| (i as f64 / ALIGN_RATE as f64 * tone * std::f64::consts::TAU).sin() as f32)
            .collect();
        let mel = log_mel(&samples, &mel_filters()).unwrap();

        // Loudest band, averaged over the steady-state frames.
        let loudest = (0..MEL_BINS)
            .map(|m| {
                let row = &mel.data[m * mel.frames..(m + 1) * mel.frames];
                (m, row.iter().sum::<f64>() / mel.frames as f64)
            })
            .fold((0usize, f64::NEG_INFINITY), |best, cur| {
                if cur.1 > best.1 {
                    cur
                } else {
                    best
                }
            })
            .0;

        // Band m is centred on the (m+1)th of MEL_BINS+1 equal mel steps.
        let expected = (hertz_to_mel(tone) / hertz_to_mel(ALIGN_RATE as f64 / 2.0)
            * (MEL_BINS + 1) as f64)
            .round() as isize
            - 1;
        assert!(
            (loudest as isize - expected).abs() <= 2,
            "a {tone} Hz tone peaked in band {loudest}, not near {expected}"
        );
    }

    #[test]
    fn audio_shorter_than_one_frame_fails_loudly() {
        assert!(log_mel(&[0.0; 100], &mel_filters()).is_err());
    }

    #[test]
    fn reflect_padding_mirrors_without_repeating_the_edge() {
        assert_eq!(reflect(-1, 5), 1);
        assert_eq!(reflect(-4, 5), 4);
        assert_eq!(reflect(0, 5), 0);
        assert_eq!(reflect(5, 5), 3);
        assert_eq!(reflect(8, 5), 0);
    }

    // ── resampler ───────────────────────────────────────────────────────────

    #[test]
    fn resampling_24k_to_16k_yields_two_thirds_of_the_samples() {
        assert_eq!(resample_24k_to_16k(&vec![0.0; 24_000]).len(), 16_000);
        assert_eq!(resample_24k_to_16k(&vec![0.0; 2_400]).len(), 1_600);
        assert!(resample_24k_to_16k(&[]).is_empty());
        assert_eq!(resample_24k_to_16k(&[0.5]).len(), 1);
    }

    /// Per-tap normalization means a constant signal survives exactly —
    /// including at the edges, where the kernel is clamped to the signal.
    #[test]
    fn a_constant_signal_keeps_its_level_everywhere() {
        let out = resample_24k_to_16k(&vec![0.75f32; 4_800]);
        for (i, v) in out.iter().enumerate() {
            assert!((v - 0.75).abs() < 1e-4, "sample {i} drifted to {v}");
        }
    }

    /// A 1 kHz tone is well inside the passband, so amplitude and phase must
    /// survive. This is the property linear interpolation would fail.
    #[test]
    fn an_in_band_tone_survives_with_its_amplitude() {
        let f = 1_000.0f64;
        let input: Vec<f32> = (0..24_000)
            .map(|i| (std::f64::consts::TAU * f * i as f64 / 24_000.0).sin() as f32)
            .collect();
        let out = resample_24k_to_16k(&input);
        assert_eq!(out.len(), 16_000);
        // Skip the kernel-width edges, where clamping is not band-limited.
        for (i, v) in out.iter().enumerate().take(15_900).skip(100) {
            let want = (std::f64::consts::TAU * f * i as f64 / 16_000.0).sin() as f32;
            assert!((v - want).abs() < 0.02, "sample {i}: {v} vs {want}");
        }
    }

    /// A 10 kHz tone is above the output's Nyquist. Band-limiting must reject
    /// it rather than fold it down to 6 kHz on top of the speech.
    #[test]
    fn an_out_of_band_tone_is_rejected_rather_than_aliased() {
        let input: Vec<f32> = (0..24_000)
            .map(|i| (std::f64::consts::TAU * 10_000.0 * i as f64 / 24_000.0).sin() as f32)
            .collect();
        let out = resample_24k_to_16k(&input);
        let peak = out[200..15_800]
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak < 0.05, "aliased through at amplitude {peak}");
    }
}
