//! Audio output — MP3 by default, WAV on request.
//!
//! Kokoro emits 24 kHz mono float samples. **MP3 is the pipeline's contract**,
//! not a preference: `ManifestAudioAsset.contentType` is literally
//! `"audio/mpeg"`, and all three existing producers (ElevenLabs, the Cloud Run
//! Kokoro runner, the create worker) hand it MP3. A local engine that emitted
//! only WAV could not publish at all, and widening the contract for one
//! producer would be the wrong side of that trade — WAV is ~3× the bytes for a
//! viewer to download.
//!
//! LAME is compiled in (autotools on unix, `cc` + `configMS.h` on MSVC), so
//! there is no ffmpeg to find on a stranger's machine — which was the whole
//! reason for a single static binary.

use std::io::Write;

use anyhow::{anyhow, Result};
use mp3lame_encoder::{Bitrate, Builder, FlushNoGap, MonoPcm, Quality};

pub const SAMPLE_RATE: u32 = 24_000;
/// Speech at 24 kHz mono. Matches what the create worker stores for cloud
/// narration, so a locally-narrated presentation is byte-comparable.
const MP3_BITRATE: Bitrate = Bitrate::Kbps128;

/// Which container `--out` asked for, decided by its extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Mp3,
    Wav,
}

impl Format {
    /// Anything that isn't explicitly `.wav` is MP3 — the pipeline default.
    pub fn for_path(path: &std::path::Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("wav") => Format::Wav,
            _ => Format::Mp3,
        }
    }

    pub fn content_type(self) -> &'static str {
        match self {
            Format::Mp3 => "audio/mpeg",
            Format::Wav => "audio/wav",
        }
    }
}

/// Peak-normalize to 0.89 full scale and quantize to 16-bit.
///
/// Shared by both encoders so the two formats are the same audio at the same
/// level — and the level matches the macOS engine, so a presentation narrated
/// on either platform plays back equally loud.
fn normalize(samples: &[f32]) -> Vec<i16> {
    let mut peak = 0.0001f32;
    for s in samples {
        peak = peak.max(s.abs());
    }
    let scale = 0.89 / peak;
    samples
        .iter()
        .map(|s| ((s * scale).clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect()
}

pub fn encode(samples: &[f32], format: Format) -> Result<Vec<u8>> {
    match format {
        Format::Wav => Ok(encode_wav(samples)),
        Format::Mp3 => encode_mp3(samples),
    }
}

pub fn encode_wav(samples: &[f32]) -> Vec<u8> {
    let pcm = normalize(samples);
    let data_size = (pcm.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_size).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    for s in pcm {
        let _ = out.write_all(&s.to_le_bytes());
    }
    out
}

pub fn encode_mp3(samples: &[f32]) -> Result<Vec<u8>> {
    let pcm = normalize(samples);

    let mut builder = Builder::new().ok_or_else(|| anyhow!("LAME: could not create an encoder"))?;
    builder
        .set_num_channels(1)
        .map_err(|e| anyhow!("LAME channels: {e}"))?;
    builder
        .set_sample_rate(SAMPLE_RATE)
        .map_err(|e| anyhow!("LAME sample rate: {e}"))?;
    builder
        .set_brate(MP3_BITRATE)
        .map_err(|e| anyhow!("LAME bitrate: {e}"))?;
    // Narration is synthesized once and listened to many times; spend the
    // encoder time rather than the listener's ears.
    builder
        .set_quality(Quality::Best)
        .map_err(|e| anyhow!("LAME quality: {e}"))?;
    let mut encoder = builder.build().map_err(|e| anyhow!("LAME build: {e}"))?;

    let mut out = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(pcm.len()));
    let written = encoder
        .encode(MonoPcm(&pcm), out.spare_capacity_mut())
        .map_err(|e| anyhow!("LAME encode: {e}"))?;
    // SAFETY: LAME reports how many bytes it wrote into the spare capacity.
    unsafe { out.set_len(out.len() + written) };

    let written = encoder
        .flush::<FlushNoGap>(out.spare_capacity_mut())
        .map_err(|e| anyhow!("LAME flush: {e}"))?;
    unsafe { out.set_len(out.len() + written) };

    Ok(out)
}

pub fn duration_ms(samples: &[f32]) -> f64 {
    samples.len() as f64 / SAMPLE_RATE as f64 * 1000.0
}

// ─── reference-recording hygiene ─────────────────────────────────────
//
// Part of `clone`'s contract: hand it a real-world recording and it
// produces a good voice or an actionable refusal — no client is expected
// to pre-clean audio. The failure this guards against is concrete: a
// reference that opens with a click and a long pause teaches the clone to
// begin narrations the same way, stochastically, and the client-side trim
// that was supposed to prevent it had to be debugged three times against
// real clips. This is capability, so it lives behind the boundary.

/// An analysis frame ~85 ms at 24 kHz.
const HYGIENE_FRAME: usize = 2048;
/// Absolute floor: frame RMS below this is silence in any clip (≈ −48 dBFS).
const HYGIENE_SILENCE_RMS: f32 = 0.004;
/// The start anchor must be SUSTAINED speech — this many consecutive
/// audible frames (~0.34 s), longer than any click or breath.
const HYGIENE_SUSTAINED_FRAMES: usize = 4;
/// An audible run that dies within this many silent frames (~0.5 s) of its
/// start is an isolated transient, skipped along with the silence after it.
const HYGIENE_TRANSIENT_GAP_FRAMES: usize = 6;
/// Silence kept on each side of the speech so it doesn't start mid-breath.
const HYGIENE_PAD_S: f64 = 0.25;

/// Cut leading junk (clicks, breaths, room tone, dead air) and trailing
/// silence from a reference recording.
///
/// The silence threshold adapts to the clip itself: AGC — on by default on
/// laptop mics — amplifies room tone well past any fixed floor (a real
/// clip measured −40 dB of "silence"), and the floor even decays as AGC
/// settles, so neither a fixed threshold nor a noise-anchored one holds.
/// The stable anchor is between the clip's quiet level (25th-percentile
/// frame RMS) and its speech level (75th): their geometric mean sits above
/// room tone in any gain regime and below even soft speech. All-silent
/// input returns an empty vec — the caller turns that into a refusal.
pub fn trim_recording(samples: &[f32]) -> Vec<f32> {
    let frames: Vec<f32> = samples
        .chunks_exact(HYGIENE_FRAME)
        .map(|c| (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt())
        .collect();
    if frames.is_empty() {
        return Vec::new();
    }
    let mut sorted = frames.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quiet = sorted[sorted.len() / 4];
    let speech = sorted[sorted.len() * 3 / 4];
    let threshold = (quiet * speech).sqrt().clamp(HYGIENE_SILENCE_RMS, 0.02);
    let audible: Vec<bool> = frames.iter().map(|r| *r >= threshold).collect();

    // Start: the first audible frame from which sustained audio arrives
    // without a transient-length gap — speech that keeps going, not a blip.
    let mut first: Option<usize> = None;
    'outer: for f in 0..audible.len() {
        if !audible[f] {
            continue;
        }
        let (mut run, mut gap) = (0usize, 0usize);
        for g in f..audible.len() {
            if audible[g] {
                run += 1;
                gap = 0;
                if run >= HYGIENE_SUSTAINED_FRAMES {
                    first = Some(f * HYGIENE_FRAME);
                    break 'outer;
                }
            } else {
                gap += 1;
                if gap >= HYGIENE_TRANSIENT_GAP_FRAMES {
                    break; // isolated transient — keep scanning past it
                }
            }
        }
    }
    // End: the last audible frame, transient or not — a trailing blip is
    // harmless and cutting real trailing speech is not.
    let last = audible.iter().rposition(|a| *a).map(|f| (f + 1) * HYGIENE_FRAME);
    let (Some(first), Some(last)) = (first, last) else {
        return Vec::new();
    };
    if last <= first {
        return Vec::new();
    }
    let pad = (HYGIENE_PAD_S * f64::from(SAMPLE_RATE)) as usize;
    samples[first.saturating_sub(pad)..(last + pad).min(samples.len())].to_vec()
}

/// Padding kept before the first aligned word when trimming, seconds.
pub const TRIM_LEAD_PAD_S: f64 = 0.15;
/// Padding kept after the last aligned word — releases and reverb tails
/// are part of the word, so the tail keeps more than the lead.
pub const TRIM_TAIL_PAD_S: f64 = 0.30;
/// A cut smaller than this is not worth re-indexing the clip over.
pub const TRIM_MIN_CUT_S: f64 = 0.05;

/// Cut everything outside the aligned speech span, shifting `words` onto
/// the trimmed clip's clock.
///
/// The clone-family engines can emit spurious audio before the first real
/// word — dead air, or a speech-loud phantom fragment — stochastically,
/// per synthesis. Amplitude can't identify that junk (the phantom is as
/// loud as speech); a forced aligner can, because it measures where the
/// *expected* words actually land. So this runs only on measured timings,
/// never on engine-inferred ones.
pub fn trim_to_aligned_words(
    samples: Vec<f32>,
    words: &mut [crate::reconcile::Word],
) -> Vec<f32> {
    let (Some(first), Some(last)) = (words.first(), words.last()) else {
        return samples;
    };
    let start_s = (first.start - TRIM_LEAD_PAD_S).max(0.0);
    let end_s = last.end + TRIM_TAIL_PAD_S;
    let start_i = (start_s * f64::from(SAMPLE_RATE)).round() as usize;
    let end_i = ((end_s * f64::from(SAMPLE_RATE)).round() as usize).min(samples.len());
    if start_i >= end_i {
        // Degenerate timings — leave the audio alone rather than emit nothing.
        return samples;
    }
    let cut = (samples.len() - (end_i - start_i)) as f64 / f64::from(SAMPLE_RATE);
    if cut < TRIM_MIN_CUT_S {
        return samples;
    }
    for w in words.iter_mut() {
        w.start = (w.start - start_s).max(0.0);
        w.end = (w.end - start_s).max(0.0);
    }
    samples[start_i..end_i].to_vec()
}

/// The loudest level a synthesis buffer can have and still be silence.
///
/// Kokoro can return a buffer of effective zeros for some (voice, text) pairs
/// while every other signal — duration, word timings, hash — stays healthy.
/// Because [`normalize`] is peak-normalizing, that buffer
/// encodes to −91 dB digital silence and nothing downstream can tell. This
/// floor is what makes that an error instead of a shipped silent narration.
///
/// −60 dBFS peak is far below any real speech — even a whisper into a distant
/// microphone peaks tens of dB above it, and the failure being guarded against
/// measures around −91 dB (±1 LSB of 16-bit). A legitimate quiet synthesis has
/// enormous margin; only an effectively-empty buffer trips it.
pub const SILENCE_FLOOR_DBFS: f32 = -60.0;

/// Peak level of a raw synthesis buffer in dBFS. Empty or all-zero input is
/// `-inf`, which is below every floor — an engine that produced no signal at
/// all must not pass a silence check on a technicality.
///
/// NaN samples contribute nothing to the peak — `f32::max` keeps the non-NaN
/// operand — and this is **relied upon**: the silent-synthesis defect was a
/// waveform of 100% NaN (the fp16 Kokoro pack's vocoder overflow), which reads
/// as `-inf` here and so lands below the floor. A NaN-propagating fold would
/// return NaN, every `<` comparison against the floor would be false, and the
/// silent buffer would ship. Guarded by `an_all_nan_buffer_reads_as_silent`.
pub fn peak_dbfs(samples: &[f32]) -> f32 {
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if peak <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * peak.log10()
    }
}

/// Decode a WAV into mono f32 at this pipeline's 24 kHz.
///
/// This exists for one caller — the reference clip handed to `bisque-voice clone`
/// — and so it reads what a person's recording actually is rather than what
/// this tool emits: 16/24/32-bit PCM or 32-bit float, mono or stereo, at
/// whatever rate the recorder used. The alternative is telling someone their
/// voice memo is the wrong format, which is not a real answer.
///
/// Deliberately not a general decoder: MP3 and M4A reference clips are rejected
/// with a message saying so, rather than silently mis-parsed.
pub fn decode_wav_to_mono_24k(bytes: &[u8]) -> Result<Vec<f32>> {
    let (samples, rate, channels) = decode_wav(bytes)?;
    let mono = if channels > 1 {
        // Average the channels. A stereo interview clip with the subject on one
        // side would otherwise clone half the room.
        samples
            .chunks(channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect()
    } else {
        samples
    };
    Ok(resample_linear(&mono, rate, SAMPLE_RATE))
}

/// `(interleaved samples, sample rate, channels)`.
fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32, u16)> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!(
            "the reference clip is not a WAV file. Convert it first, e.g.\n  \
             ffmpeg -i clip.m4a -ar 24000 -ac 1 clip.wav"
        );
    }

    let mut format_tag = 0u16;
    let mut channels = 0u16;
    let mut rate = 0u32;
    let mut bits = 0u16;
    let mut data: Option<&[u8]> = None;

    // Walk the chunk list rather than assuming `fmt ` then `data` at fixed
    // offsets — recorders interleave LIST/fact chunks freely.
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(size).min(bytes.len());
        match id {
            b"fmt " if size >= 16 => {
                let f = &bytes[body_start..body_end];
                format_tag = u16::from_le_bytes(f[0..2].try_into().unwrap());
                channels = u16::from_le_bytes(f[2..4].try_into().unwrap());
                rate = u32::from_le_bytes(f[4..8].try_into().unwrap());
                bits = u16::from_le_bytes(f[14..16].try_into().unwrap());
                // WAVE_FORMAT_EXTENSIBLE carries the real tag in its GUID's
                // first two bytes; without this a 24-bit extensible file reads
                // as an unknown format.
                if format_tag == 0xFFFE && size >= 26 {
                    format_tag = u16::from_le_bytes(f[24..26].try_into().unwrap());
                }
            }
            b"data" => data = Some(&bytes[body_start..body_end]),
            _ => {}
        }
        // Chunks are word-aligned: an odd size is followed by a pad byte.
        pos = body_start + size + (size & 1);
    }

    let data = data.ok_or_else(|| anyhow!("the WAV has no data chunk"))?;
    if channels == 0 || rate == 0 {
        anyhow::bail!("the WAV has no readable format chunk");
    }

    let samples: Vec<f32> = match (format_tag, bits) {
        (1, 16) => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32_768.0)
            .collect(),
        (1, 24) => data
            .chunks_exact(3)
            .map(|c| {
                // Sign-extend by placing the 24 bits in the top of an i32.
                let v = i32::from_le_bytes([0, c[0], c[1], c[2]]);
                v as f32 / 2_147_483_648.0
            })
            .collect(),
        (1, 32) => data
            .chunks_exact(4)
            .map(|c| {
                i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2_147_483_648.0
            })
            .collect(),
        (3, 32) => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        (tag, bits) => anyhow::bail!(
            "unsupported WAV encoding (format {tag}, {bits}-bit). Convert it \
             first, e.g.\n  ffmpeg -i clip.wav -ar 24000 -ac 1 -c:a pcm_s16le ready.wav"
        ),
    };
    Ok((samples, rate, channels))
}

/// Linear resampling. Good enough for a speaker-encoder input, where the
/// content is a voice rather than something whose spectrum is being measured.
fn resample_linear(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.len() < 2 {
        return samples.to_vec();
    }
    let out_len = ((samples.len() as f64 * to as f64 / from as f64).round() as usize).max(1);
    let ratio = samples.len() as f64 / out_len as f64;
    let last = samples.len() - 1;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let lo = (pos.floor() as usize).min(last);
            let hi = (lo + 1).min(last);
            let frac = pos - lo as f64;
            (samples[lo] as f64 * (1.0 - frac) + samples[hi] as f64 * frac) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn header_is_a_well_formed_24khz_mono_wav() {
        let wav = encode_wav(&[0.0, 0.5, -0.5, 1.0]);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(
            u32::from_le_bytes(wav[24..28].try_into().unwrap()),
            SAMPLE_RATE
        );
        assert_eq!(
            u16::from_le_bytes(wav[22..24].try_into().unwrap()),
            1,
            "mono"
        );
        assert_eq!(
            u16::from_le_bytes(wav[34..36].try_into().unwrap()),
            16,
            "16-bit"
        );
        assert_eq!(wav.len(), 44 + 4 * 2);
    }

    #[test]
    fn normalization_hits_the_target_peak() {
        let wav = encode_wav(&[0.1, -0.05]);
        let first = i16::from_le_bytes(wav[44..46].try_into().unwrap());
        let expected = (0.89 * 32767.0) as i16;
        assert!(
            (first - expected).abs() <= 1,
            "loudest sample should hit 0.89 FS"
        );
    }

    #[test]
    fn silence_does_not_divide_by_zero() {
        let wav = encode_wav(&[0.0; 8]);
        assert_eq!(wav.len(), 44 + 16);
        assert!(wav[44..].iter().all(|b| *b == 0));
    }

    #[test]
    fn duration_is_sample_count_over_rate() {
        assert_eq!(duration_ms(&vec![0.0; 24_000]), 1000.0);
    }

    #[test]
    fn mp3_starts_with_a_frame_sync_and_is_far_smaller_than_wav() {
        // One second of a 220 Hz tone — real signal, so the encoder emits frames.
        let samples: Vec<f32> = (0..24_000)
            .map(|i| (i as f32 / 24_000.0 * 220.0 * std::f32::consts::TAU).sin() * 0.5)
            .collect();
        let mp3 = encode_mp3(&samples).unwrap();
        // MPEG audio frame sync: 11 set bits. ID3-less LAME output starts there.
        assert!(
            mp3.len() > 128,
            "suspiciously short MP3: {} bytes",
            mp3.len()
        );
        assert_eq!(mp3[0], 0xFF, "no MPEG frame sync");
        assert_eq!(mp3[1] & 0xE0, 0xE0, "no MPEG frame sync");
        let wav = encode_wav(&samples);
        assert!(
            mp3.len() * 2 < wav.len(),
            "mp3 {} vs wav {}",
            mp3.len(),
            wav.len()
        );
    }

    #[test]
    fn empty_input_encodes_without_panicking() {
        assert!(encode_mp3(&[]).is_ok());
    }

    /// The failure being guarded against: a buffer of exact zeros, and one at
    /// ±1 LSB of 16-bit (the measured −91 dB case). Both are below the floor.
    #[test]
    fn silence_falls_below_the_floor() {
        assert_eq!(peak_dbfs(&[]), f32::NEG_INFINITY);
        assert_eq!(peak_dbfs(&[0.0; 24_000]), f32::NEG_INFINITY);
        let one_lsb = vec![1.0f32 / 32_768.0; 24_000];
        let peak = peak_dbfs(&one_lsb);
        assert!((peak - -90.3).abs() < 0.1, "one LSB should be ~-90.3 dBFS, got {peak}");
        for buf in [&[][..], &[0.0; 8][..], &one_lsb[..]] {
            assert!(
                peak_dbfs(buf) < SILENCE_FLOOR_DBFS,
                "silence must not pass the floor"
            );
        }
    }

    /// The failure mode that actually reached production: the fp16 Kokoro
    /// pack's vocoder overflow returns a waveform of 100% NaN, which encodes
    /// to digital silence (`NaN as i16` is 0). The peak fold must stay
    /// NaN-blind so that buffer measures `-inf` and is caught by the floor —
    /// a NaN-propagating fold would return NaN, which is not `<` any floor,
    /// and the silence would ship as a healthy synthesis again.
    #[test]
    fn an_all_nan_buffer_reads_as_silent() {
        let all_nan = vec![f32::NAN; 24_000];
        assert_eq!(peak_dbfs(&all_nan), f32::NEG_INFINITY);
        assert!(peak_dbfs(&all_nan) < SILENCE_FLOOR_DBFS);
        // And if such a buffer were encoded anyway, it becomes exact zeros —
        // the mechanism that made the defect masquerade as digital silence.
        let wav = encode_wav(&all_nan);
        assert!(wav[44..].iter().all(|b| *b == 0));
    }

    /// Real speech — even a very quiet synthesis — clears the floor by tens of
    /// dB. A raw Kokoro buffer peaks around 0.3–1.0; test an order of
    /// magnitude quieter than that.
    #[test]
    fn quiet_but_real_audio_clears_the_floor() {
        let quiet: Vec<f32> = (0..24_000)
            .map(|i| (i as f32 / 24_000.0 * 220.0 * std::f32::consts::TAU).sin() * 0.03)
            .collect();
        let peak = peak_dbfs(&quiet);
        assert!((peak - -30.5).abs() < 0.5, "0.03 peak is ~-30.5 dBFS, got {peak}");
        assert!(peak > SILENCE_FLOOR_DBFS);
    }

    /// Full-scale audio reads as ~0 dBFS, so the units are dBFS and not
    /// something dB-shaped.
    #[test]
    fn full_scale_reads_as_zero_dbfs() {
        assert_eq!(peak_dbfs(&[1.0, -0.2, 0.5]), 0.0);
        assert_eq!(peak_dbfs(&[-1.0]), 0.0, "peak is the magnitude, sign-blind");
    }

    #[test]
    fn format_comes_from_the_extension_and_defaults_to_mp3() {
        assert_eq!(Format::for_path(Path::new("a.wav")), Format::Wav);
        assert_eq!(Format::for_path(Path::new("a.WAV")), Format::Wav);
        assert_eq!(Format::for_path(Path::new("a.mp3")), Format::Mp3);
        assert_eq!(Format::for_path(Path::new("narration")), Format::Mp3);
    }

    fn secs(n: f64) -> usize {
        (n * SAMPLE_RATE as f64) as usize
    }
    fn tone(seconds: f64, amplitude: f32) -> Vec<f32> {
        (0..secs(seconds))
            .map(|i| {
                amplitude * (i as f32 / SAMPLE_RATE as f32 * 220.0 * std::f32::consts::TAU).sin()
            })
            .collect()
    }
    /// Deterministic uniform noise ≈ −42 dB RMS — AGC-boosted room tone.
    fn room_tone(seconds: f64) -> Vec<f32> {
        let mut seed: u64 = 42;
        (0..secs(seconds))
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.028
            })
            .collect()
    }
    fn joined(parts: &[Vec<f32>]) -> Vec<f32> {
        parts.concat()
    }

    /// The real-world failure: a click as recording starts, ~1.8 s of
    /// AGC-boosted room tone, then the read. Anchoring on the click kept
    /// the pause, and the clone reproduced it at the start of narrations.
    #[test]
    fn hygiene_cuts_a_click_and_a_noisy_pause_before_the_read() {
        let raw = joined(&[tone(0.15, 0.4), room_tone(1.8), tone(20.0, 0.15)]);
        let trimmed = trim_recording(&raw);
        let seconds = trimmed.len() as f64 / SAMPLE_RATE as f64;
        assert!(seconds > 20.0, "kept the read plus pads, got {seconds:.2}s");
        assert!(seconds < 21.0, "click and pause must be gone, got {seconds:.2}s");
    }

    #[test]
    fn hygiene_leaves_a_tight_take_essentially_alone() {
        let raw = tone(10.0, 0.15);
        let trimmed = trim_recording(&raw);
        let diff = (raw.len() as f64 - trimmed.len() as f64).abs() / SAMPLE_RATE as f64;
        assert!(diff < 0.5, "cut {diff:.2}s from a snug clip");
    }

    /// A deliberate short first word with a natural pause after it must
    /// anchor the start, not be skipped as a transient.
    #[test]
    fn hygiene_keeps_a_short_first_word() {
        let raw = joined(&[tone(0.4, 0.15), vec![0.0; secs(0.4)], tone(10.0, 0.15)]);
        let trimmed = trim_recording(&raw);
        let seconds = trimmed.len() as f64 / SAMPLE_RATE as f64;
        assert!(seconds > 10.7, "first word was cut: {seconds:.2}s");
    }

    #[test]
    fn hygiene_refuses_silence_and_empty_input() {
        assert!(trim_recording(&vec![0.0; secs(12.0)]).is_empty());
        assert!(trim_recording(&[]).is_empty());
    }

    fn word(word: &str, start: f64, end: f64) -> crate::reconcile::Word {
        crate::reconcile::Word {
            word: word.to_string(),
            start,
            end,
        }
    }

    /// The reported failure: ~2 s of phantom babble + dead air before the
    /// first real word. The trim cuts to the aligned span (with pads) and
    /// shifts the timings onto the new clock.
    #[test]
    fn trims_leading_junk_and_shifts_words() {
        let samples = vec![0.1_f32; 5 * SAMPLE_RATE as usize];
        let mut words = vec![word("Timing", 2.0, 2.4), word("check.", 2.4, 3.0)];
        let trimmed = trim_to_aligned_words(samples, &mut words);
        // Keeps [2.0 - 0.15, 3.0 + 0.30] = 1.45 s of audio.
        let expected = ((3.0 + TRIM_TAIL_PAD_S) - (2.0 - TRIM_LEAD_PAD_S)) * f64::from(SAMPLE_RATE);
        assert_eq!(trimmed.len(), expected.round() as usize);
        assert!((words[0].start - TRIM_LEAD_PAD_S).abs() < 1e-9);
        assert!((words[1].end - (TRIM_LEAD_PAD_S + 1.0)).abs() < 1e-9);
    }

    /// A snug clip — words spanning essentially all of it — is untouched, so
    /// well-behaved engines (Kokoro) pay nothing for this pass.
    #[test]
    fn a_snug_clip_is_left_alone() {
        let samples = vec![0.1_f32; 2 * SAMPLE_RATE as usize];
        let mut words = vec![word("Hello", 0.05, 1.95)];
        let trimmed = trim_to_aligned_words(samples.clone(), &mut words);
        assert_eq!(trimmed.len(), samples.len());
        assert_eq!(words[0].start, 0.05, "no-op must not shift timings");
    }

    /// Degenerate cases: no words, or timings past the end of the audio.
    #[test]
    fn degenerate_timings_never_produce_empty_audio() {
        let samples = vec![0.1_f32; SAMPLE_RATE as usize];
        let mut none: Vec<crate::reconcile::Word> = vec![];
        assert_eq!(
            trim_to_aligned_words(samples.clone(), &mut none).len(),
            samples.len()
        );
        let mut past = vec![word("x", 10.0, 11.0)];
        assert_eq!(
            trim_to_aligned_words(samples.clone(), &mut past).len(),
            samples.len()
        );
    }
}
