//! Repair timing metadata without changing a recording or its playback clock.

use std::{collections::BTreeSet, path::Path, process::Command};

use anyhow::{bail, Context, Result};
use serde::Serialize;

use crate::{align, audio, device::Device, fetch::Models, pronounce, quality, reconcile::Word};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Output {
    words: Vec<Word>,
    duration_ms: u64,
    exact: bool,
    aligner: String,
    content_type: &'static str,
    size: usize,
    hash: String,
}

pub fn run(
    models: &Models,
    path: &Path,
    text: &str,
    aligner_id: &str,
    device: Device,
    threads: Option<usize>,
) -> Result<Output> {
    let script = Script::new(text)?;
    let spec = align::resolve(models, Some(aligner_id))?
        .context("align requires a supported forced aligner; `none` cannot repair timings")?;
    let path = path.canonicalize().context("opening existing audio")?;
    let bytes = std::fs::read(&path)?;
    let (samples, content_type) = decode(&path, &bytes)?;
    if samples.is_empty()
        || samples.iter().any(|s| !s.is_finite())
        || audio::peak_dbfs(&samples) < audio::SILENCE_FLOOR_DBFS
    {
        bail!("the recording is empty, non-finite or silent; no timings were repaired");
    }
    let pack = models.ensure_pack(spec, &mut std::io::stderr())?;
    let mut aligner = align::Aligner::open(&pack, threads, device)?;
    let measured = aligner.measure(&samples, &script.spoken)?;
    let reading = quality::read(&measured.tokens);
    let duration_ms = audio::duration_ms(&samples);
    if reading.is_collapse()
        || reading.is_unsure()
        || quality::dropped_tail(&measured.tokens, duration_ms.round() as i64) > 0
    {
        bail!("the recording could not be reliably aligned ({reading}); no timings were repaired. Alignment cannot repair a bad take");
    }
    let mut words = script.restore(&measured.words)?;
    validate(&mut words, duration_ms / 1000.0)?;
    if std::fs::read(&path)? != bytes {
        bail!("the recording changed during alignment; no timings were repaired");
    }
    Ok(Output {
        words,
        duration_ms: duration_ms.round() as u64,
        exact: true,
        aligner: spec.id.to_string(),
        content_type,
        size: bytes.len(),
        hash: crate::sha256_prefixed(&bytes),
    })
}

fn decode(path: &Path, bytes: &[u8]) -> Result<(Vec<f32>, &'static str)> {
    if bytes.starts_with(b"RIFF") {
        return Ok((audio::decode_wav_to_mono_24k(bytes)?, "audio/wav"));
    }
    if !path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("mp3"))
    {
        bail!("align accepts WAV or MP3 recordings");
    }
    // Decode the encoded file, including its padding/delay metadata. Measuring
    // the pre-encode PCM and applying a guessed codec offset is not equivalent.
    let decoded = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args([
            "-map", "0:a:0", "-f", "f32le", "-ar", "24000", "-ac", "1", "pipe:1",
        ])
        .output()
        .context("MP3 alignment needs ffmpeg on PATH (WAV decoding is built in)")?;
    if !decoded.status.success() {
        bail!(
            "decoding the MP3: {}",
            String::from_utf8_lossy(&decoded.stderr)
        );
    }
    let samples = decoded
        .stdout
        .chunks_exact(4)
        .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
        .collect();
    Ok((samples, "audio/mpeg"))
}

fn validate(words: &mut [Word], duration: f64) -> Result<()> {
    // Qwen timestamps are quantized to 80 ms. Its final end can round just
    // beyond EOF (33.120 on a measured 33.108 s joined recording). Clip only
    // that end, within one model step; earlier spans and larger errors fail.
    if let Some(last) = words.last_mut() {
        if last.end > duration && last.end <= duration + 0.080 {
            last.end = duration;
        }
    }
    let mut end = 0.0;
    for w in words {
        if !w.start.is_finite()
            || !w.end.is_finite()
            || w.start < end
            || w.end < w.start
            || w.end > duration
        {
            bail!(
                "invalid measured span for {:?}: {}..{} on a {duration}s recording",
                w.word,
                w.start,
                w.end
            );
        }
        end = w.end;
    }
    Ok(())
}

/// Character provenance joins spoken substitutions back onto display tokens,
/// including punctuation attached to a marker and one-to-many respellings.
struct Script {
    spoken: String,
    display_words: Vec<String>,
    spoken_words: Vec<String>,
    mapping: Vec<Vec<usize>>,
}

impl Script {
    fn new(text: &str) -> Result<Self> {
        let chars: Vec<char> = text.chars().collect();
        let (mut shown, mut spoken) = (Vec::new(), Vec::new());
        let (mut i, mut owner) = (0, 0);
        while i < chars.len() {
            if chars[i] == '[' {
                if let Some((display, _, substitute, next)) = pronounce::parse_override(&chars, i) {
                    let said = substitute.as_deref().unwrap_or(&display);
                    let d: Vec<&str> = display.split_whitespace().collect();
                    let s: Vec<&str> = said.split_whitespace().collect();
                    if d.len() > 1 && d.len() != s.len() {
                        bail!("pronunciation override {display:?} has an ambiguous source-token mapping; use one override per display word");
                    }
                    let append = |out: &mut Vec<(char, usize)>, text: &str, one: bool| {
                        let (mut n, mut in_word) = (0, false);
                        for c in text.chars() {
                            if c.is_whitespace() {
                                in_word = false;
                                out.push((c, owner));
                            } else {
                                if !in_word {
                                    n += 1;
                                }
                                in_word = true;
                                out.push((c, owner + if one { 0 } else { n - 1 }));
                            }
                        }
                    };
                    append(&mut shown, &display, false);
                    append(&mut spoken, said, d.len() == 1);
                    owner += d.len();
                    i = next;
                    continue;
                }
            }
            shown.push((chars[i], owner));
            spoken.push((chars[i], owner));
            owner += 1;
            i += 1;
        }
        let tokens = |chars: &[(char, usize)]| {
            chars
                .split(|(c, _)| c.is_whitespace())
                .filter(|s| !s.is_empty())
                .map(|s| {
                    (
                        s.iter().map(|(c, _)| *c).collect::<String>(),
                        s.iter().map(|(_, id)| *id).collect::<BTreeSet<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let (display, said) = (tokens(&shown), tokens(&spoken));
        if display.is_empty() {
            bail!("the script has no words to align");
        }
        let mapping = display
            .iter()
            .map(|(_, ids)| {
                said.iter()
                    .enumerate()
                    .filter(|(_, (_, s))| !ids.is_disjoint(s))
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();
        Ok(Self {
            spoken: spoken.iter().map(|(c, _)| *c).collect(),
            display_words: display.into_iter().map(|(w, _)| w).collect(),
            spoken_words: said.into_iter().map(|(w, _)| w).collect(),
            mapping,
        })
    }

    fn restore(&self, words: &[Word]) -> Result<Vec<Word>> {
        if words.len() != self.spoken_words.len()
            || words
                .iter()
                .zip(&self.spoken_words)
                .any(|(w, s)| &w.word != s)
        {
            bail!("the aligner did not return the original spoken tokens");
        }
        self.display_words
            .iter()
            .zip(&self.mapping)
            .map(|(word, indices)| {
                let first = indices
                    .first()
                    .context("display token has no spoken timing")?;
                let last = indices.last().unwrap();
                Ok(Word {
                    word: word.clone(),
                    start: words[*first].start,
                    end: words[*last].end,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitutions_keep_display_tokens_and_measured_pauses() {
        let script =
            Script::new("Say [Qwen3](kwen three), [New York](noo york) paid $4.2M.").unwrap();
        assert_eq!(script.spoken, "Say kwen three, noo york paid $4.2M.");
        let words: Vec<Word> = script
            .spoken_words
            .iter()
            .enumerate()
            .map(|(i, w)| Word {
                word: w.clone(),
                start: 2.0 + i as f64,
                end: 2.5 + i as f64,
            })
            .collect();
        let mut mapped = script.restore(&words).unwrap();
        assert_eq!(
            mapped.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
            ["Say", "Qwen3,", "New", "York", "paid", "$4.2M."]
        );
        assert_eq!((mapped[1].start, mapped[1].end), (3.0, 4.5));
        assert_eq!(mapped[2].start, 5.0);
        validate(&mut mapped, 10.0).unwrap();
    }

    #[test]
    fn ipa_and_attached_punctuation_preserve_source_indices() {
        let script = Script::new("The ([live](/lɪv/)) stream.").unwrap();
        assert_eq!(script.spoken, "The (live) stream.");
        assert_eq!(script.mapping, vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn ambiguous_or_missing_alignment_is_refused() {
        assert!(Script::new("[New York](newyork)").is_err());
        assert!(Script::new("  ").is_err());
        assert!(Script::new("hello").unwrap().restore(&[]).is_err());
    }

    #[test]
    fn whitespace_inside_overrides_does_not_merge_display_tokens() {
        let text = "a[ New York ]( noo york )b";
        let script = Script::new(text).unwrap();
        assert_eq!(script.spoken, pronounce::spoken_text(text));
        assert_eq!(script.display_words, vec!["a", "New", "York", "b"]);
        assert_eq!(script.mapping, vec![vec![0], vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn timings_cannot_escape_or_reverse_the_encoded_clock() {
        let word = |start, end| Word {
            word: "test".into(),
            start,
            end,
        };
        assert!(validate(&mut [word(1.0, 2.0)], 3.0).is_ok());
        for w in [
            word(-0.1, 1.0),
            word(1.0, 0.5),
            word(1.0, 4.0),
            word(f64::NAN, 1.0),
        ] {
            assert!(validate(&mut [w], 3.0).is_err());
        }
        assert!(validate(&mut [word(1.0, 2.0), word(1.5, 2.5)], 3.0).is_err());
        let mut rounded_end = [word(1.0, 2.08)];
        validate(&mut rounded_end, 2.04).unwrap();
        assert_eq!(rounded_end[0].end, 2.04);
        assert!(validate(&mut [word(1.0, 2.2)], 2.04).is_err());
    }
}
