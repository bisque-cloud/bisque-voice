//! Per-word timestamps from Kokoro's own phoneme duration predictor.
//!
//! Port of `TimestampPredictor.swift` in `packages/kokoro-swift`. Both take the
//! same `pred_dur` tensor — a frame count per input token — and walk it
//! alongside the G2P token list. **Keep the two in lockstep**: they are the
//! reason a cue lands on the same word on macOS and on Linux.
//!
//! `pred_dur` is a property of Kokoro, not of any one runtime. On macOS the MLX
//! engine reads it directly; on ONNX it is an internal tensor that our model
//! pack promotes to a graph output (see `tools/promote-duration.ts`).

use crate::g2p::Token;
use crate::reconcile::Word;

/// Frames → seconds. A Kokoro frame is 600 samples at 24 kHz = 1/40 s. The
/// walk below counts in *half*-frames so a separating space can be split
/// between the words on either side, which is why the divisor is 80, not 40.
const HALF_FRAMES_PER_SECOND: f64 = 80.0;

/// Derive one `Word` per G2P token that carries phonemes.
///
/// `frames[0]` is the leading `<bos>` pad and `frames[len-1]` the trailing
/// `<eos>` pad, mirroring the `[0, …ids, 0]` shape fed to the model.
pub fn word_timings(tokens: &[Token], frames: &[f32]) -> Vec<Word> {
    // We need at least <bos>, one token, <eos>.
    if tokens.is_empty() || frames.len() < 3 {
        return Vec::new();
    }

    // Two cursors, in half-frames.
    let mut right = 2.0 * (frames[0] as f64 - 3.0).max(0.0);
    let mut left = right;

    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 1usize;
    for t in tokens {
        // A silent word — the second word of a multi-word pronunciation
        // override — owns no frames, so it still fits at the very end of the
        // audio and takes a zero-width slot there.
        let silent = t.phonemes.as_deref() == Some("") && t.whitespace.is_empty();
        if i >= frames.len() - 1 && !silent {
            break;
        }
        let phoneme_count = match &t.phonemes {
            Some(p) => p.chars().count(),
            None => {
                // A token with no phonemes still consumes the separating space.
                if !t.whitespace.is_empty() {
                    i += 1;
                    if i >= frames.len() {
                        break;
                    }
                    left = right + frames[i] as f64;
                    right = left + frames[i] as f64;
                    i += 1;
                }
                continue;
            }
        };

        let j = i + phoneme_count;
        if j >= frames.len() {
            break;
        }

        let start = left / HALF_FRAMES_PER_SECOND;
        let token_duration: f64 = frames[i..j].iter().map(|f| *f as f64).sum();
        let space_duration = if t.whitespace.is_empty() {
            0.0
        } else {
            frames[j] as f64
        };

        left = right + 2.0 * token_duration + space_duration;
        out.push(Word {
            word: t.text.clone(),
            start,
            end: left / HALF_FRAMES_PER_SECOND,
        });
        right = left + space_duration;
        i = j + if t.whitespace.is_empty() { 0 } else { 1 };
    }
    out
}

/// There is deliberately no punctuation-folding step here.
///
/// `PresenterAudio/LocalTTSService` and `services/kokoro-synth-cli` fold a
/// trailing "." into the preceding word, because MisakiSwift emits attached
/// punctuation as its own token. This engine collapses each source word's
/// sub-tokens before returning, so "Q3." arrives as one token already — and
/// folding on top of that would merge two *genuinely separate* source tokens
/// (an em dash standing alone, say), break the 1:1 character correspondence,
/// and silently downgrade every timing to the approximate path.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::g2p::Token;

    fn tok(text: &str, phonemes: &str, whitespace: &str) -> Token {
        Token {
            text: text.into(),
            phonemes: if phonemes.is_empty() {
                None
            } else {
                Some(phonemes.into())
            },
            whitespace: whitespace.into(),
            ..Token::default()
        }
    }

    #[test]
    fn timings_are_monotonic_and_bounded_by_the_audio() {
        // "hɛlˈoʊ wˈɜːld" = 6 + 6 phonemes, plus a space token, plus bos/eos.
        let tokens = vec![tok("hello", "hɛlˈoʊ", " "), tok("world", "wˈɜːld", "")];
        let frames: Vec<f32> = vec![
            17.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 12.0, 8.0,
        ];
        let words = word_timings(&tokens, &frames);
        assert_eq!(words.len(), 2);
        assert!(words[0].start <= words[0].end);
        assert!(words[0].end <= words[1].start + 1e-9);
        let total: f32 = frames.iter().sum();
        assert!(
            words[1].end <= total as f64 / 40.0 + 1e-6,
            "past the end of the audio"
        );
    }

    #[test]
    fn a_token_without_phonemes_does_not_emit_a_word() {
        let tokens = vec![
            tok("hello", "hɛlˈoʊ", " "),
            tok("$", "", " "),
            tok("world", "wˈɜːld", ""),
        ];
        let frames: Vec<f32> = vec![10.0; 20];
        let words = word_timings(&tokens, &frames);
        assert_eq!(
            words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
            vec!["hello", "world"]
        );
    }

    #[test]
    fn a_silent_word_at_the_end_of_the_audio_still_gets_a_timing() {
        // `see [New York City](new-york).`: "York" and "City." are silent
        // words of the override and own no frames, so they sit at the very
        // end — where the walk used to stop one or two words short.
        let tokens = vec![
            tok("see", "sˈi", " "),
            tok("New", "nˌujˈɔɹk.", ""),
            Token {
                phonemes: Some(String::new()),
                ..tok("York", "", "")
            },
            Token {
                phonemes: Some(String::new()),
                ..tok("City.", "", "")
            },
        ];
        // <bos>, 3 + space + 9 phonemes, <eos>.
        let frames: Vec<f32> = vec![4.0; 1 + 3 + 1 + 9 + 1];
        let words = word_timings(&tokens, &frames);
        assert_eq!(words.len(), tokens.len(), "one timing per token");
        let new_end = words[1].end;
        for w in &words[2..] {
            assert_eq!(
                (w.start, w.end),
                (new_end, new_end),
                "{} is zero-width",
                w.word
            );
        }
    }

    #[test]
    fn too_few_frames_yields_nothing_rather_than_panicking() {
        let tokens = vec![tok("hi", "hˈI", "")];
        assert!(word_timings(&tokens, &[1.0, 2.0]).is_empty());
        assert!(word_timings(&tokens, &[]).is_empty());
    }

    #[test]
    fn phoneme_run_overflowing_the_frame_buffer_stops_cleanly() {
        let tokens = vec![tok("supercalifragilistic", "sˈupəɹkˌælɪfɹˌædʒɪlˈɪstɪk", "")];
        let words = word_timings(&tokens, &[5.0, 5.0, 5.0]);
        assert!(words.is_empty());
    }
}
