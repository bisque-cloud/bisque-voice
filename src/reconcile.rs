//! Reconcile an engine's word timings onto SOURCE whitespace tokens.
//!
//! Rust port of `packages/tts-core/src/word-timing.ts`. **Keep the two in
//! lockstep** — they are two implementations of one published guarantee, and a
//! divergence here desynchronizes every cue with no error anywhere.
//!
//! The guarantee: `words` is 1:1 with the source's whitespace-delimited tokens,
//! in order. Cue markers index a *source* word index into this array
//! (`marker.wordIndex` counts source words in `parseTokenStream`,
//! `packages/presentation-format`), so if the phonemizer splits `3.5x` /
//! `$4.2M` / `state-of-the-art` into a different number of tokens, every
//! downstream cue drifts — silently. Guaranteeing `words.len() ==
//! source_words.len()` by construction eliminates the drift entirely.
//!
//! Two strategies, best first:
//!
//! - **EXACT** (`exact: true`): when the engine's non-whitespace characters
//!   equal the source's (case-insensitively), we know exactly which engine
//!   tokens cover each source word, so each source word gets real start/end
//!   times. This is the common case — Kokoro re-tokenizes but preserves
//!   characters.
//! - **APPROXIMATE** (`exact: false`): when the engine altered the text (number
//!   or symbol expansion, "$5" → "five dollars"), per-token mapping is
//!   impossible, so the utterance's `[start, end]` is distributed across source
//!   words by character length. Timings are approximate but the COUNT is still
//!   exact, so cues stay on the right words. The flag exists so we can monitor
//!   how often this path is taken rather than corrupting silently.
//!
//! Fails closed only when there is genuinely nothing to work with (the engine
//! returned no timings for non-empty source).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug)]
pub struct Reconciled {
    pub words: Vec<Word>,
    /// False when timings were distributed proportionally rather than mapped.
    pub exact: bool,
}

#[derive(Debug)]
pub struct WordTimingError(pub String);

impl std::fmt::Display for WordTimingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "word timing: {}", self.0)
    }
}
impl std::error::Error for WordTimingError {}

/// Source words = maximal non-whitespace runs, matching `parseTokenStream`.
pub fn source_words(text: &str) -> Vec<&str> {
    text.split_whitespace().filter(|w| !w.is_empty()).collect()
}

/// Lowercase, whitespace-stripped comparison key for character coverage.
fn norm(word: &str) -> Vec<char> {
    word.chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// EXACT alignment: if the concatenated normalized characters of the source and
/// the engine match, map each source word to the engine token(s) covering its
/// characters. Returns `None` when the character streams diverge.
fn align_by_char_coverage(words: &[&str], engine: &[Word]) -> Option<Vec<Word>> {
    // Per-character start/end, drawn from each engine token's own timing.
    let mut char_start: Vec<f64> = Vec::new();
    let mut char_end: Vec<f64> = Vec::new();
    let mut engine_chars: Vec<char> = Vec::new();
    for ew in engine {
        for c in norm(&ew.word) {
            engine_chars.push(c);
            char_start.push(ew.start);
            char_end.push(ew.end);
        }
    }

    let source_chars: Vec<char> = words.iter().flat_map(|w| norm(w)).collect();
    if source_chars != engine_chars {
        return None;
    }

    let mut out = Vec::with_capacity(words.len());
    let mut pos = 0usize;
    for w in words {
        let len = norm(w).len();
        if len == 0 {
            // A source "word" that normalizes to empty. Give it a zero-width
            // slot at the current cursor to keep the 1:1 count.
            let at = if pos > 0 {
                char_end[pos - 1]
            } else {
                char_start.first().copied().unwrap_or(0.0)
            };
            out.push(Word {
                word: (*w).to_string(),
                start: at,
                end: at,
            });
            continue;
        }
        out.push(Word {
            word: (*w).to_string(),
            start: char_start[pos],
            end: char_end[pos + len - 1],
        });
        pos += len;
    }
    Some(out)
}

/// APPROXIMATE alignment: distribute `[t0, t1]` across words by char length.
fn distribute_proportionally(words: &[&str], t0: f64, t1: f64) -> Vec<Word> {
    let lengths: Vec<f64> = words.iter().map(|w| norm(w).len().max(1) as f64).collect();
    let total: f64 = lengths.iter().sum();
    let span = (t1 - t0).max(0.0);
    let mut out = Vec::with_capacity(words.len());
    let mut cursor = t0;
    for (i, w) in words.iter().enumerate() {
        let start = cursor;
        // Pin the final word's end to t1 exactly so rounding never drifts past.
        let end = if i == words.len() - 1 {
            t1
        } else {
            start + span * lengths[i] / total
        };
        out.push(Word {
            word: (*w).to_string(),
            start,
            end,
        });
        cursor = end;
    }
    out
}

pub fn reconcile(source_text: &str, engine: &[Word]) -> Result<Reconciled, WordTimingError> {
    let words = source_words(source_text);
    if words.is_empty() {
        return Ok(Reconciled {
            words: Vec::new(),
            exact: true,
        });
    }
    if engine.is_empty() {
        return Err(WordTimingError(
            "engine returned no word timings for non-empty text".into(),
        ));
    }

    if let Some(exact) = align_by_char_coverage(&words, engine) {
        return Ok(Reconciled {
            words: exact,
            exact: true,
        });
    }

    let t0 = engine[0].start;
    let t1 = engine[engine.len() - 1].end;
    Ok(Reconciled {
        words: distribute_proportionally(&words, t0, t1),
        exact: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(word: &str, start: f64, end: f64) -> Word {
        Word {
            word: word.into(),
            start,
            end,
        }
    }

    #[test]
    fn empty_source_is_empty_and_exact() {
        let r = reconcile("   ", &[]).unwrap();
        assert!(r.words.is_empty());
        assert!(r.exact);
    }

    #[test]
    fn fails_closed_when_engine_returns_nothing() {
        assert!(reconcile("hello world", &[]).is_err());
    }

    #[test]
    fn identical_tokenization_maps_exactly() {
        let engine = vec![w("hello", 0.0, 0.5), w("world", 0.5, 1.0)];
        let r = reconcile("hello world", &engine).unwrap();
        assert!(r.exact);
        assert_eq!(r.words, engine);
    }

    #[test]
    fn engine_splitting_a_source_word_still_yields_one_word() {
        // The phonemizer split "state-of-the-art" into pieces. One source word
        // in, one word out, spanning the whole run.
        let engine = vec![
            w("state", 0.0, 0.2),
            w("-of-", 0.2, 0.4),
            w("the", 0.4, 0.5),
            w("-art", 0.5, 0.8),
        ];
        let r = reconcile("state-of-the-art", &engine).unwrap();
        assert!(
            r.exact,
            "characters match, so this must take the exact path"
        );
        assert_eq!(r.words.len(), 1);
        assert_eq!(r.words[0].word, "state-of-the-art");
        assert_eq!(r.words[0].start, 0.0);
        assert_eq!(r.words[0].end, 0.8);
    }

    #[test]
    fn engine_merging_source_words_still_yields_one_word_each() {
        let engine = vec![w("3.5xto", 0.0, 1.0)];
        let r = reconcile("3.5x to", &engine).unwrap();
        assert!(r.exact);
        assert_eq!(r.words.len(), 2);
        assert_eq!(r.words[0].word, "3.5x");
        assert_eq!(r.words[1].word, "to");
    }

    #[test]
    fn expanded_numbers_fall_back_to_proportional_but_keep_the_count() {
        // Kokoro says "four point two million dollars" for "$4.2M": the
        // characters no longer match, so exact mapping is impossible.
        let engine = vec![
            w("four", 0.0, 0.3),
            w("point", 0.3, 0.5),
            w("two", 0.5, 0.7),
            w("million", 0.7, 1.1),
            w("dollars", 1.1, 1.6),
        ];
        let r = reconcile("We raised $4.2M today", &engine).unwrap();
        assert!(!r.exact, "expansion must be reported as approximate");
        assert_eq!(r.words.len(), 4, "one word per source token, always");
        assert_eq!(
            r.words.iter().map(|x| x.word.as_str()).collect::<Vec<_>>(),
            vec!["We", "raised", "$4.2M", "today"]
        );
        assert_eq!(r.words[0].start, 0.0);
        assert_eq!(r.words[3].end, 1.6, "last word pinned to the utterance end");
        // Monotonic, non-overlapping.
        for pair in r.words.windows(2) {
            assert!(pair[0].end <= pair[1].start + f64::EPSILON);
        }
    }

    #[test]
    fn case_differences_do_not_break_exact_alignment() {
        let engine = vec![w("HELLO", 0.0, 0.5), w("World", 0.5, 1.0)];
        let r = reconcile("hello world", &engine).unwrap();
        assert!(r.exact);
        assert_eq!(r.words[0].end, 0.5);
    }

    #[test]
    fn word_count_always_matches_source_tokens() {
        // The property the whole cue system rests on, over a mixed sample.
        let cases = [
            "Revenue grew 3.5x to $4.2M in Q3.",
            "one",
            "a b c d e f g",
            "state-of-the-art  double   spaced",
        ];
        for text in cases {
            let engine = vec![w("anything", 0.0, 1.0)];
            let r = reconcile(text, &engine).unwrap();
            assert_eq!(r.words.len(), source_words(text).len(), "for {text:?}");
        }
    }
}
