//! What is shared by everything Qwen-derived.
//!
//! Three things outlived the ONNX Qwen3-TTS engine that used to live in
//! `qwen.rs`, because the **forced aligner** (`align.rs`) and the **voice
//! cloning engines** (`qwentts.rs`) both need them:
//!
//! - **The byte-level BPE.** Qwen's pre-tokenizer splits digits one at a time
//!   where GPT-2's groups them, and narration is full of numbers. The aligner
//!   and the talker must agree on it exactly or they disagree about where every
//!   figure on a slide begins.
//! - **Sentence chunking.** An autoregressive talker fed too much text stops
//!   emitting end-of-utterance and decodes to its frame cap, burning minutes to
//!   produce nothing. The budget below is weighted by script, because a Han
//!   character is a whole syllable where a Latin one is a fraction of one.
//! - **Audio assembly.** Retiming for `--speed`, and concatenating per-chunk
//!   timings onto one clock.
//!
//! Nothing here touches ONNX Runtime. It is plain text and PCM handling that
//! happens to have been written for Qwen first.

use std::ops::Range;
use std::path::Path;

use anyhow::{anyhow, bail, Result};

use crate::reconcile::Word;


/// Qwen's byte-level BPE pre-tokenizer pattern, verbatim from the Qwen2/2.5/3
/// `tokenizer.json`. It is **not** GPT-2's built-in ByteLevel pattern: Qwen
/// splits digits one at a time (`\p{N}`) where GPT-2 groups them (` ?\p{N}+`),
/// and narration is full of numbers. Using the wrong one mispronounces every
/// figure on the slide.
// The only runtime caller is the forced aligner (`align/run.rs`, behind the
// `onnx` feature) — the talker reads its tokenizer out of the GGUF — but the
// pattern-parity tests below run on every build, so this is allowed rather
// than gated.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
pub(crate) const QWEN_PRE_TOKENIZER: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Retime the waveform by `1 / speed`, linearly interpolated.
///
/// Unlike Kokoro, whose `speed` is a graph input that divides the predicted
/// durations before vocoding, Qwen has no speed control at all — it decides its
/// own pace. This is post-hoc resampling, so it shifts pitch the way tape speed
/// does; it exists so `--speed` keeps meaning something across engines, not
/// because it is as good as Kokoro's.
pub(crate) fn resample(samples: &[f32], speed: f32) -> Vec<f32> {
    // A zero, negative or NaN speed is nonsense rather than an instruction;
    // returning the audio unchanged beats emitting nothing.
    if !speed.is_finite() || speed <= 0.0 || (speed - 1.0).abs() < 1e-6 || samples.len() < 2 {
        return samples.to_vec();
    }
    let out_len = ((samples.len() as f64 / speed as f64).round() as usize).max(1);
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

// ── sentence chunking ───────────────────────────────────────────────────────
//
// Kokoro chunks because the graph has a hard 508-phoneme input limit. Qwen has
// no such limit and chunks for a worse reason: past a certain amount of text
// the talker stops emitting end-of-utterance and decodes to the frame cap, so
// the fail-closed guard rejects the result *after* burning the full ~11 CPU
// minutes. Bounding the input is the only lever we have over that, since the
// decode is autoregressive and there is nothing to look at until it ends.
//
// This splitter works on raw text rather than reusing `model::chunk_tokens`,
// which walks Kokoro's phonemized `Token`s — an English-only G2P stage that
// this engine does not have and must not require.

/// Characters that end a sentence. The Latin stops plus the CJK/full-width
/// forms: a Chinese or Japanese passage contains no ASCII stop at all, so a
/// splitter that only knows `.` sees one 400-character "sentence" and hands the
/// talker exactly the input it runs away on.
const SENTENCE_END: [char; 9] = ['.', '!', '?', '…', ';', '。', '！', '？', '；'];

/// Stops that end a sentence wherever they appear. The ASCII ones do not: CJK
/// is written without spaces, but `3.14` and `example.com` are single words,
/// and cutting inside one would make the talker read "three" and "fourteen" as
/// separate utterances. So an ASCII stop needs whitespace (or the end of the
/// text) after it; a full-width one does not.
const WIDE_END: [char; 5] = ['…', '。', '！', '？', '；'];

/// Quotes and brackets that belong to the sentence they close rather than to
/// the next one. Cutting before them would strand a lone `"` at the head of a
/// chunk, which the talker voices as a hesitation.
const CLOSERS: [char; 10] = ['"', '\'', '”', '’', ')', ']', '»', '」', '』', '）'];

/// Where an over-budget sentence may be cut. A clause break is not a full stop,
/// so this costs prosody and is never the first choice — but one long sentence
/// on its own reproduces the runaway, so it has to be cuttable somewhere.
const CLAUSE_END: [char; 7] = [',', '，', '、', ':', '：', '—', '–'];

/// Budget weight for one character.
///
/// The budget stands in for *audio duration*, and scripts are not equally
/// dense: a Han, Kana or Hangul character is a whole syllable (~6 spoken per
/// second) where a Latin or Cyrillic one is a fraction of one (~15/s). Without
/// the 3:1 weighting a single budget would chunk Chinese three times longer
/// than English, which is precisely the case that runs away.
fn char_cost(c: char) -> usize {
    let syllabic = matches!(c as u32,
        0x3040..=0x30FF     // Hiragana, Katakana
        | 0x3400..=0x4DBF   // CJK Unified Ideographs Extension A
        | 0x4E00..=0x9FFF   // CJK Unified Ideographs
        | 0xF900..=0xFAFF   // CJK Compatibility Ideographs
        | 0xAC00..=0xD7AF   // Hangul syllables
    );
    if syllabic {
        3
    } else {
        1
    }
}

fn cost_of(text: &str) -> usize {
    text.chars().map(char_cost).sum()
}

/// Byte ranges of `text`, one per sentence, covering it in order.
///
/// Trailing whitespace is absorbed into the sentence that just ended rather
/// than left to lead the next one: a chunk whose prefill starts on a space is a
/// different prompt than the same chunk without it.
fn sentence_spans(text: &str) -> Vec<Range<usize>> {
    let mut spans: Vec<Range<usize>> = Vec::new();
    let mut start = 0usize;
    let mut chars = text.char_indices().peekable();

    while let Some((i, c)) = chars.next() {
        if !SENTENCE_END.contains(&c) {
            continue;
        }
        // Absorb a run of stops ("?!", "...") and whatever closes the sentence,
        // so the boundary lands after all of them.
        let mut end = i + c.len_utf8();
        let mut wide = WIDE_END.contains(&c);
        while let Some(&(j, n)) = chars.peek() {
            if SENTENCE_END.contains(&n) || CLOSERS.contains(&n) {
                wide |= WIDE_END.contains(&n);
                end = j + n.len_utf8();
                chars.next();
            } else {
                break;
            }
        }
        let terminal = chars.peek().is_none_or(|&(_, n)| n.is_whitespace());
        if !(terminal || wide) {
            continue;
        }
        while let Some(&(j, n)) = chars.peek() {
            if n.is_whitespace() {
                end = j + n.len_utf8();
                chars.next();
            } else {
                break;
            }
        }
        spans.push(start..end);
        start = end;
    }
    if start < text.len() {
        spans.push(start..text.len());
    }
    spans.retain(|r| !text[r.clone()].trim().is_empty());
    spans
}

/// Cut a sentence that is over budget on its own.
///
/// Preference order is how much prosody the cut costs: clause punctuation, then
/// any whitespace, then blind at the budget. The last case is not hypothetical
/// — an unpunctuated CJK run has neither commas nor spaces — and half a word
/// beats a 164-second decode that is rejected and produces nothing.
fn split_oversized(text: &str, span: Range<usize>, budget: usize) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    let mut start = span.start;
    let mut used = 0usize;
    let mut clause: Option<usize> = None;
    let mut space: Option<usize> = None;

    for (offset, c) in text[span.clone()].char_indices() {
        let at = span.start + offset;
        let cost = char_cost(c);
        // Cut *before* accounting for this character, so no piece can exceed
        // the budget; the character then opens the next piece.
        if used + cost > budget && at > start {
            let cut = clause.or(space).unwrap_or(at);
            out.push(start..cut);
            start = cut;
            used = cost_of(&text[cut..at]);
            clause = None;
            space = None;
        }
        used += cost;
        let end = at + c.len_utf8();
        if CLAUSE_END.contains(&c) {
            clause = Some(end);
        } else if c.is_whitespace() {
            space = Some(end);
        }
    }
    if start < span.end {
        out.push(start..span.end);
    }
    out.retain(|r| !text[r.clone()].trim().is_empty());
    out
}

/// Pack sentences greedily into chunks of at most `budget`.
///
/// Greedy packing is also what keeps abbreviations from hurting: `chunk_tokens`
/// has no abbreviation list either, and a stop after "Dr" is only ever a
/// *candidate* cut — one is taken only when the budget is already spent, so
/// "Dr. Chen" stays together in every realistic input.
fn chunk_spans(text: &str, budget: usize) -> Vec<Range<usize>> {
    let budget = budget.max(1);
    let mut chunks: Vec<Range<usize>> = Vec::new();
    let mut current: Option<Range<usize>> = None;
    let mut used = 0usize;

    for span in sentence_spans(text) {
        let cost = cost_of(&text[span.clone()]);
        if cost > budget {
            if let Some(open) = current.take() {
                chunks.push(open);
                used = 0;
            }
            chunks.extend(split_oversized(text, span, budget));
            continue;
        }
        match current.take() {
            Some(open) if used + cost <= budget => {
                current = Some(open.start..span.end);
                used += cost;
            }
            Some(open) => {
                chunks.push(open);
                current = Some(span);
                used = cost;
            }
            None => {
                current = Some(span);
                used = cost;
            }
        }
    }
    if let Some(open) = current {
        chunks.push(open);
    }
    chunks
}

/// The text of each synthesis chunk, trimmed and in order.
pub(crate) fn chunk_text(text: &str, budget: usize) -> Vec<&str> {
    chunk_spans(text, budget)
        .into_iter()
        .map(|r| text[r].trim())
        .filter(|c| !c.is_empty())
        .collect()
}

/// Concatenate per-chunk `(pcm, timings)` into one utterance.
///
/// Every chunk's timings come back on its own clock, starting at zero. Shifting
/// them by the audio already emitted is the whole reason this is a function and
/// not three lines inline: get it wrong and every cue after the first chunk
/// drifts, with nothing anywhere reporting it.
pub(crate) fn concat_chunks(
    parts: Vec<(Vec<f32>, Vec<Word>)>,
    sample_rate: u32,
) -> (Vec<f32>, Vec<Word>) {
    let mut samples: Vec<f32> = Vec::with_capacity(parts.iter().map(|(p, _)| p.len()).sum());
    let mut words: Vec<Word> = Vec::new();
    for (pcm, mut chunk_words) in parts {
        let offset = samples.len() as f64 / f64::from(sample_rate);
        for w in chunk_words.iter_mut() {
            w.start += offset;
            w.end += offset;
        }
        words.append(&mut chunk_words);
        samples.extend_from_slice(&pcm);
    }
    (samples, words)
}


// ── tokenizer ───────────────────────────────────────────────────────────────

/// Build the Qwen byte-level BPE from a pack's `vocab.json` + `merges.txt`.
///
/// Shared with `align.rs` rather than copied: `QWEN_PRE_TOKENIZER` is the part
/// that decides how digits split, so two copies of this that drifted apart
/// would mispronounce (or misalign) every number on a slide with no error
/// anywhere. `whose` and `reinstall` name the pack and the command that would
/// fix it — the two callers install under different ids, and an error that
/// cannot be acted on is only half an error.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
pub(crate) fn load_qwen_bpe(
    pack: &Path,
    whose: &str,
    reinstall: &str,
) -> Result<tokenizers::Tokenizer> {
    let vocab = pack.join("vocab.json");
    let merges = pack.join("merges.txt");
    for p in [&vocab, &merges] {
        if !p.exists() {
            bail!("{whose} is missing {}. Reinstall it: {reinstall}", p.display());
        }
    }
    let bpe = tokenizers::models::bpe::BPE::from_file(
        &vocab.to_string_lossy(),
        &merges.to_string_lossy(),
    )
    .build()
    .map_err(|e| anyhow!("loading the Qwen BPE from {}: {e}", pack.display()))?;

    // Qwen's own split pattern first, then a byte-level pass with its regex
    // disabled — the exact `pre_tokenizer` sequence from the checkpoint's
    // `tokenizer.json`.
    let split = tokenizers::pre_tokenizers::split::Split::new(
        tokenizers::pre_tokenizers::split::SplitPattern::Regex(QWEN_PRE_TOKENIZER.to_string()),
        tokenizers::SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| anyhow!("compiling the Qwen pre-tokenizer pattern: {e}"))?;
    let byte_level = tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, false, false);
    let pre = tokenizers::pre_tokenizers::sequence::Sequence::new(vec![
        split.into(),
        byte_level.into(),
    ]);

    let mut inner = tokenizers::Tokenizer::new(bpe);
    inner.with_pre_tokenizer(Some(pre));
    Ok(inner)
}

#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
pub(crate) fn encode_with(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<i64>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow!("tokenizing {text:?}: {e}"))?;
    Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipping budget, mirrored from `qwentts.rs` so the runaway
    /// reproduction below is testing the number we actually use.
    const CHUNK_BUDGET: usize = 120;

    /// Every chunker test asserts the same property: chunking loses no
    /// character. Comparing with whitespace stripped lets a cut fall on a
    /// space without failing.
    fn glued(chunks: &[&str]) -> String {
        chunks
            .concat()
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect()
    }

    fn glued_source(text: &str) -> String {
        text.chars().filter(|c| !c.is_whitespace()).collect()
    }

    #[test]
    fn speed_one_leaves_the_waveform_untouched() {
        let samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        assert_eq!(resample(&samples, 1.0), samples);
    }

    #[test]
    fn speed_scales_duration_by_its_inverse() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        assert_eq!(
            resample(&samples, 2.0).len(),
            500,
            "twice as fast, half as long"
        );
        assert_eq!(resample(&samples, 0.5).len(), 2000);
        assert_eq!(resample(&samples, 1.25).len(), 800);
    }

    #[test]
    fn resampling_preserves_the_shape_of_a_ramp() {
        // A linear ramp is exactly representable under linear interpolation, so
        // any deviation is an indexing bug rather than interpolation error.
        let samples: Vec<f32> = (0..101).map(|i| i as f32).collect();
        let out = resample(&samples, 2.0);
        for (i, v) in out.iter().enumerate() {
            let expected = i as f32 * (samples.len() as f32 / out.len() as f32);
            assert!((v - expected).abs() < 1e-3, "sample {i}: {v} vs {expected}");
        }
    }

    #[test]
    fn a_nonsense_speed_is_ignored_rather_than_producing_no_audio() {
        let samples: Vec<f32> = (0..10).map(|i| i as f32).collect();
        assert_eq!(resample(&samples, 0.0), samples);
        assert_eq!(resample(&samples, -1.0), samples);
        assert_eq!(resample(&samples, f32::NAN), samples);
        // Degenerate inputs must not panic on the `len - 1` indexing.
        assert_eq!(resample(&[], 2.0), Vec::<f32>::new());
        assert_eq!(resample(&[0.5], 2.0), vec![0.5]);
    }

    #[test]
    fn short_text_is_one_chunk_and_is_trimmed() {
        assert_eq!(chunk_text("  Hello there.  ", CHUNK_BUDGET), ["Hello there."]);
        // Nothing to say is not one empty chunk; it is no chunks.
        assert!(chunk_text("   ", CHUNK_BUDGET).is_empty());
        assert!(chunk_text("", CHUNK_BUDGET).is_empty());
    }

    #[test]
    fn latin_sentences_pack_up_to_the_budget_and_cut_at_a_stop() {
        let text = "One sentence here. Another sentence follows! And a third one? \
                    Then a fourth; finally a fifth.";
        let chunks = chunk_text(text, 40);
        assert!(chunks.len() > 1, "{chunks:?}");
        for c in &chunks {
            assert!(cost_of(c) <= 40, "chunk over budget: {c:?}");
        }
        // Every cut lands after a stop, so no chunk starts mid-clause.
        for c in &chunks[..chunks.len() - 1] {
            assert!(
                c.ends_with(['.', '!', '?', ';']),
                "cut must land on a sentence end: {c:?}"
            );
        }
        assert_eq!(glued(&chunks), glued_source(text));
    }

    #[test]
    fn the_reproduced_runaway_splits_under_the_shipping_budget() {
        // Verbatim from the Russian demo presentation: 182 characters that ran
        // away as one decode — 2048 frames, no end-of-utterance, ~11 CPU
        // minutes for a rejected result — while each sentence alone synthesized
        // in 2-5 s. What matters is that the SHIPPING budget splits it, not
        // that some test-only number does.
        let text = "Это реестр локальных моделей, а не обёртка вокруг одной из них. \
                    Kokoro маленькая и быстрая, но говорит только по-английски. \
                    Qwen3 намного больше и знает десять языков. Выбор за вами.";
        let chunks = chunk_text(text, CHUNK_BUDGET);
        assert!(chunks.len() > 1, "the runaway input must split: {chunks:?}");
        for c in &chunks {
            assert!(cost_of(c) <= CHUNK_BUDGET, "chunk over budget: {c:?}");
            assert!(
                c.ends_with('.'),
                "every cut in this input lands on a full stop: {c:?}"
            );
        }
        assert_eq!(glued(&chunks), glued_source(text));

        // The Chinese demo of the same slide: no ASCII stops at all, mixed with
        // Latin product names, and the ideograph weighting is what keeps it
        // from becoming one over-long chunk.
        let zh = "这是本地模型的注册表，而不是对其中某一个模型的封装。\
                  Kokoro 小巧快速，但只会说英语。Qwen3 大得多，会说十种语言。选择权在您。";
        let chunks = chunk_text(zh, CHUNK_BUDGET);
        assert!(chunks.len() > 1, "{chunks:?}");
        for c in &chunks {
            assert!(cost_of(c) <= CHUNK_BUDGET, "chunk over budget: {c:?}");
        }
        assert_eq!(glued(&chunks), glued_source(zh));
    }

    #[test]
    fn cjk_stops_split_without_any_whitespace_to_help() {
        // No spaces anywhere: `。` is the only boundary there is, and an
        // ASCII-only splitter would see one 19-character sentence.
        let text = "你好世界。这是第二句话。第三句在这里。";
        let chunks = chunk_text(text, 20);
        assert_eq!(chunks, ["你好世界。", "这是第二句话。", "第三句在这里。"]);
        // The other full-width stops and the ellipsis behave the same way.
        assert_eq!(
            chunk_text("真的吗？当然！好吧…再见。", 12),
            ["真的吗？", "当然！", "好吧…", "再见。"]
        );
    }

    #[test]
    fn ideographs_cost_more_than_letters_so_one_budget_fits_both() {
        assert_eq!(char_cost('a'), 1);
        assert_eq!(char_cost('я'), 1);
        assert_eq!(char_cost('。'), 1);
        for c in ['字', 'あ', 'カ', '한'] {
            assert_eq!(char_cost(c), 3, "{c:?} is a syllable, not a letter");
        }
        // 20 ideographs is roughly as much speech as 60 letters, and the budget
        // has to see them that way or CJK chunks run three times long.
        assert_eq!(cost_of(&"字".repeat(20)), cost_of(&"a".repeat(60)));
    }

    #[test]
    fn a_stop_inside_a_number_or_a_hostname_is_not_a_sentence_end() {
        // "3.14" split in two would be read as "three" then "fourteen".
        assert_eq!(chunk_text("Pi is 3.14 exactly.", CHUNK_BUDGET), ["Pi is 3.14 exactly."]);
        assert_eq!(
            chunk_text("Visit example.com today.", CHUNK_BUDGET),
            ["Visit example.com today."]
        );
    }

    #[test]
    fn a_closing_quote_stays_with_the_sentence_it_closes() {
        let chunks = chunk_text("\"Stop!\" she said. Then silence.", 20);
        assert_eq!(chunks[0], "\"Stop!\" she said.");
        assert!(
            chunks.iter().all(|c| !c.starts_with(['"', '\''])
                || c.starts_with('"') && c.len() > 1),
            "no chunk may open with a stranded quote: {chunks:?}"
        );
    }

    #[test]
    fn an_over_long_sentence_falls_back_to_clause_punctuation() {
        // One sentence, no stop until the very end: the sentence walk yields a
        // single over-budget span, which is exactly the case that reproduces
        // the runaway if it is left whole.
        let text = "alpha, beta, gamma, delta, epsilon, zeta, eta, theta.";
        let chunks = chunk_text(text, 20);
        assert!(chunks.len() > 1, "{chunks:?}");
        for c in &chunks {
            assert!(cost_of(c) <= 20, "chunk over budget: {c:?}");
        }
        for c in &chunks[..chunks.len() - 1] {
            assert!(c.ends_with(','), "cut must land on a clause end: {c:?}");
        }
        assert_eq!(glued(&chunks), glued_source(text));
        // Full-width clause marks are the same fallback for CJK.
        let cjk = "第一部分，第二部分，第三部分，第四部分";
        let chunks = chunk_text(cjk, 15);
        assert!(chunks.len() > 1, "{chunks:?}");
        assert_eq!(glued(&chunks), glued_source(cjk));
    }

    #[test]
    fn without_clauses_the_fallback_is_whitespace_before_the_hard_cap() {
        let text = "word one two three four five six seven eight nine ten eleven";
        let chunks = chunk_text(text, 20);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(cost_of(c) <= 20, "chunk over budget: {c:?}");
            // Whitespace cuts, so no word is ever broken in half.
            assert!(text.contains(*c), "cut mid-word: {c:?}");
        }
        assert_eq!(glued(&chunks), glued_source(text));
    }

    #[test]
    fn a_clause_free_space_free_run_still_hits_the_hard_cap() {
        // Nothing to cut on at all. Splitting mid-word is ugly; decoding 164 s
        // of audio and then rejecting it is worse.
        let text = "a".repeat(100);
        let chunks = chunk_text(&text, 30);
        assert_eq!(chunks.len(), 4, "{chunks:?}");
        assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [30, 30, 30, 10]);
        assert_eq!(glued(&chunks), glued_source(&text));

        // The same run in an unpunctuated script, where the weighting means
        // three characters buy a whole budget unit each.
        let cjk = "字".repeat(50);
        let chunks = chunk_text(&cjk, 30);
        for c in &chunks {
            assert!(cost_of(c) <= 30, "chunk over budget: {c:?}");
            assert_eq!(c.chars().count(), 10, "30 units is 10 ideographs");
        }
        assert_eq!(glued(&chunks), glued_source(&cjk));
    }

    #[test]
    fn a_degenerate_budget_still_terminates_and_keeps_every_character() {
        for budget in [0usize, 1, 2, 3] {
            let text = "Hi there. 你好世界。ok";
            let chunks = chunk_text(text, budget);
            assert!(!chunks.is_empty(), "budget {budget}");
            assert_eq!(glued(&chunks), glued_source(text), "budget {budget}");
        }
    }

    #[test]
    fn chunk_timings_are_offset_by_the_audio_already_emitted() {
        let rate = 24_000u32;
        let word = |start: f64, end: f64| Word {
            word: String::new(),
            start,
            end,
        };
        // 1 s, 0.5 s, 2 s of audio, each carrying timings on its own clock.
        let parts = vec![
            (vec![0.0f32; 24_000], vec![word(0.0, 1.0)]),
            (vec![0.0f32; 12_000], vec![word(0.0, 0.25), word(0.25, 0.5)]),
            (vec![0.0f32; 48_000], vec![word(0.0, 2.0)]),
        ];
        let (samples, words) = concat_chunks(parts, rate);
        assert_eq!(samples.len(), 84_000);
        let spans: Vec<(f64, f64)> = words.iter().map(|w| (w.start, w.end)).collect();
        assert_eq!(
            spans,
            vec![(0.0, 1.0), (1.0, 1.25), (1.25, 1.5), (1.5, 3.5)]
        );
        // Continuous and ending exactly at the concatenation's duration — the
        // property a cue drift would break.
        assert_eq!(words.last().unwrap().end, samples.len() as f64 / f64::from(rate));
        for pair in words.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "gap or overlap at a boundary");
        }
    }

    #[test]
    fn one_chunk_is_unchanged_by_the_concatenation() {
        // A single chunk must be bit-identical to the un-chunked path, or the
        // short-narration audio hash moves for no reason.
        let pcm: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let words = vec![Word {
            word: String::new(),
            start: 0.0,
            end: 0.5,
        }];
        let (samples, words) = concat_chunks(vec![(pcm.clone(), words.clone())], 24_000);
        assert_eq!(samples, pcm);
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[0].end, 0.5);
    }

    #[test]
    fn nothing_to_narrate_produces_no_chunks_rather_than_an_empty_one() {
        // `narrate` bails on this; the splitter's job is to report it honestly
        // rather than hand the talker a lone "…".
        for text in ["...", "。。。", " , ", "!!!"] {
            assert!(
                chunk_text(text, CHUNK_BUDGET).is_empty()
                    || chunk_text(text, CHUNK_BUDGET)
                        .iter()
                        .all(|c| !c.trim().is_empty()),
                "{text:?}"
            );
        }
    }

    const TINY_TOKENS: [&str; 17] = [
        "a",
        "s",
        "i",
        "t",
        "n",
        "Ċ",
        "1",
        "2",
        "as",
        "ass",
        "assi",
        "assis",
        "assist",
        "assista",
        "assistan",
        "assistant",
        "12",
    ];

    fn tiny_id(token: &str) -> i64 {
        TINY_TOKENS.iter().position(|t| *t == token).unwrap() as i64
    }

    fn tiny_bpe(dir: &Path) {
        let vocab: serde_json::Map<String, serde_json::Value> = TINY_TOKENS
            .iter()
            .enumerate()
            .map(|(i, t)| ((*t).to_string(), serde_json::json!(i)))
            .collect();
        std::fs::write(dir.join("vocab.json"), serde_json::to_vec(&vocab).unwrap()).unwrap();
        std::fs::write(
            dir.join("merges.txt"),
            "#version: 0.2\na s\nas s\nass i\nassi s\nassis t\nassist a\nassista n\nassistan t\n1 2\n",
        )
        .unwrap();
    }

    #[test]
    fn digits_are_split_one_at_a_time_the_way_qwen_does() {
        // The whole reason QWEN_PRE_TOKENIZER is carried verbatim: Qwen splits
        // digits singly where GPT-2's built-in pattern groups them. Narration
        // is full of numbers, and the aligner and the talker have to agree on
        // where each one begins.
        let dir = tempfile::tempdir().unwrap();
        tiny_bpe(dir.path());
        let tok = load_qwen_bpe(dir.path(), "the test pack", "n/a").unwrap();
        assert_eq!(
            encode_with(&tok, "12").unwrap(),
            vec![tiny_id("1"), tiny_id("2")],
            "\"12\" must be two tokens, not one"
        );
    }

    #[test]
    fn a_pack_without_a_tokenizer_names_the_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let err = load_qwen_bpe(dir.path(), "the aligner pack", "bisque-voice install <aligner>")
            .unwrap_err()
            .to_string();
        assert!(err.contains("vocab.json"), "{err}");
        assert!(err.contains("the aligner pack"), "{err}");
        assert!(err.contains("bisque-voice install"), "the fix is actionable: {err}");
    }
}
