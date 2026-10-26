//! Chinese G2P, ported from `misaki/zh.py`.
//!
//! Chinese is written without spaces, so the text is first **segmented** into
//! words (jieba), each word is looked up as **pinyin** (pypinyin), and each
//! pinyin syllable is mapped to **IPA** (`transcription.py`). Words are joined
//! with a space, which is a real token to Kokoro and is what puts the pauses in
//! the right places.
//!
//! Segmentation is not a nicety. `银行` is read `yín háng` ("bank") but the
//! same second character alone is `xíng`; only knowing the word boundary gets
//! it right.
//!
//! ## One deliberate divergence from Misaki
//!
//! `ZHG2P.__call__` decides whether the first segment is Chinese by testing
//! `text[0]` — the text as given — and then iterates over segments of `zhstr`,
//! the text *after* `cn2an` has rewritten Arabic numerals into Chinese
//! numerals. It alternates the flag from there.
//!
//! When that rewrite changes the class of the leading character — which it does
//! for **any string starting with a digit** — the flag is inverted for the
//! whole string. Chinese runs are then emitted as raw hanzi and punctuation is
//! fed to the pinyin converter. The damage is not cosmetic: "2026" comes back
//! as the literal characters 二千零二十六, which are not in Kokoro's
//! vocabulary, and "2026年的测试。" raises outright. Text beginning with a year
//! or a figure is ordinary narration.
//!
//! This reads the flag off `zhstr`, the string actually being iterated.
//! Measured over the committed corpus plus digit-leading variants, 15 of 18
//! cases are byte-identical to stock Misaki and the 3 that change are exactly
//! the broken ones. `fixtures/g2p/zh.json` records stock Misaki's output for
//! every diverging case in a `misaki` field, and a test asserts each divergence
//! is a Misaki *failure* rather than a difference of opinion.

pub mod an2cn;
pub mod ipa;
pub mod pinyin;

use std::path::Path;

use anyhow::Result;
use jieba_rs::Jieba;

use super::Token;

/// The CJK Unified Ideographs block, which is Misaki's definition of "Chinese"
/// here (`zh='一-鿿'`).
fn is_han(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c)
}

pub struct ChineseG2P {
    jieba: Jieba,
    dictionaries: pinyin::Dictionaries,
    numbers: an2cn::An2Cn,
}

impl ChineseG2P {
    /// `dir` is the installed `g2p-zh` pack, holding pypinyin's two JSON
    /// dictionaries.
    pub fn new(dir: &Path) -> Result<Self> {
        Ok(ChineseG2P {
            // jieba-rs' embedded dictionary is the same one Python jieba
            // ships, and its segmentation was verified to match across the
            // fixture corpus — but only with HMM enabled, which is Python's
            // default and is easy to lose.
            jieba: Jieba::new(),
            dictionaries: pinyin::Dictionaries::load(dir)?,
            numbers: an2cn::An2Cn::new(),
        })
    }

    /// One word's IPA. `ZHG2P.word2ipa`.
    fn word_to_ipa(&self, word: &str) -> String {
        self.dictionaries
            .lazy_pinyin_tone3(word)
            .iter()
            .filter_map(|syllable| self.syllable_to_ipa(syllable))
            .collect()
    }

    /// One pinyin syllable's IPA. `ZHG2P.py2ipa`.
    fn syllable_to_ipa(&self, tone3: &str) -> Option<String> {
        let (normal, tone) = pinyin::split_tone(tone3)?;
        let initial = pinyin::initials(&normal);
        let final_ = pinyin::finals(&normal);
        let raw = ipa::syllable_to_ipa(&normal, initial, &final_, tone)?;
        Some(ipa::retone(&raw))
    }

    /// The whole text, phonemized. `ZHG2P.__call__`.
    pub fn phonemize_text(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        let transformed = self.numbers.transform(text);
        let Some(first) = transformed.chars().next() else {
            return String::new();
        };

        // Segments alternate between Chinese and not-Chinese, so the class of
        // the first one decides the whole alternation. Taken from the
        // transformed string — see the module docs for why that is not what
        // upstream does.
        let mut chinese = is_han(first);
        let mut out = String::new();
        for segment in split_by_script(&transformed) {
            if chinese {
                let words = self.jieba.cut(segment, true);
                let joined: Vec<String> =
                    words.iter().map(|w| self.word_to_ipa(w)).collect();
                out.push_str(&joined.join(" "));
            } else {
                out.push_str(segment);
            }
            chinese = !chinese;
        }
        // A combining macron below can survive the tables; Kokoro has no such
        // character and upstream strips it.
        out.replace('\u{32F}', "")
    }

    /// Phonemes as `Token`s. See `super::align_tokens`.
    pub fn phonemize(&self, text: &str) -> Vec<Token> {
        super::align_tokens(text, &self.phonemize_text(text))
    }
}

/// Split into maximal runs of Han and non-Han characters, alternating.
/// Equivalent to Python's `re.findall('[zh]+|[^zh]+', s)`.
fn split_by_script(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut current: Option<bool> = None;
    for (i, c) in text.char_indices() {
        let han = is_han(c);
        match current {
            Some(prev) if prev == han => {}
            Some(_) => {
                out.push(&text[start..i]);
                start = i;
            }
            None => {}
        }
        current = Some(han);
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_runs_alternate_and_lose_nothing() {
        assert_eq!(split_by_script("你好，世界"), vec!["你好", "，", "世界"]);
        assert_eq!(split_by_script("abc测试"), vec!["abc", "测试"]);
        assert_eq!(split_by_script("测试abc"), vec!["测试", "abc"]);
        assert_eq!(split_by_script("测试"), vec!["测试"]);
        assert_eq!(split_by_script("abc"), vec!["abc"]);
        assert!(split_by_script("").is_empty());
        // Whatever the split, concatenating it back must be lossless.
        for text in ["你好，世界！", "中文和English混在一起。", "2026年"] {
            assert_eq!(split_by_script(text).concat(), text);
        }
    }

    #[test]
    fn the_han_block_is_the_one_misaki_uses() {
        assert!(is_han('中'));
        assert!(is_han('\u{4E00}'));
        assert!(is_han('\u{9FFF}'));
        // Punctuation, even the full-width kind, is not Han.
        assert!(!is_han('，'));
        assert!(!is_han('。'));
        assert!(!is_han('、'));
        assert!(!is_han('a'));
        assert!(!is_han('2'));
    }
}
