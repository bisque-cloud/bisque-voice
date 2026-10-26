//! Japanese G2P, ported from `misaki/ja.py`.
//!
//! Japanese is the one language here that cannot be done with rules. Kanji
//! carry no pronunciation on their face — 行 is read differently in almost
//! every word it appears in — so the text has to go through a **morphological
//! analyzer**, which splits it into words *and* reports each word's reading
//! from a dictionary.
//!
//! Misaki uses MeCab through fugashi; this uses lindera, which is the same
//! algorithm over the same UniDic data. Verified: identical segmentation and
//! identical `pron` fields.
//!
//! The pipeline:
//!
//! 1. Normalize (`normalize`): fold width variants, rewrite numbers as words.
//! 2. Analyze: words, each with a katakana reading, folded to hiragana.
//! 3. Regroup against `ja_words.txt`, so a compound the analyzer split is put
//!    back together before it is read.
//! 4. Walk the kana, mapping each to IPA with its neighbours in view — the
//!    sound of `ん` depends on what follows it, and small kana modify the kana
//!    before them.
//! 5. Space the tokens, with punctuation attaching the way it does in print.
//!
//! ## What lindera does not give us, and why it does not matter
//!
//! `ja.py` reads fugashi's `char_type`, MeCab's character category for the
//! token. lindera does not expose it. It does not need to: the expression is
//! `6 if w.char_type == 7 or not w.is_unk else w.char_type`, so **every token
//! the dictionary knows becomes 6** regardless of its category, and the real
//! value is consulted only for unknown tokens — where the Unicode block is
//! enough. See `normalize::category`.

pub mod hepburn;
pub mod normalize;
pub mod num2kana;

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use lindera::dictionary::load_fs_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use lindera::tokenizer::Tokenizer;

use super::Token;

/// UniDic's `pron` field — the phonetic reading, which is what Misaki wants.
///
/// It is not the same as the spelling-derived reading: 東京 has `pron`
/// トーキョー (with the long vowel actually spoken) where the kana form is
/// トウキョウ, and は as a particle has `pron` ワ, which is how it is said.
const PRON_FIELD: usize = 9;

/// One analyzed word: its surface text, its reading in hiragana, and its MeCab
/// character category.
#[derive(Debug, Clone)]
struct Word {
    surface: String,
    hira: String,
    char_type: u8,
}

/// A token on its way out, and whether a space follows it.
#[derive(Debug, Clone)]
struct Spaced {
    text: String,
    space: bool,
}

pub struct JapaneseG2P {
    tokenizer: Tokenizer,
    /// Misaki's word list, used to put split compounds back together.
    words: HashSet<String>,
}

impl JapaneseG2P {
    /// `dir` is the installed `g2p-ja` pack: a lindera dictionary directory
    /// plus `ja_words.txt`.
    pub fn new(dir: &Path) -> Result<Self> {
        let dictionary = load_fs_dictionary(&dir.join("lindera-unidic"))
            .map_err(|e| anyhow::anyhow!("loading the UniDic dictionary: {e}"))?;
        let segmenter = Segmenter::new(Mode::Normal, dictionary, None);

        let list = std::fs::read_to_string(dir.join("ja_words.txt"))
            .with_context(|| format!("reading {}", dir.join("ja_words.txt").display()))?;
        Ok(JapaneseG2P {
            tokenizer: Tokenizer::new(segmenter),
            words: list.lines().map(|l| l.trim().to_string()).collect(),
        })
    }

    fn analyze(&self, text: &str) -> Result<Vec<Word>> {
        let mut tokens = self
            .tokenizer
            .tokenize(text)
            .map_err(|e| anyhow::anyhow!("analyzing Japanese text: {e}"))?;
        let mut out = Vec::with_capacity(tokens.len());
        for token in tokens.iter_mut() {
            let surface = token.surface.to_string();
            let unknown = token.word_id.is_unknown();
            let details = token.details();
            let pron = details
                .get(PRON_FIELD)
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty() && s != "*")
                .unwrap_or_else(|| surface.clone());
            // Every known token is category 6; katakana is too. Only an
            // unknown token keeps its real category.
            let char_type = if unknown {
                let real = normalize::category(&surface);
                if real == normalize::char_type::KATAKANA {
                    normalize::char_type::HIRAGANA
                } else {
                    real
                }
            } else {
                normalize::char_type::HIRAGANA
            };
            out.push(Word {
                hira: normalize::kata2hira(&pron),
                surface,
                char_type,
            });
        }
        Ok(out)
    }

    /// Put back together any run of words that `ja_words.txt` lists as one.
    ///
    /// The analyzer splits on grammar, not on how something is read; a compound
    /// it cut in two can have a different reading than its halves. Only runs of
    /// a single character category are considered, and the **longest** match
    /// wins.
    fn regroup(&self, words: Vec<Word>) -> Vec<Word> {
        let mut groups: Vec<Vec<Word>> = Vec::new();
        let mut i = 0usize;
        while i < words.len() {
            // The run of like-categorized words starting at i.
            let limit = words[i + 1..]
                .iter()
                .position(|w| w.char_type != words[i].char_type)
                .map(|p| i + 1 + p)
                .unwrap_or(words.len());

            // Longest join that the word list knows, shortest fallback of one.
            let mut joined = None;
            for j in (i + 1..=limit).rev() {
                let candidate: String = words[i..j].iter().map(|w| w.surface.as_str()).collect();
                if self.words.contains(&candidate) {
                    joined = Some(j);
                    break;
                }
            }
            match joined {
                Some(j) => {
                    groups.push(words[i..j].to_vec());
                    i = j;
                }
                None => {
                    groups.push(vec![words[i].clone()]);
                    i += 1;
                }
            }
        }
        groups
            .into_iter()
            .map(|g| Word {
                surface: g.iter().map(|w| w.surface.as_str()).collect(),
                hira: g.iter().map(|w| w.hira.as_str()).collect(),
                char_type: g[0].char_type,
            })
            .collect()
    }

    /// One word's IPA. `JAG2P._romaji_word`.
    fn word_to_ipa(&self, word: &Word) -> String {
        // ASCII passes through untouched — it is already something Kokoro can
        // read, and the kana tables have nothing to say about it.
        if word.surface.is_ascii() {
            return word.surface.clone();
        }
        if word.char_type == normalize::char_type::SYMBOL {
            return word
                .surface
                .chars()
                .map(|c| hepburn::single(c).map(str::to_string).unwrap_or_else(|| c.to_string()))
                .collect();
        }
        if word.char_type != normalize::char_type::HIRAGANA {
            // An unknown kanji has no reading to give. Upstream drops it.
            return String::new();
        }

        let kana: Vec<char> = word.hira.chars().collect();
        let mut out = String::new();
        for (i, c) in kana.iter().enumerate() {
            let previous = if i > 0 { Some(kana[i - 1]) } else { None };
            let next = kana.get(i + 1).copied();
            out.push_str(&map_kana(previous, *c, next));
        }
        out
    }

    /// Space the tokens. `JAG2P._romaji_tokens`' second half.
    ///
    /// Punctuation does not take a space the way a word does: an opening
    /// bracket binds to what follows, a closing one and a comma bind to what
    /// precedes.
    fn space(&self, words: &[Word]) -> Vec<Spaced> {
        let mut out: Vec<Spaced> = Vec::with_capacity(words.len());
        for word in words {
            let text = self.word_to_ipa(word);
            let mut token = Spaced {
                space: false,
                text: text.clone(),
            };
            // Python writes these as `surface in '「『«'`, which is *substring*
            // containment, not membership — so an empty string matches. That
            // is not a quirk to tidy away: a word the dictionary cannot read
            // (an unknown kanji) maps to the empty string, and upstream sends
            // it down this branch, attaching it to the previous token rather
            // than letting it take a space of its own.
            let opens = word.surface.is_empty()
                || matches!(word.surface.as_str(), "「" | "『" | "«")
                || text.is_empty()
                || matches!(text.as_str(), "(" | "[");
            if opens {
                if let Some(previous) = out.last_mut() {
                    previous.space = true;
                }
            } else if matches!(word.surface.as_str(), "」" | "』" | "»")
                || matches!(text.as_str(), "]" | ")" | "." | "," | "?" | "!" | ":")
            {
                if let Some(previous) = out.last_mut() {
                    previous.space = false;
                }
                token.space = true;
            } else if text == " " {
                token.space = false;
            } else {
                token.space = true;
            }
            out.push(token);
        }
        // Any small tsu that survived the walk is dropped; it is a length mark,
        // not a sound of its own.
        for token in out.iter_mut() {
            token.text = token.text.replace('っ', "");
        }
        out
    }

    /// The whole text, phonemized. `JAG2P.__call__`.
    pub fn phonemize_text(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }
        let normalized = normalize::normalize_text(text);
        let words = self.regroup(self.analyze(&normalized)?);
        let spaced = self.space(&words);

        let joined: String = spaced
            .iter()
            .map(|t| format!("{}{}", t.text, if t.space { " " } else { "" }))
            .collect();

        // Collapse whitespace runs, then swap parentheses for angles the way
        // the eSpeak path does.
        let mut collapsed = String::with_capacity(joined.len());
        let mut last_was_space = false;
        for c in joined.trim().chars() {
            if c.is_whitespace() {
                if !last_was_space {
                    collapsed.push(' ');
                }
                last_was_space = true;
            } else {
                collapsed.push(c);
                last_was_space = false;
            }
        }
        let collapsed = collapsed.replace('(', "«").replace(')', "»");
        Ok(strip_glottal_stop_spacing(&collapsed))
    }

    /// Phonemes as `Token`s. See `super::align_tokens`.
    pub fn phonemize(&self, text: &str) -> Result<Vec<Token>> {
        Ok(super::align_tokens(text, &self.phonemize_text(text)?))
    }
}

/// One kana → IPA, with its neighbours in view. `JAG2P._get_single_mapping`.
fn map_kana(previous: Option<char>, kana: char, next: Option<char>) -> String {
    // Iteration marks repeat what came before.
    if hepburn::is_odori(kana) {
        if kana == 'ゝ' || kana == 'ヽ' {
            return previous.map(|p| p.to_string()).unwrap_or_default();
        }
        if kana == 'ゞ' || kana == 'ヾ' {
            return previous
                .and_then(hepburn::add_dakuten)
                .and_then(hepburn::single)
                .map(str::to_string)
                .unwrap_or_default();
        }
        // 々 repeats a kanji and 〃 a symbol; neither span can be recovered
        // from the kana alone.
        return String::new();
    }

    // This kana already combined with the one before it.
    if let Some(p) = previous {
        if hepburn::digraph(p, kana).is_some() {
            return String::new();
        }
    }
    // ...or it is about to combine with the one after.
    if let Some(n) = next {
        if let Some(mapped) = hepburn::digraph(kana, n) {
            return mapped.to_string();
        }
        // A small kana that forms no listed digraph still palatalizes: the
        // preceding kana loses its vowel and takes the small kana's.
        if hepburn::is_sutegana(n) {
            if kana == 'っ' {
                return String::new();
            }
            let base = hepburn::single(kana).unwrap_or("");
            let tail = hepburn::single(n).unwrap_or("");
            let mut trimmed: Vec<char> = base.chars().collect();
            trimmed.pop();
            return format!("{}{tail}", trimmed.into_iter().collect::<String>());
        }
    }
    if hepburn::is_sutegana(kana) {
        return String::new();
    }
    // The prolonged sound mark lengthens the vowel before it.
    if kana == 'ー' {
        return "ː".to_string();
    }
    // The small tsu is a glottal stop: it doubles the following consonant.
    if kana == 'っ' {
        return "ʔ".to_string();
    }
    // ん assimilates to whatever follows it.
    if kana == 'ん' {
        let following = next.and_then(hepburn::single).unwrap_or("");
        if let Some(first) = following.chars().next() {
            if "mpb".contains(first) {
                return "m".to_string();
            }
            if first == 'k' || first == 'ɡ' {
                return "ŋ".to_string();
            }
            if following.starts_with('ɲ')
                || following.starts_with('ʨ')
                || following.starts_with('ʥ')
            {
                return "ɲ".to_string();
            }
            if "ntdɾz".contains(first) {
                return "n".to_string();
            }
        }
        return "ɴ".to_string();
    }
    hepburn::single(kana).unwrap_or("").to_string()
}

/// Close up the space around a glottal stop.
///
/// `re.sub(r'(?<![!",.:;?»—…”]) (?=ʔ)|(?<=ʔ) (?!["«“])', '', ps)` — a space
/// before a glottal stop is removed unless punctuation precedes it, and a space
/// after one is removed unless a quote follows.
fn strip_glottal_stop_spacing(text: &str) -> String {
    const PUNCTUATION_BEFORE: &str = "!\",.:;?»—…\u{201D}";
    const QUOTE_AFTER: &str = "\"«\u{201C}";
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    for (i, c) in chars.iter().enumerate() {
        if *c == ' ' {
            let before = if i > 0 { Some(chars[i - 1]) } else { None };
            let after = chars.get(i + 1).copied();
            let drop_before_stop = after == Some('ʔ')
                && !before.is_some_and(|b| PUNCTUATION_BEFORE.contains(b));
            let drop_after_stop =
                before == Some('ʔ') && !after.is_some_and(|a| QUOTE_AFTER.contains(a));
            if drop_before_stop || drop_after_stop {
                continue;
            }
        }
        out.push(*c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_kana_map_straight_through() {
        assert_eq!(map_kana(None, 'か', None), "ka");
        assert_eq!(map_kana(None, 'せ', None), "se");
    }

    /// A two-kana sound is emitted on the first kana and suppressed on the
    /// second, so the pair contributes one sound rather than two.
    #[test]
    fn a_digraph_is_emitted_once_across_its_two_kana() {
        assert_eq!(map_kana(None, 'き', Some('ゃ')), "kʲa");
        assert_eq!(map_kana(Some('き'), 'ゃ', None), "");
    }

    #[test]
    fn the_prolonged_sound_mark_lengthens_rather_than_speaks() {
        assert_eq!(map_kana(Some('と'), 'ー', None), "ː");
    }

    /// ん takes its place of articulation from what follows: m before m/p/b,
    /// ŋ before k/g, n before n/t/d/r/z, ɴ otherwise.
    #[test]
    fn the_moraic_nasal_assimilates_to_what_follows_it() {
        assert_eq!(map_kana(None, 'ん', Some('ま')), "m");
        assert_eq!(map_kana(None, 'ん', Some('ぱ')), "m");
        assert_eq!(map_kana(None, 'ん', Some('か')), "ŋ");
        assert_eq!(map_kana(None, 'ん', Some('が')), "ŋ");
        assert_eq!(map_kana(None, 'ん', Some('に')), "ɲ");
        assert_eq!(map_kana(None, 'ん', Some('た')), "n");
        assert_eq!(map_kana(None, 'ん', Some('さ')), "ɴ", "no rule matches");
        assert_eq!(map_kana(None, 'ん', None), "ɴ", "at the end of a word");
    }

    #[test]
    fn the_small_tsu_is_a_glottal_stop() {
        assert_eq!(map_kana(None, 'っ', Some('た')), "ʔ");
    }

    #[test]
    fn an_iteration_mark_repeats_the_previous_kana() {
        assert_eq!(map_kana(Some('か'), 'ゝ', None), "か");
        // With voicing: か + ゞ is が's sound.
        assert_eq!(map_kana(Some('か'), 'ゞ', None), "ɡa");
        assert_eq!(map_kana(None, 'ゝ', None), "", "nothing to repeat");
    }

    #[test]
    fn a_bare_small_kana_contributes_nothing() {
        assert_eq!(map_kana(None, 'ゃ', None), "");
    }

    #[test]
    fn glottal_stop_spacing_closes_up() {
        assert_eq!(strip_glottal_stop_spacing("iʔ ta"), "iʔta");
        assert_eq!(strip_glottal_stop_spacing("a ʔta"), "aʔta");
        // ...but not across punctuation, or into a quote.
        assert_eq!(strip_glottal_stop_spacing("a. ʔta"), "a. ʔta");
        assert_eq!(strip_glottal_stop_spacing("aʔ \u{201C}b"), "aʔ \u{201C}b");
    }
}
