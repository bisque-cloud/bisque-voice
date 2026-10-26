//! Chinese characters → pinyin, ported from `pypinyin`.
//!
//! **Pinyin** is the standard way of writing Mandarin pronunciation in Latin
//! letters. `银行` ("bank") is `yín háng`. Misaki goes characters → pinyin →
//! IPA, so this is the first half.
//!
//! The reason this is a port rather than a crate: the Rust `pinyin` crate maps
//! one character at a time, and **that cannot be correct**. 行 is `háng` in 银行
//! (bank) but `xíng` in 行走 (to walk) — a *polyphone*, whose reading is decided
//! by the word it sits in. pypinyin resolves those with a 47,000-entry phrase
//! dictionary; no Rust crate ships one. So the algorithm is ported and
//! pypinyin's own two dictionaries are carried in the `g2p-zh` pack.
//!
//! Only the one call Misaki makes is implemented:
//! `lazy_pinyin(word, style=TONE3, neutral_tone_with_five=True)` — pinyin with
//! the tone as a trailing digit 1-5, where 5 is the neutral tone. Everything
//! pypinyin does for other styles is deliberately absent.

use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

/// Tone-marked vowels → the plain letter plus a tone digit.
///
/// `pypinyin.phonetic_symbol.phonetic_symbol`, verbatim. Note `ü` maps to bare
/// `v` with no digit: pypinyin carries ü as `v` internally and restores it at
/// the end, which is why `to_finals` starts by turning `v` back into `ü`.
const PHONETIC_SYMBOLS: &[(&str, &str)] = &[
    ("ā", "a1"), ("á", "a2"), ("ǎ", "a3"), ("à", "a4"),
    ("ē", "e1"), ("é", "e2"), ("ě", "e3"), ("è", "e4"),
    ("ō", "o1"), ("ó", "o2"), ("ǒ", "o3"), ("ò", "o4"),
    ("ī", "i1"), ("í", "i2"), ("ǐ", "i3"), ("ì", "i4"),
    ("ū", "u1"), ("ú", "u2"), ("ǔ", "u3"), ("ù", "u4"),
    ("ü", "v"), ("ǖ", "v1"), ("ǘ", "v2"), ("ǚ", "v3"), ("ǜ", "v4"),
    ("ń", "n2"), ("ň", "n3"), ("ǹ", "n4"),
    // These three are two code points each: a base letter plus a combining
    // mark. They must be replaced before the single-character symbols, or the
    // base letter would be consumed on its own.
    ("m̄", "m1"), ("ḿ", "m2"), ("m̀", "m4"),
    ("ê̄", "ê1"), ("ế", "ê2"), ("ê̌", "ê3"), ("ề", "ê4"),
];

/// The initial consonants, longest first so `zh` wins over `z`.
///
/// `pypinyin.style._constants._INITIALS`, which deliberately excludes `y` and
/// `w`: in the official scheme those are not initials but spellings of a
/// vowel-initial syllable, and `convert_zero_consonant` undoes them.
const INITIALS: &[&str] = &[
    "zh", "ch", "sh", // two letters first
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s",
];

/// Every legal final. Membership decides whether a zero-consonant expansion
/// produced something real, so this is a check, not just a list.
const FINALS: &[&str] = &[
    "i", "u", "ü", "a", "ia", "ua", "o", "uo", "e", "ie", "üe", "ai", "uai", "ei", "uei", "ao",
    "iao", "ou", "iou", "an", "ian", "uan", "üan", "en", "in", "uen", "ün", "ang", "iang", "uang",
    "eng", "ing", "ueng", "ong", "iong", "er", "ê",
];

/// pypinyin's two dictionaries, as shipped in the `g2p-zh` pack.
///
/// `characters` is keyed by Unicode code point (pypinyin stores the key as a
/// decimal string) and holds comma-separated readings, most common first.
/// `phrases` maps a whole word to one reading per character, and is what makes
/// polyphones come out right.
pub struct Dictionaries {
    characters: HashMap<u32, Vec<String>>,
    phrases: HashMap<String, Vec<String>>,
}

#[derive(Deserialize)]
struct RawPhrase(Vec<Vec<String>>);

impl Dictionaries {
    /// Load from the pack directory holding `pinyin_dict.json` and
    /// `phrases_dict.json`.
    pub fn load(dir: &std::path::Path) -> Result<Self> {
        let chars_raw: HashMap<String, String> =
            serde_json::from_slice(&std::fs::read(dir.join("pinyin_dict.json")).with_context(
                || format!("reading {}", dir.join("pinyin_dict.json").display()),
            )?)
            .context("parsing pinyin_dict.json")?;
        let mut characters = HashMap::with_capacity(chars_raw.len());
        for (code, readings) in chars_raw {
            let code: u32 = code
                .parse()
                .map_err(|_| anyhow!("pinyin_dict.json has a non-numeric key {code:?}"))?;
            characters.insert(
                code,
                readings.split(',').map(|s| s.to_string()).collect::<Vec<_>>(),
            );
        }

        let phrases_raw: HashMap<String, RawPhrase> =
            serde_json::from_slice(&std::fs::read(dir.join("phrases_dict.json")).with_context(
                || format!("reading {}", dir.join("phrases_dict.json").display()),
            )?)
            .context("parsing phrases_dict.json")?;
        // Only the first reading per character is ever used: Misaki calls
        // lazy_pinyin, which is the heteronym=False path.
        let phrases = phrases_raw
            .into_iter()
            .map(|(word, RawPhrase(per_char))| {
                let first: Vec<String> = per_char
                    .into_iter()
                    .map(|mut r| if r.is_empty() { String::new() } else { r.remove(0) })
                    .collect();
                (word, first)
            })
            .collect();

        Ok(Dictionaries {
            characters,
            phrases,
        })
    }

    /// `lazy_pinyin(word, style=TONE3, neutral_tone_with_five=True)`.
    ///
    /// Returns one entry per character that has a reading; characters with no
    /// pinyin at all contribute nothing, matching pypinyin's
    /// `_remove_dup_and_empty`.
    pub fn lazy_pinyin_tone3(&self, word: &str) -> Vec<String> {
        // The phrase dictionary is consulted for the whole word first. This is
        // the entire reason for carrying the data: per-character lookup would
        // read 银行 as "yín xíng".
        let toned: Vec<String> = match self.phrases.get(word) {
            Some(readings) => readings.clone(),
            None => word
                .chars()
                .filter_map(|c| {
                    self.characters
                        .get(&(c as u32))
                        .and_then(|r| r.first())
                        .cloned()
                })
                .collect(),
        };
        toned
            .iter()
            .map(|p| to_tone3_with_neutral_five(p))
            .filter(|p| !p.is_empty())
            .collect()
    }
}

/// Replace tone-marked characters with letter+digit.
///
/// `pypinyin.style._utils.replace_symbol_to_number`. The multi-code-point
/// symbols are handled first for the reason noted on the table.
fn replace_symbol_to_number(pinyin: &str) -> String {
    let mut out = pinyin.to_string();
    for (symbol, to) in PHONETIC_SYMBOLS.iter().filter(|(k, _)| k.chars().count() > 1) {
        if out.contains(symbol) {
            out = out.replace(symbol, to);
        }
    }
    out.chars()
        .map(|c| {
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            PHONETIC_SYMBOLS
                .iter()
                .find(|(k, _)| *k == s)
                .map(|(_, v)| (*v).to_string())
                .unwrap_or_else(|| s.to_string())
        })
        .collect()
}

/// Strip the tone digits, leaving plain pinyin.
fn replace_symbol_to_no_symbol(pinyin: &str) -> String {
    replace_symbol_to_number(pinyin)
        .chars()
        .filter(|c| !c.is_ascii_digit())
        .collect()
}

/// Tone-marked pinyin → TONE3: letters then a single trailing digit, with 5 for
/// the neutral tone.
///
/// `zhōng` → `zhong1`, `shang` → `shang5`. pypinyin reaches this through a
/// chain of style converters; the chain collapses to "move the digit to the
/// end, defaulting to 5" because the input is always a dictionary reading.
fn to_tone3_with_neutral_five(toned: &str) -> String {
    let numbered = replace_symbol_to_number(toned);
    let digit = numbered.chars().find(|c| c.is_ascii_digit());
    let letters: String = numbered.chars().filter(|c| !c.is_ascii_digit()).collect();
    if letters.is_empty() {
        return String::new();
    }
    match digit {
        Some(d) => format!("{letters}{d}"),
        // No tone mark means the neutral tone, which
        // `neutral_tone_with_five=True` writes as 5 rather than leaving bare.
        None => format!("{letters}5"),
    }
}

/// Split a TONE3 syllable into its plain form and tone number.
///
/// Misaki's `get_tone` and `to_normal` both operate on `lazy_pinyin`'s TONE3
/// output, which is always letters followed by one digit — so the general style
/// machinery upstream runs is not needed, and its absence is checked rather
/// than assumed: a syllable that does not have that shape is refused.
pub fn split_tone(tone3: &str) -> Option<(String, usize)> {
    let mut chars: Vec<char> = tone3.chars().collect();
    let tone = chars.pop()?.to_digit(10)? as usize;
    if !(1..=5).contains(&tone) {
        return None;
    }
    let normal: String = chars.into_iter().collect();
    if normal.is_empty() || normal.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((normal, tone))
}

/// `get_initials(pinyin, strict=True)`.
pub fn initials(normal: &str) -> &'static str {
    INITIALS
        .iter()
        .find(|i| normal.starts_with(**i))
        .copied()
        .unwrap_or("")
}

// --- pypinyin/standard.py: recovering the underlying final ------------------
//
// Pinyin spelling hides the real final in four places. Each of these undoes one
// of them, so the IPA tables can be keyed on the underlying form.

/// `y`/`w` are not consonants; they are how a vowel-initial syllable is spelled.
/// `yi` is really `i`, `wu` is really `u`, `yu` is really `ü`.
fn convert_zero_consonant(pinyin: &str) -> String {
    let raw = pinyin.to_string();
    let mut out = raw.clone();
    if let Some(rest) = raw.strip_prefix('y') {
        let first = rest.chars().next();
        out = match first {
            Some('u') => format!("ü{}", rest.chars().skip(1).collect::<String>()),
            Some('i') => rest.to_string(),
            _ => format!("i{rest}"),
        };
    }
    if let Some(rest) = raw.strip_prefix('w') {
        out = match rest.chars().next() {
            Some('u') => rest.to_string(),
            _ => format!("u{rest}"),
        };
    }
    if FINALS.contains(&out.as_str()) {
        out
    } else {
        raw
    }
}

/// After j, q or x, `ü` is written `u` — there is no `u` final in those
/// positions, so the `u` must be a disguised `ü`.
fn convert_uv(pinyin: &str) -> String {
    for initial in ["j", "q", "x"] {
        if let Some(rest) = pinyin.strip_prefix(initial) {
            if let Some(tail) = rest.strip_prefix('u') {
                return format!("{initial}ü{tail}");
            }
        }
    }
    pinyin.to_string()
}

/// `iou`, `uei` and `uen` are contracted to `iu`, `ui`, `un` after an initial:
/// `niu` is `n` + `iou`, `gui` is `g` + `uei`, `lun` is `l` + `uen`.
fn convert_contractions(pinyin: &str) -> String {
    for (short, long) in [("iu", "iou"), ("ui", "uei"), ("un", "uen")] {
        if let Some(head) = pinyin.strip_suffix(short) {
            // Only after a real initial — a bare "un" is not a contraction.
            if !head.is_empty() && head.chars().all(|c| c.is_ascii_lowercase()) {
                return format!("{head}{long}");
            }
        }
    }
    pinyin.to_string()
}

fn convert_finals(pinyin: &str) -> String {
    let p = convert_zero_consonant(pinyin);
    let p = convert_uv(&p);
    convert_contractions(&p)
}

/// `get_finals(pinyin, strict=True)`, with `v` restored to `ü` first.
pub fn finals(normal: &str) -> String {
    let pinyin = replace_symbol_to_no_symbol(normal).replace('v', "ü");
    let converted = convert_finals(&pinyin);
    let initial = initials(&converted);
    let rest = &converted[initial.len()..];
    if FINALS.contains(&rest) {
        return rest.to_string();
    }
    // `y`/`w` can make the strict split misfire (yo). Retry treating them as
    // initials, as pypinyin does.
    let loose = ["zh", "ch", "sh", "y", "w"]
        .iter()
        .chain(INITIALS.iter())
        .find(|i| converted.starts_with(**i))
        .copied()
        .unwrap_or("");
    let rest = &converted[loose.len()..];
    if FINALS.contains(&rest) {
        rest.to_string()
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_marks_become_trailing_digits() {
        assert_eq!(to_tone3_with_neutral_five("zhōng"), "zhong1");
        assert_eq!(to_tone3_with_neutral_five("háng"), "hang2");
        assert_eq!(to_tone3_with_neutral_five("nǐ"), "ni3");
        assert_eq!(to_tone3_with_neutral_five("hào"), "hao4");
    }

    /// The point of `neutral_tone_with_five`: an unmarked syllable is the
    /// neutral tone, and it must say so rather than come back bare, or the tone
    /// lookup downstream has nothing to read.
    #[test]
    fn an_unmarked_syllable_is_the_neutral_tone_five() {
        assert_eq!(to_tone3_with_neutral_five("shang"), "shang5");
        assert_eq!(to_tone3_with_neutral_five("le"), "le5");
    }

    #[test]
    fn u_umlaut_is_carried_as_v_and_restored() {
        // pypinyin represents ü as v internally.
        assert_eq!(to_tone3_with_neutral_five("lüè"), "lve4");
        assert_eq!(finals("lve"), "üe");
    }

    #[test]
    fn split_tone_refuses_anything_that_is_not_letters_then_one_digit() {
        assert_eq!(split_tone("zhong1"), Some(("zhong".into(), 1)));
        assert_eq!(split_tone("shang5"), Some(("shang".into(), 5)));
        assert_eq!(split_tone("zhong"), None, "no tone digit");
        assert_eq!(split_tone("zh1ong2"), None, "an interior digit");
        assert_eq!(split_tone("6"), None, "not a tone");
        assert_eq!(split_tone(""), None);
    }

    #[test]
    fn two_letter_initials_win_over_their_first_letter() {
        assert_eq!(initials("zhong"), "zh");
        assert_eq!(initials("chi"), "ch");
        assert_eq!(initials("shi"), "sh");
        assert_eq!(initials("zi"), "z");
        assert_eq!(initials("si"), "s");
    }

    /// y and w are spellings, not consonants. Getting this wrong makes every
    /// vowel-initial syllable unmappable.
    #[test]
    fn y_and_w_are_not_initials() {
        assert_eq!(initials("yi"), "", "y is not an initial");
        assert_eq!(initials("wu"), "", "w is not an initial");
        assert_eq!(finals("yi"), "i");
        assert_eq!(finals("wu"), "u");
        assert_eq!(finals("yu"), "ü");
        assert_eq!(finals("ye"), "ie");
        assert_eq!(finals("wo"), "uo");
    }

    /// The contracted finals. `niu` is n + iou, not n + iu.
    #[test]
    fn contracted_finals_are_expanded_to_their_real_form() {
        assert_eq!(finals("niu"), "iou");
        assert_eq!(finals("gui"), "uei");
        assert_eq!(finals("lun"), "uen");
    }

    /// After j, q, x there is no u final, so the u must be ü.
    #[test]
    fn u_after_j_q_x_is_really_u_umlaut() {
        assert_eq!(finals("ju"), "ü");
        assert_eq!(finals("qu"), "ü");
        assert_eq!(finals("xu"), "ü");
        // ... but after l and n, u is genuinely u.
        assert_eq!(finals("lu"), "u");
    }

    #[test]
    fn ordinary_syllables_split_the_obvious_way() {
        assert_eq!((initials("hang"), finals("hang")), ("h", "ang".to_string()));
        assert_eq!((initials("ni"), finals("ni")), ("n", "i".to_string()));
        assert_eq!((initials("zhong"), finals("zhong")), ("zh", "ong".to_string()));
    }

    #[test]
    fn multi_code_point_tone_marks_are_replaced_whole() {
        // m̄ is two code points; replacing the base letter alone would leave a
        // stray combining macron behind.
        assert_eq!(replace_symbol_to_number("m̄"), "m1");
        assert!(!replace_symbol_to_number("m̄").contains('\u{304}'));
    }
}
