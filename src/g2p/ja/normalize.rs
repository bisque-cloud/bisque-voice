//! Japanese text normalization, ported from `JAG2P._normalize_text`.
//!
//! Japanese text carries variations that mean nothing for pronunciation, and
//! the morphological analyzer only recognizes one of each. Ordinary ASCII has a
//! second, "full-width" form used in CJK typesetting (`Ａ` beside `A`), katakana
//! has a legacy "half-width" form (`ｱ` beside `ア`), and numerals have to become
//! words before anything can read them aloud.
//!
//! The order is upstream's and matters: the wave dash is rewritten before NFKC
//! folds it into something else, and numbers are converted last, after every
//! digit has been folded to its ASCII form.

use unicode_normalization::UnicodeNormalization;

use super::{hepburn, num2kana};

/// Katakana → hiragana. `jaconv.kata2hira`.
///
/// The two scripts are laid out in the same order 96 code points apart, so this
/// is arithmetic rather than a table. The prolonged sound mark `ー` and the
/// small Ainu kana sit outside that range and are left alone, which is correct:
/// `ー` has its own meaning in the Hepburn table.
pub fn kata2hira(text: &str) -> String {
    text.chars()
        .map(|c| {
            if ('\u{30A1}'..='\u{30F6}').contains(&c) {
                char::from_u32(c as u32 - 0x60).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// MeCab's character categories, as far as this port needs them.
///
/// `misaki/ja.py` reads fugashi's `char_type`, which lindera does not expose.
/// It does not need to: Misaki computes
/// `6 if char_type == 7 or not is_unk else char_type`, so **every token the
/// dictionary knows becomes 6** and the real category is consulted only for
/// unknown tokens. For those, the Unicode block is enough.
///
/// The numbers are MeCab's, from unidic's `char.def`, so the comparisons in the
/// port read the same as the Python they came from.
pub mod char_type {
    pub const KANJI: u8 = 2;
    pub const SYMBOL: u8 = 3;
    pub const NUMERIC: u8 = 4;
    pub const ALPHA: u8 = 5;
    /// Hiragana, and the category every known token is folded to.
    pub const HIRAGANA: u8 = 6;
    pub const KATAKANA: u8 = 7;
}

/// The MeCab category of a token, from its first character.
pub fn category(text: &str) -> u8 {
    let Some(c) = text.chars().next() else {
        return char_type::SYMBOL;
    };
    match c {
        '\u{3041}'..='\u{309F}' => char_type::HIRAGANA,
        '\u{30A0}'..='\u{30FF}' | '\u{31F0}'..='\u{31FF}' => char_type::KATAKANA,
        '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' => char_type::KANJI,
        c if c.is_ascii_digit() => char_type::NUMERIC,
        c if c.is_ascii_alphabetic() => char_type::ALPHA,
        _ => char_type::SYMBOL,
    }
}

/// Full-width ASCII → half-width. `mojimoji.zen_to_han(kana=False)`.
///
/// NFKC has already done most of this; it is kept because upstream keeps it and
/// because the ideographic space is not something NFKC folds to a plain space.
fn zen_to_han_ascii(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{FF01}'..='\u{FF5E}' => {
                char::from_u32(c as u32 - 0xFEE0).unwrap_or(c)
            }
            '\u{3000}' => ' ',
            _ => c,
        })
        .collect()
}

/// Half-width katakana → full-width. `mojimoji.han_to_zen(digit=False,
/// ascii=False)`.
///
/// Half-width katakana writes a voiced sound as two characters — `ｶ` plus a
/// standalone `ﾞ` — so the combining marks have to be folded into the kana
/// before them, not just remapped.
fn han_to_zen_kana(text: &str) -> String {
    const HALF: &str = "｡｢｣､･ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ";
    const FULL: &str = "。「」、・ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン";
    let half: Vec<char> = HALF.chars().collect();
    let full: Vec<char> = FULL.chars().collect();
    debug_assert_eq!(half.len(), full.len());

    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        // A standalone voicing mark folds into the kana already emitted.
        if c == 'ﾞ' || c == 'ﾟ' {
            let previous = out.pop();
            match previous {
                Some(prev) => match hepburn::voice_katakana(prev, c == 'ﾞ') {
                    Some(voiced) => out.push(voiced),
                    // The mark does not apply to this kana. Keep both rather
                    // than silently drop one.
                    None => {
                        out.push(prev);
                        out.push(if c == 'ﾞ' { '゛' } else { '゜' });
                    }
                },
                None => out.push(if c == 'ﾞ' { '゛' } else { '゜' }),
            }
            continue;
        }
        match half.iter().position(|h| *h == c) {
            Some(i) => out.push(full[i]),
            None => out.push(c),
        }
    }
    out
}

/// The whole of `JAG2P._normalize_text`.
pub fn normalize_text(text: &str) -> String {
    // A wave dash before a digit is a range, and is read as the word "from".
    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if (*c == '\u{301C}' || *c == '\u{FF5E}')
            && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit())
        {
            out.push_str("から");
        } else {
            out.push(*c);
        }
    }

    let out = hepburn::fold_phonetic_extensions(&out);
    let out: String = out.nfkc().collect();
    let out = zen_to_han_ascii(&out);
    let out = han_to_zen_kana(&out);

    // Digit runs become words. A leading space keeps the number a separate
    // token from whatever preceded it.
    let mut result = String::with_capacity(out.len());
    for run in split_digit_runs(&out) {
        if run.chars().all(|c| c.is_ascii_digit()) && !run.is_empty() {
            match num2kana::convert(run) {
                Some(kana) => {
                    result.push(' ');
                    result.push_str(&kana);
                }
                // Too long for the converter. Leaving the digits in place is
                // better than dropping the number outright; the analyzer will
                // read them as an unknown numeric token.
                None => result.push_str(run),
            }
        } else {
            result.push_str(run);
        }
    }
    result
}

/// `re.findall(r'\d+|\D+', text)`: alternating runs of digits and non-digits.
fn split_digit_runs(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut current: Option<bool> = None;
    for (i, c) in text.char_indices() {
        let digit = c.is_ascii_digit();
        match current {
            Some(prev) if prev == digit => {}
            Some(_) => {
                out.push(&text[start..i]);
                start = i;
            }
            None => {}
        }
        current = Some(digit);
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
    fn katakana_becomes_hiragana_by_arithmetic() {
        assert_eq!(kata2hira("コンニチワ"), "こんにちわ");
        assert_eq!(kata2hira("セカイ"), "せかい");
        // Already hiragana, unchanged.
        assert_eq!(kata2hira("せかい"), "せかい");
    }

    /// The prolonged sound mark must survive: the Hepburn table maps it to a
    /// length marker, and folding it would lose vowel length.
    #[test]
    fn the_prolonged_sound_mark_is_not_converted() {
        assert_eq!(kata2hira("コンピューター"), "こんぴゅーたー");
        assert!(kata2hira("トーキョー").contains('ー'));
    }

    #[test]
    fn full_width_ascii_folds_to_ascii() {
        assert_eq!(zen_to_han_ascii("ＡＢＣ１２３"), "ABC123");
        assert_eq!(zen_to_han_ascii("\u{3000}"), " ", "ideographic space");
        assert_eq!(zen_to_han_ascii("テスト"), "テスト", "kana untouched");
    }

    #[test]
    fn half_width_katakana_folds_to_full_width() {
        assert_eq!(han_to_zen_kana("ｱｲｳ"), "アイウ");
        assert_eq!(han_to_zen_kana("ｺﾝﾆﾁﾜ"), "コンニチワ");
        assert_eq!(han_to_zen_kana("アイウ"), "アイウ", "already full width");
    }

    /// Half-width katakana writes が as two characters. Folding them
    /// separately would produce a stray mark and a wrong reading.
    #[test]
    fn a_standalone_voicing_mark_folds_into_the_kana_before_it() {
        assert_eq!(han_to_zen_kana("ｶﾞ"), "ガ");
        assert_eq!(han_to_zen_kana("ﾊﾟ"), "パ");
        assert_eq!(han_to_zen_kana("ﾃﾞｽ"), "デス");
    }

    #[test]
    fn digit_runs_split_from_the_text_around_them() {
        assert_eq!(split_digit_runs("abc123def"), vec!["abc", "123", "def"]);
        assert_eq!(split_digit_runs("123"), vec!["123"]);
        assert_eq!(split_digit_runs("abc"), vec!["abc"]);
        assert!(split_digit_runs("").is_empty());
        for text in ["2026年", "3.5キロ", "テスト"] {
            assert_eq!(split_digit_runs(text).concat(), text, "lossless");
        }
    }

    #[test]
    fn numbers_become_words() {
        assert_eq!(normalize_text("2026年"), " にせんにじゅうろく年");
        assert_eq!(normalize_text("10個"), " じゅう個");
    }

    /// A decimal never reaches the converter as one token — the digit-run
    /// split cuts at the point — so "3.5" is read "three point five" by way of
    /// two separate conversions.
    #[test]
    fn a_decimal_is_split_at_the_point_before_conversion() {
        let out = normalize_text("3.5キロ");
        assert!(out.contains("さん"), "{out:?}");
        assert!(out.contains('.'), "the point itself survives: {out:?}");
        assert!(out.contains('ご'), "{out:?}");
    }

    #[test]
    fn a_wave_dash_before_a_digit_is_read_as_a_range() {
        assert!(normalize_text("1〜5").starts_with(' '));
        assert!(normalize_text("1〜5").contains("から"));
        // Not before a non-digit.
        assert!(!normalize_text("あ〜い").contains("から"));
    }

    #[test]
    fn categories_come_from_the_unicode_block() {
        assert_eq!(category("あ"), char_type::HIRAGANA);
        assert_eq!(category("ア"), char_type::KATAKANA);
        assert_eq!(category("漢"), char_type::KANJI);
        assert_eq!(category("1"), char_type::NUMERIC);
        assert_eq!(category("a"), char_type::ALPHA);
        assert_eq!(category("。"), char_type::SYMBOL);
        assert_eq!(category(""), char_type::SYMBOL);
    }

    #[test]
    fn text_without_anything_to_normalize_is_unchanged() {
        assert_eq!(normalize_text("こんにちは"), "こんにちは");
        assert_eq!(normalize_text(""), "");
    }
}
