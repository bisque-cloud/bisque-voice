//! Kana → IPA, ported from `misaki/ja.py`'s `HEPBURN` table.
//!
//! **Hepburn** is the standard scheme for writing Japanese in Latin letters.
//! Misaki uses a modified version that emits IPA directly rather than romaji,
//! so `か` is `ka` but `し` is `ɕi` (not "shi") and `つ` is `ʦɨ` (not "tsu").
//!
//! Generated from the Python source rather than transcribed by hand — 189
//! entries of near-identical kana is exactly the kind of table where a typo
//! survives review and produces one wrong syllable in one word.
//!
//! Two-kana entries come first and must be tried first: `きゃ` is the single
//! sound `kʲa`, not `kʲi` followed by `a`.
//!
//! The tail of the table is punctuation — Japanese full-width marks mapped to
//! their ASCII equivalents, since that is what Kokoro's vocabulary holds.

/// Two-kana sequences that are one sound. Tried before the single-kana table.
const HEPBURN_DIGRAPHS: &[(&str, &str)] = &[
    ("いぇ", "je"), // いぇ
    ("うぃ", "βi"), // うぃ
    ("うぇ", "βe"), // うぇ
    ("うぉ", "βo"), // うぉ
    ("きぇ", "kʲe"), // きぇ
    ("きゃ", "kʲa"), // きゃ
    ("きゅ", "kʲɨ"), // きゅ
    ("きょ", "kʲo"), // きょ
    ("ぎゃ", "ɡʲa"), // ぎゃ
    ("ぎゅ", "ɡʲɨ"), // ぎゅ
    ("ぎょ", "ɡʲo"), // ぎょ
    ("くぁ", "kᵝa"), // くぁ
    ("くぃ", "kᵝi"), // くぃ
    ("くぇ", "kᵝe"), // くぇ
    ("くぉ", "kᵝo"), // くぉ
    ("ぐぁ", "ɡᵝa"), // ぐぁ
    ("ぐぃ", "ɡᵝi"), // ぐぃ
    ("ぐぇ", "ɡᵝe"), // ぐぇ
    ("ぐぉ", "ɡᵝo"), // ぐぉ
    ("しぇ", "ɕe"), // しぇ
    ("しゃ", "ɕa"), // しゃ
    ("しゅ", "ɕɨ"), // しゅ
    ("しょ", "ɕo"), // しょ
    ("じぇ", "ʥe"), // じぇ
    ("じゃ", "ʥa"), // じゃ
    ("じゅ", "ʥɨ"), // じゅ
    ("じょ", "ʥo"), // じょ
    ("ちぇ", "ʨe"), // ちぇ
    ("ちゃ", "ʨa"), // ちゃ
    ("ちゅ", "ʨɨ"), // ちゅ
    ("ちょ", "ʨo"), // ちょ
    ("ぢゃ", "ʥa"), // ぢゃ
    ("ぢゅ", "ʥɨ"), // ぢゅ
    ("ぢょ", "ʥo"), // ぢょ
    ("つぁ", "ʦa"), // つぁ
    ("つぃ", "ʦʲi"), // つぃ
    ("つぇ", "ʦe"), // つぇ
    ("つぉ", "ʦo"), // つぉ
    ("てぃ", "tʲi"), // てぃ
    ("てゅ", "tʲɨ"), // てゅ
    ("でぃ", "dʲi"), // でぃ
    ("でゅ", "dʲɨ"), // でゅ
    ("とぅ", "tɯ"), // とぅ
    ("どぅ", "dɯ"), // どぅ
    ("にぇ", "ɲe"), // にぇ
    ("にゃ", "ɲa"), // にゃ
    ("にゅ", "ɲɨ"), // にゅ
    ("にょ", "ɲo"), // にょ
    ("ひぇ", "çe"), // ひぇ
    ("ひゃ", "ça"), // ひゃ
    ("ひゅ", "çɨ"), // ひゅ
    ("ひょ", "ço"), // ひょ
    ("びゃ", "bʲa"), // びゃ
    ("びゅ", "bʲɨ"), // びゅ
    ("びょ", "bʲo"), // びょ
    ("ぴゃ", "pʲa"), // ぴゃ
    ("ぴゅ", "pʲɨ"), // ぴゅ
    ("ぴょ", "pʲo"), // ぴょ
    ("ふぁ", "ɸa"), // ふぁ
    ("ふぃ", "ɸʲi"), // ふぃ
    ("ふぇ", "ɸe"), // ふぇ
    ("ふぉ", "ɸo"), // ふぉ
    ("ふゅ", "ɸʲɨ"), // ふゅ
    ("ふょ", "ɸʲo"), // ふょ
    ("みゃ", "mʲa"), // みゃ
    ("みゅ", "mʲɨ"), // みゅ
    ("みょ", "mʲo"), // みょ
    ("りゃ", "ɾʲa"), // りゃ
    ("りゅ", "ɾʲɨ"), // りゅ
    ("りょ", "ɾʲo"), // りょ
    ("ゔぁ", "va"), // ゔぁ
    ("ゔぃ", "vʲi"), // ゔぃ
    ("ゔぇ", "ve"), // ゔぇ
    ("ゔぉ", "vo"), // ゔぉ
    ("ゔゅ", "bʲɨ"), // ゔゅ
    ("ゔょ", "bʲo"), // ゔょ
];
/// Single kana, plus the punctuation Misaki rewrites.
const HEPBURN_SINGLES: &[(char, &str)] = &[
    ('ぁ', "a"),
    ('あ', "a"),
    ('ぃ', "i"),
    ('い', "i"),
    ('ぅ', "ɯ"),
    ('う', "ɯ"),
    ('ぇ', "e"),
    ('え', "e"),
    ('ぉ', "o"),
    ('お', "o"),
    ('か', "ka"),
    ('が', "ɡa"),
    ('き', "kʲi"),
    ('ぎ', "ɡʲi"),
    ('く', "kɯ"),
    ('ぐ', "ɡɯ"),
    ('け', "ke"),
    ('げ', "ɡe"),
    ('こ', "ko"),
    ('ご', "ɡo"),
    ('さ', "sa"),
    ('ざ', "ʣa"),
    ('し', "ɕi"),
    ('じ', "ʥi"),
    ('す', "sɨ"),
    ('ず', "zɨ"),
    ('せ', "se"),
    ('ぜ', "ʣe"),
    ('そ', "so"),
    ('ぞ', "ʣo"),
    ('た', "ta"),
    ('だ', "da"),
    ('ち', "ʨi"),
    ('ぢ', "ʥi"),
    ('つ', "ʦɨ"),
    ('づ', "zɨ"),
    ('て', "te"),
    ('で', "de"),
    ('と', "to"),
    ('ど', "do"),
    ('な', "na"),
    ('に', "ɲi"),
    ('ぬ', "nɯ"),
    ('ね', "ne"),
    ('の', "no"),
    ('は', "ha"),
    ('ば', "ba"),
    ('ぱ', "pa"),
    ('ひ', "çi"),
    ('び', "bʲi"),
    ('ぴ', "pʲi"),
    ('ふ', "ɸɯ"),
    ('ぶ', "bɯ"),
    ('ぷ', "pɯ"),
    ('へ', "he"),
    ('べ', "be"),
    ('ぺ', "pe"),
    ('ほ', "ho"),
    ('ぼ', "bo"),
    ('ぽ', "po"),
    ('ま', "ma"),
    ('み', "mʲi"),
    ('む', "mɯ"),
    ('め', "me"),
    ('も', "mo"),
    ('ゃ', "ja"),
    ('や', "ja"),
    ('ゅ', "jɯ"),
    ('ゆ', "jɯ"),
    ('ょ', "jo"),
    ('よ', "jo"),
    ('ら', "ɾa"),
    ('り', "ɾʲi"),
    ('る', "ɾɯ"),
    ('れ', "ɾe"),
    ('ろ', "ɾo"),
    ('ゎ', "βa"),
    ('わ', "βa"),
    ('ゐ', "i"),
    ('ゑ', "e"),
    ('を', "o"),
    ('ゔ', "vɯ"),
    ('ゕ', "ka"),
    ('ゖ', "ke"),
    ('ヷ', "va"),
    ('ヸ', "vʲi"),
    ('ヹ', "ve"),
    ('ヺ', "vo"),
    ('。', "."),
    ('、', ","),
    ('？', "?"),
    ('！', "!"),
    ('「', "“"),
    ('」', "”"),
    ('『', "“"),
    ('』', "”"),
    ('：', ":"),
    ('；', ";"),
    ('（', "("),
    ('）', ")"),
    ('《', "("),
    ('》', ")"),
    ('【', "["),
    ('】', "]"),
    ('・', " "),
    ('，', ","),
    ('～', "—"),
    ('〜', "—"),
    ('—', "—"),
    ('«', "“"),
    ('»', "”"),
    ('゚', ""),
    ('゙', ""),
];
/// Small katakana used in Ainu transcription, folded to their full-size form
/// before anything else runs.
const KATAKANA_PHONETIC_EXTENSIONS: &[(char, char)] = &[
    ('ㇰ', 'ク'),
    ('ㇱ', 'シ'),
    ('ㇲ', 'ス'),
    ('ㇳ', 'ト'),
    ('ㇴ', 'ヌ'),
    ('ㇵ', 'ハ'),
    ('ㇶ', 'ヒ'),
    ('ㇷ', 'フ'),
    ('ㇸ', 'ヘ'),
    ('ㇹ', 'ホ'),
    ('ㇺ', 'ム'),
    ('ㇻ', 'ラ'),
    ('ㇼ', 'リ'),
    ('ㇽ', 'ル'),
    ('ㇾ', 'レ'),
    ('ㇿ', 'ロ'),
];
/// Small kana. They never stand alone; they modify the kana before them.
const SUTEGANA: &str = "ぁぃぅぇぉゃゅょ";
/// Iteration marks, which repeat the previous character.
const ODORI: &str = "〃々ゝゞヽ";

/// Look up a two-kana sequence.
pub fn digraph(a: char, b: char) -> Option<&'static str> {
    let key: String = [a, b].iter().collect();
    HEPBURN_DIGRAPHS
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| *v)
}

/// Look up a single kana or punctuation mark.
pub fn single(c: char) -> Option<&'static str> {
    HEPBURN_SINGLES
        .iter()
        .find(|(k, _)| *k == c)
        .map(|(_, v)| *v)
}

/// Fold Ainu small katakana to their full-size form.
pub fn fold_phonetic_extensions(text: &str) -> String {
    text.chars()
        .map(|c| {
            KATAKANA_PHONETIC_EXTENSIONS
                .iter()
                .find(|(k, _)| *k == c)
                .map(|(_, v)| *v)
                .unwrap_or(c)
        })
        .collect()
}

pub fn is_sutegana(c: char) -> bool {
    SUTEGANA.contains(c)
}

pub fn is_odori(c: char) -> bool {
    ODORI.contains(c)
}

/// Add a dakuten (the voicing mark) to a kana: か → が.
///
/// Used only by the iteration mark `ゞ`, which means "repeat the previous kana,
/// voiced".
pub fn add_dakuten(c: char) -> Option<char> {
    const PLAIN: &str = "かきくけこさしすせそたちつてとはひふへほ";
    const VOICED: &str = "がぎぐげござじずぜぞだぢづでどばびぶべぼ";
    let index = PLAIN.chars().position(|p| p == c)?;
    VOICED.chars().nth(index)
}

/// Apply a voicing mark to a **katakana**: カ + ﾞ → ガ, ハ + ﾟ → パ.
///
/// Needed by half-width katakana, which writes a voiced sound as the plain kana
/// followed by a standalone mark. Returns `None` when the mark does not apply
/// to that kana, so the caller can keep both characters rather than lose one.
pub fn voice_katakana(kana: char, dakuten: bool) -> Option<char> {
    const PLAIN: &str = "カキクケコサシスセソタチツテトハヒフヘホウ";
    const VOICED: &str = "ガギグゲゴザジズゼゾダヂヅデドバビブベボヴ";
    const PLAIN_P: &str = "ハヒフヘホ";
    const VOICED_P: &str = "パピプペポ";
    if dakuten {
        let i = PLAIN.chars().position(|p| p == kana)?;
        VOICED.chars().nth(i)
    } else {
        let i = PLAIN_P.chars().position(|p| p == kana)?;
        VOICED_P.chars().nth(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table's own invariants, asserted the way the Python source asserts
    /// them, so a regenerated table that lost entries fails loudly.
    #[test]
    fn the_table_is_the_size_the_source_says_it_is() {
        assert_eq!(HEPBURN_DIGRAPHS.len(), 76);
        assert_eq!(HEPBURN_SINGLES.len(), 113);
        assert_eq!(KATAKANA_PHONETIC_EXTENSIONS.len(), 16);
    }

    /// Every kana of a two-kana entry must itself be in the single table.
    /// Python asserts this at import; a hand-edit that broke it would
    /// otherwise only show up as one wrong syllable in one word.
    #[test]
    fn every_digraph_is_built_from_known_kana() {
        for (key, _) in HEPBURN_DIGRAPHS {
            let mut chars = key.chars();
            let (a, b) = (chars.next().unwrap(), chars.next().unwrap());
            assert!(chars.next().is_none(), "{key:?} is not two characters");
            assert!(single(a).is_some(), "{a:?} of {key:?} is not a known kana");
            assert!(single(b).is_some(), "{b:?} of {key:?} is not a known kana");
        }
    }

    #[test]
    fn no_key_appears_twice() {
        let mut singles: Vec<char> = HEPBURN_SINGLES.iter().map(|(k, _)| *k).collect();
        singles.sort_unstable();
        let before = singles.len();
        singles.dedup();
        assert_eq!(singles.len(), before, "a kana is mapped twice");

        let mut digraphs: Vec<&str> = HEPBURN_DIGRAPHS.iter().map(|(k, _)| *k).collect();
        digraphs.sort_unstable();
        let before = digraphs.len();
        digraphs.dedup();
        assert_eq!(digraphs.len(), before, "a digraph is mapped twice");
    }

    #[test]
    fn kana_map_to_ipa_not_romaji() {
        assert_eq!(single('か'), Some("ka"));
        assert_eq!(single('し'), Some("ɕi"), "shi is one IPA symbol, not two letters");
        assert_eq!(single('つ'), Some("ʦɨ"));
        assert_eq!(single('ふ'), Some("ɸɯ"));
    }

    #[test]
    fn a_digraph_is_one_sound_not_two() {
        assert_eq!(digraph('き', 'ゃ'), Some("kʲa"));
        assert_eq!(digraph('し', 'ょ'), Some("ɕo"));
        assert_eq!(digraph('じ', 'ぇ'), Some("ʥe"));
        assert_eq!(digraph('か', 'か'), None, "not every pair is a digraph");
    }

    #[test]
    fn japanese_punctuation_becomes_its_ascii_equivalent() {
        assert_eq!(single('。'), Some("."));
        assert_eq!(single('、'), Some(","));
        assert_eq!(single('？'), Some("?"));
        assert_eq!(single('！'), Some("!"));
        assert_eq!(single('「'), Some("\u{201C}"));
        assert_eq!(single('」'), Some("\u{201D}"));
    }

    #[test]
    fn ainu_small_katakana_fold_to_full_size() {
        assert_eq!(fold_phonetic_extensions("ㇰㇱ"), "クシ");
        assert_eq!(fold_phonetic_extensions("テスト"), "テスト", "unchanged");
    }

    #[test]
    fn dakuten_voices_a_kana() {
        assert_eq!(add_dakuten('か'), Some('が'));
        assert_eq!(add_dakuten('は'), Some('ば'));
        assert_eq!(add_dakuten('と'), Some('ど'));
        assert_eq!(add_dakuten('あ'), None, "vowels cannot be voiced");
    }

    #[test]
    fn small_kana_and_iteration_marks_are_recognized() {
        for c in ['ゃ', 'ゅ', 'ょ', 'ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ'] {
            assert!(is_sutegana(c), "{c:?} is a small kana");
        }
        assert!(!is_sutegana('や'), "the full-size kana is not small");
        for c in ['々', 'ゝ', 'ゞ'] {
            assert!(is_odori(c), "{c:?} is an iteration mark");
        }
    }
}
