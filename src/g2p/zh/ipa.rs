//! Pinyin syllable → IPA, ported from `misaki/transcription.py`.
//!
//! A Mandarin syllable splits into an **initial** (the opening consonant, if
//! any) and a **final** (the vowel plus anything after it): `bāng` is initial
//! `b` + final `ang`. Each part maps to IPA through a fixed table, and the tone
//! is written into the vowel afterwards.
//!
//! Upstream's tables list several IPA variants per part — `h` may be `x` or
//! `h`, `er` may be `ɚ` or `aɚ̯` — and build the Cartesian product of them.
//! Misaki then takes `pinyin_to_ipa(py)[0]`, the first combination, and
//! discards the rest. Since the product enumerates the first initial with the
//! first final first, only the first variant of each part can ever be reached,
//! so only those are carried here. The alternatives are recorded in comments
//! rather than dropped silently, because a future caller wanting a different
//! dialect would need them.
//!
//! `0` inside a mapping is a placeholder for the tone mark, substituted by
//! `apply_tone`.

/// Tone number → IPA pitch contour. Tone 5 is the neutral tone and is unmarked.
const TONES: [&str; 5] = [
    "˥",    // 1, ā — high level
    "˧˥",   // 2, á — rising
    "˧˩˧",  // 3, ǎ — dipping
    "˥˩",   // 4, à — falling
    "",     // 5, a — neutral
];

/// The initial consonants, in `to_initials(strict=True)`'s vocabulary.
///
/// Alternatives not taken: `h` → `h`, `r` → `ʐ`.
const INITIALS: &[(&str, &str)] = &[
    ("b", "p"),
    ("c", "ʦʰ"),
    ("ch", "\u{ab67}ʰ"), // ʈʂʰ
    ("d", "t"),
    ("f", "f"),
    ("g", "k"),
    ("h", "x"),
    ("j", "ʨ"),
    ("k", "kʰ"),
    ("l", "l"),
    ("m", "m"),
    ("n", "n"),
    ("p", "pʰ"),
    ("q", "ʨʰ"),
    ("r", "ɻ"),
    ("s", "s"),
    ("sh", "ʂ"),
    ("t", "tʰ"),
    ("x", "ɕ"),
    ("z", "ʦ"),
    ("zh", "\u{ab67}"), // tʂ
];

/// The finals. Each entry is the pieces of the syllable in order; `0` marks
/// where the tone goes.
const FINALS: &[(&str, &[&str])] = &[
    ("a", &["a0"]),
    ("ai", &["ai̯0"]),
    ("an", &["a0", "n"]),
    ("ang", &["a0", "ŋ"]),
    ("ao", &["au̯0"]),
    ("e", &["ɤ0"]),
    ("ei", &["ei̯0"]),
    ("en", &["ə0", "n"]),
    ("eng", &["ə0", "ŋ"]),
    ("i", &["i0"]),
    ("ia", &["j", "a0"]),
    ("ian", &["j", "ɛ0", "n"]),
    ("iang", &["j", "a0", "ŋ"]),
    ("iao", &["j", "au̯0"]),
    ("ie", &["j", "e0"]),
    ("in", &["i0", "n"]),
    ("iou", &["j", "ou̯0"]),
    ("ing", &["i0", "ŋ"]),
    ("iong", &["j", "ʊ0", "ŋ"]),
    ("ong", &["ʊ0", "ŋ"]),
    ("ou", &["ou̯0"]),
    ("u", &["u0"]),
    ("uei", &["w", "ei̯0"]),
    ("ua", &["w", "a0"]),
    ("uai", &["w", "ai̯0"]),
    ("uan", &["w", "a0", "n"]),
    ("uen", &["w", "ə0", "n"]),
    ("uang", &["w", "a0", "ŋ"]),
    ("ueng", &["w", "ə0", "ŋ"]),
    // uo is written `o` after b, p, m, f; lo and yo are treated the same way.
    ("uo", &["w", "o0"]),
    ("o", &["w", "o0"]),
    // ü is written `u` after j, q, x, but pypinyin hands back ü there.
    ("ü", &["y0"]),
    ("üe", &["ɥ", "e0"]),
    ("üan", &["ɥ", "ɛ0", "n"]),
    ("ün", &["y0", "n"]),
];

/// After zh/ch/sh/r, the final `i` is an apical retroflex continuant rather
/// than the vowel — `zhi` is not "zhee". Alternative not taken: `ʐ̩`.
const FINAL_I_AFTER_RETROFLEX: &[&str] = &["ɻ̩0"];
/// After z/c/s, `i` is a laminal denti-alveolar continuant. Alternative not
/// taken: `z̩`.
const FINAL_I_AFTER_SIBILANT: &[&str] = &["ɹ̩0"];

/// Syllabic consonants, which stand alone with no initial or final.
const SYLLABIC_CONSONANTS: &[(&str, &[&str])] = &[
    ("hm", &["h", "m0"]),
    ("hng", &["h", "ŋ0"]),
    ("m", &["m0"]),
    ("n", &["n0"]),
    ("ng", &["ŋ0"]),
];

/// Interjections and the handful of syllables that are not initial+final.
/// Alternative not taken: `er` → `aɚ̯`.
const INTERJECTIONS: &[(&str, &[&str])] = &[
    ("io", &["j", "ɔ0"]),
    ("ê", &["ɛ0"]),
    ("er", &["ɚ0"]),
    ("o", &["ɔ0"]),
];

fn lookup<'a>(table: &'a [(&str, &'a [&str])], key: &str) -> Option<&'a [&'a str]> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

fn apply_tone(parts: &[&str], tone: usize) -> String {
    let mark = TONES[tone - 1];
    parts.iter().map(|p| p.replace('0', mark)).collect()
}

/// One syllable's IPA, given its normal-form pinyin split and its tone.
///
/// `initial` is the empty string when the syllable has none. Returns `None`
/// when the final is not one this table knows, which upstream signals by
/// raising.
pub fn syllable_to_ipa(normal: &str, initial: &str, final_: &str, tone: usize) -> Option<String> {
    debug_assert!((1..=5).contains(&tone), "tone {tone} out of range");

    // Interjections and syllabic consonants are matched on the whole syllable
    // and have neither an initial nor a final.
    if let Some(parts) = lookup(INTERJECTIONS, normal) {
        return Some(apply_tone(parts, tone));
    }
    if let Some(parts) = lookup(SYLLABIC_CONSONANTS, normal) {
        return Some(apply_tone(parts, tone));
    }

    let mut out = String::new();
    if !initial.is_empty() {
        let mapped = INITIALS
            .iter()
            .find(|(k, _)| *k == initial)
            .map(|(_, v)| *v)?;
        out.push_str(mapped);
    }

    let final_parts: &[&str] = match (initial, final_) {
        ("zh" | "ch" | "sh" | "r", "i") => FINAL_I_AFTER_RETROFLEX,
        ("z" | "c" | "s", "i") => FINAL_I_AFTER_SIBILANT,
        _ => lookup(FINALS, final_)?,
    };
    out.push_str(&apply_tone(final_parts, tone));
    Some(out)
}

/// `ZHG2P.retone`: rewrite IPA pitch contours as single arrow characters, which
/// is what Kokoro's vocabulary actually contains.
///
/// Order matters: `˧˥` must be replaced before `˥`, or the rising tone would be
/// consumed a character at a time.
pub fn retone(p: &str) -> String {
    let out = p
        .replace("˧˩˧", "↓") // third
        .replace("˧˥", "↗") // second
        .replace("˥˩", "↘") // fourth
        .replace('˥', "→") // first
        // The two apical continuants collapse to one symbol.
        .replace("\u{27b}\u{329}", "ɨ")
        .replace("\u{279}\u{329}", "ɨ");
    debug_assert!(
        !out.contains('\u{329}'),
        "a syllabic diacritic survived retone: {out:?}"
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tones_are_written_into_the_vowel() {
        // ni3 -> initial n, final i, third tone.
        assert_eq!(syllable_to_ipa("ni", "n", "i", 3).unwrap(), "ni˧˩˧");
        assert_eq!(syllable_to_ipa("ni", "n", "i", 1).unwrap(), "ni˥");
        // Neutral tone leaves no mark at all.
        assert_eq!(syllable_to_ipa("le", "l", "e", 5).unwrap(), "lɤ");
    }

    #[test]
    fn i_after_a_retroflex_is_not_the_vowel_i() {
        // zhi is not "zhee"; the final is an apical continuant.
        let zhi = syllable_to_ipa("zhi", "zh", "i", 1).unwrap();
        assert!(zhi.contains('\u{329}'), "expected a syllabic mark: {zhi:?}");
        assert_ne!(zhi, syllable_to_ipa("ji", "j", "i", 1).unwrap());
    }

    #[test]
    fn i_after_a_sibilant_differs_from_i_after_a_retroflex() {
        let zi = syllable_to_ipa("zi", "z", "i", 1).unwrap();
        let zhi = syllable_to_ipa("zhi", "zh", "i", 1).unwrap();
        assert_ne!(zi, zhi);
    }

    #[test]
    fn retone_replaces_the_rising_tone_before_the_level_one() {
        // If ˥ were replaced first, ˧˥ would come out as "˧→" rather than "↗".
        assert_eq!(retone("ma˧˥"), "ma↗");
        assert_eq!(retone("ma˥"), "ma→");
        assert_eq!(retone("ma˧˩˧"), "ma↓");
        assert_eq!(retone("ma˥˩"), "ma↘");
    }

    #[test]
    fn retone_collapses_both_apical_continuants_to_one_symbol() {
        assert_eq!(retone("\u{27b}\u{329}"), "ɨ");
        assert_eq!(retone("\u{279}\u{329}"), "ɨ");
    }

    #[test]
    fn an_interjection_is_matched_on_the_whole_syllable() {
        // er has no initial and no final of its own.
        assert_eq!(syllable_to_ipa("er", "", "er", 4).unwrap(), "ɚ˥˩");
    }

    #[test]
    fn an_unknown_final_is_refused_rather_than_guessed() {
        assert!(syllable_to_ipa("zzz", "z", "zzz", 1).is_none());
    }

    #[test]
    fn every_table_key_is_unique() {
        for (name, keys) in [
            ("INITIALS", INITIALS.iter().map(|(k, _)| *k).collect::<Vec<_>>()),
            ("FINALS", FINALS.iter().map(|(k, _)| *k).collect()),
            (
                "SYLLABIC_CONSONANTS",
                SYLLABIC_CONSONANTS.iter().map(|(k, _)| *k).collect(),
            ),
            (
                "INTERJECTIONS",
                INTERJECTIONS.iter().map(|(k, _)| *k).collect(),
            ),
        ] {
            let mut sorted = keys.clone();
            sorted.sort_unstable();
            let before = sorted.len();
            sorted.dedup();
            assert_eq!(sorted.len(), before, "{name} has a duplicate key");
        }
    }
}
