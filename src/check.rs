//! Show what a voice is about to say, before it says it.
//!
//! A mispronunciation ships silently: catching one means listening to every
//! sentence of every render. The G2P already knows the answer before any audio
//! exists, so the only thing missing was showing it to whoever is authoring.
//!
//! The report states facts and never judges them. It used to filter the word
//! list through spelling heuristics (vowel counts, digit tests) to guess which
//! words were "at risk", and the guess did not work: `Kokoro`, `ElevenLabs`
//! and `companies` are all absent from the lexicon, exactly one of them is
//! mispronounced, and nothing in their spelling says which. So the report now
//! emits every word, and judging the respellings belongs to whoever reads the
//! report — a language model or a person, both of whom know how the words
//! should sound.
//!
//! Per word, two kinds of fact:
//!
//! - **What it will sound like, in letters.** `ˌiɡmˈil` tells a reader nothing;
//!   `ig-MEEL` is wrong on sight. The respelling is approximate by design. It
//!   exists to make a defect obvious, never to be authored back into narration.
//! - **What the lexicon knows.** A word it has never seen was guessed from
//!   spelling (`ggml`); a word it holds with two distinct-sounding readings
//!   had one chosen for it (`live`).
//!
//! Kokoro only. An engine with no phoneme stage has nothing to inspect, and
//! the report says so rather than passing the text (`Report::Unavailable`).

/// What a pronunciation report pass could determine.
///
/// This type exists to keep two different answers from printing the same way.
/// The report used to be a bare `Vec<Finding>`, so a clean sentence, an engine
/// with no letters-to-sounds stage, a voice whose language has no lexicon, and
/// an outright failure all came back empty and all read as "nothing flagged" —
/// the exact confusion the module doc above warns about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Report {
    /// The letters-to-sounds stage ran, and these are its facts, one entry
    /// per word. Empty means the text had no words, and that is a real answer.
    Checked(Vec<Finding>),
    /// Nothing was inspected, and why. Never a claim about the text.
    Unavailable(String),
}

/// One word's facts, exactly as the letters-to-sounds stage states them.
///
/// Serializes as one element of the `pronunciation-report --json` envelope,
/// so the field names are part of the CLI contract.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Finding {
    pub word: String,
    pub phonemes: String,
    /// The phonemes written in ordinary letters, stressed syllable in capitals.
    pub respelled: String,
    /// Whether the lexicon holds this word. False means the neural fallback
    /// guessed the phonemes from the spelling.
    #[serde(rename = "inLexicon")]
    pub in_lexicon: bool,
    /// How many distinct-sounding readings the lexicon holds for it. More
    /// than one means a part-of-speech tagger chose the reading reported here.
    pub readings: u32,
}

/// Kokoro's phoneme alphabet in ordinary letters.
///
/// Misaki's inventory is IPA with a few substitutions of its own: `I A O W`
/// are the diphthongs, `T` is the flap, `ᵊ ᵻ` are reduced vowels. Longest match
/// first, so `ʤ` never gets read as two symbols.
/// Kokoro's phoneme alphabet in ordinary letters, and whether each is a vowel.
///
/// Misaki's inventory is IPA with substitutions of its own: `I A O W Y` are the
/// diphthongs, `T` is the flap, `ᵊ ᵻ` are reduced vowels. Longest match first,
/// so `ʤ` is never read as two symbols.
///
/// The vowel flag is what makes syllable splitting possible: a sound is the
/// unit, not a letter. Splitting on letters would cut "eh" after its `e`.
const SOUNDS: &[(&str, &str, bool)] = &[
    // diphthongs, including Kokoro's own capital forms
    ("I", "eye", true), ("A", "ay", true), ("O", "oh", true),
    ("W", "ow", true), ("Y", "oy", true),
    ("aʊ", "ow", true), ("aɪ", "eye", true), ("eɪ", "ay", true),
    ("oʊ", "oh", true), ("ɔɪ", "oy", true),
    // consonants that are not already their own letter
    ("ʤ", "j", false), ("ʧ", "ch", false), ("ʃ", "sh", false),
    ("ʒ", "zh", false), ("θ", "th", false), ("ð", "dh", false),
    ("ŋ", "ng", false), ("ɹ", "r", false), ("j", "y", false),
    ("ɡ", "g", false), ("T", "t", false), ("ɾ", "t", false), ("ʔ", "", false),
    // vowels
    ("ɑ", "ah", true), ("æ", "a", true), ("ə", "uh", true), ("ᵊ", "uh", true),
    ("ɛ", "eh", true), ("ɪ", "ih", true), ("ᵻ", "ih", true), ("i", "ee", true),
    ("ʊ", "uu", true), ("u", "oo", true), ("ɔ", "aw", true), ("ʌ", "uh", true),
    ("ɜ", "ur", true), ("ɐ", "uh", true), ("ɒ", "o", true),
];

/// One sound: the letters that stand for it, and whether it carries a syllable.
type Sound = (String, bool);

/// Rewrite a phoneme string as letters, with the stressed syllable capitalised.
///
/// Approximate on purpose. It has one job: make a wrong pronunciation obvious
/// to someone who does not read IPA.
pub fn respell(phonemes: &str) -> String {
    let mut syllables: Vec<(bool, Vec<Sound>)> = Vec::new();
    let mut current: Vec<Sound> = Vec::new();
    let mut stressed = false;
    let mut rest = phonemes;

    // A stress mark opens a syllable, but Kokoro writes it AFTER the onset
    // consonant (`lˈɪv`, not `ˈlɪv`), and consonants already collected since
    // the last vowel belong to the syllable now opening, not the one closing.
    // So: close at the last vowel, and carry the tail forward.
    let open_syllable = |current: &mut Vec<Sound>,
                             syllables: &mut Vec<(bool, Vec<Sound>)>,
                             stressed: &mut bool| {
        if !current.iter().any(|(_, vowel)| *vowel) {
            return;
        }
        let last_vowel = current.iter().rposition(|(_, vowel)| *vowel).unwrap_or(0);
        let onset: Vec<Sound> = current.split_off(last_vowel + 1);
        syllables.push((*stressed, std::mem::take(current)));
        *stressed = false;
        *current = onset;
    };

    while !rest.is_empty() {
        if let Some(tail) = rest.strip_prefix('ˈ') {
            open_syllable(&mut current, &mut syllables, &mut stressed);
            stressed = true;
            rest = tail;
            continue;
        }
        if let Some(tail) = rest.strip_prefix('ˌ') {
            open_syllable(&mut current, &mut syllables, &mut stressed);
            rest = tail;
            continue;
        }
        if let Some((sym, letters, vowel)) = SOUNDS.iter().find(|(s, _, _)| rest.starts_with(s)) {
            if !letters.is_empty() {
                current.push(((*letters).to_string(), *vowel));
            }
            rest = &rest[sym.len()..];
            continue;
        }
        let ch = rest.chars().next().expect("rest is non-empty");
        // Punctuation rides along. A bare ASCII vowel is still a vowel.
        if !ch.is_whitespace() {
            let vowel = matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u');
            current.push((ch.to_string(), vowel));
        }
        rest = &rest[ch.len_utf8()..];
    }
    if !current.is_empty() {
        syllables.push((stressed, current));
    }

    // With no stress mark anywhere there is nothing to emphasise, so stay flat.
    let any_stress = syllables.iter().any(|(s, _)| *s);
    syllables
        .into_iter()
        .map(|(is_stressed, sounds)| {
            let text: String = sounds.into_iter().map(|(l, _)| l).collect();
            if is_stressed && any_stress {
                text.to_uppercase()
            } else {
                text
            }
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_published_defects_are_legible_as_letters() {
        // These are the exact phoneme strings the engine produced for the
        // llama.cpp presentation. If a reader cannot see what is wrong here,
        // the whole subcommand is pointless.
        assert_eq!(respell("ˌiɡmˈil"), "ee-GMEEL");
        assert_eq!(respell("lˈIv"), "LEYEV");
        assert_eq!(respell("lˈɪv"), "LIHV");
    }

    #[test]
    fn a_correct_reading_and_a_wrong_one_do_not_look_alike() {
        assert_ne!(respell("lˈɪv"), respell("lˈIv"));
    }

    #[test]
    fn stress_marks_split_syllables_and_capitalise_one() {
        // Kokoro marks the VOWEL, not the syllable, so consonants standing
        // before a mark open the syllable it belongs to. The split is
        // approximate and only has to stay readable: this is "G G M L".
        assert_eq!(respell("ʤˌiʤˌiˌɛmˈɛl"), "jee-jee-eh-MEHL");
    }

    #[test]
    fn an_onset_consonant_is_never_stranded_on_its_own() {
        // The failure this guards is `l-IHV`, where the leading consonant
        // became a syllable with no vowel in it.
        for p in ["lˈɪv", "lˈIv", "ʤˌiʤˌiˌɛmˈɛl", "ˌiɡmˈil", "tˈɛnsəɹz"] {
            for part in respell(p).split('-') {
                assert!(
                    part.chars().any(|c| "aeiouAEIOU".contains(c)),
                    "{p:?} produced a vowel-less syllable {part:?} in {:?}",
                    respell(p)
                );
            }
        }
    }

    #[test]
    fn a_word_with_no_stress_mark_stays_flat() {
        assert_eq!(respell("ðə"), "dhuh");
    }

    #[test]
    fn multi_character_symbols_win_over_their_first_character() {
        // `ʤ` must not be read as two sounds, and `ʃ` is "sh" not "s"+"h".
        assert_eq!(respell("ʤ"), "j");
        assert_eq!(respell("ʃ"), "sh");
        assert_eq!(respell("ŋ"), "ng");
    }

    #[test]
    fn punctuation_rides_along() {
        assert_eq!(respell("lˈɪv."), "LIHV.");
    }

    #[test]
    fn an_empty_phoneme_string_yields_nothing() {
        assert_eq!(respell(""), "");
    }
}
