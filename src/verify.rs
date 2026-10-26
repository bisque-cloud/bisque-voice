//! Name the Kokoro mispronunciations a listener has already reported, right
//! after the narration is made.
//!
//! Every rule here is a mistake a person heard in published Kokoro audio, with
//! a fix that was narrated and heard to work (`tools/verify-regression.ts`
//! narrates each one again):
//!
//! - `plugin` read with a soft g, PLOOJUHN. Not in the lexicon, and the
//!   neural fallback reads the `g` as in "gin".
//! - `TDD` spelled as one run, `tˌidˌiˈdi`: the two D's blur into one. Each
//!   letter needs its own stress and a space, `tˈi dˈi dˈi`.
//! - An IPA marker with its stress mark before the first consonant,
//!   `[gstack](/ˈɡistˌæk/)`. Standard IPA, and Kokoro says it as an inserted
//!   vowel: "Uggy Stack" (#116). Kokoro's own lexicon puts the mark after the
//!   onset, `ɡˈistˌæk`.
//!
//! It reads the phonemes Kokoro was given, not the audio. For these three the
//! phonemes decide the outcome, so the warning is exact; a mistake outside
//! them is not caught, and no recognizer runs to look for one. Whisper hears
//! through context and transcribed several of these as correct, so an audio
//! check could not stand behind a verdict. That is why a match warns with its
//! fix and never stops the narration.

use serde::Serialize;

use crate::check::{respell, Finding};

/// One reported mistake found in this narration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Warning {
    /// Which reported mistake: `plugin-soft-g`, `tdd-run-together`,
    /// `stress-before-onset`.
    pub rule: &'static str,
    /// The word as written in the narration.
    pub word: String,
    /// What Kokoro was given for it.
    pub phonemes: String,
    /// The marker to write instead.
    pub fix: String,
    pub message: String,
}

fn is_edge_punct(c: char) -> bool {
    c.is_ascii_punctuation() || "—–…“”‘’«»".contains(c)
}

/// The word without the punctuation riding on it: `plugin,` → `plugin`.
fn bare(word: &str) -> &str {
    word.trim_matches(is_edge_punct)
}

/// The phonemes without the punctuation G2P carries along: `plˈuʤən,`.
fn bare_phonemes(p: &str) -> &str {
    p.trim_matches(|c: char| is_edge_punct(c) || c.is_whitespace())
}

fn unstressed(p: &str) -> String {
    p.chars().filter(|c| *c != 'ˈ' && *c != 'ˌ').collect()
}

/// Kokoro's vowel symbols, diphthongs included (`check.rs`, `SOUNDS`).
fn is_vowel(c: char) -> bool {
    "IAOWYaeoɑæəᵊɛɪᵻiʊuɔʌɜɐɒ".contains(c)
}

/// Every reported mistake in one narration. `findings` is Kokoro's
/// pronunciation report for `text`, which is the narration with its markers.
pub fn warnings(findings: &[Finding], text: &str) -> Vec<Warning> {
    let mut out: Vec<Warning> = Vec::new();
    for f in findings {
        let found = plugin_soft_g(f).or_else(|| tdd_run_together(f));
        // One line per word: every occurrence has the same fix.
        if let Some(w) = found {
            if !out.iter().any(|o| o.rule == w.rule && o.word == w.word) {
                out.push(w);
            }
        }
    }
    out.extend(stress_before_onset(text));
    out
}

fn plugin_soft_g(f: &Finding) -> Option<Warning> {
    let word = bare(&f.word);
    let phonemes = bare_phonemes(&f.phonemes);
    let folded = word.to_lowercase().replace('-', "");
    if !(folded == "plugin" || folded == "plugins") || !phonemes.contains('ʤ') {
        return None;
    }
    let ipa = if folded.ends_with('s') {
        "plˈʌɡɪnz"
    } else {
        "plˈʌɡɪn"
    };
    let fix = format!("[{word}](/{ipa}/)");
    Some(Warning {
        rule: "plugin-soft-g",
        message: format!(
            "\"{word}\" is said {}, with a soft g. Write {fix} so it is said \
             PLUH-gihn.",
            respell(phonemes)
        ),
        word: word.to_string(),
        phonemes: phonemes.to_string(),
        fix,
    })
}

fn tdd_run_together(f: &Finding) -> Option<Warning> {
    let word = bare(&f.word);
    let (stem, plural) = match word.strip_suffix('s') {
        Some(stem) if stem.eq_ignore_ascii_case("tdd") => (stem, true),
        _ => (word, false),
    };
    let phonemes = bare_phonemes(&f.phonemes);
    // Already separated, as the fix writes it.
    if !stem.eq_ignore_ascii_case("tdd") || phonemes.contains(char::is_whitespace) {
        return None;
    }
    let said = unstressed(phonemes);
    let spelled = if plural { "tididiz" } else { "tididi" };
    if said != spelled {
        return None;
    }
    let fix = format!("[{word}](/tˈi dˈi dˈi{}/)", if plural { "z" } else { "" });
    Some(Warning {
        rule: "tdd-run-together",
        message: format!(
            "\"{word}\" runs its letters together ({}), so the two D's blur \
             into one. Write {fix} so each letter is said on its own.",
            respell(phonemes)
        ),
        word: word.to_string(),
        phonemes: phonemes.to_string(),
        fix,
    })
}

/// IPA markers whose stress mark comes before an onset consonant. The fix
/// moves the mark to just before the first vowel, where Kokoro's lexicon
/// writes it.
fn stress_before_onset(text: &str) -> Vec<Warning> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<Warning> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '[' {
            i += 1;
            continue;
        }
        let Some((display, ipa, _, end)) = crate::pronounce::parse_override(&chars, i) else {
            i += 1;
            continue;
        };
        i = end;
        let Some(ipa) = ipa else { continue };
        let pieces: Vec<&str> = ipa.split_whitespace().collect();
        if !pieces.iter().any(|p| moved_stress(p).is_some()) {
            continue;
        }
        let fixed: Vec<String> = pieces
            .iter()
            .map(|p| moved_stress(p).unwrap_or_else(|| p.to_string()))
            .collect();
        let fix = format!("[{display}](/{}/)", fixed.join(" "));
        if out.iter().any(|o| o.fix == fix) {
            continue;
        }
        out.push(Warning {
            rule: "stress-before-onset",
            message: format!(
                "[{display}](/{ipa}/) puts the stress mark before a consonant, \
                 which Kokoro says as an extra vowel (\"gstack\" came out \
                 \"Uggy Stack\"). Write {fix}, with the mark just before the \
                 vowel."
            ),
            word: display,
            phonemes: ipa,
            fix,
        });
    }
    out
}

/// `ˈɡistˌæk` → `ɡˈistˌæk`; `None` when the piece does not open with a stress
/// mark followed by a consonant.
fn moved_stress(piece: &str) -> Option<String> {
    let mut chars = piece.chars();
    let mark = chars.next().filter(|c| *c == 'ˈ' || *c == 'ˌ')?;
    let rest: String = chars.collect();
    let first = rest.chars().next()?;
    if is_vowel(first) || !first.is_alphabetic() {
        return None;
    }
    let at = rest.find(is_vowel)?;
    Some(format!("{}{mark}{}", &rest[..at], &rest[at..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(word: &str, phonemes: &str) -> Finding {
        Finding {
            word: word.into(),
            phonemes: phonemes.into(),
            respelled: respell(phonemes),
            in_lexicon: false,
            readings: 1,
        }
    }

    fn rules(w: &[Warning]) -> Vec<&str> {
        w.iter().map(|w| w.rule).collect()
    }

    // Phonemes below are what `pronunciation-report --voice kokoro:af_heart`
    // printed on bisque-voice 0.1.22 for the sentences that shipped.

    #[test]
    fn plugin_with_a_soft_g_is_named_with_its_fix() {
        let w = warnings(&[finding("plugin,", "plˈuʤən,")], "Code plugin, and");
        assert_eq!(rules(&w), ["plugin-soft-g"]);
        assert_eq!(w[0].word, "plugin");
        assert_eq!(w[0].fix, "[plugin](/plˈʌɡɪn/)");
        assert!(w[0].message.contains("PLOOJUHN"), "{}", w[0].message);
    }

    #[test]
    fn plugins_gets_the_plural_fix() {
        let w = warnings(&[finding("Plugins", "plˈuʤənz")], "Plugins");
        assert_eq!(w[0].fix, "[Plugins](/plˈʌɡɪnz/)");
    }

    #[test]
    fn plugin_fixed_with_its_marker_passes() {
        let text = "Code [plugin](/plˈʌɡɪn/), and";
        assert!(warnings(&[finding("plugin,", "plˈʌɡɪn,")], text).is_empty());
    }

    #[test]
    fn the_episodes_tdd_marker_is_named() {
        let text = "the [tdd](/tˌidˌiˈdi/) skill";
        let w = warnings(&[finding("tdd", "tˌidˌiˈdi")], text);
        assert_eq!(rules(&w), ["tdd-run-together"]);
        assert_eq!(w[0].fix, "[tdd](/tˈi dˈi dˈi/)");
    }

    #[test]
    fn tdd_from_the_lexicon_is_named_too() {
        let w = warnings(&[finding("TDD.", "tˌidˌidˈi.")], "We use TDD.");
        assert_eq!(rules(&w), ["tdd-run-together"]);
        assert_eq!(w[0].fix, "[TDD](/tˈi dˈi dˈi/)");
    }

    #[test]
    fn tdd_separated_passes() {
        let text = "the [tdd](/tˈi dˈi dˈi/) skill";
        assert!(warnings(&[finding("tdd", "tˈi dˈi dˈi")], text).is_empty());
    }

    /// Bounded to the reported word: other initialisms are spelled the same
    /// way and nobody has heard them wrong.
    #[test]
    fn other_initialisms_are_left_alone() {
        let w = warnings(
            &[finding("CSS", "sˌiˌɛsˈɛs"), finding("HTTP", "ˌAʧtˌitˈipˈi")],
            "CSS HTTP",
        );
        assert!(w.is_empty(), "{w:?}");
    }

    #[test]
    fn a_marker_with_stress_before_its_onset_is_named() {
        let text = "Install [gstack](/ˈɡistˌæk/) first.";
        let w = warnings(&[finding("gstack", "ˈɡistˌæk")], text);
        assert_eq!(rules(&w), ["stress-before-onset"]);
        assert_eq!(w[0].fix, "[gstack](/ɡˈistˌæk/)");
    }

    #[test]
    fn a_marker_stressed_after_its_onset_passes() {
        let text = "Install [gstack](/ɡˈistˌæk/) first, then [it](/ˈɪt/).";
        assert!(warnings(&[], text).is_empty());
    }

    #[test]
    fn only_the_offending_piece_of_a_spaced_marker_moves() {
        let w = warnings(&[], "[Q-tip](/kjˈu ˈtɪp/)");
        assert_eq!(w[0].fix, "[Q-tip](/kjˈu tˈɪp/)");
    }

    #[test]
    fn each_word_is_named_once() {
        let f = finding("plugin", "plˈuʤən");
        let w = warnings(&[f.clone(), f], "plugin plugin");
        assert_eq!(w.len(), 1);
    }
}
