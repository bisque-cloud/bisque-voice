//! The Rust G2P against golden output from Python Misaki, character for
//! character.
//!
//! Misaki ships no tests, so there was nothing to port; these fixtures are the
//! substitute. `tools/gen-g2p-fixtures.py` generates them by running the real
//! Misaki, and they are committed — Python is needed only to regenerate them
//! after a deliberate Misaki version bump.
//!
//! A failure here is not a style disagreement. Kokoro was trained on Misaki's
//! exact phoneme sequences, so a single character of divergence is a
//! mispronunciation at best; and because `model.rs` walks the duration tensor
//! one model token at a time, it is a silently desynchronized cue track at
//! worst.
//!
//! These live inside the crate rather than in `tests/` because `bisque-voice` is
//! a binary with no library target, so an integration test could not reach
//! `crate::g2p`.

use std::path::PathBuf;

use serde::Deserialize;

use super::{align_tokens, espeak, phoneme_string};

#[derive(Debug, Deserialize)]
struct Case {
    text: String,
    phonemes: String,
    /// Present only where we deliberately differ from stock Misaki, recording
    /// what stock Misaki returns. See `gen_zh()` in the generator.
    #[serde(default)]
    misaki: Option<String>,
}

fn fixtures(name: &str) -> Vec<Case> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/g2p")
        .join(format!("{name}.json"));
    let raw =
        std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_slice(&raw)
        .unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
}

/// The directory *containing* `espeak-ng-data`.
///
/// eSpeak's phoneme output is a function of its data files, so a test that
/// silently skipped when they were missing would be a test that never runs in
/// CI and never fails. It panics with instructions instead.
fn espeak_data() -> PathBuf {
    if let Ok(dir) = std::env::var("BISQUE_VOICE_ESPEAK_DATA") {
        return PathBuf::from(dir);
    }
    if let Ok(models) = crate::fetch::Models::from_env() {
        let packed = models.root().join(super::pack::ESPEAK.pack_version);
        if packed.join("espeak-ng-data").is_dir() {
            return packed;
        }
    }
    // Developer convenience: a Homebrew or distro install of the same 1.52.0
    // release the pack is built from.
    for candidate in ["/opt/homebrew/share", "/usr/local/share", "/usr/share"] {
        let p = PathBuf::from(candidate);
        if p.join("espeak-ng-data").is_dir() {
            return p;
        }
    }
    panic!(
        "no espeak-ng-data found. Install the pack (`bisque-voice install kokoro`), \
         or `brew install espeak-ng`, or point BISQUE_VOICE_ESPEAK_DATA at the \
         directory containing espeak-ng-data."
    );
}

/// The directory holding pypinyin's dictionaries.
///
/// Same resolution order as `espeak_data`, plus the staging directory
/// `tools/build-g2p-packs.ts` writes, so the tests can run against a pack that
/// has been built but not installed.
fn chinese_data() -> PathBuf {
    if let Ok(dir) = std::env::var("BISQUE_VOICE_ZH_DATA") {
        return PathBuf::from(dir);
    }
    if let Ok(models) = crate::fetch::Models::from_env() {
        let packed = models.root().join(super::pack::CHINESE.pack_version);
        if packed.join("pinyin_dict.json").is_file() {
            return packed;
        }
    }
    let staged = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("dist")
        .join(super::pack::CHINESE.pack_version);
    if staged.join("pinyin_dict.json").is_file() {
        return staged;
    }
    panic!(
        "no Chinese G2P data found. Build the pack with \
         `bun run tools/build-g2p-packs.ts --only zh`, install it, or point \
         BISQUE_VOICE_ZH_DATA at a directory holding pinyin_dict.json and \
         phrases_dict.json."
    );
}

#[test]
fn chinese_matches_misaki() {
    let g2p = super::zh::ChineseG2P::new(&chinese_data())
        .unwrap_or_else(|e| panic!("opening the Chinese G2P: {e:#}"));

    let cases = fixtures("zh");
    assert!(!cases.is_empty(), "zh.json has no cases");

    let mut failures: Vec<String> = Vec::new();
    for case in &cases {
        let got = g2p.phonemize_text(&case.text);
        if got != case.phonemes {
            failures.push(format!(
                "  {:?}\n    expected {:?}\n    got      {:?}",
                case.text, case.phonemes, got
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} zh cases diverge from Misaki:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

/// The polyphone the whole phrase dictionary exists for. If per-character
/// lookup ever creeps back in, 银行 becomes "yín xíng" and this catches it.
#[test]
fn the_phrase_dictionary_resolves_polyphones() {
    let g2p = super::zh::ChineseG2P::new(&chinese_data()).unwrap();
    let bank = g2p.phonemize_text("银行");
    let walk = g2p.phonemize_text("行走");
    assert_ne!(
        bank, walk,
        "行 must be read differently in 银行 (háng) and 行走 (xíng)"
    );
    assert!(bank.contains('x'), "银行's 行 is háng, so it starts with x: {bank:?}");

    // In a sentence, each 行 is still resolved by the word it belongs to.
    let sentence = g2p.phonemize_text("我去银行，他在行走。");
    assert!(sentence.contains("i↗nxa↗ŋ"), "银行 is yín háng: {sentence:?}");
    assert!(sentence.contains("ɕi↗ŋʦou↓"), "行走 is xíng zǒu: {sentence:?}");
}

/// A polyphone **standing alone** falls back to its most common reading, and
/// there is no fixing that here.
///
/// A phrase dictionary resolves a character by the word it sits in. An isolated
/// one has no word, so 行 gets `xíng` — even in 银行的行, where a reader knows it
/// is quoting the 行 of 银行 and says `háng`. Recovering that needs to follow a
/// reference back through the sentence, which is semantics, not a lookup.
///
/// Python Misaki produces the identical string, so this is upstream behaviour
/// faithfully reproduced rather than a gap in the port. Pinned because it looks
/// exactly like a bug when someone hears it, and the useful answer is "known,
/// and not something the dictionary can decide".
#[test]
fn an_isolated_polyphone_takes_its_most_common_reading() {
    let g2p = super::zh::ChineseG2P::new(&chinese_data()).unwrap();
    assert_eq!(
        g2p.phonemize_text("银行的行，和行走的行。"),
        "i↗nxa↗ŋ tɤ ɕi↗ŋ，xɤ↗ ɕi↗ŋʦou↓ tɤ ɕi↗ŋ。",
        "byte-identical to Python Misaki, including the standalone 行 that a \
         person would read as háng"
    );
}

/// The directory holding the compiled UniDic dictionary and Misaki's word list.
fn japanese_data() -> PathBuf {
    if let Ok(dir) = std::env::var("BISQUE_VOICE_JA_DATA") {
        return PathBuf::from(dir);
    }
    if let Ok(models) = crate::fetch::Models::from_env() {
        let packed = models.root().join(super::pack::JAPANESE.pack_version);
        if packed.join("ja_words.txt").is_file() {
            return packed;
        }
    }
    let staged = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("dist")
        .join(super::pack::JAPANESE.pack_version);
    if staged.join("ja_words.txt").is_file() {
        return staged;
    }
    panic!(
        "no Japanese G2P data found. Build the pack with \
         `bun run tools/build-g2p-packs.ts --only ja`, install it, or point \
         BISQUE_VOICE_JA_DATA at a directory holding lindera-unidic/ and \
         ja_words.txt."
    );
}

#[test]
fn japanese_matches_misaki() {
    let g2p = super::ja::JapaneseG2P::new(&japanese_data())
        .unwrap_or_else(|e| panic!("opening the Japanese G2P: {e:#}"));

    let cases = fixtures("ja");
    assert!(!cases.is_empty(), "ja.json has no cases");

    let mut failures: Vec<String> = Vec::new();
    for case in &cases {
        match g2p.phonemize_text(&case.text) {
            Ok(got) if got == case.phonemes => {}
            Ok(got) => failures.push(format!(
                "  {:?}\n    expected {:?}\n    got      {:?}",
                case.text, case.phonemes, got
            )),
            Err(e) => failures.push(format!("  {:?}\n    errored: {e:#}", case.text)),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} ja cases diverge from Misaki:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

fn check_espeak(lang: char, name: &str) {
    let voice =
        espeak::voice_for(lang).unwrap_or_else(|| panic!("{lang} is not an eSpeak language"));
    let g2p = espeak::EspeakG2P::new(voice, &espeak_data())
        .unwrap_or_else(|e| panic!("opening eSpeak for {voice}: {e:#}"));

    let cases = fixtures(name);
    assert!(!cases.is_empty(), "{name}.json has no cases");

    let mut failures: Vec<String> = Vec::new();
    for case in &cases {
        match g2p.phonemize_text(&case.text) {
            Ok(got) if got == case.phonemes => {}
            Ok(got) => failures.push(format!(
                "  {:?}\n    expected {:?}\n    got      {:?}",
                case.text, case.phonemes, got
            )),
            Err(e) => failures.push(format!("  {:?}\n    errored: {e:#}", case.text)),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} {name} cases diverge from Misaki:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn spanish_matches_misaki() {
    check_espeak('e', "es");
}

#[test]
fn french_matches_misaki() {
    check_espeak('f', "fr");
}

#[test]
fn italian_matches_misaki() {
    check_espeak('i', "it");
}

#[test]
fn portuguese_matches_misaki() {
    check_espeak('p', "pt");
}

#[test]
fn hindi_matches_misaki() {
    check_espeak('h', "hi");
}

/// Whatever the phonemes are, the token split must reproduce them exactly.
///
/// This is the invariant `model.rs` rests on: the string handed to the model is
/// rebuilt from the tokens, and the duration tensor is walked against those
/// same tokens. Drift of one character moves every later cue.
#[test]
fn tokenizing_every_fixture_round_trips_its_phoneme_string() {
    for name in ["es", "fr", "it", "pt", "hi", "zh", "ja"] {
        for case in fixtures(name) {
            let tokens = align_tokens(&case.text, &case.phonemes);
            assert_eq!(
                phoneme_string(&tokens),
                case.phonemes,
                "{name}: {:?}",
                case.text
            );
            assert!(
                tokens.iter().all(|t| t.phonemes.is_some()),
                "{name}: every token must carry phonemes so timings stay 1:1"
            );
        }
    }
}

/// The Chinese fixtures record where we deliberately depart from stock Misaki.
/// Each such case must be one we actually signed up for — a Misaki crash, or
/// raw hanzi it failed to phonemize — never a quiet difference in output.
#[test]
fn every_chinese_divergence_is_a_misaki_failure_not_a_difference_of_opinion() {
    let cases = fixtures("zh");
    let diverging = cases.iter().filter(|c| c.misaki.is_some()).count();
    assert!(
        diverging > 0,
        "the digit-leading cases must still exercise the divergence"
    );
    for case in cases.iter().filter(|c| c.misaki.is_some()) {
        let stock = case.misaki.as_deref().unwrap();
        assert!(
            stock.starts_with("<raises")
                || stock.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c)),
            "a divergence must be a Misaki crash or raw hanzi it could not \
             phonemize, but for {:?} stock Misaki returned {stock:?}",
            case.text
        );
    }
}
