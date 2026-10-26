//! Heteronym accuracy against the REAL `us_gold` lexicon.
//!
//! `us_gold` has 790 POS-conditioned entries whose reading depends on the tag
//! ("present", "record", "read", "live"…). This harness runs the full English
//! G2P over a hand-labelled sentence corpus and checks that each target word
//! resolves to the reading its context demands.
//!
//! It needs the installed Kokoro engine pack (the lexicons ship inside it), so
//! it is `#[ignore]`d in a bare `cargo test`. Run it with:
//!
//! ```sh
//! BISQUE_VOICE_MODEL_DIR=~/.bisque/models \
//!   cargo test -p bisque-voice -- --ignored heteronym --nocapture
//! ```
//!
//! Cases are labelled with what MUST hold (`must: true`) versus documented
//! limits (`must: false`) — sentences whose resolution needs semantics no
//! rule-based tagger has ("they read it yesterday"). Limits are printed, not
//! asserted, so an improvement shows up as an unexpected pass rather than
//! silence.

use std::path::PathBuf;

use super::lexicon::Lexicon;
use super::EnglishG2P;

/// The engine-pack directory holding `us_gold.json` / `us_silver.json` —
/// the same resolution the engine itself uses, minus the download.
fn english_pack_dir() -> PathBuf {
    let root = crate::fetch::Models::from_env()
        .expect("resolving the models directory (set BISQUE_VOICE_MODEL_DIR)")
        .root()
        .to_path_buf();
    // `engine.rs` KOKORO.pack_version. Kept literal here so the eval fails
    // loudly (not silently against stale data) if the pack version moves.
    let dir = root.join("kokoro-v1.0-fp32-1");
    assert!(
        dir.join("us_gold.json").exists(),
        "no us_gold.json under {} — install the kokoro pack or point \
         BISQUE_VOICE_MODEL_DIR at a directory that has it",
        dir.display()
    );
    dir
}

fn english_g2p() -> (EnglishG2P, serde_json::Value) {
    let dir = english_pack_dir();
    let gold: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("us_gold.json")).unwrap()).unwrap();
    let silver: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("us_silver.json")).unwrap()).unwrap();
    let lexicon = Lexicon::new(gold.clone(), silver, false);
    // No BART fallback: every corpus word is in the lexicon, and the eval must
    // measure the lexicon path, not a neural guess.
    (EnglishG2P::new(lexicon, None), gold)
}

struct Case {
    sentence: &'static str,
    /// The target word exactly as it appears in the sentence.
    target: &'static str,
    /// The `us_gold` key holding the correct reading in this context.
    /// "DEFAULT" marks both guard cases (common reading must survive) and
    /// -ate fixes (the suffix rule used to force the rarer reading).
    want_key: &'static str,
    /// Asserted. `false` = documented limit, printed only.
    must: bool,
}

const fn case(
    sentence: &'static str,
    target: &'static str,
    want_key: &'static str,
    must: bool,
) -> Case {
    Case {
        sentence,
        target,
        want_key,
        must,
    }
}

/// Hand-labelled corpus. Three populations:
/// - verb/noun/participle contexts a context rule should resolve (fixes),
/// - contexts where the pre-context behavior was already right (guards),
/// - known limits (`must: false`).
const CORPUS: &[Case] = &[
    // ── verb contexts ────────────────────────────────────────────────────
    case("I live in the city.", "live", "VERB", true),
    case("He will present the plan.", "present", "VERB", true),
    case("She read the book yesterday.", "read", "VBD", true),
    case("Please record the meeting minutes.", "record", "VERB", true),
    case("They object to the change.", "object", "VERB", true),
    case("You can record your screen.", "record", "VERB", true),
    case("We should close the loop.", "close", "VERB", true),
    case("He lives in Austin.", "lives", "VERB", true),
    case("She presents the results.", "presents", "VERB", true),
    case("They will separate the two cases.", "separate", "VERB", true),
    case("We need to document the API.", "document", "VERB", true),
    case("Please implement the fix.", "implement", "VERB", true),
    case("He read the memo.", "read", "VBD", true),
    case("The report was read aloud.", "read", "VBN", true),
    case("She has read every draft.", "read", "VBN", true),
    case("He wound the clock.", "wound", "VBD", true),
    case("Don't present without notes.", "present", "VERB", true),
    case("We'll record the demo.", "record", "VERB", true),
    case("Let's close the deal.", "close", "VERB", true),
    case("We must not separate them.", "separate", "VERB", true),
    case("Present the summary first.", "Present", "VERB", true),
    case("Close the tab now.", "Close", "VERB", true),
    case("Record your thoughts.", "Record", "VERB", true),
    case("I object.", "object", "VERB", true),
    case("I used to live there.", "used", "VBD", true),
    case("I used to live there.", "live", "VERB", true),
    case("They rebel against the rules.", "rebel", "VERB", true),
    case("I suspect the cache.", "suspect", "VERB", true),
    case("It will progress quickly.", "progress", "VERB", true),
    case("We moderate the discussion.", "moderate", "VERB", true),
    // Coordinated predicate: the object determiner after the coordinator is
    // what licenses the verb reading ("past and present" has none).
    case(
        "Read the last week of git history and present a week in review.",
        "present",
        "VERB",
        true,
    ),
    // ── noun contexts ────────────────────────────────────────────────────
    case("Their use of color is bold.", "use", "NOUN", true),
    case("The close of trading was calm.", "close", "NOUN", true),
    case("An estimate of the cost.", "estimate", "NOUN", true),
    case("The content is king.", "content", "NOUN", true),
    case("The exploit was patched.", "exploit", "NOUN", true),
    case("The insert on page two.", "insert", "NOUN", true),
    case("In use since March.", "use", "NOUN", true),
    case("The produce at the market.", "produce", "NOUN", true),
    case("The refuse was collected.", "refuse", "NOUN", true),
    case("An increase in cost.", "increase", "NOUN", true),
    // ── -ate suffix fixes: DEFAULT is correct, the old Verb tag was not ──
    case("The appropriate level of detail.", "appropriate", "DEFAULT", true),
    case("A separate issue entirely.", "separate", "DEFAULT", true),
    case("The graduate students arrived.", "graduate", "DEFAULT", true),
    case("A moderate increase in cost.", "moderate", "DEFAULT", true),
    case("Use the alternate route.", "alternate", "DEFAULT", true),
    case("His associate agreed.", "associate", "DEFAULT", true),
    // ── guards: correct before, must stay correct ────────────────────────
    case("The record shows growth.", "record", "DEFAULT", true),
    case("A close friend of mine.", "close", "DEFAULT", true),
    case("The record label signed them.", "record", "DEFAULT", true),
    case("Their lives are better now.", "lives", "DEFAULT", true),
    case("Live music every night.", "Live", "DEFAULT", true),
    case("Present company excluded.", "Present", "DEFAULT", true),
    case("Record numbers attended.", "Record", "DEFAULT", true),
    case("A present from the team.", "present", "DEFAULT", true),
    case("The wound healed slowly.", "wound", "DEFAULT", true),
    case("The house was empty.", "house", "DEFAULT", true),
    case("The subject of the talk.", "subject", "DEFAULT", true),
    case("It was a close call.", "close", "DEFAULT", true),
    case("Wait a minute.", "minute", "DEFAULT", true),
    case("We use it every day.", "use", "DEFAULT", true),
    case("They produce great work.", "produce", "DEFAULT", true),
    case("We estimate the cost.", "estimate", "DEFAULT", true),
    // Predicative "live" reached through a verb, not a noun subject — must
    // stay the common /laɪv/ reading on every platform.
    case("The stream went live.", "live", "DEFAULT", true),
    case("Is the demo live?", "live", "DEFAULT", true),
    // ── known limits (not asserted) ──────────────────────────────────────
    // Fused-head "some" is not a safe verb trigger.
    case("Some live in the city.", "live", "VERB", false),
    // Non-3sg past is string-identical to the present: needs semantics.
    case("They read it yesterday.", "read", "VBD", false),
    // Proper-noun subjects are not in the pronoun lists.
    case("Maria read the book.", "read", "VBD", false),
    // "to" + noun-adjunct: the infinitive reading wins by frequency.
    case("He went to record stores.", "record", "DEFAULT", false),
    // An adjective between determiner and head hides the noun position.
    case("A moderate increase in cost.", "increase", "NOUN", false),
];

/// Cases that need `NLTagger` (macOS only): open-class noun subjects and
/// other patterns beyond safe one-word context rules. On other platforms
/// these remain among the known limits above.
#[cfg(target_os = "macos")]
const PLATFORM_CORPUS: &[Case] = &[
    case(
        "It is something you do continually, to refine the pitch and find the channel \
         where your real users live.",
        "live",
        "VERB",
        true,
    ),
    case("Some live in the city.", "live", "VERB", true),
    case("A moderate increase in cost.", "increase", "NOUN", true),
    // NLTagger's own known limit (shared with the Swift engine): an
    // object-complement adjective reads as a verb.
    case("Keep the servers live.", "live", "DEFAULT", false),
];
#[cfg(not(target_os = "macos"))]
const PLATFORM_CORPUS: &[Case] = &[];

/// The reading `want_key` selects, with the same stress transforms `lookup`
/// applies to a capitalized surface form (0.5 is a no-op on an already
/// stressed phoneme string, so plain equality holds for this corpus).
fn expected(gold: &serde_json::Value, target: &str, want_key: &str) -> String {
    let entry = gold
        .get(target.to_lowercase())
        .unwrap_or_else(|| panic!("{target:?} is not a us_gold headword"));
    match entry {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(map) => map
            .get(want_key)
            .or_else(|| map.get("DEFAULT"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or_else(|| panic!("{target:?} has no {want_key:?}/DEFAULT key"))
            .to_string(),
        v => panic!("unexpected entry shape for {target:?}: {v}"),
    }
}

/// What the G2P produced for the target word, punctuation stripped.
fn produced(g2p: &EnglishG2P, sentence: &str, target: &str) -> String {
    let tokens = g2p.phonemize(sentence);
    let token = tokens
        .iter()
        .find(|t| t.text.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'') == target)
        .unwrap_or_else(|| panic!("{target:?} not found among tokens of {sentence:?}"));
    token
        .phonemes
        .clone()
        .unwrap_or_else(|| panic!("{target:?} in {sentence:?} resolved to nothing"))
        .chars()
        .filter(|c| !super::PUNCTUATION.contains(*c))
        .collect()
}

#[test]
#[ignore = "needs the kokoro engine pack on disk (BISQUE_VOICE_MODEL_DIR)"]
fn heteronym_accuracy_on_the_labelled_corpus() {
    let (g2p, gold) = english_g2p();
    let mut failures: Vec<String> = Vec::new();
    let mut limit_hits = 0usize;
    let mut limit_total = 0usize;

    for c in CORPUS.iter().chain(PLATFORM_CORPUS) {
        let want = expected(&gold, c.target, c.want_key);
        let got = produced(&g2p, c.sentence, c.target);
        let pass = got == want;
        let kind = if c.must { "case " } else { "limit" };
        println!(
            "{} {} — {:20} want {:6} {:16} got {:16} {}",
            if pass { "PASS" } else { "FAIL" },
            kind,
            c.target,
            c.want_key,
            want,
            got,
            c.sentence
        );
        if c.must {
            if !pass {
                failures.push(format!(
                    "{:?}: {} should read {} ({}), got {}",
                    c.sentence, c.target, want, c.want_key, got
                ));
            }
        } else {
            limit_total += 1;
            if pass {
                limit_hits += 1;
            }
        }
    }

    let must_total = CORPUS
        .iter()
        .chain(PLATFORM_CORPUS)
        .filter(|c| c.must)
        .count();
    println!(
        "\nasserted: {}/{} — known limits also passing: {}/{}",
        must_total - failures.len(),
        must_total,
        limit_hits,
        limit_total
    );
    assert!(
        failures.is_empty(),
        "{} heteronym case(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// "macOS" against the real lexicon: the "OS" subtoken must be spelled out
/// ("oh ess"), never read as the `us_gold` word "os" (/ɑs/).
#[test]
#[ignore = "needs the kokoro engine pack on disk (BISQUE_VOICE_MODEL_DIR)"]
fn camelcase_acronyms_spell_out_against_the_real_lexicon() {
    let (g2p, _) = english_g2p();
    for surface in ["macOS", "iOS", "watchOS", "visionOS"] {
        let sentence = format!("It runs on {surface} today.");
        let ps = produced(&g2p, &sentence, surface);
        assert!(
            ps.contains("ˈɛs") && !ps.contains("ˈɑs"),
            "{surface}: expected a spelled-out OS, got {ps}"
        );
    }
}

/// An all-caps initialism is read as the word its lowercase spells — `US`
/// is "us" — so the author spells it out with a marker. `[US](U-S)` must
/// keep reading as the two letters against the real lexicon: @meta-muse's
/// "Introducing Muse" said "rolling out in the us" until its final slide was
/// republished with this marker (2026-09-28).
#[test]
#[ignore = "needs the kokoro engine pack on disk (BISQUE_VOICE_MODEL_DIR)"]
fn a_letter_marker_spells_out_an_initialism_against_the_real_lexicon() {
    let (g2p, _) = english_g2p();
    let plain = produced(&g2p, "It's rolling out in the US on iOS.", "US");
    println!("limit — unmarked US reads {plain}");
    let marked = produced(&g2p, "It's rolling out in the [US](U-S) on iOS.", "US");
    assert_eq!(marked, "jˌuˈɛs", "[US](U-S) should say the letters U.S.");
}
