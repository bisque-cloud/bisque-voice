//! Coarse part-of-speech tagging, without a POS model.
//!
//! The Misaki pipeline needs a tag for two things: routing in `retokenize`
//! (punctuation vs number vs currency vs word), and picking a pronunciation for
//! the small set of lexicon entries that are POS-conditioned. macOS gets its
//! tags from Apple's `NLTagger`; there is no equivalent to link against on
//! Linux or Windows, so this is a rule-based stand-in.
//!
//! **Design rule: be confident or say `Xx`.** `Lexicon::parent_tag` collapses
//! every tag into five buckets, and an unknown bucket selects the entry's
//! `DEFAULT` pronunciation — which is the common reading. Guessing wrong picks
//! a *rarer* reading, so an uncertain guess is strictly worse than abstaining.
//!
//! Word-in-isolation confidence is not the whole story, though. The ~790
//! POS-conditioned lexicon entries — "present", "record", "live", "read" — are
//! exactly the words a single-token tagger must abstain on, so they all fell to
//! `DEFAULT` and every verb use of a noun-defaulted heteronym was wrong.
//! `NLTagger` on macOS resolves these from sentence context; `tag_sequence`
//! below is the cross-platform counterpart: a small set of high-precision
//! context rules ("will ___" is a verb, "the ___ of" is a noun) that fire
//! *only* on POS-conditioned headwords, so an ordinary word's tag — and
//! therefore its pronunciation — cannot change. Where no rule fires the word
//! still falls to `DEFAULT`: abstention stays the default posture, context
//! just shrinks how often it is needed.
//!
//! The routing half is not a guess at all: punctuation, digits and currency
//! symbols are decided from character classes, which is also how the vendored
//! Swift patches decide them (Apple's tags disagreed with spaCy's there, and
//! the mismatch silently deleted numbers — see `packages/kokoro-swift/VENDORED.md`).

/// Coarse classes, matching what `Lexicon::parent_tag` can act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    Noun,
    Verb,
    /// A finite past-tense verb (Penn `VBD`) — "she read the book". Collapses
    /// to `VERB` unless the entry carries the fine key, as `us_gold` does for
    /// "read"/"reread"/"wound"/"used".
    VerbPast,
    /// A past participle (Penn `VBN`) — "the report was read aloud".
    VerbPastPart,
    Adjective,
    Adverb,
    Determiner,
    Preposition,
    Pronoun,
    Conjunction,
    Number,
    Punctuation,
    Currency,
    /// Not confidently classified. Selects a lexicon entry's `DEFAULT`.
    Xx,
}

impl Tag {
    /// The five buckets POS-conditioned lexicon entries are keyed by.
    pub fn parent(self) -> &'static str {
        match self {
            Tag::Verb | Tag::VerbPast | Tag::VerbPastPart => "VERB",
            Tag::Noun => "NOUN",
            Tag::Adverb => "ADV",
            Tag::Adjective => "ADJ",
            _ => "XX",
        }
    }

    /// The exact Penn Treebank key, for the few entries keyed finer than the
    /// parent buckets ("read" has `VBD`/`VBN`/`VBP` alongside `DEFAULT`).
    /// Python Misaki tries the exact spaCy tag before the parent; the Swift
    /// port lost that by collapsing straight to the parent, which made "read"
    /// unreachable as /ɹɛd/ on every platform. This restores the Python order.
    pub fn fine(self) -> Option<&'static str> {
        match self {
            Tag::VerbPast => Some("VBD"),
            Tag::VerbPastPart => Some("VBN"),
            _ => None,
        }
    }

    pub fn is_proper_noun(self, text: &str) -> bool {
        self == Tag::Noun && text.chars().next().is_some_and(char::is_uppercase)
    }
}

const DETERMINERS: &[&str] = &[
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "each",
    "every",
    "some",
    "any",
    "no",
    "another",
    "both",
    "either",
    "neither",
    "which",
    "whatever",
    "whichever",
];
const PREPOSITIONS: &[&str] = &[
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "as", "into", "like",
    "through", "after", "over", "between", "out", "against", "during", "without", "before",
    "under", "around", "among", "across", "behind", "beyond", "within", "along", "upon",
];
const PRONOUNS: &[&str] = &[
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "who",
    "whom",
    "whose",
    "whoever",
    "whomever",
    "what",
];
const CONJUNCTIONS: &[&str] = &[
    "and", "or", "but", "nor", "yet", "so", "because", "although", "though", "if", "while", "when",
    "whenever", "since", "unless", "until", "whether",
];
const AUX_VERBS: &[&str] = &[
    "am", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has",
    "had", "will", "would", "can", "could", "shall", "should", "may", "might", "must",
];
const ADVERBS: &[&str] = &[
    "not", "very", "too", "also", "just", "only", "even", "still", "already", "always", "never",
    "often", "again", "here", "there", "now", "then", "when", "where", "why", "how", "well",
    "more", "most", "less", "least", "quite", "rather", "almost",
];

/// Currency symbols the lexicon knows how to attach a unit for.
pub fn is_currency(text: &str) -> bool {
    matches!(text, "$" | "£" | "€")
}

/// A token that is entirely digits, separators and an optional magnitude
/// suffix. Deliberately textual — see the module docs.
pub fn looks_numeric(text: &str) -> bool {
    !text.is_empty()
        && text.chars().any(|c| c.is_numeric())
        && text.chars().all(|c| c.is_numeric() || c == ',' || c == '.')
}

pub fn tag(text: &str) -> Tag {
    if text.is_empty() {
        return Tag::Xx;
    }
    if is_currency(text) {
        return Tag::Currency;
    }
    if looks_numeric(text) {
        return Tag::Number;
    }
    if text
        .chars()
        .all(|c| c.is_ascii_punctuation() || "—–…“”‘’«»".contains(c))
    {
        return Tag::Punctuation;
    }

    let lower = text.to_lowercase();
    if DETERMINERS.contains(&lower.as_str()) {
        return Tag::Determiner;
    }
    if PRONOUNS.contains(&lower.as_str()) {
        return Tag::Pronoun;
    }
    // "to" and "as" are prepositions here; the Misaki special cases that care
    // about them check the surface text as well.
    if PREPOSITIONS.contains(&lower.as_str()) {
        return Tag::Preposition;
    }
    if CONJUNCTIONS.contains(&lower.as_str()) {
        return Tag::Conjunction;
    }
    if AUX_VERBS.contains(&lower.as_str()) {
        return Tag::Verb;
    }
    if ADVERBS.contains(&lower.as_str()) {
        return Tag::Adverb;
    }

    // Open-class morphology. Only suffixes that are strongly diagnostic — a
    // weak guess is worse than Xx (see the module docs).
    if lower.ends_with("ly") && lower.len() > 4 {
        return Tag::Adverb;
    }
    for suffix in [
        "tion", "sion", "ment", "ness", "ity", "ance", "ence", "ship", "hood",
    ] {
        if lower.ends_with(suffix) {
            return Tag::Noun;
        }
    }
    // "-ate" is deliberately absent: it is the one verb suffix that is NOT
    // diagnostic. 91 of the POS-conditioned heteronyms end in -ate
    // ("appropriate", "separate", "alternate", "graduate"…) and for most of
    // them the -ate form is the *adjective or noun* — tagging them Verb chose
    // the rarer reading in every sentence, the exact failure this module's
    // design rule exists to prevent. For genuinely verbal -ate words
    // ("generate", "activate") the tag is irrelevant: their lexicon entries
    // are plain strings.
    for suffix in ["ize", "ise", "ify"] {
        if lower.ends_with(suffix) && lower.len() > 5 {
            return Tag::Verb;
        }
    }
    for suffix in ["ous", "ful", "ive", "able", "ible", "al", "ic"] {
        if lower.ends_with(suffix) && lower.len() > 4 {
            return Tag::Adjective;
        }
    }
    // A capitalized word mid-sentence is very likely a proper noun, and the
    // lexicon uses that to decide whether to spell it out.
    if text.chars().next().is_some_and(char::is_uppercase)
        && text
            .chars()
            .skip(1)
            .all(|c| c.is_lowercase() || !c.is_alphabetic())
    {
        return Tag::Noun;
    }

    Tag::Xx
}

// ── sentence context ─────────────────────────────────────────────────────────
//
// The word lists below are the context *signals*, distinct from the
// closed-class membership lists above. They are deliberately narrow: every
// entry is a word that almost never means anything else in the position the
// rule reads it in. Precision over recall — a rule that misfires re-creates
// the wrong-guess problem `tag` abstains to avoid.

/// Words after which a bare heteronym is an infinitive or finite verb:
/// modals, do-support, "to", and polite imperative lead-ins.
const VERB_TRIGGERS: &[&str] = &[
    "will",
    "would",
    "can",
    "could",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "do",
    "does",
    "did",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "couldn't",
    "shouldn't",
    "can't",
    "cannot",
    "to",
    "please",
    "let",
    "lets",
    "let's",
    "help",
];
/// Third-person-singular subjects. A *bare* heteronym after one is finite —
/// and because the 3sg present would carry "-s" ("she reads"), the bare form
/// is specifically past tense: "she read", "he wound".
const SUBJECTS_3SG: &[&str] = &["he", "she", "it"];
/// Coordinators that chain a heteronym onto an earlier predicate: "read the
/// history and present a review". Only safe together with a following object
/// determiner/pronoun (see the rule) — a bare "and present" could equally be
/// noun coordination ("past and present").
const COORDINATORS: &[&str] = &["and", "or", "then"];
/// Non-3sg subject pronouns: the bare form after these is a plain verb.
const SUBJECTS_PLAIN: &[&str] = &["i", "you", "we", "they"];
const BE_FORMS: &[&str] = &["am", "is", "are", "was", "were", "be", "been", "being"];
const HAVE_FORMS: &[&str] = &["have", "has", "had", "having"];
/// The heteronyms whose lexicon entries carry a `VBN` key, so a passive
/// ("was read aloud") can actually select it. For anything else "be + word"
/// is more often predicate adjective ("is present") and must stay `DEFAULT`.
const VBN_HETERONYMS: &[&str] = &["read", "reread", "wound"];
/// Determiners that are unambiguously pre-nominal. "some"/"these"/"each" are
/// excluded: they stand alone as subjects ("some live in the city"), where
/// tagging the next word Noun would be exactly backwards.
const NOUN_DETERMINERS: &[&str] = &[
    "the", "a", "an", "every", "no", "another", "my", "your", "his", "her", "its", "our", "their",
];
/// Prepositions that put a following bare heteronym in noun position
/// ("in use", "on record", "of note").
const NOUN_PREPOSITIONS: &[&str] = &[
    "of", "in", "on", "at", "by", "for", "with", "from", "into", "without", "during",
];
/// First words of a plausible imperative *object* — "Record the meeting",
/// "Present your findings", "Close it".
const IMPERATIVE_OBJECTS: &[&str] = &[
    "the", "a", "an", "this", "that", "these", "those", "every", "no", "another", "my", "your",
    "his", "her", "its", "our", "their", "some", "any", "all", "each", "me", "us", "them", "him",
    "it", "more",
];

/// Whether `lower` belongs to any closed class the single-token tagger already
/// decides confidently. Context rules never touch these.
fn closed_class(lower: &str) -> bool {
    DETERMINERS.contains(&lower)
        || PRONOUNS.contains(&lower)
        || PREPOSITIONS.contains(&lower)
        || CONJUNCTIONS.contains(&lower)
        || AUX_VERBS.contains(&lower)
        || ADVERBS.contains(&lower)
}

/// Tag a whole token sequence, resolving POS-conditioned heteronyms from
/// sentence context.
///
/// `pos_conditioned` reports whether a word has a POS-keyed lexicon entry —
/// i.e. whether its pronunciation depends on the tag at all. Context rules
/// fire **only** for those words; every other token keeps exactly the tag
/// `tag()` gives it, so this pass cannot change the pronunciation of any
/// ordinary word. Where no rule fires, a heteronym also keeps its `tag()`
/// result (usually `Xx` → the entry's `DEFAULT`, the common reading).
///
/// The rules read one word of left context — with intervening adverbs skipped,
/// so "will not record" still sees "will" — plus one word of right context for
/// the noun-head and imperative rules. Any punctuation clears the left
/// context: rules never fire across a clause boundary.
pub fn tag_sequence(words: &[&str], pos_conditioned: &dyn Fn(&str) -> bool) -> Vec<Tag> {
    let mut out: Vec<Tag> = words.iter().map(|w| tag(w)).collect();
    // The last non-adverb word before the current token, lowercased; None at
    // the start and after any punctuation.
    let mut prev: Option<String> = None;
    for i in 0..words.len() {
        let word = words[i];
        let base = out[i];
        let lower = word.to_lowercase();
        let eligible = !matches!(base, Tag::Punctuation | Tag::Number | Tag::Currency)
            && !closed_class(&lower)
            // ALLCAPS is acronym territory; its tag drives spell-out, not
            // reading choice, and must stay what `tag()` said.
            && word != word.to_uppercase()
            && pos_conditioned(&lower);

        if eligible {
            let p = prev.as_deref();
            let next = words.get(i + 1).map(|n| n.to_lowercase());
            // "end of utterance", punctuation and closed-class words all mean
            // the current word is its phrase's head, not a modifier of what
            // follows.
            let next_is_head_boundary = match next.as_deref() {
                None => true,
                Some(n) => tag(n) == Tag::Punctuation || closed_class(n),
            };
            let chosen = if p.is_some_and(|p| {
                VERB_TRIGGERS.contains(&p) || p.ends_with("'ll") || p.ends_with("'d")
            }) {
                Some(Tag::Verb)
            } else if p.is_some_and(|p| SUBJECTS_3SG.contains(&p)) {
                Some(Tag::VerbPast)
            } else if p.is_some_and(|p| SUBJECTS_PLAIN.contains(&p)) {
                Some(Tag::Verb)
            } else if p.is_some_and(|p| HAVE_FORMS.contains(&p)) {
                Some(Tag::VerbPastPart)
            } else if p.is_some_and(|p| BE_FORMS.contains(&p))
                && VBN_HETERONYMS.contains(&lower.as_str())
            {
                Some(Tag::VerbPastPart)
            } else if p.is_some_and(|p| {
                NOUN_DETERMINERS.contains(&p) || NOUN_PREPOSITIONS.contains(&p)
            }) && next_is_head_boundary
            {
                Some(Tag::Noun)
            } else if p.is_none()
                && next
                    .as_deref()
                    .is_some_and(|n| IMPERATIVE_OBJECTS.contains(&n))
            {
                // Sentence-initial word directly followed by an object's
                // determiner/pronoun: an imperative — "Record the meeting".
                Some(Tag::Verb)
            } else if p.is_some_and(|p| COORDINATORS.contains(&p))
                && next
                    .as_deref()
                    .is_some_and(|n| IMPERATIVE_OBJECTS.contains(&n))
            {
                // Coordinated predicate with an object: "read the history and
                // present a review". The object determiner is what rules out
                // noun coordination — "past and present" has no determiner.
                Some(Tag::Verb)
            } else {
                None
            };
            if let Some(tag) = chosen {
                out[i] = tag;
            }
        }

        if base == Tag::Punctuation {
            prev = None;
        } else if base != Tag::Adverb {
            prev = Some(lower);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_classes_come_from_characters_not_guesses() {
        assert_eq!(tag("$"), Tag::Currency);
        assert_eq!(tag("£"), Tag::Currency);
        assert_eq!(tag("4.50"), Tag::Number);
        assert_eq!(tag("1,250"), Tag::Number);
        assert_eq!(tag("."), Tag::Punctuation);
        assert_eq!(tag("—"), Tag::Punctuation);
        assert_eq!(tag("…"), Tag::Punctuation);
    }

    #[test]
    fn magnitude_tokens_are_not_numeric_on_their_own() {
        // "4.2M" is handled by Lexicon::split_magnitude, not by looks_numeric.
        assert!(!looks_numeric("4.2M"));
        assert!(looks_numeric("4.2"));
        assert!(!looks_numeric("abc"));
        assert!(!looks_numeric(""));
    }

    #[test]
    fn closed_class_words_are_confident() {
        assert_eq!(tag("the"), Tag::Determiner);
        assert_eq!(tag("of"), Tag::Preposition);
        assert_eq!(tag("and"), Tag::Conjunction);
        assert_eq!(tag("is"), Tag::Verb);
        assert_eq!(tag("their"), Tag::Pronoun);
        assert_eq!(tag("never"), Tag::Adverb);
    }

    #[test]
    fn diagnostic_morphology_only() {
        assert_eq!(tag("quickly"), Tag::Adverb);
        assert_eq!(tag("presentation"), Tag::Noun);
        assert_eq!(tag("beautiful"), Tag::Adjective);
        // No confident signal: abstain rather than guess a rarer reading.
        assert_eq!(tag("record"), Tag::Xx);
        assert_eq!(tag("lead"), Tag::Xx);
    }

    #[test]
    fn parent_buckets_match_the_lexicon_keys() {
        assert_eq!(Tag::Verb.parent(), "VERB");
        assert_eq!(Tag::Noun.parent(), "NOUN");
        assert_eq!(Tag::Adverb.parent(), "ADV");
        assert_eq!(Tag::Adjective.parent(), "ADJ");
        assert_eq!(Tag::Xx.parent(), "XX");
        assert_eq!(Tag::Determiner.parent(), "XX");
        // Fine verb tags still collapse to the VERB bucket…
        assert_eq!(Tag::VerbPast.parent(), "VERB");
        assert_eq!(Tag::VerbPastPart.parent(), "VERB");
        // …but expose the exact Penn key for the entries that carry one.
        assert_eq!(Tag::VerbPast.fine(), Some("VBD"));
        assert_eq!(Tag::VerbPastPart.fine(), Some("VBN"));
        assert_eq!(Tag::Verb.fine(), None);
        assert_eq!(Tag::Noun.fine(), None);
    }

    #[test]
    fn ate_is_not_a_verb_suffix() {
        // "-ate" heteronyms are usually adjectives/nouns; forcing Verb chose
        // the rarer reading in every sentence. Truly verbal -ate words have
        // string entries, so Xx costs them nothing.
        assert_eq!(tag("separate"), Tag::Xx);
        assert_eq!(tag("appropriate"), Tag::Xx);
        // The diagnostic verb suffixes stay.
        assert_eq!(tag("organize"), Tag::Verb);
        assert_eq!(tag("classify"), Tag::Verb);
    }

    // ── tag_sequence ────────────────────────────────────────────────────────

    /// A stand-in for `Lexicon::pos_conditioned`: the heteronyms used in the
    /// context tests below.
    fn het(word: &str) -> bool {
        [
            "present", "record", "read", "live", "lives", "close", "use", "wound", "separate",
            "estimate", "produce", "content", "used", "document",
        ]
        .contains(&word)
    }

    fn tags_for(sentence: &[&str]) -> Vec<Tag> {
        tag_sequence(sentence, &het)
    }

    #[test]
    fn context_resolves_verbs_after_modals_subjects_and_to() {
        assert_eq!(tags_for(&["He", "will", "present", "the", "plan"])[2], Tag::Verb);
        assert_eq!(tags_for(&["I", "live", "in", "the", "city"])[1], Tag::Verb);
        assert_eq!(tags_for(&["Please", "record", "the", "minutes"])[1], Tag::Verb);
        assert_eq!(tags_for(&["We", "need", "to", "document", "it"])[3], Tag::Verb);
        assert_eq!(tags_for(&["don't", "present", "without", "notes"])[1], Tag::Verb);
        assert_eq!(tags_for(&["we'll", "record", "the", "demo"])[1], Tag::Verb);
    }

    #[test]
    fn a_bare_form_after_a_3sg_subject_is_past_tense() {
        // "she reads" would be present; bare "she read" is necessarily past.
        assert_eq!(tags_for(&["She", "read", "the", "book"])[1], Tag::VerbPast);
        assert_eq!(tags_for(&["he", "wound", "the", "clock"])[1], Tag::VerbPast);
        // Non-3sg subjects are genuinely ambiguous — plain Verb, which falls
        // to the entry's DEFAULT for VBD-keyed words.
        assert_eq!(tags_for(&["they", "read", "it"])[1], Tag::Verb);
    }

    #[test]
    fn passives_and_perfects_reach_the_participle() {
        assert_eq!(tags_for(&["it", "was", "read", "aloud"])[2], Tag::VerbPastPart);
        assert_eq!(tags_for(&["she", "has", "read", "it"])[2], Tag::VerbPastPart);
        // "be + heteronym" without a VBN key is predicate-adjective territory:
        // "is present" must stay unresolved, not become a verb.
        assert_eq!(tags_for(&["he", "is", "present"])[2], Tag::Xx);
    }

    #[test]
    fn a_determiner_plus_phrase_head_is_a_noun() {
        // "their use of color" — head position, rule fires.
        assert_eq!(tags_for(&["their", "use", "of", "color"])[1], Tag::Noun);
        assert_eq!(tags_for(&["the", "close", "of", "trading"])[1], Tag::Noun);
        assert_eq!(tags_for(&["the", "content", "is", "king"])[1], Tag::Noun);
        // "a close friend" — an open-class word follows, so "close" may be a
        // modifier: abstain (DEFAULT is the adjective reading, which is right).
        assert_eq!(tags_for(&["a", "close", "friend"])[1], Tag::Xx);
        assert_eq!(tags_for(&["the", "record", "label"])[1], Tag::Xx);
    }

    #[test]
    fn sentence_initial_imperatives_are_verbs() {
        assert_eq!(tags_for(&["Record", "the", "meeting"])[0], Tag::Verb);
        assert_eq!(tags_for(&["Close", "it", "now"])[0], Tag::Verb);
        // A capitalized heteronym NOT in imperative shape keeps tag()'s
        // proper-noun reading of the capitalization.
        assert_eq!(tags_for(&["Live", "music", "nightly"])[0], Tag::Noun);
    }

    #[test]
    fn adverbs_are_transparent_and_punctuation_resets_context() {
        // "must not separate" — the negation does not hide the modal.
        assert_eq!(tags_for(&["We", "must", "not", "separate", "them"])[3], Tag::Verb);
        // Punctuation is a clause boundary: no rule reaches across it.
        assert_eq!(tags_for(&["will", ",", "record"])[2], Tag::Xx);
    }

    #[test]
    fn context_rules_only_touch_pos_conditioned_words() {
        // "plan" is not POS-conditioned: same tag with and without context.
        let with = tags_for(&["He", "will", "plan", "the", "work"]);
        assert_eq!(with[2], tag("plan"));
        // Fused-head determiners ("some") are not noun triggers, and closed-
        // class words are never re-tagged even if the lexicon keys them.
        assert_eq!(tags_for(&["Some", "live", "in", "the", "city"])[1], Tag::Xx);
        assert_eq!(tags_for(&["the", "used", "car"])[1], Tag::Xx);
        // ALLCAPS stays on the acronym path.
        assert_eq!(tags_for(&["will", "RECORD", "the"])[1], tag("RECORD"));
    }
}
