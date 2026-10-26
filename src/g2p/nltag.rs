//! Context-aware lexical-class tags from Apple's NLTagger — macOS only.
//!
//! The rule tagger abstains (`Xx`) wherever one-word context is not enough
//! ("the channel where your real users live"), and abstention selects the
//! DEFAULT reading — wrong in every context that needed the other one. The
//! reference Swift engine resolves these with `NLTagger`; this module gives
//! the Rust engine the same authority on macOS, but more conservatively:
//! NLTagger is consulted only for POS-conditioned, all-lowercase words the
//! rules abstained on. A word the rules already resolved keeps its rule tag
//! (rules carry the VBD/VBN fine keys NLTagger's lexical classes cannot
//! express), and an ordinary word's tag cannot change at all.
//!
//! Known limit, inherited from NLTagger and shared with the Swift engine:
//! object-complement adjectives read as verbs ("keep the servers live").

use objc2::AnyThread;
use objc2_foundation::{NSArray, NSString};
use objc2_natural_language::{NLTagSchemeLexicalClass, NLTagger, NLTokenUnit};

use super::tagger::Tag;
use super::Token;

/// Map an NLTag lexical class onto the coarse buckets the lexicon keys by.
/// Anything else (pronoun, particle, `nil`…) is "no opinion".
fn class_to_tag(class: &str) -> Option<Tag> {
    match class {
        "Noun" => Some(Tag::Noun),
        "Verb" => Some(Tag::Verb),
        "Adjective" => Some(Tag::Adjective),
        "Adverb" => Some(Tag::Adverb),
        _ => None,
    }
}

/// Refine the flat tag sequence (one tag per subtoken across `groups`) with
/// NLTagger. Only `Xx` slots holding an all-lowercase POS-conditioned word are
/// consulted, so refinement can only turn an abstention into a context
/// reading.
pub fn refine(
    seq: Vec<Tag>,
    groups: &[Vec<Token>],
    pos_conditioned: &dyn Fn(&str) -> bool,
) -> Vec<Tag> {
    // Rebuild the text NLTagger sees — each group's subtokens concatenated,
    // followed by the (already-normalized, single-space) whitespace recorded
    // on its last subtoken — while collecting each candidate's UTF-16 offset,
    // which is what NLTagger's characterIndex counts.
    let mut text = String::new();
    let mut offset_utf16 = 0usize;
    // (flat index, utf16 offset of the word's first character)
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    let mut flat = 0usize;
    for group in groups {
        for token in group {
            let word = token.text.as_str();
            if seq[flat] == Tag::Xx && word == word.to_lowercase() && pos_conditioned(word) {
                candidates.push((flat, offset_utf16));
            }
            text.push_str(word);
            offset_utf16 += word.encode_utf16().count();
            flat += 1;
        }
        if let Some(last) = group.last() {
            text.push_str(&last.whitespace);
            offset_utf16 += last.whitespace.encode_utf16().count();
        }
    }
    if candidates.is_empty() {
        return seq;
    }

    let mut seq = seq;
    unsafe {
        // The scheme constant is weak-linked; it exists on every macOS this
        // binary supports, but stay a no-op rather than crash if it doesn't.
        let Some(scheme) = NLTagSchemeLexicalClass else {
            return seq;
        };
        let schemes = NSArray::from_slice(&[scheme]);
        let tagger = NLTagger::initWithTagSchemes(NLTagger::alloc(), &schemes);
        tagger.setString(Some(&NSString::from_str(&text)));
        for (flat, offset) in candidates {
            let class = tagger.tagAtIndex_unit_scheme_tokenRange(
                offset,
                NLTokenUnit::Word,
                scheme,
                std::ptr::null_mut(),
            );
            if let Some(class) = class {
                if let Some(tag) = class_to_tag(&class.to_string()) {
                    seq[flat] = tag;
                }
            }
        }
    }
    seq
}
