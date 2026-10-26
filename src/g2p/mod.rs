//! English grapheme-to-phoneme, cross-platform.
//!
//! Port of `EnglishG2P.swift` in `packages/kokoro-swift`, which is itself a
//! port of Python `misaki`. It reads the same Apache-2.0 lexicons the macOS
//! engine reads, so the two agree by construction.
//!
//! English does **not** go through eSpeak NG. Misaki reads English from its
//! lexicons and a neural fallback, so this does too; eSpeak NG (`espeak`) is
//! used only for the five languages Misaki phonemizes with it, and is
//! statically linked into the default build.
//!
//! Pipeline: tokenize → sub-tokenize → route (punctuation / symbol / currency)
//! → resolve right-to-left through the lexicon, so each token can see whether
//! the next sound is a vowel → neural fallback for what is left.

pub mod espeak;
pub mod fallback;
pub mod ja;
pub mod lexicon;
pub mod num2word;
pub mod pack;
pub mod router;
pub mod tagger;
pub mod zh;

#[cfg(test)]
mod eval_tests;
#[cfg(test)]
mod fixture_tests;

#[cfg(target_os = "macos")]
pub mod nltag;

use crate::pronounce::parse_override;
use lexicon::{Ctx, Lexicon};
use tagger::Tag;

/// One G2P token: surface text, its phonemes, and the whitespace that followed
/// it in the source. `timing` consumes exactly this shape.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Token {
    pub text: String,
    pub phonemes: Option<String>,
    pub whitespace: String,
    pub currency: Option<String>,
    /// Misaki's confidence rating; 1 means the neural fallback guessed.
    pub rating: Option<i32>,
    /// An author override written `[live](liv)`: say THIS instead of `text`.
    ///
    /// `text` stays the word a reader sees, so timings, transcripts and
    /// captions are unaffected; only synthesis sees the substitute. The
    /// `[live](/lˈɪv/)` form skips this and pins `phonemes` directly.
    pub spoken: Option<String>,
}

const PUNCTUATION: &str = ";:,.!?—…\"“”";
const SUBTOKEN_JUNK: &str = "',-._/‘’";

pub struct EnglishG2P {
    lexicon: Lexicon,
    fallback: Option<fallback::Fallback>,
}

impl EnglishG2P {
    pub fn new(lexicon: Lexicon, fallback: Option<fallback::Fallback>) -> Self {
        EnglishG2P { lexicon, fallback }
    }

    fn parse_override(chars: &[char], i: usize) -> Option<(String, Option<String>, Option<String>, usize)> {
        parse_override(chars, i)
    }

    /// Lexicon facts for one token: whether the lexicon holds the word, and
    /// how many distinct-sounding readings it holds for it.
    ///
    /// `None` for a token with no word in it — bare punctuation. Facts only:
    /// this stage cannot know how a word is supposed to sound, so deciding
    /// which words are wrong belongs to whoever reads the report.
    pub fn facts(&self, word: &str) -> Option<(bool, u32)> {
        let w = word.trim_matches(|c: char| PUNCTUATION.contains(c) || c == '\'');
        if w.is_empty() {
            return None;
        }
        Some((self.lexicon.is_known(w), self.lexicon.readings(w)))
    }

    /// Split into words plus the whitespace that follows each.
    fn tokenize(text: &str) -> Vec<Token> {
        let mut out: Vec<Token> = Vec::new();
        let mut current = String::new();
        let chars: Vec<char> = text.trim().chars().collect();
        let mut idx = 0usize;
        while idx < chars.len() {
            let ch = chars[idx];
            if ch == '[' {
                if let Some((display, ipa, spoken, next)) = Self::parse_override(&chars, idx) {
                    if !current.is_empty() {
                        out.push(Token {
                            text: std::mem::take(&mut current),
                            ..Token::default()
                        });
                    }
                    // Punctuation touching the marker belongs to the word, the
                    // same as anywhere else: `[live](liv).` has to tokenize
                    // like `live.` or the trailing stop becomes its own token,
                    // gaining a bare "." in the transcript and one extra word
                    // timing that every later cue is then off by.
                    let mut next = next;
                    let mut tail = String::new();
                    while let Some(c) = chars.get(next) {
                        if PUNCTUATION.contains(*c) {
                            tail.push(*c);
                            next += 1;
                        } else {
                            break;
                        }
                    }
                    let ipa = ipa.map(|p| format!("{p}{tail}"));
                    let spoken = spoken.map(|s| format!("{s}{tail}"));

                    let display = format!("{display}{tail}");
                    let words: Vec<&str> = display.split_whitespace().collect();
                    let said = ipa.as_deref().or(spoken.as_deref()).unwrap_or("");
                    let sections: Vec<&str> = said.split_whitespace().collect();
                    if words.len() > 1 && sections.len() == words.len() {
                        // One spoken section per shown word — `[October
                        // 2](October second)` — so each word is said and timed
                        // on its own, exactly as the unmarked sentence would be.
                        for (w, (word, section)) in words.iter().zip(&sections).enumerate() {
                            out.push(Token {
                                text: (*word).to_string(),
                                phonemes: ipa.as_ref().map(|_| (*section).to_string()),
                                spoken: spoken.as_ref().map(|_| (*section).to_string()),
                                whitespace: if w + 1 < words.len() { " ".into() } else { String::new() },
                                ..Token::default()
                            });
                        }
                    } else {
                        // Otherwise the override rides on the first word and
                        // the rest fall silent, as Misaki folds a multi-word
                        // link. The spaces between the shown words are not
                        // spoken separately: the override already carries its
                        // own, and a second one would be an extra model token
                        // the timestamp walk never accounts for. Only the last
                        // word keeps the whitespace that follows the marker.
                        for (w, word) in words.iter().enumerate() {
                            out.push(Token {
                                text: (*word).to_string(),
                                phonemes: if w == 0 {
                                    ipa.clone()
                                } else {
                                    Some(String::new())
                                },
                                spoken: if w == 0 { spoken.clone() } else { None },
                                whitespace: String::new(),
                                ..Token::default()
                            });
                        }
                    }
                    idx = next;
                    continue;
                }
            }
            idx += 1;
            if ch.is_whitespace() {
                if !current.is_empty() {
                    out.push(Token {
                        text: std::mem::take(&mut current),
                        ..Token::default()
                    });
                }
                // Any run of whitespace becomes exactly one space. Kokoro's
                // vocabulary has only ' ', so a double space or a newline would
                // contribute a different number of model tokens than the
                // timestamp walk accounts for, and every later cue would drift.
                if let Some(last) = out.last_mut() {
                    last.whitespace = " ".into();
                }
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            out.push(Token {
                text: current,
                ..Token::default()
            });
        }
        out
    }

    /// Break a word at acronym, number, sign and camelCase boundaries, matching
    /// Misaki's sub-tokenizer. Hand-rolled rather than regex so the boundaries
    /// stay explicit and testable.
    fn subtokenize(word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        let mut out: Vec<String> = Vec::new();
        let mut i = 0usize;
        while i < chars.len() {
            let c = chars[i];
            if c.is_numeric() {
                // A number run, including internal separators between digits.
                let mut j = i;
                while j < chars.len() {
                    let cj = chars[j];
                    if cj.is_numeric() {
                        j += 1;
                    } else if (cj == '.' || cj == ',')
                        && j + 1 < chars.len()
                        && chars[j + 1].is_numeric()
                    {
                        j += 1;
                    } else {
                        break;
                    }
                }
                out.push(chars[i..j].iter().collect());
                i = j;
            } else if c.is_alphabetic() {
                // A letter run, breaking before a lower→upper transition so
                // "camelCase" splits the way the lexicon expects.
                let mut j = i;
                while j < chars.len() && (chars[j].is_alphabetic() || chars[j] == '\'') {
                    if j > i && chars[j].is_uppercase() && chars[j - 1].is_lowercase() {
                        break;
                    }
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
            } else {
                // Runs of the same separator stay together ("--").
                let mut j = i;
                while j < chars.len() && chars[j] == c {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
            }
        }
        out
    }

    /// Resolve punctuation, symbols and currency, and group the sub-tokens of a
    /// word so the lexicon gets a chance at the whole thing before the pieces.
    fn retokenize(tokens: Vec<Token>) -> Vec<Vec<Token>> {
        let mut groups: Vec<Vec<Token>> = Vec::new();
        let mut currency: Option<String> = None;

        let texts: Vec<String> = tokens.iter().map(|t| t.text.clone()).collect();
        for (i, token) in tokens.iter().enumerate() {
            // An author override owns the whole word. Splitting it at a camelCase
            // or acronym boundary would drop the pin and hand the pieces back to
            // the lexicon, which is the guess the override exists to replace.
            if token.phonemes.is_some() || token.spoken.is_some() {
                groups.push(vec![token.clone()]);
                continue;
            }
            let parts = Self::subtokenize(&token.text);
            let last_part = parts.len().saturating_sub(1);
            let mut group: Vec<Token> = Vec::new();

            for (j, part) in parts.iter().enumerate() {
                let mut sub = Token {
                    text: part.clone(),
                    whitespace: if j == last_part {
                        token.whitespace.clone()
                    } else {
                        String::new()
                    },
                    ..Token::default()
                };
                let tag = tagger::tag(part);

                if tagger::is_currency(part) {
                    // Currency symbols are recognized by text, never by a POS
                    // tag — see VENDORED.md; a tag mismatch used to delete "$".
                    currency = Some(part.clone());
                    sub.phonemes = Some(String::new());
                    sub.rating = Some(4);
                } else if part == "-" || part == "–" || part == "—" {
                    // A dash standing alone in the text is a deliberate pause
                    // and keeps the em-dash phoneme. A hyphen INSIDE a word
                    // ("single-machine", "state-of-the-art") is only spelling:
                    // the em-dash phoneme there makes Kokoro insert ~500-900ms
                    // of dead air mid-word, so it becomes a plain word gap —
                    // the compound is spoken exactly like the spaced phrase.
                    // At a word edge ("co-") there is no second word to gap
                    // to, so it contributes nothing.
                    sub.phonemes = Some(if parts.len() == 1 {
                        "—".into()
                    } else if j > 0 && j < last_part {
                        " ".into()
                    } else {
                        String::new()
                    });
                    sub.rating = Some(3);
                } else if tag == Tag::Punctuation && lexicon::symbol_word(part).is_none() {
                    // Symbols (%, &, +, @) fall through to the lexicon, which
                    // speaks them.
                    sub.phonemes =
                        Some(part.chars().filter(|c| PUNCTUATION.contains(*c)).collect());
                    sub.rating = Some(4);
                } else if currency.is_some() {
                    let is_magnitude_suffix = part.chars().count() == 1
                        && lexicon::split_magnitude(&format!("0{part}")).is_some()
                        && j > 0
                        && tagger::looks_numeric(&parts[j - 1]);
                    if is_magnitude_suffix || lexicon::is_magnitude_word(part) {
                        sub.currency = currency.take();
                    } else if !tagger::looks_numeric(part) {
                        currency = None;
                    } else if j == last_part {
                        // The next source token still carries its trailing
                        // punctuation ("million."), so compare on its word part.
                        let next = texts
                            .get(i + 1)
                            .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_string());
                        let next_is_magnitude_word =
                            next.as_deref().is_some_and(lexicon::is_magnitude_word);
                        let next_is_numeric = next.as_deref().is_some_and(tagger::looks_numeric);
                        if !next_is_magnitude_word && !next_is_numeric {
                            sub.currency = currency.take();
                        }
                    }
                }
                group.push(sub);
            }
            groups.push(group);
        }
        groups
    }

    /// Update the right-to-left context from a token's phonemes.
    fn advance(ctx: Ctx, phonemes: Option<&str>, text: &str) -> Ctx {
        const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
        const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
        let mut vowel = ctx.future_vowel;
        if let Some(ps) = phonemes {
            for c in ps.chars() {
                if PUNCTUATION.contains(c) && c != '"' && c != '“' && c != '”' {
                    vowel = None;
                    break;
                }
                if VOWELS.contains(c) {
                    vowel = Some(true);
                    break;
                }
                if CONSONANTS.contains(c) {
                    vowel = Some(false);
                    break;
                }
            }
        }
        Ctx {
            future_vowel: vowel,
            future_to: matches!(text, "to" | "To" | "TO"),
        }
    }

    /// Phonemize `text`. Returns the tokens; `phoneme_string` joins them into
    /// what the model is fed.
    pub fn phonemize(&self, text: &str) -> Vec<Token> {
        let mut groups = Self::retokenize(Self::tokenize(text));

        // Resolve `[live](liv)` overrides: phonemize the substitute and pin the
        // result onto the token, whose `text` stays the word a reader sees. From
        // here on both override forms look identical to the resolution walk.
        for group in groups.iter_mut() {
            for token in group.iter_mut() {
                let Some(spoken) = token.spoken.clone() else {
                    continue;
                };
                // Section by section: a space in the substitute is a spoken
                // boundary the author asked for, and a hyphen inside a section
                // glues its pieces into one word.
                let said = spoken
                    .split_whitespace()
                    .map(|section| {
                        as_one_word(phoneme_string(&self.phonemize(section)).trim_end())
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                token.phonemes = Some(said);
                token.rating = Some(4);
            }
        }

        // Tags are context-resolved over the whole sequence, left to right,
        // BEFORE the right-to-left resolution walk below (which threads sound
        // context, not grammatical context). Only POS-conditioned heteronyms
        // can come back different from `tagger::tag` — see `tag_sequence`.
        let flat: Vec<&str> = groups
            .iter()
            .flatten()
            .map(|t| t.text.as_str())
            .collect();
        let seq = tagger::tag_sequence(&flat, &|w| self.lexicon.pos_conditioned(w));
        // On macOS, Apple's context-aware tagger fills the abstentions the
        // rules could not resolve — and nothing else (see `nltag`).
        #[cfg(target_os = "macos")]
        let seq = nltag::refine(seq, &groups, &|w| self.lexicon.pos_conditioned(w));
        let mut tags: Vec<Vec<Tag>> = Vec::with_capacity(groups.len());
        let mut taken = 0usize;
        for group in &groups {
            tags.push(seq[taken..taken + group.len()].to_vec());
            taken += group.len();
        }

        let mut resolved: Vec<Vec<Token>> = groups.clone();
        let mut ctx = Ctx::default();

        // Right to left, so each token knows what sound follows it.
        for (gi, group) in resolved.iter_mut().enumerate().rev() {
            // Try the whole word first: "3.5x" beats "3.5" + "x".
            //
            // Trailing punctuation is resolved in `retokenize`, before this
            // walk, so it is excluded here — otherwise it blocks the lookup
            // and the word falls back to its camelCase pieces. That is how
            // `iOS,` came out `ˈIˌOˈɛs` ("EYE. OH. ESS.") where `iOS` reads
            // `ˌIOˈɛs`: one comma, and the lexicon never saw the word.
            let word_len = group
                .iter()
                .rposition(|t| t.phonemes.is_none())
                .map_or(0, |i| i + 1);
            let word = &group[..word_len];
            let merged_text: String = word.iter().map(|t| t.text.as_str()).collect();
            let unresolved = word.iter().all(|t| t.phonemes.is_none());
            let currency = group.iter().find_map(|t| t.currency.clone());

            if unresolved && word.len() > 1 {
                if let Some((ps, rating)) = self.lexicon.transcribe(
                    &merged_text,
                    tagger::tag(&merged_text),
                    ctx,
                    currency.as_deref(),
                    None,
                    true,
                ) {
                    ctx = Self::advance(ctx, Some(&ps), &merged_text);
                    group[0].phonemes = Some(ps);
                    group[0].rating = Some(rating);
                    for t in group.iter_mut().take(word_len).skip(1) {
                        t.phonemes = Some(String::new());
                        t.rating = Some(rating);
                    }
                    continue;
                }
            }

            // An ALLCAPS piece of a mixed-case word ("macOS" → "mac"+"OS") is
            // certainly an acronym — no English word appears as the ALLCAPS
            // tail of a camelCase split — so it is spelled out rather than
            // allowed to match a dictionary word ("OS" must not read as "os").
            let mixed_case_split = group.len() > 1
                && group
                    .iter()
                    .any(|t| t.text.chars().any(char::is_lowercase));

            for (idx, token) in group.iter_mut().enumerate().rev() {
                if token.phonemes.is_some() {
                    ctx = Self::advance(ctx, token.phonemes.as_deref(), &token.text);
                    continue;
                }
                if mixed_case_split
                    && token.text.chars().count() >= 2
                    && token.text.chars().all(|c| c.is_ascii_uppercase())
                {
                    if let Some((ps, rating)) = self.lexicon.spell_out(&token.text) {
                        ctx = Self::advance(ctx, Some(&ps), &token.text);
                        token.phonemes = Some(ps);
                        token.rating = Some(rating);
                        continue;
                    }
                }
                let tag = tags[gi][idx];
                let resolved_ps = self
                    .lexicon
                    .transcribe(
                        &token.text,
                        tag,
                        ctx,
                        token.currency.as_deref(),
                        None,
                        idx == 0,
                    )
                    .or_else(|| {
                        if token.text.chars().all(|c| SUBTOKEN_JUNK.contains(c)) {
                            Some((String::new(), 3))
                        } else {
                            self.fallback
                                .as_ref()
                                .and_then(|f| f.phonemize(&token.text))
                        }
                    });
                match resolved_ps {
                    Some((ps, rating)) => {
                        ctx = Self::advance(ctx, Some(&ps), &token.text);
                        token.phonemes = Some(ps);
                        token.rating = Some(rating);
                    }
                    None => {
                        // Nothing could pronounce it. An empty string here would
                        // silently delete the word, so mark it and let the
                        // caller decide — never drop it on the floor.
                        token.phonemes = None;
                        token.rating = None;
                    }
                }
            }
        }

        // Collapse each group back to one token per source word.
        resolved
            .into_iter()
            .filter_map(|group| {
                let text: String = group.iter().map(|t| t.text.as_str()).collect();
                let whitespace = group
                    .last()
                    .map(|t| t.whitespace.clone())
                    .unwrap_or_default();
                let any = group.iter().any(|t| t.phonemes.is_some());
                if !any {
                    return Some(Token {
                        text,
                        phonemes: None,
                        whitespace,
                        ..Token::default()
                    });
                }
                let phonemes: String = group
                    .iter()
                    .filter_map(|t| t.phonemes.as_deref())
                    .collect::<Vec<_>>()
                    .concat();
                // Kokoro's vocabulary uses T for the flap and t for the glottal
                // stop, matching the Swift engine's final substitution.
                let phonemes = phonemes.replace('ɾ', "T").replace('ʔ', "t");
                Some(Token {
                    text,
                    phonemes: Some(phonemes),
                    whitespace,
                    currency: None,
                    rating: group.iter().filter_map(|t| t.rating).min(),
                    // Overrides are already resolved into `phonemes` by here.
                    spoken: None,
                })
            })
            .collect()
    }
}

/// Turn one whole-text phoneme string into `Token`s, preserving word timings.
///
/// English is the odd one out: its G2P walks source tokens and can hand back a
/// phoneme run per word. Misaki's other engines — eSpeak, `ZHG2P`, `JAG2P` —
/// phonemize the *entire* text in one pass and return a single string, because
/// the result is context-dependent (French liaison, jieba segmentation, MeCab's
/// lattice). Phonemizing word by word to recover a token list would produce
/// different phonemes, which is precisely what "port Misaki exactly" forbids.
///
/// So the split happens on the output instead. Two invariants hold no matter
/// what the counts do, and both are load-bearing:
///
/// 1. **`phoneme_string(&tokens) == phonemes`, exactly.** Every space in the
///    phoneme string becomes exactly one token boundary, including the empty
///    run a doubled space implies. `model.rs` walks the duration tensor
///    assuming one model token per phoneme character plus one per space; a
///    single character of drift silently desynchronizes every later cue.
/// 2. **One `Word` per token.** Every token carries `Some(phonemes)`, so
///    `timing::word_timings` emits one timing each and `narrate`'s
///    `timings == tokens` check cannot trip.
///
/// The surface text is the part that can degrade. When eSpeak's word count
/// matches the source's, each token takes its source word and
/// `reconcile::align_by_char_coverage` finds an exact character match, so every
/// source word gets a real start and end. When the counts differ — eSpeak
/// expanded "19,99" into four spoken words, say — there is no honest per-word
/// mapping, so the text is left empty and the reconciler falls back to
/// distributing the utterance proportionally. That path keeps the word *count*
/// exact, so cues stay on the right words, and reports `exact: false` rather
/// than inventing an alignment.
pub fn align_tokens(text: &str, phonemes: &str) -> Vec<Token> {
    if phonemes.is_empty() {
        return Vec::new();
    }
    let parts: Vec<&str> = phonemes.split(' ').collect();
    let source: Vec<&str> = text.split_whitespace().collect();
    let aligned = source.len() == parts.len();

    let last = parts.len() - 1;
    parts
        .iter()
        .enumerate()
        .map(|(i, part)| Token {
            text: if aligned {
                source[i].to_string()
            } else {
                String::new()
            },
            phonemes: Some((*part).to_string()),
            whitespace: if i == last { String::new() } else { " ".into() },
            ..Token::default()
        })
        .collect()
}

/// Join tokens into the phoneme string fed to the model.
/// Fold one section of a substitute into a single spoken word.
///
/// A section is what the author wrote between spaces, and a hyphen inside it
/// is the ask to say those pieces as one word. `read-me` is phonemized as two
/// words and arrives here as `ɹˈid mˈi`; the interior boundary goes and every
/// primary stress but the last drops to secondary, which is the shape the
/// lexicon itself uses for a word of that size (`iOS` is `ˌIOˈɛs`).
///
/// A space between sections is untouched, because the author put it there:
/// `[README.md](read-me dot em-dee)` is said "readme — dot — emdee", the way a
/// person reads the filename out loud.
///
/// Trailing punctuation belongs to the sentence, not to the section, so it
/// stays; punctuation inside a section goes with the boundary it marked.
fn as_one_word(said: &str) -> String {
    let chars: Vec<char> = said.chars().collect();
    let split = chars
        .iter()
        .rposition(|c| !PUNCTUATION.contains(*c))
        .map_or(0, |i| i + 1);
    let (head, tail) = chars.split_at(split);
    let joined: String = head
        .iter()
        .filter(|c| !c.is_whitespace() && !PUNCTUATION.contains(**c))
        .collect();
    let mut out: String = match joined.rfind('ˈ') {
        Some(last) => joined
            .char_indices()
            .map(|(i, c)| if c == 'ˈ' && i != last { 'ˌ' } else { c })
            .collect(),
        None => joined,
    };
    out.extend(tail.iter().copied());
    out
}

pub fn phoneme_string(tokens: &[Token]) -> String {
    tokens
        .iter()
        .map(|t| format!("{}{}", t.phonemes.as_deref().unwrap_or(""), t.whitespace))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    //
    // Author pronunciation overrides — `[display](spoken)`.
    //
    // The contract these guard: `text` is the word a reader sees and NEVER
    // changes, so transcripts, captions, search and word timings are untouched.
    // Only the phonemes handed to synthesis differ.
    //

    #[test]
    fn ipa_override_pins_phonemes_and_leaves_the_display_word_alone() {
        let toks = EnglishG2P::tokenize("where tensors [live](/lˈɪv/) now");
        let live = toks.iter().find(|t| t.text == "live").unwrap();
        assert_eq!(live.phonemes.as_deref(), Some("lˈɪv"));
        assert_eq!(live.spoken, None);
        let words: Vec<&str> = toks.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            words,
            vec!["where", "tensors", "live", "now"],
            "the marker must not survive into the words a reader sees"
        );
    }

    #[test]
    fn respelling_override_records_the_substitute_not_the_display_word() {
        let toks = EnglishG2P::tokenize("where tensors [live](liv) now");
        let live = toks.iter().find(|t| t.text == "live").unwrap();
        assert_eq!(live.spoken.as_deref(), Some("liv"));
        assert_eq!(live.phonemes, None, "resolved later, in phonemize");
    }

    #[test]
    fn a_multi_word_override_carries_on_its_first_word() {
        let toks = EnglishG2P::tokenize("say [New York City](/nˈu jˈɔɹk/) twice");
        let words: Vec<&str> = toks.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(words, vec!["say", "New", "York", "City", "twice"]);
        assert_eq!(toks[1].phonemes.as_deref(), Some("nˈu jˈɔɹk"));
        assert_eq!(
            toks[2].phonemes.as_deref(),
            Some(""),
            "trailing words of an override fall silent, like a merged group"
        );
        // The override's own space is the only one spoken. A space after
        // "New" as well would be one more model token than the walk counts.
        let ws: Vec<&str> = toks.iter().map(|t| t.whitespace.as_str()).collect();
        assert_eq!(ws, vec![" ", "", "", " ", ""]);
    }

    #[test]
    fn a_multi_word_override_with_a_section_per_word_reads_like_plain_text() {
        // The October 5 Hacker News slide that failed narration three times.
        let toks = EnglishG2P::tokenize("on Friday, [October 2](October second).");
        let got: Vec<(&str, Option<&str>, &str)> = toks
            .iter()
            .map(|t| (t.text.as_str(), t.spoken.as_deref(), t.whitespace.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("on", None, " "),
                ("Friday,", None, " "),
                ("October", Some("October"), " "),
                ("2.", Some("second."), ""),
            ]
        );
        let toks = EnglishG2P::tokenize("say [New York](/nˈu jˈɔɹk/) twice");
        let ipa: Vec<Option<&str>> = toks.iter().map(|t| t.phonemes.as_deref()).collect();
        assert_eq!(ipa, vec![None, Some("nˈu"), Some("jˈɔɹk"), None]);
    }

    #[test]
    fn a_multi_word_override_feeds_the_model_one_space_per_boundary() {
        // `see [New York](new york).` fed the model "sˈi nˈu jˈɔɹk. " — a
        // trailing space the timestamp walk could not place, so it produced 2
        // timings for 3 tokens and bisque-voice 0.1.34 refused to narrate.
        let lex = Lexicon::new(
            serde_json::json!({ "see": "sˈi", "new": "nˈu", "york": "jˈɔɹk", "city": "sˈɪɾi" }),
            serde_json::json!({}),
            false,
        );
        let g2p = EnglishG2P::new(lex, None);
        let fed = |s: &str| phoneme_string(&g2p.phonemize(s));
        assert_eq!(fed("see [New York](new york)."), "sˈi nˈu jˈɔɹk.");
        assert_eq!(fed("see [New York](new york), then."), fed("see New York, then."));
        assert_eq!(fed("see [New York City](new-york)."), "sˈi nˌujˈɔɹk.");
        assert!(!fed("see [New York City](new-york) now").contains("  "));
    }

    #[test]
    fn trailing_punctuation_does_not_hide_the_word_from_the_lexicon() {
        // "iOS" subtokenizes to "i" + "OS", and only the whole-word lookup
        // gets it back. A trailing comma is resolved before that lookup runs,
        // and it used to block it: `iOS,` was read as its pieces, "EYE. OH.
        // ESS.", where `iOS` read as one word. A published presentation said
        // it that way on every slide that mentioned the app.
        let lex = Lexicon::new(
            serde_json::json!({ "iOS": "ˌIOˈɛs", "ios": "ˌIOˈɛs", "i": "ˈI", "os": "ˌOˈɛs" }),
            serde_json::json!({}),
            false,
        );
        let g2p = EnglishG2P::new(lex, None);
        let say = |s: &str| phoneme_string(&g2p.phonemize(s)).trim().to_string();
        assert_eq!(say("On iOS now"), "ˌIOˈɛs");
        assert_eq!(say("On iOS, now"), "ˌIOˈɛs,");
        assert_eq!(say("On iOS."), "ˌIOˈɛs.");
    }

    #[test]
    fn a_hyphen_inside_a_substitute_glues_its_pieces_into_one_word() {
        // A section is what the author wrote between spaces. `read-me` is one
        // section, so it is said as one word.
        assert_eq!(as_one_word("ɹˈid mˈi"), "ɹˌidmˈi");
        // Punctuation the author put inside a section marked a boundary that
        // is now gone, so it goes with it.
        assert_eq!(as_one_word("ˈI, ˈO, ˈɛs"), "ˌIˌOˈɛs");
        // Trailing punctuation is the sentence's: `[live](liv).` keeps its
        // full stop.
        assert_eq!(as_one_word("lˈɪv."), "lˈɪv.");
        assert_eq!(as_one_word("ˈI ˈO ˈɛs."), "ˌIˌOˈɛs.");
        // One word in, the same word out.
        assert_eq!(as_one_word("lˈɪv"), "lˈɪv");
    }

    #[test]
    fn a_space_in_a_substitute_stays_a_spoken_boundary() {
        // The author decides where a filename breaks when it is read out
        // loud, and nothing here second-guesses it: "read-me dot em-dee" is
        // three sections, said as three words.
        let lex = Lexicon::new(
            serde_json::json!({
                "read": "ɹˈid", "me": "mˈi", "dot": "dˈɑt", "em": "ˈɛm", "dee": "dˈi",
            }),
            serde_json::json!({}),
            false,
        );
        let g2p = EnglishG2P::new(lex, None);
        let say = |s: &str| phoneme_string(&g2p.phonemize(s)).trim().to_string();
        assert_eq!(
            say("[README.md](read-me dot em-dee)"),
            "ɹˌidmˈi dˈɑt ˌɛmdˈi"
        );
        assert_eq!(
            say("[README.md](read me dot em dee)"),
            "ɹˈid mˈi dˈɑt ˈɛm dˈi"
        );
        assert_eq!(say("[README.md](read-me-dot-em-dee)"), "ɹˌidmˌidˌɑtˌɛmdˈi");
    }

    #[test]
    fn ordinary_brackets_are_read_as_text() {
        for prose in [
            "an array [0] of things",
            "see [the docs] for more",
            "a [half open marker",
            "[](empty)",
            "[word]()",
            "[word](//)",
        ] {
            let toks = EnglishG2P::tokenize(prose);
            assert!(
                toks.iter().all(|t| t.phonemes.is_none() && t.spoken.is_none()),
                "{prose:?} is prose, not an override"
            );
        }
    }

    /// A miniature lexicon: one heteronym with two distinct-sounding readings,
    /// one POS-conditioned word whose readings sound alike, one plain word.
    fn tiny_lexicon() -> Lexicon {
        Lexicon::new(
            serde_json::json!({
                "read": { "VERB": "ɹˈid", "VBD": "ɹˈɛd" },
                "would": { "VERB": "wˈʊd", "NOUN": "wˈʊd" },
                "servers": "sˈɜɹvəɹz",
            }),
            serde_json::json!({}),
            false,
        )
    }

    #[test]
    fn facts_state_lexicon_membership_and_reading_count_without_judging() {
        let g2p = EnglishG2P::new(tiny_lexicon(), None);
        // A held word with one reading, a held heteronym, and a word the
        // lexicon has never seen all report — no spelling heuristic decides
        // that "Kokoro" is fine and "ggml" is not.
        assert_eq!(g2p.facts("servers"), Some((true, 1)));
        assert_eq!(g2p.facts("read"), Some((true, 2)));
        assert_eq!(g2p.facts("Kokoro"), Some((false, 1)));
        assert_eq!(g2p.facts("ggml"), Some((false, 1)));
        // POS-conditioned but every reading sounds the same: one reading.
        assert_eq!(g2p.facts("would"), Some((true, 1)));
        // Punctuation riding on the word does not change the answer, and a
        // token that is only punctuation has no facts to state.
        assert_eq!(g2p.facts("read."), Some((true, 2)));
        assert_eq!(g2p.facts("…"), None);
    }

    #[test]
    fn punctuation_after_an_override_stays_attached_to_the_word() {
        let plain = EnglishG2P::tokenize("tensors already live.");
        let marked = EnglishG2P::tokenize("tensors already [live](liv).");
        assert_eq!(
            plain.len(),
            marked.len(),
            "an override must not change how many tokens a sentence has, or every \
             later word timing shifts"
        );
        assert_eq!(marked.last().unwrap().text, "live.");
        assert_eq!(marked.last().unwrap().spoken.as_deref(), Some("liv."));
    }

    #[test]
    fn an_override_is_never_split_at_a_subtoken_boundary() {
        // `ggml` has no camelCase seam, but an override on a word that does
        // must still arrive at the resolver in one piece with its pin intact.
        let groups = EnglishG2P::retokenize(EnglishG2P::tokenize("[macOS](/mˈakˌoʊˈɛs/) ships"));
        let over = groups
            .iter()
            .find(|g| g.iter().any(|t| t.text == "macOS"))
            .expect("the override survives retokenize as its own group");
        assert_eq!(over.len(), 1, "an override is not subtokenized");
        assert_eq!(over[0].phonemes.as_deref(), Some("mˈakˌoʊˈɛs"));
    }

    #[test]
    fn tokenize_normalizes_whitespace_runs_to_one_space() {
        let toks = EnglishG2P::tokenize("hello  world\nagain");
        assert_eq!(toks.len(), 3);
        assert_eq!(toks[0].text, "hello");
        assert_eq!(
            toks[0].whitespace, " ",
            "a whitespace run collapses to one space"
        );
        assert_eq!(toks[1].whitespace, " ", "a newline is a space too");
        assert_eq!(toks[2].whitespace, "");
    }

    #[test]
    fn subtokenize_splits_where_the_lexicon_expects() {
        assert_eq!(EnglishG2P::subtokenize("3.5x"), vec!["3.5", "x"]);
        assert_eq!(EnglishG2P::subtokenize("$4.2M"), vec!["$", "4.2", "M"]);
        assert_eq!(EnglishG2P::subtokenize("15%"), vec!["15", "%"]);
        assert_eq!(EnglishG2P::subtokenize("1,250"), vec!["1,250"]);
        assert_eq!(EnglishG2P::subtokenize("camelCase"), vec!["camel", "Case"]);
        assert_eq!(
            EnglishG2P::subtokenize("state-of-the-art"),
            vec!["state", "-", "of", "-", "the", "-", "art"]
        );
        assert_eq!(EnglishG2P::subtokenize("Q3"), vec!["Q", "3"]);
        assert_eq!(EnglishG2P::subtokenize("don't"), vec!["don't"]);
    }

    #[test]
    fn a_trailing_period_is_not_glued_to_the_number() {
        assert_eq!(EnglishG2P::subtokenize("2026."), vec!["2026", "."]);
    }

    /// An intra-word hyphen is spelling, not prosody. The em-dash phoneme it
    /// used to map to makes Kokoro synthesize ~500-900ms of dead air inside
    /// the word — "single-machine" arrived with a hole in the middle. It maps
    /// to a plain word gap instead; a dash standing alone keeps the pause.
    #[test]
    fn an_intra_word_hyphen_is_a_word_gap_not_a_pause() {
        let groups = EnglishG2P::retokenize(EnglishG2P::tokenize("a single-machine setup"));
        let hyphen = groups
            .iter()
            .flatten()
            .find(|t| t.text == "-")
            .expect("the hyphen subtoken exists");
        assert_eq!(hyphen.phonemes.as_deref(), Some(" "));

        // A dash standing alone stays a deliberate pause.
        let groups = EnglishG2P::retokenize(EnglishG2P::tokenize("wait — go"));
        let dash = groups
            .iter()
            .flatten()
            .find(|t| t.text == "—")
            .expect("the standalone dash exists");
        assert_eq!(dash.phonemes.as_deref(), Some("—"));
    }

    #[test]
    fn retokenize_records_the_currency_on_the_amount() {
        let groups = EnglishG2P::retokenize(EnglishG2P::tokenize("It cost $4.50 today."));
        let amount = groups.iter().flatten().find(|t| t.text == "4.50").unwrap();
        assert_eq!(amount.currency.as_deref(), Some("$"));
    }

    #[test]
    fn a_magnitude_suffix_carries_the_currency_not_the_amount() {
        let groups = EnglishG2P::retokenize(EnglishG2P::tokenize("We raised $4.2M."));
        let flat: Vec<&Token> = groups.iter().flatten().collect();
        assert_eq!(
            flat.iter().find(|t| t.text == "4.2").unwrap().currency,
            None
        );
        assert_eq!(
            flat.iter()
                .find(|t| t.text == "M")
                .unwrap()
                .currency
                .as_deref(),
            Some("$"),
            "the unit belongs after 'million'"
        );
    }

    #[test]
    fn a_following_magnitude_word_carries_the_currency() {
        let groups = EnglishG2P::retokenize(EnglishG2P::tokenize("We raised $4.2 million."));
        let flat: Vec<&Token> = groups.iter().flatten().collect();
        assert_eq!(
            flat.iter().find(|t| t.text == "4.2").unwrap().currency,
            None
        );
        assert_eq!(
            flat.iter()
                .find(|t| t.text == "million")
                .unwrap()
                .currency
                .as_deref(),
            Some("$")
        );
    }

    #[test]
    fn symbols_are_not_claimed_by_the_punctuation_branch() {
        let groups = EnglishG2P::retokenize(EnglishG2P::tokenize("Growth was 15%."));
        let pct = groups.iter().flatten().find(|t| t.text == "%").unwrap();
        assert!(
            pct.phonemes.is_none(),
            "% must reach the lexicon, not be filtered to \"\""
        );
    }

    /// Invariant 1: what the model is fed must survive the token round trip
    /// exactly, or the duration walk drifts against the phoneme stream.
    #[test]
    fn align_tokens_round_trips_the_phoneme_string() {
        for ps in [
            "ˈola",
            "ˈola kˈomo estˈas",
            "a b c",
            // A doubled space is not hypothetical: punctuation restoration can
            // leave one, and it is a real model token that must be accounted.
            "a  b",
            " leading",
            "trailing ",
        ] {
            let tokens = align_tokens("whatever it says", ps);
            assert_eq!(phoneme_string(&tokens), ps, "round trip failed for {ps:?}");
        }
    }

    #[test]
    fn align_tokens_attaches_source_words_when_the_counts_match() {
        let tokens = align_tokens("Hola mundo", "ˈola mˈundo");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Hola");
        assert_eq!(tokens[1].text, "mundo");
        assert_eq!(tokens[0].whitespace, " ");
        assert_eq!(tokens[1].whitespace, "");
    }

    /// When eSpeak expands a number the word counts diverge. Inventing a
    /// mapping would put cues on the wrong words; leaving the text empty makes
    /// the reconciler distribute proportionally and report `exact: false`.
    #[test]
    fn align_tokens_leaves_text_empty_when_the_counts_diverge() {
        let tokens = align_tokens("Tengo 19,99 euros", "tˈɛŋɡo ðjˌɛθinwˈeβe kˌoma euros");
        assert_eq!(tokens.len(), 4);
        assert!(
            tokens.iter().all(|t| t.text.is_empty()),
            "no honest per-word mapping exists, so none is claimed"
        );
    }

    /// Invariant 2: every token carries phonemes, so `word_timings` emits one
    /// timing each and `narrate`'s count check cannot trip.
    #[test]
    fn every_aligned_token_carries_phonemes() {
        let tokens = align_tokens("a b", "x  y");
        assert_eq!(tokens.len(), 3, "the empty run between two spaces counts");
        assert!(tokens.iter().all(|t| t.phonemes.is_some()));
    }

    #[test]
    fn align_tokens_on_empty_phonemes_is_empty() {
        assert!(align_tokens("anything", "").is_empty());
    }

    /// End-to-end: sentence context reaches the POS-conditioned readings
    /// through the full phonemize pipeline (tag_sequence → lexicon fine keys).
    #[test]
    fn context_resolves_heteronyms_through_phonemize() {
        let lexicon = lexicon::Lexicon::new(
            serde_json::json!({
                "i": "ˈI", "he": "hˌi", "she": "ʃˌi", "will": "wɪl",
                "the": "ðə", "in": "ɪn", "city": "sˈɪTi", "plan": "plˈæn",
                "book": "bˈʊk", "yesterday": "jˈɛstəɹdˌA", "please": "plˈiz",
                "meeting": "mˈiTɪŋ", "minute": "mˈɪnət",
                "live": { "DEFAULT": "lˈIv", "VERB": "lˈɪv" },
                "present": { "DEFAULT": "pɹˈɛzᵊnt", "VERB": "pɹizˈɛnt" },
                "read": { "ADJ": "ɹˈɛd", "DEFAULT": "ɹˈid", "VBD": "ɹˈɛd", "VBN": "ɹˈɛd" },
                "record": { "DEFAULT": "ɹˈɛkəɹd", "VERB": "ɹəkˈɔɹd" }
            }),
            serde_json::json!({}),
            false,
        );
        let g2p = EnglishG2P::new(lexicon, None);
        let word = |text: &str, target: &str| -> String {
            g2p.phonemize(text)
                .into_iter()
                .find(|t| t.text.trim_matches('.') == target)
                .unwrap_or_else(|| panic!("{target} missing"))
                .phonemes
                .unwrap_or_else(|| panic!("{target} unresolved"))
                .replace('.', "")
        };
        assert_eq!(word("I live in the city.", "live"), "lˈɪv");
        assert_eq!(word("He will present the plan.", "present"), "pɹizˈɛnt");
        assert_eq!(word("She read the book yesterday.", "read"), "ɹˈɛd");
        assert_eq!(
            word("Please record the meeting minute.", "record"),
            "ɹəkˈɔɹd"
        );
        // And the guard direction: no context signal ⇒ the common reading.
        assert_eq!(word("The record in the book.", "record"), "ɹˈɛkəɹd");
    }

    /// "macOS" splits into "mac" + "OS", and the ALLCAPS tail of a mixed-case
    /// word is an acronym by construction — it must be spelled out, never
    /// read as the dictionary word "os".
    #[test]
    fn camelcase_acronym_tail_is_spelled_out() {
        let lexicon = lexicon::Lexicon::new(
            serde_json::json!({
                "it": "ˈɪt", "works": "wˈɜɹks", "on": "ˈɑn",
                "mac": "mˈæk", "os": "ˈɑs", "O": "ˈO", "S": "ˈɛs"
            }),
            serde_json::json!({}),
            false,
        );
        let g2p = EnglishG2P::new(lexicon, None);
        let tokens = g2p.phonemize("It works on macOS.");
        let word = tokens
            .iter()
            .find(|t| t.text.starts_with("macOS"))
            .expect("macOS token missing");
        let ps = word.phonemes.clone().expect("macOS unresolved");
        assert!(ps.contains("mˈæk"), "mac should read as the word: {ps}");
        assert!(!ps.contains("ɑs"), "OS must not read as the word 'os': {ps}");
        assert!(ps.contains("ɛs"), "OS should be spelled out: {ps}");
    }

    #[test]
    fn phoneme_string_reproduces_the_source_spacing() {
        let tokens = vec![
            Token {
                text: "a".into(),
                phonemes: Some("ɐ".into()),
                whitespace: " ".into(),
                ..Default::default()
            },
            Token {
                text: "b".into(),
                phonemes: Some("bˈi".into()),
                whitespace: "".into(),
                ..Default::default()
            },
        ];
        assert_eq!(phoneme_string(&tokens), "ɐ bˈi");
    }
}
