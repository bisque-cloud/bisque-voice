//! Grapheme-to-phoneme dictionary lookup, stress, morphology and numbers.
//!
//! Port of `Lexicon.swift` in `packages/kokoro-swift`, **including the vendored
//! patches** recorded in that package's `VENDORED.md` (decimals, `%`, currency
//! symbols, hyphenated numerals, magnitude suffixes). It reads the *same*
//! Apache-2.0 `us_gold` / `us_silver` / `gb_*` JSON the macOS engine reads, so
//! the two agree by construction rather than by coincidence.
//!
//! Keep this and the Swift file in lockstep. A word that reads differently on
//! Linux than on macOS is worse than a missing feature.

use std::collections::HashMap;

use serde_json::Value;

use super::num2word;
use super::tagger::Tag;

pub const PRIMARY_STRESS: char = 'ˈ';
pub const SECONDARY_STRESS: char = 'ˌ';
const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const US_TAUS: &str = "AIOWYiuæɑəɛɪɹʊʌ";
const ORDINAL_SUFFIXES: &[&str] = &["st", "nd", "rd", "th"];

/// `%`, `&`, `+`, `@` are spoken, not punctuation.
pub fn symbol_word(text: &str) -> Option<&'static str> {
    Some(match text {
        "%" => "percent",
        "&" => "and",
        "+" => "plus",
        "@" => "at",
        _ => return None,
    })
}

/// (unit, subunit) per currency symbol.
pub fn currency_units(symbol: &str) -> Option<(&'static str, &'static str)> {
    Some(match symbol {
        "$" => ("dollar", "cent"),
        "£" => ("pound", "pence"),
        "€" => ("euro", "cent"),
        _ => return None,
    })
}

/// Magnitude/multiplier suffixes attached directly to a number — "$4.2M",
/// "12K", "3.5x". These are the canonical numbers on a business slide.
fn magnitude_word(letter: char) -> Option<&'static str> {
    Some(match letter.to_ascii_lowercase() {
        'k' => "thousand",
        'm' => "million",
        'b' => "billion",
        't' => "trillion",
        'x' => "times",
        _ => return None,
    })
}

/// The spelled-out counterparts. In "$4.2 million" the currency unit belongs
/// after the magnitude word, not folded into the amount as cents.
pub fn is_magnitude_word(text: &str) -> bool {
    matches!(
        text.to_lowercase().as_str(),
        "thousand" | "million" | "billion" | "trillion"
    )
}

/// "4.2M" → ("4.2", "million"). Only fires when a digit sits immediately before
/// the suffix, so "Mx" or a bare "K" is untouched.
pub fn split_magnitude(word: &str) -> Option<(String, &'static str)> {
    let mut chars = word.chars();
    let last = chars.next_back()?;
    let core: String = chars.collect();
    if core.is_empty() || !core.chars().next_back()?.is_numeric() {
        return None;
    }
    Some((core, magnitude_word(last)?))
}

/// Context threaded right-to-left through the sentence: whether the next spoken
/// sound is a vowel ("the" → "ðə" vs "ði"), and whether a "to" follows.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ctx {
    pub future_vowel: Option<bool>,
    pub future_to: bool,
}

pub struct Lexicon {
    golds: HashMap<String, Value>,
    silvers: HashMap<String, Value>,
    british: bool,
}

/// A resolved pronunciation plus Misaki's confidence rating (5 = author
/// override, 4 = gold, 3 = silver/spelled, 1 = neural fallback).
pub type Resolved = Option<(String, i32)>;

impl Lexicon {
    pub fn new(gold: Value, silver: Value, british: bool) -> Self {
        Lexicon {
            golds: grow(gold),
            silvers: grow(silver),
            british,
        }
    }

    // ── stress ───────────────────────────────────────────────────────────────

    /// Reposition stress markers so each sits immediately before its vowel.
    fn restress(ps: &str) -> String {
        let chars: Vec<char> = ps.chars().collect();
        // (position, char) with fractional positions used to move markers.
        let mut indexed: Vec<(f64, char)> = chars
            .iter()
            .enumerate()
            .map(|(i, c)| (i as f64, *c))
            .collect();
        for (i, c) in chars.iter().enumerate() {
            if *c == PRIMARY_STRESS || *c == SECONDARY_STRESS {
                if let Some(j) = (i + 1..chars.len()).find(|j| VOWELS.contains(chars[*j])) {
                    indexed[i].0 = j as f64 - 0.5;
                }
            }
        }
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        indexed.into_iter().map(|(_, c)| c).collect()
    }

    pub fn apply_stress(ps: &str, stress: Option<f64>) -> String {
        let Some(stress) = stress else {
            return ps.to_string();
        };
        let has_primary = ps.contains(PRIMARY_STRESS);
        let has_any = has_primary || ps.contains(SECONDARY_STRESS);
        let has_vowel = ps.chars().any(|c| VOWELS.contains(c));

        if stress < -1.0 {
            return ps
                .chars()
                .filter(|c| *c != PRIMARY_STRESS && *c != SECONDARY_STRESS)
                .collect();
        }
        if stress == -1.0 || ((stress == 0.0 || stress == -0.5) && has_primary) {
            return ps
                .chars()
                .filter(|c| *c != SECONDARY_STRESS)
                .map(|c| {
                    if c == PRIMARY_STRESS {
                        SECONDARY_STRESS
                    } else {
                        c
                    }
                })
                .collect();
        }
        if (stress == 0.0 || stress == 0.5 || stress == 1.0) && !has_any {
            if !has_vowel {
                return ps.to_string();
            }
            return Self::restress(&format!("{SECONDARY_STRESS}{ps}"));
        }
        if stress >= 1.0 && !has_primary && ps.contains(SECONDARY_STRESS) {
            return ps
                .chars()
                .map(|c| {
                    if c == SECONDARY_STRESS {
                        PRIMARY_STRESS
                    } else {
                        c
                    }
                })
                .collect();
        }
        if stress > 1.0 && !has_any {
            if !has_vowel {
                return ps.to_string();
            }
            return Self::restress(&format!("{PRIMARY_STRESS}{ps}"));
        }
        ps.to_string()
    }

    // ── dictionary ───────────────────────────────────────────────────────────

    fn raw(&self, word: &str) -> Option<(&Value, i32)> {
        if let Some(v) = self.golds.get(word) {
            return Some((v, 4));
        }
        self.silvers.get(word).map(|v| (v, 3))
    }

    /// Whether a word's pronunciation is POS-conditioned — its gold entry is a
    /// tag-keyed object rather than a plain string. This is the gate for
    /// `tagger::tag_sequence`: context rules only ever fire on these words, so
    /// re-tagging cannot move any ordinary word's pronunciation.
    pub fn pos_conditioned(&self, word: &str) -> bool {
        let object = |w: &str| matches!(self.golds.get(w), Some(Value::Object(_)));
        object(word) || object(&word.to_lowercase())
    }

    /// How many distinct-SOUNDING readings the lexicon holds for a word.
    ///
    /// 1 for a plain entry, for a word the lexicon has never seen (the
    /// fallback produces exactly one guess), and for a POS-conditioned entry
    /// whose parts of speech all resolve to the same sounds or differ only in
    /// which syllable takes the stress — "would" and "through" are conditioned
    /// but have one reading a listener would notice. Counting those as several
    /// would bury the words that genuinely change ("separate", "live",
    /// "record").
    pub fn readings(&self, word: &str) -> u32 {
        let entry = self
            .golds
            .get(word)
            .or_else(|| self.golds.get(&word.to_lowercase()));
        let Some(Value::Object(map)) = entry else {
            return 1;
        };
        let bare = |s: &str| -> String {
            s.chars().filter(|c| *c != '\u{2c8}' && *c != '\u{2cc}').collect()
        };
        let mut distinct: Vec<String> = Vec::new();
        for v in map.values() {
            let Some(p) = v.as_str() else { continue };
            let p = bare(p);
            if !distinct.contains(&p) {
                distinct.push(p);
            }
        }
        distinct.len().max(1) as u32
    }

    pub fn is_known(&self, word: &str) -> bool {
        if self.golds.contains_key(word)
            || self.silvers.contains_key(word)
            || symbol_word(word).is_some()
        {
            return true;
        }
        // Only apostrophe, hyphen and letters can be a dictionary headword.
        if !word
            .chars()
            .all(|c| c == '\'' || c == '-' || c.is_ascii_alphabetic())
        {
            return false;
        }
        if word.chars().count() == 1 {
            return true;
        }
        if word == word.to_uppercase() && self.golds.contains_key(&word.to_lowercase()) {
            return true;
        }
        // ALLCAPS after the first letter is treated as an acronym, which the
        // spell-out path can always render.
        let tail: String = word.chars().skip(1).collect();
        tail.to_uppercase() == tail
    }

    /// Spell-out for pieces the tokenizer already knows are acronyms (the
    /// ALLCAPS tail of a camelCase split — "macOS" → "OS"), bypassing word
    /// lookup entirely so "OS" cannot resolve to the dictionary word "os".
    pub fn spell_out(&self, word: &str) -> Resolved {
        self.nnp(word)
    }

    /// Spell out acronyms and abbreviations letter by letter.
    fn nnp(&self, word: &str) -> Resolved {
        let mut pieces: Vec<String> = Vec::new();
        for ch in word.chars().filter(|c| c.is_alphabetic()) {
            let key = ch.to_uppercase().to_string();
            match self.golds.get(&key).and_then(Value::as_str) {
                Some(v) => pieces.push(v.to_string()),
                // A letter we have no pronunciation for: refuse rather than
                // spell a partial word.
                None => return None,
            }
        }
        // VENDORED PATCH parity: no letters at all means "not an acronym".
        // Without this a decimal ("3.5") resolves to an empty string and the
        // number path never runs — the bug that made the macOS voice silently
        // drop every decimal.
        if pieces.is_empty() {
            return None;
        }
        let joined = Self::apply_stress(&pieces.concat(), Some(0.0));
        Some((replace_last(&joined, SECONDARY_STRESS, PRIMARY_STRESS), 3))
    }

    fn lookup(&self, w: &str, tag: Tag, stress: Option<f64>, ctx: Option<Ctx>) -> Resolved {
        let mut word = w.to_string();
        let mut is_nnp = None;
        if word == word.to_uppercase() && !self.golds.contains_key(&word) {
            word = word.to_lowercase();
            is_nnp = Some(tag.is_proper_noun(w));
        }

        let (value, rating) = match self.raw(&word) {
            Some(v) => v,
            None => return self.nnp(&word),
        };
        // A proper noun is spelled out rather than read from the silver list.
        if is_nnp == Some(true) && rating < 4 {
            return self.nnp(&word);
        }

        let phonetic: Option<String> = match value {
            Value::String(s) => Some(s.clone()),
            Value::Object(map) => {
                // POS-conditioned entry. An unknown tag selects DEFAULT, which
                // is the common reading — see g2p::tagger. Python Misaki tries
                // the exact Penn tag before the parent bucket, which is the
                // only route to entries keyed finer than the buckets ("read"'s
                // VBD/VBN) — mirror that order.
                let phrase_final =
                    ctx.is_some_and(|c| c.future_vowel.is_none()) && map.contains_key("None");
                let key = if phrase_final { "XX" } else { tag.parent() };
                let fine = if phrase_final {
                    None
                } else {
                    tag.fine().and_then(|f| map.get(f))
                };
                fine.or_else(|| map.get(key))
                    .or_else(|| map.get("DEFAULT"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            }
            _ => None,
        };

        let Some(phonetic) = phonetic else {
            return self.nnp(&word);
        };
        if is_nnp == Some(true) && !phonetic.contains(PRIMARY_STRESS) {
            if let Some(nn) = self.nnp(&word) {
                return Some(nn);
            }
        }
        Some((Self::apply_stress(&phonetic, stress), rating))
    }

    // ── morphology ───────────────────────────────────────────────────────────

    fn pluralize_s(&self, stem: &str) -> Option<String> {
        let last = stem.chars().next_back()?;
        if "ptkfθ".contains(last) {
            return Some(format!("{stem}s"));
        }
        if "szʃʒʧʤ".contains(last) {
            return Some(format!("{stem}{}z", if self.british { 'ɪ' } else { 'ᵻ' }));
        }
        Some(format!("{stem}z"))
    }

    fn past_ed(&self, stem: &str) -> Option<String> {
        let last = stem.chars().next_back()?;
        if "pkfθʃsʧ".contains(last) {
            return Some(format!("{stem}t"));
        }
        if last == 'd' {
            return Some(format!("{stem}{}d", if self.british { 'ɪ' } else { 'ᵻ' }));
        }
        if last != 't' {
            return Some(format!("{stem}d"));
        }
        if self.british || stem.chars().count() < 2 {
            return Some(format!("{stem}ɪd"));
        }
        let penult = stem.chars().nth_back(1)?;
        if US_TAUS.contains(penult) {
            let trimmed: String = stem.chars().take(stem.chars().count() - 1).collect();
            return Some(format!("{trimmed}ɾᵻd"));
        }
        Some(format!("{stem}ᵻd"))
    }

    fn prog_ing(&self, stem: &str) -> Option<String> {
        if stem.is_empty() {
            return None;
        }
        Some(format!("{stem}ɪŋ"))
    }

    fn stem_s(&self, word: &str, tag: Tag, stress: Option<f64>, ctx: Option<Ctx>) -> Resolved {
        let n = word.chars().count();
        if n < 3 || !word.ends_with('s') {
            return None;
        }
        let drop = |k: usize| -> String { word.chars().take(n - k).collect() };
        let stem = if !word.ends_with("ss") && self.is_known(&drop(1)) {
            drop(1)
        } else if (word.ends_with("'s")
            || (n > 4 && word.ends_with("es") && !word.ends_with("ies")))
            && self.is_known(&drop(2))
        {
            drop(2)
        } else if n > 4 && word.ends_with("ies") && self.is_known(&format!("{}y", drop(3))) {
            format!("{}y", drop(3))
        } else {
            return None;
        };
        let (ps, rating) = self.lookup(&stem, tag, stress, ctx)?;
        Some((self.pluralize_s(&ps)?, rating))
    }

    fn stem_ed(&self, word: &str, tag: Tag, stress: Option<f64>, ctx: Option<Ctx>) -> Resolved {
        let n = word.chars().count();
        if n < 4 || !word.ends_with('d') {
            return None;
        }
        let drop = |k: usize| -> String { word.chars().take(n - k).collect() };
        let stem = if !word.ends_with("dd") && self.is_known(&drop(1)) {
            drop(1)
        } else if n > 4 && word.ends_with("ed") && !word.ends_with("eed") && self.is_known(&drop(2))
        {
            drop(2)
        } else {
            return None;
        };
        let (ps, rating) = self.lookup(&stem, tag, stress, ctx)?;
        Some((self.past_ed(&ps)?, rating))
    }

    fn stem_ing(&self, word: &str, tag: Tag, stress: Option<f64>, ctx: Option<Ctx>) -> Resolved {
        let n = word.chars().count();
        if n < 5 || !word.ends_with("ing") {
            return None;
        }
        let base: String = word.chars().take(n - 3).collect();
        let stem = if n > 5 && self.is_known(&base) {
            base.clone()
        } else if self.is_known(&format!("{base}e")) {
            format!("{base}e")
        } else if n > 5 && doubled_consonant(&base) {
            let undoubled: String = base.chars().take(base.chars().count() - 1).collect();
            if self.is_known(&undoubled) {
                undoubled
            } else {
                return None;
            }
        } else {
            return None;
        };
        let (ps, rating) = self.lookup(&stem, tag, stress, ctx)?;
        Some((self.prog_ing(&ps)?, rating))
    }

    fn stems(&self, word: &str, tag: Tag, stress: Option<f64>, ctx: Option<Ctx>) -> Resolved {
        self.stem_s(word, tag, stress, ctx)
            .or_else(|| self.stem_ed(word, tag, stress, ctx))
            .or_else(|| {
                self.stem_ing(
                    word,
                    tag,
                    if stress.is_none() { Some(0.5) } else { stress },
                    ctx,
                )
            })
    }

    // ── entry point ──────────────────────────────────────────────────────────

    fn special_case(&self, word: &str, tag: Tag, stress: Option<f64>, ctx: Ctx) -> Resolved {
        if tag == Tag::Punctuation {
            if let Some(target) = match word {
                "." => Some("dot"),
                "/" => Some("slash"),
                _ => None,
            } {
                return self.lookup(target, Tag::Xx, Some(-0.5), Some(ctx));
            }
        }
        if let Some(sym) = symbol_word(word) {
            return self.lookup(sym, Tag::Xx, None, Some(ctx));
        }
        // Dotted acronym ("U.S.A."). VENDORED PATCH parity: digits mean
        // "number", so this must not claim "3.5" or "3.5x".
        if word.trim_matches('.').contains('.') && !word.chars().any(|c| c.is_numeric()) {
            if word.split('.').map(str::len).max().unwrap_or(0) < 3 {
                return self.nnp(word);
            }
        }
        match word {
            "a" | "A" => {
                return Some(if tag == Tag::Determiner {
                    ("ɐ".into(), 4)
                } else {
                    (format!("{PRIMARY_STRESS}A"), 4)
                })
            }
            "an" | "An" | "AN" => return Some(("ɐn".into(), 4)),
            "to" | "To" => {
                let chosen = match ctx.future_vowel {
                    None => self
                        .golds
                        .get("to")
                        .and_then(Value::as_str)
                        .unwrap_or("to")
                        .to_string(),
                    Some(false) => "tə".to_string(),
                    Some(true) => "tʊ".to_string(),
                };
                return Some((chosen, 4));
            }
            "in" | "In" => {
                let s = if ctx.future_vowel.is_none() || tag != Tag::Preposition {
                    PRIMARY_STRESS.to_string()
                } else {
                    String::new()
                };
                return Some((format!("{s}ɪn"), 4));
            }
            "the" | "The" => {
                return Some((
                    if ctx.future_vowel == Some(true) {
                        "ði".into()
                    } else {
                        "ðə".into()
                    },
                    4,
                ))
            }
            "used" | "Used" | "USED" => {
                // "used to" is /juːst/, "used a hammer" is /juːzd/.
                let verbish = matches!(
                    tag,
                    Tag::Verb | Tag::VerbPast | Tag::VerbPastPart | Tag::Adjective
                );
                let key = if verbish && ctx.future_to {
                    "VBD"
                } else {
                    "DEFAULT"
                };
                if let Some(ps) = self
                    .golds
                    .get("used")
                    .and_then(Value::as_object)
                    .and_then(|m| m.get(key).or_else(|| m.get("DEFAULT")))
                    .and_then(Value::as_str)
                {
                    return Some((ps.to_string(), 4));
                }
            }
            _ => {}
        }
        let _ = stress;
        None
    }

    /// Resolve one token. `None` means the caller should fall back.
    pub fn transcribe(
        &self,
        text: &str,
        tag: Tag,
        ctx: Ctx,
        currency: Option<&str>,
        token_stress: Option<f64>,
        is_head: bool,
    ) -> Resolved {
        let word: String = text.replace(['\u{2018}', '\u{2019}'], "'");

        // ALLCAPS and Capitalized words carry an implicit stress hint.
        let stress = if word == word.to_lowercase() {
            None
        } else if word == word.to_uppercase() {
            Some(2.0)
        } else {
            Some(0.5)
        };

        if let Some((ps, rating)) = self.get_word(&word, tag, stress, ctx) {
            let with_currency = match currency.and_then(currency_units) {
                Some((unit, _)) => match self.stem_s(&format!("{unit}s"), Tag::Xx, None, None) {
                    Some((plural, _)) => format!("{ps} {plural}"),
                    None => ps,
                },
                None => ps,
            };
            return Some((Self::apply_stress(&with_currency, token_stress), rating));
        }

        if is_number(&word, is_head) {
            let (ps, rating) = self.get_number(&word, currency, is_head)?;
            return Some((Self::apply_stress(&ps, token_stress), rating));
        }
        None
    }

    fn get_word(&self, word: &str, tag: Tag, stress: Option<f64>, ctx: Ctx) -> Resolved {
        if let Some(hit) = self.special_case(word, tag, stress, ctx) {
            return Some(hit);
        }
        let lower = word.to_lowercase();
        let mut candidate = word.to_string();

        // A capitalized ordinary word is looked up in lowercase, unless it is
        // long enough to be a genuine proper noun.
        if word.chars().count() > 1
            && word.replace('\'', "").chars().all(char::is_alphabetic)
            && word != lower
            && (!tag.is_proper_noun(word) || word.chars().count() > 7)
            && !self.golds.contains_key(word)
            && !self.silvers.contains_key(word)
            && (word == word.to_uppercase() || {
                let tail: String = word.chars().skip(1).collect();
                tail.to_lowercase() == tail
            })
            && (self.golds.contains_key(&lower)
                || self.silvers.contains_key(&lower)
                || self.stems(&lower, tag, stress, Some(ctx)).is_some())
        {
            candidate = lower;
        }

        if self.is_known(&candidate) {
            return self.lookup(&candidate, tag, stress, Some(ctx));
        }
        let n = candidate.chars().count();
        if candidate.ends_with("s'") {
            let base: String = candidate.chars().take(n - 2).collect();
            let possessive = format!("{base}'s");
            if self.is_known(&possessive) {
                return self.lookup(&possessive, tag, stress, Some(ctx));
            }
        }
        if candidate.ends_with('\'') {
            let base: String = candidate.chars().take(n - 1).collect();
            if self.is_known(&base) {
                return self.lookup(&base, tag, stress, Some(ctx));
            }
        }
        self.stems(&candidate, tag, stress, Some(ctx))
    }

    // ── numbers ──────────────────────────────────────────────────────────────

    /// True when the fractional part reads as subunits ("$4.50" → 50 cents)
    /// rather than as digits.
    fn is_currency_amount(word: &str) -> bool {
        if !word.contains('.') {
            return true;
        }
        if word.matches('.').count() > 1 {
            return false;
        }
        match word.split('.').next_back() {
            Some(cents) => cents.len() < 3 || cents.chars().all(|c| c == '0'),
            None => false,
        }
    }

    fn get_number(&self, input: &str, currency: Option<&str>, is_head: bool) -> Resolved {
        let mut result: Vec<(String, i32)> = Vec::new();
        // A free function rather than a closure so `result` stays borrowable
        // alongside the other pushes below.
        fn push(lex: &Lexicon, out: &mut Vec<(String, i32)>, w: &str, s: Option<f64>) {
            if let Some((ps, r)) = lex.lookup(w, Tag::Xx, s, None) {
                out.push((ps, r));
            }
        }

        // Peel "4.2M" → 4.2 + "million" first, so the rest sees a plain number.
        let magnitude = split_magnitude(input);
        let mut word = match &magnitude {
            Some((core, _)) => core.clone(),
            None => input.to_string(),
        };

        // A trailing letter run is an ordinal or plural marker.
        let suffix = trailing_lowercase(&word);
        if let Some(s) = &suffix {
            word.truncate(word.len() - s.len());
        }

        if let Some(stripped) = word.strip_prefix('-') {
            push(self, &mut result, "minus", None);
            word = stripped.to_string();
        }

        let plain = !word.is_empty() && word.chars().all(|c| c.is_ascii_digit());
        let spoken: String;

        if plain
            && suffix
                .as_deref()
                .is_some_and(|s| ORDINAL_SUFFIXES.contains(&s))
        {
            spoken = num2word::to_ordinal(word.parse().unwrap_or(0)).to_lowercase();
        } else if plain && word.len() == 4 && currency.is_none() {
            spoken = num2word::to_year(word.parse().unwrap_or(0));
        } else if magnitude.is_none()
            && currency.is_some_and(|c| currency_units(c).is_some())
            && Self::is_currency_amount(&word)
        {
            // "$4.50" → four dollars and fifty cents.
            let (unit, subunit) = currency_units(currency.unwrap()).unwrap();
            let cleaned = word.replace(',', "");
            let mut parts = cleaned.split('.');
            let major: i64 = parts.next().unwrap_or("0").parse().unwrap_or(0);
            let minor: i64 = parts.next().unwrap_or("").parse().unwrap_or(0);
            let mut pairs: Vec<(i64, &str)> = vec![(major, unit), (minor, subunit)];
            if pairs[1].0 == 0 {
                pairs.truncate(1);
            } else if pairs[0].0 == 0 {
                pairs.remove(0);
            }
            for (i, (num, unit)) in pairs.into_iter().enumerate() {
                if i > 0 {
                    push(self, &mut result, "and", None);
                }
                for w in num2word::to_cardinal(num).split(|c: char| !c.is_alphabetic()) {
                    if !w.is_empty() {
                        push(self, &mut result, w, None);
                    }
                }
                if num.abs() != 1 && unit != "pence" {
                    if let Some((s, _)) = self.stem_s(&format!("{unit}s"), Tag::Xx, None, None) {
                        result.push((s, 4));
                    }
                } else {
                    push(self, &mut result, unit, None);
                }
            }
            return self.finish_number(result, magnitude, currency, suffix.as_deref());
        } else if !is_head && !word.contains('.') {
            // A number that is not the head of its word is an identifier, not a
            // quantity: "Q3" or a version reads digit by digit rather than as
            // "three thousand". Long runs and leading zeros always do.
            let num = word.replace(',', "");
            if num.starts_with('0') || num.chars().count() > 3 {
                spoken = num
                    .chars()
                    .filter_map(|c| c.to_digit(10))
                    .map(|d| num2word::to_cardinal(d as i64))
                    .collect::<Vec<_>>()
                    .join(" ");
            } else {
                spoken = num2word::to_decimal(&num);
            }
        } else {
            spoken = num2word::to_decimal(&word);
        }

        // VENDORED PATCH parity: split on any non-letter. num2words hyphenates
        // ("ninety-nine"), and a hyphenated word misses the lexicon and gets
        // spelled out letter by letter.
        for w in spoken.split(|c: char| !c.is_alphabetic()) {
            if w.is_empty() {
                continue;
            }
            let s = if w == "point" { Some(-2.0) } else { None };
            push(self, &mut result, w, s);
        }
        self.finish_number(result, magnitude, currency, suffix.as_deref())
    }

    fn finish_number(
        &self,
        mut result: Vec<(String, i32)>,
        magnitude: Option<(String, &'static str)>,
        currency: Option<&str>,
        suffix: Option<&str>,
    ) -> Resolved {
        // "…million", then the currency unit: "$4.2M" → "four point two
        // million dollars".
        if let Some((_, spoken)) = magnitude {
            if let Some((ps, r)) = self.lookup(spoken, Tag::Xx, None, None) {
                result.push((ps, r));
            }
            if let Some((unit, _)) = currency.and_then(currency_units) {
                if let Some((plural, _)) = self.stem_s(&format!("{unit}s"), Tag::Xx, None, None) {
                    result.push((plural, 4));
                }
            }
        }
        if result.is_empty() {
            return None;
        }
        let rating = result.iter().map(|(_, r)| *r).min().unwrap_or(4);
        let mut text = result
            .into_iter()
            .map(|(p, _)| p)
            .collect::<Vec<_>>()
            .join(" ");
        match suffix {
            Some("s") | Some("'s") => text = self.pluralize_s(&text).unwrap_or(text),
            Some("ed") | Some("'d") => text = self.past_ed(&text).unwrap_or(text),
            Some("ing") => text = self.prog_ing(&text).unwrap_or(text),
            _ => {}
        }
        Some((text, rating))
    }
}

/// Digits, separators, an optional leading minus and an optional trailing
/// ordinal/plural or magnitude marker.
pub fn is_number(word: &str, is_head: bool) -> bool {
    if !word.chars().any(|c| c.is_numeric()) {
        return false;
    }
    let mut core = match split_magnitude(word) {
        Some((c, _)) => c,
        None => word.to_string(),
    };
    for s in ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"] {
        if core.ends_with(s) {
            core.truncate(core.len() - s.len());
            break;
        }
    }
    core.chars()
        .enumerate()
        .all(|(i, c)| c.is_numeric() || c == ',' || c == '.' || (is_head && i == 0 && c == '-'))
}

fn trailing_lowercase(word: &str) -> Option<String> {
    let tail: String = word
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_lowercase() || *c == '\'')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if tail.is_empty() {
        None
    } else {
        Some(tail)
    }
}

fn doubled_consonant(base: &str) -> bool {
    let chars: Vec<char> = base.chars().collect();
    if chars.len() < 2 {
        return false;
    }
    let (a, b) = (chars[chars.len() - 2], chars[chars.len() - 1]);
    (a == b && "bcdgklmnprstvxz".contains(a)) || base.ends_with("ck")
}

fn replace_last(s: &str, from: char, to: char) -> String {
    match s.rfind(from) {
        Some(idx) => {
            let mut out = String::with_capacity(s.len());
            out.push_str(&s[..idx]);
            out.push(to);
            out.push_str(&s[idx + from.len_utf8()..]);
            out
        }
        None => s.to_string(),
    }
}

/// Add the capitalized/lowercase variant of every headword, matching
/// `growDictionary`. Original entries win on collision.
fn grow(value: Value) -> HashMap<String, Value> {
    let Value::Object(map) = value else {
        return HashMap::new();
    };
    let mut out: HashMap<String, Value> = HashMap::with_capacity(map.len() * 2);
    for (k, v) in &map {
        if k.chars().count() < 2 {
            continue;
        }
        let lower = k.to_lowercase();
        let capitalized = capitalize(&lower);
        if *k == lower {
            if *k != capitalized {
                out.insert(capitalized, v.clone());
            }
        } else if *k == capitalize(&k.to_lowercase()) {
            out.insert(lower, v.clone());
        }
    }
    for (k, v) in map {
        out.insert(k, v);
    }
    out
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnitude_splitting() {
        assert_eq!(split_magnitude("4.2M"), Some(("4.2".into(), "million")));
        assert_eq!(split_magnitude("12K"), Some(("12".into(), "thousand")));
        assert_eq!(split_magnitude("3.5x"), Some(("3.5".into(), "times")));
        assert_eq!(split_magnitude("Mx"), None, "no digit before the suffix");
        assert_eq!(split_magnitude("K"), None);
        assert_eq!(split_magnitude("4.2"), None);
    }

    #[test]
    fn number_recognition_covers_the_hard_cases() {
        assert!(is_number("3.5", true));
        assert!(is_number("3.5x", true));
        assert!(is_number("4.2M", true));
        assert!(is_number("1,250", true));
        assert!(is_number("21st", true));
        assert!(is_number("90s", true));
        assert!(is_number("-5", true));
        assert!(!is_number("hello", true));
        assert!(!is_number("", true));
    }

    #[test]
    fn currency_amount_vs_magnitude() {
        assert!(Lexicon::is_currency_amount("4.50"));
        assert!(Lexicon::is_currency_amount("5"));
        assert!(
            !Lexicon::is_currency_amount("4.201"),
            "three decimals is not cents"
        );
        assert!(
            Lexicon::is_currency_amount("4.000"),
            "all zeros still reads as units"
        );
    }

    #[test]
    fn stress_removal_and_promotion() {
        assert_eq!(Lexicon::apply_stress("hˈɛloʊ", Some(-2.0)), "hɛloʊ");
        assert_eq!(Lexicon::apply_stress("hˈɛloʊ", Some(-1.0)), "hˌɛloʊ");
        assert_eq!(Lexicon::apply_stress("hˌɛloʊ", Some(1.0)), "hˈɛloʊ");
        assert_eq!(Lexicon::apply_stress("hˈɛloʊ", None), "hˈɛloʊ");
    }

    #[test]
    fn restress_moves_a_marker_in_front_of_its_vowel() {
        // A marker prepended to a consonant onset slides to just before the vowel.
        let out = Lexicon::apply_stress("stɹit", Some(1.0));
        assert!(out.contains(SECONDARY_STRESS));
        let idx = out.find(SECONDARY_STRESS).unwrap();
        let after: char = out[idx + SECONDARY_STRESS.len_utf8()..]
            .chars()
            .next()
            .unwrap();
        assert!(
            VOWELS.contains(after),
            "marker should sit before a vowel, got {out:?}"
        );
    }

    #[test]
    fn replace_last_only_touches_the_final_occurrence() {
        assert_eq!(
            replace_last("ˌaˌbˌc", SECONDARY_STRESS, PRIMARY_STRESS),
            "ˌaˌbˈc"
        );
        assert_eq!(replace_last("abc", SECONDARY_STRESS, PRIMARY_STRESS), "abc");
    }

    #[test]
    fn grow_adds_case_variants_without_clobbering_originals() {
        let raw = serde_json::json!({ "hello": "h1", "World": "w1", "USA": "u1" });
        let grown = grow(raw);
        assert_eq!(grown["hello"], serde_json::json!("h1"));
        assert_eq!(
            grown["Hello"],
            serde_json::json!("h1"),
            "capitalized variant added"
        );
        assert_eq!(
            grown["world"],
            serde_json::json!("w1"),
            "lowercase variant added"
        );
        assert!(
            !grown.contains_key("usa"),
            "ALLCAPS is not a capitalization variant"
        );
    }

    fn test_lexicon() -> Lexicon {
        Lexicon::new(
            serde_json::json!({
                "A": "ˈA", "B": "bˈi", "C": "sˈi",
                "hello": "hˈɛloʊ", "cat": "kˈæt", "walk": "wˈɔk",
                "dollar": "dˈɑləɹ", "and": "ænd", "percent": "pəɹsˈɛnt",
                "point": "pYnt", "four": "fˈɔɹ", "two": "tˈu", "five": "fˈIv",
                "three": "θɹˈi", "million": "mˈɪljᵊn", "times": "tˈImz",
                "fifty": "fˈɪfti", "cent": "sˈɛnt",
                "record": { "NOUN": "ɹˈɛkəɹd", "VERB": "ɹɪkˈɔɹd", "DEFAULT": "ɹˈɛkəɹd" },
                "read": { "ADJ": "ɹˈɛd", "DEFAULT": "ɹˈid", "VBD": "ɹˈɛd", "VBN": "ɹˈɛd", "VBP": "ɹˈɛd" }
            }),
            serde_json::json!({}),
            false,
        )
    }

    #[test]
    fn fine_tags_reach_the_penn_keyed_readings() {
        let lex = test_lexicon();
        let say = |tag: Tag| {
            lex.transcribe("read", tag, Ctx::default(), None, None, true)
                .unwrap()
                .0
        };
        // The exact key wins where the entry carries it…
        assert_eq!(say(Tag::VerbPast), "ɹˈɛd");
        assert_eq!(say(Tag::VerbPastPart), "ɹˈɛd");
        // …a plain Verb has no VERB key here and falls to DEFAULT…
        assert_eq!(say(Tag::Verb), "ɹˈid");
        // …and entries without fine keys collapse to the parent as before.
        let (verb, _) = lex
            .transcribe("record", Tag::VerbPast, Ctx::default(), None, None, true)
            .unwrap();
        assert_eq!(verb, "ɹɪkˈɔɹd", "VerbPast collapses to the VERB bucket");
    }

    #[test]
    fn pos_conditioned_flags_object_entries_only() {
        let lex = test_lexicon();
        assert!(lex.pos_conditioned("record"));
        assert!(lex.pos_conditioned("read"));
        assert!(lex.pos_conditioned("Record"), "case-insensitive");
        assert!(!lex.pos_conditioned("hello"));
        assert!(!lex.pos_conditioned("missing"));
    }

    #[test]
    fn plain_lookup_and_plural_stemming() {
        let lex = test_lexicon();
        let (ps, _) = lex
            .transcribe("hello", Tag::Xx, Ctx::default(), None, None, true)
            .unwrap();
        assert_eq!(ps, "hˈɛloʊ");
        let (ps, _) = lex
            .transcribe("cats", Tag::Xx, Ctx::default(), None, None, true)
            .unwrap();
        assert_eq!(ps, "kˈæts", "voiceless stem takes /s/");
    }

    #[test]
    fn pos_conditioned_entries_default_when_the_tag_is_unknown() {
        let lex = test_lexicon();
        let noun = lex
            .transcribe("record", Tag::Noun, Ctx::default(), None, None, true)
            .unwrap()
            .0;
        let verb = lex
            .transcribe("record", Tag::Verb, Ctx::default(), None, None, true)
            .unwrap()
            .0;
        let unk = lex
            .transcribe("record", Tag::Xx, Ctx::default(), None, None, true)
            .unwrap()
            .0;
        assert_eq!(noun, "ɹˈɛkəɹd");
        assert_eq!(verb, "ɹɪkˈɔɹd");
        assert_eq!(unk, "ɹˈɛkəɹd", "abstaining picks the common reading");
    }

    #[test]
    fn the_hard_number_cases_from_the_plan() {
        let lex = test_lexicon();
        let say = |t: &str, cur: Option<&str>| {
            lex.transcribe(t, Tag::Number, Ctx::default(), cur, None, true)
                .map(|(p, _)| p)
        };
        assert_eq!(say("3.5", None).as_deref(), Some("θɹˈi pYnt fˈIv"));
        assert_eq!(say("3.5x", None).as_deref(), Some("θɹˈi pYnt fˈIv tˈImz"));
        assert_eq!(
            say("4.2M", Some("$")).as_deref(),
            Some("fˈɔɹ pYnt tˈu mˈɪljᵊn dˈɑləɹz"),
            "magnitude carries the currency unit"
        );
        assert_eq!(
            say("4.50", Some("$")).as_deref(),
            Some("fˈɔɹ dˈɑləɹz ænd fˈɪfti sˈɛnts")
        );
    }

    #[test]
    fn a_decimal_is_never_silently_dropped() {
        // The regression that made the macOS voice skip every decimal.
        let lex = test_lexicon();
        for t in ["3.5", "4.2", "2.5"] {
            let out = lex.transcribe(t, Tag::Number, Ctx::default(), None, None, true);
            assert!(
                matches!(out, Some((ref p, _)) if !p.is_empty()),
                "{t} produced nothing"
            );
        }
    }

    #[test]
    fn symbols_are_spoken() {
        let lex = test_lexicon();
        let (ps, _) = lex
            .transcribe("%", Tag::Punctuation, Ctx::default(), None, None, true)
            .unwrap();
        assert_eq!(ps, "pəɹsˈɛnt");
    }

    #[test]
    fn the_article_agrees_with_the_following_vowel() {
        let lex = test_lexicon();
        let before_vowel = Ctx {
            future_vowel: Some(true),
            future_to: false,
        };
        let before_consonant = Ctx {
            future_vowel: Some(false),
            future_to: false,
        };
        assert_eq!(
            lex.transcribe("the", Tag::Determiner, before_vowel, None, None, true)
                .unwrap()
                .0,
            "ði"
        );
        assert_eq!(
            lex.transcribe("the", Tag::Determiner, before_consonant, None, None, true)
                .unwrap()
                .0,
            "ðə"
        );
    }

    #[test]
    fn acronyms_spell_out_but_digits_do_not() {
        let lex = test_lexicon();
        let abc = lex
            .transcribe("A.B.C.", Tag::Noun, Ctx::default(), None, None, true)
            .unwrap()
            .0;
        // Three letters run together, restressed so only the last is primary.
        assert!(
            abc.contains('b') && abc.contains('s'),
            "all letters spelled: {abc}"
        );
        assert_eq!(
            abc.matches(PRIMARY_STRESS).count(),
            1,
            "one primary stress: {abc}"
        );
        // "3.5" has the same dotted shape but must reach the number path.
        let dec = lex
            .transcribe("3.5", Tag::Number, Ctx::default(), None, None, true)
            .unwrap()
            .0;
        assert_eq!(dec, "θɹˈi pYnt fˈIv");
    }
}
