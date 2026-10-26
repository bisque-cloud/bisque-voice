//! eSpeak NG-backed G2P for Spanish, French, Italian, Portuguese and Hindi.
//!
//! Port of `misaki/espeak.py`'s `EspeakG2P`, **plus** the parts of the Python
//! `phonemizer` package it leans on. That second half is not optional: Misaki
//! constructs `phonemizer.backend.EspeakBackend(preserve_punctuation=True,
//! with_stress=True, tie='^', language_switch='remove-flags')`, and three of
//! those four options are implemented in *phonemizer*, not in eSpeak. Calling
//! `espeak_TextToPhonemes` and cleaning up afterwards by eye produces different
//! strings — the phoneme sequence Kokoro was trained on is the one that comes
//! out of this specific pipeline.
//!
//! The pipeline, in order:
//!
//! 1. `EspeakG2P`: `«»` → curly quotes, then `()` → `«»` (below).
//! 2. `Punctuation::preserve`: cut the text at punctuation marks, remembering
//!    each mark and whether it began, ended, or sat inside the line.
//! 3. eSpeak, once per surviving chunk, in IPA + tie mode.
//! 4. `postprocess_line`: collapse separators, strip `(en)`-style language
//!    switch flags, retie.
//! 5. `Punctuation::restore`: put the marks back where they came from.
//! 6. `EspeakG2P`: apply the `E2M` table, delete leftover ties and hyphens,
//!    then `«»` → `()`.
//!
//! Kokoro's language codes come from `kokoro/pipeline.py`'s `LANG_CODES`; note
//! `f` is `fr-fr` and `p` is `pt-br`, not `fr`/`pt`. `tools/gen-g2p-fixtures.py`
//! asserts that table still matches upstream when fixtures are regenerated.

use std::sync::Mutex;

use anyhow::{anyhow, bail, Result};
use fancy_regex::Regex;

use super::Token;

/// eSpeak voice names, keyed by the Kokoro language letter.
///
/// Straight from `kokoro/pipeline.py`. `fr-fr` and `pt-br` are the exact
/// strings Kokoro passes; `fr` is not even a valid eSpeak voice name and
/// `pt` is European Portuguese, which is not what the `p` voices speak.
pub const VOICES: &[(char, &str)] = &[
    ('e', "es"),
    ('f', "fr-fr"),
    ('h', "hi"),
    ('i', "it"),
    ('p', "pt-br"),
];

pub fn voice_for(lang: char) -> Option<&'static str> {
    VOICES.iter().find(|(c, _)| *c == lang).map(|(_, v)| *v)
}

// ---------------------------------------------------------------------------
// phonemizer/punctuation.py
// ---------------------------------------------------------------------------

/// `phonemizer.punctuation._DEFAULT_MARKS`.
const DEFAULT_MARKS: &str = ";:,.!?¡¿—…\"«»“”(){}[]";
/// Marks that double as a decimal separator, and so are not punctuation when
/// they sit between two digits — otherwise "19,99" reaches eSpeak as "19" and
/// "99" and is read as two numbers.
const DECIMAL_SEPARATORS: &str = ",.";

/// Where a mark sat in its line, so `restore` can put it back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Position {
    /// The line began with it.
    Begin,
    /// The line ended with it.
    End,
    /// Somewhere in the middle.
    Interior,
    /// The line was nothing but punctuation.
    Alone,
}

#[derive(Debug, Clone)]
struct MarkIndex {
    index: usize,
    mark: String,
    position: Position,
}

struct Punctuation {
    re: Regex,
}

impl Punctuation {
    fn new(marks: &str) -> Result<Self> {
        // Python builds this as
        //   (\s*(?:[others]|(?<![0-9])[decimals]|[decimals](?![0-9]))+\s*)+
        // The lookarounds are why this uses `fancy-regex` rather than `regex`,
        // which has none. `fancy-regex` is already in the tree via `tokenizers`.
        let decimals: String = marks
            .chars()
            .filter(|c| DECIMAL_SEPARATORS.contains(*c))
            .collect();
        let others: String = marks
            .chars()
            .filter(|c| !DECIMAL_SEPARATORS.contains(*c))
            .collect();

        let mut alternatives: Vec<String> = Vec::new();
        if !others.is_empty() {
            alternatives.push(format!("[{}]", escape_class(&others)));
        }
        if !decimals.is_empty() {
            let escaped = escape_class(&decimals);
            alternatives.push(format!("(?<![0-9])[{escaped}]"));
            alternatives.push(format!("[{escaped}](?![0-9])"));
        }
        let pattern = format!(r"(\s*(?:{})+\s*)+", alternatives.join("|"));
        Ok(Punctuation {
            re: Regex::new(&pattern).map_err(|e| anyhow!("punctuation regex: {e}"))?,
        })
    }

    /// All matches of the mark pattern, as `(start, end)` byte ranges.
    fn matches(&self, line: &str) -> Result<Vec<(usize, usize)>> {
        let mut out = Vec::new();
        for m in self.re.find_iter(line) {
            let m = m.map_err(|e| anyhow!("scanning punctuation: {e}"))?;
            out.push((m.start(), m.end()));
        }
        Ok(out)
    }

    /// `Punctuation.preserve` for a single line.
    ///
    /// Returns the text chunks between marks (empties dropped, as Python does
    /// via `if line`) and the marks themselves.
    fn preserve(&self, line: &str) -> Result<(Vec<String>, Vec<MarkIndex>)> {
        let found = self.matches(line)?;
        if found.is_empty() {
            return Ok((
                if line.is_empty() {
                    Vec::new()
                } else {
                    vec![line.to_string()]
                },
                Vec::new(),
            ));
        }

        // A line that is nothing but punctuation.
        if found.len() == 1 && found[0].0 == 0 && found[0].1 == line.len() {
            return Ok((
                Vec::new(),
                vec![MarkIndex {
                    index: 0,
                    mark: line.to_string(),
                    position: Position::Alone,
                }],
            ));
        }

        let last = found.len() - 1;
        let mut marks: Vec<MarkIndex> = Vec::with_capacity(found.len());
        for (i, (start, end)) in found.iter().copied().enumerate() {
            let text = &line[start..end];
            let position = if i == 0 && start == 0 {
                Position::Begin
            } else if i == last && end == line.len() {
                Position::End
            } else {
                Position::Interior
            };
            marks.push(MarkIndex {
                index: 0,
                mark: text.to_string(),
                position,
            });
        }

        // Python splits on the mark *text* rather than the match offsets:
        //   split = line.split(mark.mark); prefix, suffix = split[0], join(rest)
        // so an identical mark occurring earlier wins. Reproduced exactly,
        // because the chunk boundaries decide what eSpeak is handed.
        let mut rest = line.to_string();
        let mut chunks: Vec<String> = Vec::new();
        for mark in &marks {
            match rest.find(mark.mark.as_str()) {
                Some(at) => {
                    chunks.push(rest[..at].to_string());
                    rest = rest[at + mark.mark.len()..].to_string();
                }
                None => {
                    // Cannot happen for marks we just matched, but refusing to
                    // guess is better than silently dropping text.
                    bail!("punctuation mark {:?} vanished from the line", mark.mark);
                }
            }
        }
        chunks.push(rest);
        Ok((chunks.into_iter().filter(|c| !c.is_empty()).collect(), marks))
    }

    /// `Punctuation.restore`, specialized to one input line and the default
    /// separator (`word = " "`, `strip = False`).
    fn restore(mut text: Vec<String>, mut marks: Vec<MarkIndex>) -> Vec<String> {
        const SEP: &str = " ";
        let mut out: Vec<String> = Vec::new();
        let mut pos = 0usize;

        while !text.is_empty() || !marks.is_empty() {
            if marks.is_empty() {
                for line in text.drain(..) {
                    // strip is False, so every line ends with a word separator.
                    if line.ends_with(SEP) {
                        out.push(line);
                    } else {
                        out.push(line + SEP);
                    }
                }
            } else if text.is_empty() {
                out.push(
                    marks
                        .iter()
                        .map(|m| m.mark.as_str())
                        .collect::<String>()
                        .replace(' ', SEP),
                );
                marks.clear();
            } else if marks[0].index == pos {
                let current = marks.remove(0);
                let mark = current.mark.replace(' ', SEP);

                if text[0].ends_with(SEP) {
                    let keep = text[0].len() - SEP.len();
                    text[0].truncate(keep);
                }

                match current.position {
                    Position::Begin => {
                        text[0] = mark + &text[0];
                    }
                    Position::End => {
                        let tail = if mark.ends_with(SEP) { "" } else { SEP };
                        out.push(format!("{}{}{}", text.remove(0), mark, tail));
                        pos += 1;
                    }
                    Position::Alone => {
                        let tail = if mark.ends_with(SEP) { "" } else { SEP };
                        out.push(format!("{mark}{tail}"));
                        pos += 1;
                    }
                    Position::Interior => {
                        if text.len() == 1 {
                            // The tail after an interior mark was never
                            // phonemized; keep the mark rather than lose it.
                            text[0] = format!("{}{}", text[0], mark);
                        } else {
                            let first = text.remove(0);
                            text[0] = format!("{}{}{}", first, mark, text[0]);
                        }
                    }
                }
            } else {
                out.push(text.remove(0));
                pos += 1;
            }
        }
        out
    }
}

/// Escape the characters that are special inside a regex character class.
///
/// Only these five are special between `[` and `]`; the rest of the regex
/// metacharacters are literal there. Written as a match rather than a string of
/// specials so the backslash and the bracket cannot be miscounted.
fn escape_class(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        if matches!(c, '\\' | '^' | '-' | ']' | '[') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

// ---------------------------------------------------------------------------
// phonemizer/backend/espeak
// ---------------------------------------------------------------------------

/// eSpeak NG holds its voice in a process global and is not reentrant, so every
/// call is serialized. Phonemization is far cheaper than synthesis; this has
/// never been the bottleneck.
static ESPEAK: Mutex<()> = Mutex::new(());

mod ffi {
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::path::Path;
    use std::sync::OnceLock;

    use anyhow::{anyhow, bail, Result};

    static INIT: OnceLock<Result<(), String>> = OnceLock::new();

    /// `phonemizer.backend.espeak.wrapper`'s phoneme mode when a tie is asked
    /// for: IPA output (bit 1), "bits 8-23 are a tie character" (bit 7), and
    /// U+0361 as that character. Misaki then rewrites U+0361 to `^`.
    fn phoneme_mode() -> i32 {
        (0x02u32 | (0x01u32 << 7) | (0x0361u32 << 8)) as i32
    }

    /// Initialize eSpeak against the data directory in the installed pack.
    ///
    /// eSpeak can only be initialized once per process, so the first caller's
    /// data path wins; every language we support lives in the same directory.
    pub fn init(data_parent: &Path) -> Result<()> {
        let result = INIT.get_or_init(|| {
            let path = CString::new(data_parent.to_string_lossy().as_ref())
                .map_err(|_| "data path contains a null byte".to_string())?;
            // AUDIO_OUTPUT_RETRIEVAL + DONT_EXIT: we never play audio, and a
            // library that calls exit() on bad input has no place in a CLI.
            let rate = unsafe {
                espeak_rs_sys::espeak_Initialize(
                    espeak_rs_sys::espeak_AUDIO_OUTPUT_AUDIO_OUTPUT_RETRIEVAL,
                    0,
                    path.as_ptr(),
                    espeak_rs_sys::espeakINITIALIZE_DONT_EXIT as i32,
                )
            };
            if rate <= 0 {
                return Err(format!(
                    "eSpeak NG refused to initialize (code {rate}) against {}. \
                     The pack's espeak-ng-data directory is missing or does not \
                     match the linked library.",
                    data_parent.display()
                ));
            }
            Ok(())
        });
        result.clone().map_err(|e| anyhow!("{e}"))
    }

    /// `EspeakWrapper.set_voice`: resolve a *language code* to a voice
    /// **identifier**, then select that.
    ///
    /// The distinction is not pedantry. `espeak_SetVoiceByName("fr-fr")` fails
    /// outright — the voice file is `roa/fr` and `fr-fr` is only the language
    /// it declares. phonemizer walks `espeak_ListVoices`, maps each voice's
    /// first declared language to its identifier keeping the **first** voice
    /// per language (the list is ordered by relevance), and selects by that.
    /// Picking a different voice for the same language would produce different
    /// phonemes, so the tie-break is part of the specification.
    ///
    /// mbrola voices are skipped: they list under the same language code as the
    /// plain voice and can sort first, but they need the mbrola binary, which
    /// is not part of an eSpeak installation.
    pub fn set_voice(language: &str) -> Result<()> {
        let identifier = voice_identifier(language)?;
        let c = CString::new(identifier.clone())
            .map_err(|_| anyhow!("voice identifier has a null byte"))?;
        let rc = unsafe { espeak_rs_sys::espeak_SetVoiceByName(c.as_ptr()) };
        if rc != espeak_rs_sys::espeak_ERROR_EE_OK {
            bail!("eSpeak NG refused voice {identifier:?} for language {language:?} (code {rc})");
        }
        Ok(())
    }

    fn voice_identifier(language: &str) -> Result<String> {
        // A null selector asks for every voice. The result is a null-terminated
        // array of pointers owned by eSpeak; we only read from it.
        let voices = unsafe { espeak_rs_sys::espeak_ListVoices(std::ptr::null_mut()) };
        if voices.is_null() {
            bail!("eSpeak NG returned no voice list");
        }
        let mut first_match: Option<String> = None;
        let mut index = 0isize;
        loop {
            let entry = unsafe { *voices.offset(index) };
            if entry.is_null() {
                break;
            }
            index += 1;
            let voice = unsafe { &*entry };
            if voice.languages.is_null() || voice.identifier.is_null() {
                continue;
            }
            let identifier = unsafe { CStr::from_ptr(voice.identifier) }.to_string_lossy();
            if identifier.starts_with("mb/") || identifier.starts_with("mb\\") {
                continue;
            }
            // `languages` is a priority byte followed by the language name. The
            // C string stops at the first NUL, so this is the voice's primary
            // language, which is what phonemizer keys on.
            let raw = unsafe { CStr::from_ptr(voice.languages) }.to_bytes();
            if raw.is_empty() {
                continue;
            }
            let declared = String::from_utf8_lossy(&raw[1..]);
            if declared == language && first_match.is_none() {
                first_match = Some(identifier.into_owned());
            }
        }
        first_match.ok_or_else(|| {
            anyhow!(
                "eSpeak NG has no voice for language {language:?}. The pack's \
                 espeak-ng-data may be missing this language's dictionary."
            )
        })
    }

    /// `EspeakWrapper.text_to_phonemes`: call until the text pointer is
    /// exhausted, joining the clauses with a space.
    pub fn text_to_phonemes(text: &str) -> Result<String> {
        // eSpeak reads through the buffer clause by clause, so it must outlive
        // the loop.
        let owned = CString::new(text).map_err(|_| anyhow!("text has a null byte"))?;
        let mut ptr: *const c_char = owned.as_ptr();
        let mode = phoneme_mode();
        let mut parts: Vec<String> = Vec::new();
        while !ptr.is_null() {
            let res = unsafe {
                espeak_rs_sys::espeak_TextToPhonemes(
                    &mut ptr as *mut *const c_char as *mut *const c_void,
                    espeak_rs_sys::espeakCHARS_UTF8 as i32,
                    mode,
                )
            };
            if res.is_null() {
                continue;
            }
            parts.push(unsafe { CStr::from_ptr(res) }.to_string_lossy().into_owned());
        }
        Ok(parts.join(" "))
    }
}

/// `EspeakBackend._postprocess_line`, for `tie='^'`, `with_stress=True`,
/// `language_switch='remove-flags'`, `strip=False`, default separator.
fn postprocess_line(line: &str, flags: &Regex) -> Result<String> {
    let mut line = line.trim().replace('\n', " ").replace("  ", " ");

    // espeak-ng #694: stray separators at the end of a word.
    while line.contains("__") {
        line = line.replace("__", "_");
    }
    line = line.replace("_ ", " ");

    // 'remove-flags': drop every (xx) language-switch marker.
    if flags.is_match(&line).map_err(|e| anyhow!("{e}"))? {
        line = flags.replace_all(&line, "").into_owned();
    }
    if line.is_empty() {
        return Ok(String::new());
    }

    // with_stress=True, so stress marks survive; tie is set, so no '_' is
    // appended and the default tie is rewritten to the requested one.
    let mut out = String::with_capacity(line.len() + 8);
    for word in line.split(' ') {
        out.push_str(&word.trim().replace('\u{361}', "^"));
        out.push(' ');
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// misaki/espeak.py EspeakG2P
// ---------------------------------------------------------------------------

/// `EspeakG2P.E2M`, verbatim.
///
/// Applied in Python's `sorted(dict.items())` order — plain lexicographic by
/// key, **not** longest-first (that is `EspeakFallback`'s table, a different
/// one). Every key here is two code points joined by the tie, so no key is a
/// prefix of another and the order does not change the result; it is preserved
/// so the two tables cannot be conflated later.
const E2M: &[(&str, &str)] = &[
    ("a^ɪ", "I"),
    ("a^ʊ", "W"),
    ("d^z", "ʣ"),
    ("d^ʒ", "ʤ"),
    ("e^ɪ", "A"),
    ("o^ʊ", "O"),
    ("s^s", "S"),
    ("t^s", "ʦ"),
    ("t^ʃ", "ʧ"),
    ("ɔ^ɪ", "Y"),
    ("ə^ʊ", "Q"),
];

pub struct EspeakG2P {
    voice: &'static str,
    punctuation: Punctuation,
    flags: Regex,
}

impl EspeakG2P {
    /// `data_parent` is the directory *containing* `espeak-ng-data`.
    pub fn new(voice: &'static str, data_parent: &std::path::Path) -> Result<Self> {
        ffi::init(data_parent)?;
        Ok(EspeakG2P {
            voice,
            punctuation: Punctuation::new(DEFAULT_MARKS)?,
            // BaseLanguageSwitch._ESPEAK_FLAGS_RE
            flags: Regex::new(r"\(.+?\)").map_err(|e| anyhow!("flags regex: {e}"))?,
        })
    }

    /// The whole pipeline, text in, phoneme string out.
    pub fn phonemize_text(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }
        // Angles to curly quotes, then parentheses to angles. Both are in the
        // punctuation set, so this smuggles parentheses through eSpeak as
        // marks; step 6 puts them back.
        let text = text.replace('«', "\u{201C}").replace('»', "\u{201D}");
        let text = text.replace('(', "«").replace(')', "»");

        let (chunks, marks) = self.punctuation.preserve(&text)?;

        let phonemized = {
            let _serialized = ESPEAK.lock().unwrap_or_else(|e| e.into_inner());
            ffi::set_voice(self.voice)?;
            let mut out = Vec::with_capacity(chunks.len());
            for chunk in &chunks {
                let raw = ffi::text_to_phonemes(chunk)?;
                out.push(postprocess_line(&raw, &self.flags)?);
            }
            out
        };

        let restored = Punctuation::restore(phonemized, marks);
        let mut ps = restored.first().cloned().unwrap_or_default().trim().to_string();

        for (from, to) in E2M {
            if ps.contains(from) {
                ps = ps.replace(from, to);
            }
        }
        // Any tie that did not form a mapped digraph, and hyphens, are dropped —
        // Misaki's comment says "not sure what they mean", and Kokoro's
        // vocabulary has neither.
        ps = ps.replace('^', "").replace('-', "");
        ps = ps.replace('«', "(").replace('»', ")");
        Ok(ps)
    }

    /// Phonemes as `Token`s, one per space-separated run.
    ///
    /// See `super::align_tokens` for why the surface text is attached the way
    /// it is, and what it costs when eSpeak's word count does not match the
    /// source's.
    pub fn phonemize(&self, text: &str) -> Result<Vec<Token>> {
        Ok(super::align_tokens(text, &self.phonemize_text(text)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kokoro_language_codes_are_the_ones_kokoro_uses() {
        // fr-fr and pt-br specifically: "fr" is not a valid eSpeak voice name
        // and "pt" is European Portuguese, which the p_ voices do not speak.
        assert_eq!(voice_for('f'), Some("fr-fr"));
        assert_eq!(voice_for('p'), Some("pt-br"));
        assert_eq!(voice_for('e'), Some("es"));
        assert_eq!(voice_for('i'), Some("it"));
        assert_eq!(voice_for('h'), Some("hi"));
        assert_eq!(voice_for('a'), None, "English does not go through eSpeak");
        assert_eq!(voice_for('z'), None, "Chinese does not go through eSpeak");
        assert_eq!(voice_for('j'), None, "Japanese does not go through eSpeak");
    }

    #[test]
    fn a_decimal_separator_between_digits_is_not_punctuation() {
        // The bug this guards: splitting "19,99" makes eSpeak read two numbers,
        // so "19,99 euro" becomes "nineteen ninety-nine euro".
        let p = Punctuation::new(DEFAULT_MARKS).unwrap();
        let (chunks, marks) = p.preserve("Tengo 19,99 euros.").unwrap();
        assert_eq!(chunks, vec!["Tengo 19,99 euros"]);
        assert_eq!(marks.len(), 1);
        assert_eq!(marks[0].mark, ".");
        assert_eq!(marks[0].position, Position::End);
    }

    #[test]
    fn a_separator_next_to_a_non_digit_is_still_punctuation() {
        let p = Punctuation::new(DEFAULT_MARKS).unwrap();
        let (chunks, _) = p.preserve("Hola, mundo").unwrap();
        assert_eq!(chunks, vec!["Hola", "mundo"]);
    }

    #[test]
    fn mark_positions_are_recorded() {
        let p = Punctuation::new(DEFAULT_MARKS).unwrap();
        let (_, marks) = p.preserve("¿Cómo estás? Bien.").unwrap();
        let positions: Vec<Position> = marks.iter().map(|m| m.position).collect();
        assert_eq!(
            positions,
            vec![Position::Begin, Position::Interior, Position::End]
        );
    }

    #[test]
    fn a_line_of_pure_punctuation_is_marked_alone() {
        let p = Punctuation::new(DEFAULT_MARKS).unwrap();
        let (chunks, marks) = p.preserve("...").unwrap();
        assert!(chunks.is_empty());
        assert_eq!(marks.len(), 1);
        assert_eq!(marks[0].position, Position::Alone);
    }

    #[test]
    fn restore_is_the_inverse_of_preserve_for_plain_text() {
        let p = Punctuation::new(DEFAULT_MARKS).unwrap();
        let (chunks, marks) = p.preserve("uno, dos. tres").unwrap();
        // Stand in for phonemization: each chunk comes back with the trailing
        // word separator the backend adds.
        let phonemized: Vec<String> = chunks.iter().map(|c| format!("{c} ")).collect();
        let out = Punctuation::restore(phonemized, marks);
        assert_eq!(out.first().map(|s| s.trim()), Some("uno, dos. tres"));
    }

    #[test]
    fn language_switch_flags_are_removed_not_kept() {
        let flags = Regex::new(r"\(.+?\)").unwrap();
        let out = postprocess_line("ʒɛm lə (en)fʊtbɔːl(fr)", &flags).unwrap();
        assert!(!out.contains('('), "{out:?}");
        assert!(out.contains("fʊtbɔːl"), "the word itself is preserved: {out:?}");
    }

    #[test]
    fn the_default_tie_is_rewritten_to_the_requested_one() {
        let flags = Regex::new(r"\(.+?\)").unwrap();
        let out = postprocess_line("a\u{361}ɪ", &flags).unwrap();
        assert_eq!(out, "a^ɪ ");
    }

    #[test]
    fn e2m_keys_are_all_tied_digraphs_so_order_cannot_matter() {
        // Guards against someone pasting EspeakFallback's table (which IS
        // order-sensitive and sorted longest-first) in here by mistake.
        for (from, _) in E2M {
            assert!(from.contains('^'), "{from:?} is not a tied digraph");
        }
        for (a, _) in E2M {
            for (b, _) in E2M {
                if a != b {
                    assert!(!a.starts_with(b), "{a:?} has prefix {b:?}");
                }
            }
        }
    }
}
