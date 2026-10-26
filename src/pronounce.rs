//! Author pronunciation overrides: `[live](/lˈɪv/)` and `[live](liv)`.
//!
//! Deliberately NOT inside `g2p`, which is gated on the `kokoro` feature. The
//! Intel macOS build has no ONNX Runtime and therefore no Kokoro, but it still
//! ships the Studio engines — and those are exactly the engines that can only
//! honour an override by substituting the text they are given. Gating this
//! alongside Kokoro left the build that needs it most unable to compile.
//!
//! Pure string work: no lexicon, no model, no I/O.
//!
//! The same grammar lives in `packages/presentation-format/src/pronunciation.ts`
//! in the Bisque monorepo, and the two must agree.

const PUNCTUATION: &str = ";:,.!?—…\"“”";

/// Parse an author override starting at `[` in `chars[i..]`.
///
/// `[live](/lˈɪv/)` pins phonemes; `[live](liv)` substitutes the text sent to
/// synthesis. Returns the display text, the IPA, the substitute, and the index
/// just past the closing `)`. Anything that does not parse cleanly is left
/// alone and read as literal characters, so ordinary prose containing a bracket
/// is never swallowed.
///
/// The same grammar lives in `packages/presentation-format/src/pronunciation.ts`
/// in the Bisque monorepo, and the two must agree.
pub fn parse_override(
    chars: &[char],
    i: usize,
) -> Option<(String, Option<String>, Option<String>, usize)> {
    let close = chars[i + 1..].iter().position(|c| *c == ']')? + i + 1;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let end = chars[close + 2..].iter().position(|c| *c == ')')? + close + 2;
    let display: String = chars[i + 1..close].iter().collect();
    let body: String = chars[close + 2..end].iter().collect();
    if display.trim().is_empty() || body.trim().is_empty() {
        return None;
    }
    // A bracket run holding a newline is prose, not a marker.
    if display.contains('\n') || body.contains('\n') {
        return None;
    }
    // A display holding another `[` is a marker nested inside a marker
    // (`[[Changelog](/…/)'s](/…/)`). Refusing the outer one lets the inner
    // one parse; the leftover `](` is what publish rejects on the TypeScript
    // side (`findMalformedPronunciation`), so it never reaches a listener.
    if display.contains('[') {
        return None;
    }
    // A body wrapped in slashes is IPA. Slashes with nothing between them is
    // an override the author started and did not finish, and speaking the
    // slashes would be worse than leaving the word alone, so it stays prose.
    // Kept identical to `stripPronunciation` in
    // `packages/presentation-format/src/pronunciation.ts`; the two decide
    // token counts that must match, and a publish is rejected if they differ.
    let (ipa, spoken) = match body.strip_prefix('/').and_then(|b| b.strip_suffix('/')) {
        Some(p) if !p.trim().is_empty() => (Some(p.to_string()), None),
        Some(_) => return None,
        None => (None, Some(body)),
    };
    Some((display, ipa, spoken, end + 1))
}

/// Rewrite every override to one of its two sides.
///
/// `display` picks the word a reader sees; otherwise the substitute is used
/// where one was given. An IPA override has no substitute, and an engine with
/// no phoneme stage cannot be handed IPA, so it falls back to the display word
/// — which is why the respelling form is the portable one.
fn rewrite_overrides(text: &str, display: bool) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == '[' {
            if let Some((shown, _ipa, spoken, next)) = parse_override(&chars, i) {
                out.push_str(if display {
                    &shown
                } else {
                    spoken.as_deref().unwrap_or(&shown)
                });
                i = next;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// The narration with every override reduced to the word a reader sees.
///
/// Everything that counts or shows words uses this: the forced aligner's
/// transcript, the reconciled word list, and the 1:1 count the publish path
/// checks. Never the synthesis input.
pub fn display_text(text: &str) -> String {
    rewrite_overrides(text, true)
}

/// The narration with every override reduced to what should be said.
///
/// For engines that take text and nothing else, this is the only lever there
/// is. Only synthesis sees it.
pub fn spoken_text(text: &str) -> String {
    rewrite_overrides(text, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_spoken_are_two_different_strings() {
        let src = "where its tensors already [live](liv).";
        assert_eq!(display_text(src), "where its tensors already live.");
        assert_eq!(spoken_text(src), "where its tensors already liv.");
    }

    #[test]
    fn an_ipa_override_falls_back_to_the_display_word_when_asked_to_speak() {
        // An engine with no phoneme stage cannot be handed IPA, so the spoken
        // side degrades to the word itself rather than to gibberish. This is
        // why the respelling form is the portable one.
        let src = "the [ggml](/ʤˌiʤˌiˌɛmˈɛl/) library";
        assert_eq!(display_text(src), "the ggml library");
        assert_eq!(spoken_text(src), "the ggml library");
    }

    #[test]
    fn display_text_word_count_is_what_the_aligner_will_report() {
        // main.rs feeds `display_text` to both the aligner and reconcile, and
        // bails when they disagree. The multi-word case is the one that breaks
        // if this counts the raw marker instead.
        let words = |s: &str| s.split_whitespace().count();
        assert_eq!(words(&display_text("say [New York](noo york) twice")), 4);
        assert_eq!(words(&display_text("[ggml](G-G-M-L) ships")), 2);
    }

    #[test]
    fn a_marker_nested_in_a_marker_parses_only_the_inner_one() {
        // The outer `[` must not swallow `[Changelog](/…/)` as its display
        // text; the inner marker is the one that carries the phonemes. The
        // leftover `'s](/…/)` stays literal here — the TypeScript validator
        // rejects the publish before this text is ever synthesized.
        let src = "the [[Changelog](/ʧˈAnʤlɔɡ/)'s](/ʧˈAnʤlɔɡz/) shows";
        let chars: Vec<char> = src.chars().collect();
        assert!(parse_override(&chars, 4).is_none(), "outer marker must not parse");
        let (display, ipa, _, _) = parse_override(&chars, 5).expect("inner marker parses");
        assert_eq!(display, "Changelog");
        assert_eq!(ipa.as_deref(), Some("ʧˈAnʤlɔɡ"));
        assert_eq!(
            display_text(src),
            "the [Changelog's](/ʧˈAnʤlɔɡz/) shows"
        );
    }

    #[test]
    fn text_without_markers_is_returned_unchanged() {
        let plain = "an array [0] of things, see [the docs]";
        assert_eq!(display_text(plain), plain);
        assert_eq!(spoken_text(plain), plain);
    }
}
