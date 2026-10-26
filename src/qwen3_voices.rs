//! Per-voice metadata for the Qwen3-TTS CustomVoice named speakers.
//!
//! The nine speakers ship inside the `qwen3-voices-*` packs' talker GGUF
//! (`qwen3-tts.codec.speaker_names` / `speaker_dialects` metadata, read by the
//! vendored qwentts.cpp at load time) — this table is the *listing* metadata
//! for them, in the same shape as `kokoro_voices::VOICES` so every enumeration
//! surface (`engines --json`, the human catalogue, doctor) renders them the
//! same way.
//!
//! Same policy as Kokoro's table (`VOICE-CURATION.md` is canonical):
//! metadata informs choice and never gates synthesis — the engine resolves a
//! speaker against the model's own speaker table, not against this list.
//!
//! Two differences from the Kokoro set, both facts about upstream rather than
//! choices here:
//!
//! - **No grades.** hexgrad publishes per-voice target-quality grades for
//!   Kokoro; Qwen publishes none for the CustomVoice speakers. The grade
//!   column is "—" (present, so the serialized shape stays uniform across
//!   engines) rather than an invented letter.
//! - **All nine are recommended**, including the two dialect voices: `eric`
//!   (Sichuan-dialect Chinese) and `dylan` (Beijing-dialect Chinese) are high
//!   quality *in their language* — the accent column and note say what that
//!   language is, and choosing them for English narration is the mismatch,
//!   not the voice.

use crate::kokoro_voices::VoiceSpec;

/// No upstream grade exists for these speakers; "—" keeps the field a string
/// so consumers of `engines --json` read one shape for every engine.
const UNGRADED: &str = "—";

const fn v(
    id: &'static str,
    display_name: &'static str,
    accent: &'static str,
    gender: &'static str,
    note: Option<&'static str>,
) -> VoiceSpec {
    VoiceSpec {
        id,
        display_name,
        accent,
        gender,
        grade: UNGRADED,
        recommended: true,
        listed: true,
        note,
    }
}

/// The nine CustomVoice speakers, verbatim from the checkpoint's speaker
/// table. `accent` carries each voice's language affinity: the checkpoint is
/// multilingual, but `ono_anna`/`sohee` lean toward their home language and
/// `eric`/`dylan` *are* their dialect — the model overrides the language id
/// with the dialect when the text is Chinese (or language is unstated).
#[rustfmt::skip]
pub const VOICES: &[VoiceSpec] = &[
    v("serena",    "Serena",    "Multilingual",               "female", None),
    v("vivian",    "Vivian",    "Multilingual",               "female", None),
    v("uncle_fu",  "Uncle Fu",  "Multilingual",               "male",   None),
    v("ryan",      "Ryan",      "Multilingual",               "male",   None),
    v("aiden",     "Aiden",     "Multilingual",               "male",   None),
    v("ono_anna",  "Ono Anna",  "Japanese-leaning",           "female", None),
    v("sohee",     "Sohee",     "Korean-leaning",             "female", None),
    v("eric",      "Eric",      "Chinese (Sichuan dialect)",  "male",
      Some("best in Chinese — the dialect is the voice")),
    v("dylan",     "Dylan",     "Chinese (Beijing dialect)",  "male",
      Some("best in Chinese — the dialect is the voice")),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The speaker ids must be exactly the checkpoint's speaker table — the
    /// engine resolves `--voice` against the model, and a listed id the model
    /// does not know would be an offer that fails at synthesis time.
    #[test]
    fn the_nine_speakers_match_the_checkpoint_verbatim() {
        let ids: Vec<&str> = VOICES.iter().map(|v| v.id).collect();
        assert_eq!(
            ids,
            [
                "serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna",
                "sohee", "eric", "dylan"
            ],
            "ids come from qwen3-tts.codec.speaker_names, in table order"
        );
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "duplicate speaker id");
    }

    /// Every speaker is listed and recommended — there is no graded tail here,
    /// and hiding a shipped speaker would contradict the checkpoint. The two
    /// dialect voices carry a note saying where their quality lives.
    #[test]
    fn all_listed_all_recommended_with_dialects_annotated() {
        for voice in VOICES {
            assert!(voice.listed, "{}", voice.id);
            assert!(voice.recommended, "{}", voice.id);
            assert_eq!(voice.grade, UNGRADED, "{}: no upstream grade exists", voice.id);
            assert!(!voice.display_name.is_empty(), "{}", voice.id);
            assert!(!voice.accent.is_empty(), "{}", voice.id);
            assert!(
                matches!(voice.gender, "female" | "male"),
                "{}: {}",
                voice.id,
                voice.gender
            );
        }
        for id in ["eric", "dylan"] {
            let v = VOICES.iter().find(|v| v.id == id).unwrap();
            assert!(
                v.accent.contains("dialect"),
                "{id}'s accent must name its dialect: {}",
                v.accent
            );
            assert!(
                v.note.expect("dialect voices carry a note").contains("Chinese"),
                "{id}'s note must say where its quality lives"
            );
        }
    }
}
