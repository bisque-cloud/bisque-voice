//! Per-voice quality metadata for Kokoro's English voices.
//!
//! The grades are upstream's own target-quality grades, copied **verbatim**
//! from hexgrad/Kokoro-82M's `VOICES.md`
//! (<https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md>). A low
//! grade means audible quality issues — upstream trained that voice on very
//! little data — not a licensing or availability distinction.
//!
//! Policy (see `VOICE-CURATION.md` for the canonical text):
//!
//! - **Metadata never gates synthesis.** Any voice named explicitly via
//!   `--voice` synthesizes, including unlisted ones. Resolution and the
//!   engine's voice lookup (`model::KokoroEngine::voice_row`, which reads the
//!   pack's `voices/<id>.bin` directly) do not consult this table.
//! - **Every listing surface carries the grade and a recommended flag**, with
//!   recommended voices first, so humans and LLM agents alike can choose well.
//! - **`listed: false` voices appear in no listing at all** — today that is
//!   `af_nicole`, a whisper/ASMR-register voice that is wrong for presentation
//!   narration — but stay synthesizable when named explicitly.
//!
//! Kokoro's pack also carries non-English voices (55 voices total); language
//! is part of the voice id (`g2p::router`), and those voices are outside this
//! graded set — absent from the table is "ungraded here", not "unlisted".

use serde::Serialize;

/// One graded Kokoro voice. `listed: false` removes a voice from every
/// listing surface (and is therefore never serialized — an emitted voice is
/// listed by construction); it never blocks explicit synthesis.
#[derive(Debug, Clone, Serialize)]
pub struct VoiceSpec {
    pub id: &'static str,
    #[serde(rename = "displayName")]
    pub display_name: &'static str,
    pub accent: &'static str,
    pub gender: &'static str,
    /// Upstream's target-quality grade, verbatim (A … F+).
    pub grade: &'static str,
    /// Grade C+ or above, and listed. Everything else has audible quality
    /// issues per upstream's own grading.
    pub recommended: bool,
    #[serde(skip)]
    pub listed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<&'static str>,
}

const fn v(
    id: &'static str,
    display_name: &'static str,
    accent: &'static str,
    gender: &'static str,
    grade: &'static str,
    recommended: bool,
    listed: bool,
    note: Option<&'static str>,
) -> VoiceSpec {
    VoiceSpec {
        id,
        display_name,
        accent,
        gender,
        grade,
        recommended,
        listed,
        note,
    }
}

const AF: &str = "American English";
const BF: &str = "British English";

/// The graded English voices, in listing order: recommended first (by grade,
/// best first), then the rest (same order). `engines --json` serializes this
/// slice filtered to `listed` voices, so the order here IS the order every
/// consumer sees.
#[rustfmt::skip]
pub const VOICES: &[VoiceSpec] = &[
    // ── Recommended: grade C+ and above ─────────────────────────────────
    v("af_heart",    "Heart",    AF, "female", "A",  true,  true,  Some("default")),
    v("af_bella",    "Bella",    AF, "female", "A-", true,  true,  None),
    v("bf_emma",     "Emma",     BF, "female", "B-", true,  true,  None),
    v("af_aoede",    "Aoede",    AF, "female", "C+", true,  true,  None),
    v("af_kore",     "Kore",     AF, "female", "C+", true,  true,  None),
    v("af_sarah",    "Sarah",    AF, "female", "C+", true,  true,  None),
    v("am_fenrir",   "Fenrir",   AF, "male",   "C+", true,  true,  None),
    v("am_michael",  "Michael",  AF, "male",   "C+", true,  true,  None),
    v("am_puck",     "Puck",     AF, "male",   "C+", true,  true,  None),
    v("bf_isabella", "Isabella", BF, "female", "C",  true , true,  None),
    v("bm_fable",    "Fable",    BF, "male",   "C",  true , true,  None),
    v("bm_george",   "George",   BF, "male",   "C",  true , true,  None),
    // ── Unlisted: never enumerated, still synthesizable explicitly ──────
    v("af_nicole",   "Nicole",   AF, "female", "B-", false, false,
      Some("whisper register — unsuitable for narration")),
    // ── Not recommended: audible quality issues (small upstream data) ───
    v("af_alloy",    "Alloy",    AF, "female", "C",  false, true,  None),
    v("af_nova",     "Nova",     AF, "female", "C",  false, true,  None),
    v("af_sky",      "Sky",      AF, "female", "C-", false, true,  None),
    v("bm_lewis",    "Lewis",    BF, "male",   "D+", false, true,  None),
    v("af_jessica",  "Jessica",  AF, "female", "D",  false, true,  None),
    v("af_river",    "River",    AF, "female", "D",  false, true,  None),
    v("am_echo",     "Echo",     AF, "male",   "D",  false, true,  None),
    v("am_eric",     "Eric",     AF, "male",   "D",  false, true,  None),
    v("am_liam",     "Liam",     AF, "male",   "D",  false, true,  None),
    v("am_onyx",     "Onyx",     AF, "male",   "D",  false, true,  None),
    v("bf_alice",    "Alice",    BF, "female", "D",  false, true,  None),
    v("bf_lily",     "Lily",     BF, "female", "D",  false, true,  None),
    v("bm_daniel",   "Daniel",   BF, "male",   "D",  false, true,  None),
    v("am_santa",    "Santa",    AF, "male",   "D-", false, true,  None),
    v("am_adam",     "Adam",     AF, "male",   "F+", false, true,  None),
];

/// Look a voice up by id. Metadata only — nothing in the synthesis path
/// calls this, which is what keeps the never-block guarantee structural.
#[cfg_attr(not(test), allow(dead_code))]
pub fn spec(id: &str) -> Option<&'static VoiceSpec> {
    VOICES.iter().find(|v| v.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Best-first position of a grade; used only to assert internal
    /// consistency of the hand-written table.
    fn rank(grade: &str) -> usize {
        const ORDER: &[&str] = &[
            "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F+",
        ];
        ORDER
            .iter()
            .position(|g| *g == grade)
            .unwrap_or_else(|| panic!("unknown grade {grade:?}"))
    }

    #[test]
    fn voice_ids_are_unique_and_shaped_like_kokoro_ids() {
        let mut ids: Vec<&str> = VOICES.iter().map(|v| v.id).collect();
        assert_eq!(ids.len(), 28, "the graded English set is 28 voices");
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), before, "duplicate voice id");
        for voice in VOICES {
            // The id prefix carries accent and gender; the columns must agree
            // with it or the table is lying about one of them.
            let (accent, gender) = match &voice.id[..3] {
                "af_" => (AF, "female"),
                "am_" => (AF, "male"),
                "bf_" => (BF, "female"),
                "bm_" => (BF, "male"),
                other => panic!("{}: unexpected prefix {other:?}", voice.id),
            };
            assert_eq!(voice.accent, accent, "{}", voice.id);
            assert_eq!(voice.gender, gender, "{}", voice.id);
        }
    }

    /// The policy line: every C+-or-above listed voice is recommended, and
    /// nothing below C is. The band between (grade C) admits deliberate
    /// coverage picks — currently bf_isabella / bm_fable / bm_george, kept so
    /// the recommended cast has British voices of both genders. Widening the
    /// coverage set is a curation decision and must change this list too.
    #[test]
    fn recommendation_follows_grade_with_named_coverage_picks() {
        const COVERAGE_PICKS: [&str; 3] = ["bf_isabella", "bm_fable", "bm_george"];
        for voice in VOICES {
            let c_plus_or_better = rank(voice.grade) <= rank("C+");
            let coverage = COVERAGE_PICKS.contains(&voice.id);
            let expected = (c_plus_or_better || coverage) && voice.listed;
            assert_eq!(
                voice.recommended,
                expected,
                "{} ({}, listed: {}, coverage: {})",
                voice.id,
                voice.grade,
                voice.listed,
                coverage
            );
            if voice.recommended && !c_plus_or_better {
                assert!(coverage, "{} recommended below C+ without a coverage entry", voice.id);
            }
        }
    }

    /// Listing surfaces serialize this slice in order, so the order is the
    /// contract: every recommended voice before every non-recommended one,
    /// and the default first.
    #[test]
    fn recommended_voices_come_first_and_the_default_leads() {
        assert_eq!(VOICES[0].id, "af_heart");
        assert_eq!(VOICES[0].note, Some("default"));
        let listed: Vec<&VoiceSpec> = VOICES.iter().filter(|v| v.listed).collect();
        let first_other = listed
            .iter()
            .position(|v| !v.recommended)
            .expect("non-recommended voices exist");
        assert!(
            listed[first_other..].iter().all(|v| !v.recommended),
            "a recommended voice appears after a non-recommended one"
        );
    }

    /// The unlisted voice: absent from every listing, present in the table
    /// (so its note can explain itself), never recommended.
    #[test]
    fn af_nicole_is_unlisted_but_known() {
        let nicole = spec("af_nicole").expect("af_nicole stays in the table");
        assert!(!nicole.listed);
        assert!(!nicole.recommended);
        assert!(
            nicole.note.expect("a note says why").contains("whisper"),
            "the note must say what is wrong with it for narration"
        );
    }
}
