# Voice curation

How per-voice quality metadata is used across bisque-voice and every surface
that lists voices. The tables live in `src/kokoro_voices.rs` (Kokoro's graded
English set) and `src/qwen3_voices.rs` (the Qwen3 Studio named speakers); this
file is the policy's canonical home.

## Policy

1. **Never block.** Any voice remains synthesizable when passed explicitly
   (`--voice` / API), including unlisted ones. Metadata informs choice; it
   never gates execution. Resolution and the pack's voice lookup do not
   consult the table.
2. **Inform everywhere voices are listed.** Every enumeration surface —
   `engines --json` (the `voices` array), the human `engines` catalogue, the
   `present.mjs` doctor output — carries each voice's quality grade and a
   `recommended` flag, recommended voices first, so humans and LLM agents
   alike can choose well.
3. **"Not recommended" means quality.** Non-recommended voices have audible
   quality issues — upstream trained them on very little data. It is not a
   licensing or availability distinction.
4. **Unlisted voices exist.** `af_nicole` is a whisper/ASMR-register voice —
   wrong for presentation narration, period. It appears in **no** listing
   (not even a "show all"), but explicit `--voice kokoro:af_nicole` still
   synthesizes.

Recommended = grade C+ and above, minus unlisted. The default suggestion is
`af_heart`.

## The graded set

Grades are upstream's own target-quality grades, verbatim from
hexgrad/Kokoro-82M's `VOICES.md`
(<https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md>). Kokoro's
pack also carries non-English voices; those are outside this graded set —
absent from the table means "ungraded here", not "unlisted".

### Recommended

| Voice        | Accent · gender   | Grade | Note    |
| ------------ | ----------------- | ----- | ------- |
| `af_heart`   | American · female | A     | default |
| `af_bella`   | American · female | A-    |         |
| `bf_emma`    | British · female  | B-    |         |
| `af_aoede`   | American · female | C+    |         |
| `af_kore`    | American · female | C+    |         |
| `af_sarah`   | American · female | C+    |         |
| `am_fenrir`  | American · male   | C+    |         |
| `am_michael` | American · male   | C+    |         |
| `am_puck`    | American · male   | C+    |         |

#**Coverage picks (recommended despite grade C):** bf_isabella, bm_fable,
bm_george — kept so the recommended cast includes British voices of both
genders; three honest C voices beat a cast with no British male.

## Not recommended (audible quality issues)

| Voice         | Accent · gender   | Grade |
| ------------- | ----------------- | ----- |
| `af_alloy`    | American · female | C     |
| `af_nova`     | American · female | C     |
| `bf_isabella` | British · female  | C     |
| `bm_fable`    | British · male    | C     |
| `bm_george`   | British · male    | C     |
| `af_sky`      | American · female | C-    |
| `bm_lewis`    | British · male    | D+    |
| `af_jessica`  | American · female | D     |
| `af_river`    | American · female | D     |
| `am_echo`     | American · male   | D     |
| `am_eric`     | American · male   | D     |
| `am_liam`     | American · male   | D     |
| `am_onyx`     | American · male   | D     |
| `bf_alice`    | British · female  | D     |
| `bf_lily`     | British · female  | D     |
| `bm_daniel`   | British · male    | D     |
| `am_santa`    | American · male   | D-    |
| `am_adam`     | American · male   | F+    |

### Unlisted

| Voice       | Accent · gender   | Grade | Why unlisted                                |
| ----------- | ----------------- | ----- | ------------------------------------------- |
| `af_nicole` | American · female | B-    | whisper register — unsuitable for narration |

## The Studio named speakers

The `qwen3-voices-06b` / `qwen3-voices-17b` engines carry the nine speakers
baked into the Qwen3-TTS CustomVoice checkpoint. The table lives in
`src/qwen3_voices.rs` and follows the same policy: it informs choice and
never gates synthesis — the engine resolves `--voice` against the model's own
speaker table, not against the metadata.

Upstream publishes **no per-voice grades** for these speakers, so the grade
column is `—` rather than an invented letter, and the shape of `engines
--json` stays uniform across engines. All nine are listed and recommended —
including the two dialect voices, which are high quality _in their language_;
the accent column is what says which language that is.

| Voice      | Language affinity · gender       | Note                                       |
| ---------- | -------------------------------- | ------------------------------------------ |
| `serena`   | Multilingual · female            |                                            |
| `vivian`   | Multilingual · female            |                                            |
| `uncle_fu` | Multilingual · male              |                                            |
| `ryan`     | Multilingual · male              |                                            |
| `aiden`    | Multilingual · male              |                                            |
| `ono_anna` | Japanese-leaning · female        |                                            |
| `sohee`    | Korean-leaning · female          |                                            |
| `eric`     | Chinese (Sichuan dialect) · male | best in Chinese — the dialect is the voice |
| `dylan`    | Chinese (Beijing dialect) · male | best in Chinese — the dialect is the voice |

For `eric` and `dylan` the model itself swaps the language id for the
dialect's when the text's language is Chinese (or unstated) — choosing them
for English narration is the mismatch, not the voice.
