# bisque-voice

Local speech synthesis with word-level timings, on macOS, Linux and Windows.

A **host for local speech models**, not a wrapper around one. This is the piece
of `/present` that cannot run in the cloud: authoring is already Claude Code,
and hosting, rendering and the watch page are already server-side — but
on-device synthesis is what makes unlimited free narration free, so it is the
only thing that has to be native.

New to speech synthesis, or wondering what any of the dependencies are for?
[**GLOSSARY.md**](./GLOSSARY.md) defines every term and lists every dependency
with a one-line reason it is here.

## Install

macOS and Linux:

```sh
curl -fsSL https://download.bisque.today/bisque-voice/install.sh | sh
```

Windows (PowerShell 5.1 or 7 — nothing to install first):

```powershell
irm https://download.bisque.today/bisque-voice/install.ps1 | iex
```

Both install to `~/.bisque/bin` (`%USERPROFILE%\.bisque\bin`), verify the
download **before** anything reaches that directory, and print the line that
puts it on your `PATH` rather than editing your profile behind your back.
Re-running either one upgrades in place. `BISQUE_BIN_DIR` changes the
destination and `BISQUE_VOICE_VERSION` pins a version instead of resolving
`latest.json`.

## Verifying the download

Every release publishes a SHA-256 for each archive and every macOS binary is
signed with our Developer ID and notarized by Apple. The installer checks both
and refuses to install if either fails — it does not warn and continue.

The two checks catch different things:

| Check                  | Catches                                                                 | Where the trust comes from                                                         |
| ---------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| SHA-256                | a corrupted or truncated download                                       | the same server that served the archive                                            |
| Developer ID signature | a **swapped binary**, including one whose checksum was swapped to match | Apple's root CA — forging it needs our signing key, not write access to our bucket |

The checksum alone proves only that the bytes arrived intact. Whoever can
replace the archive can replace the `.sha256` beside it, so the signature is
the check that does not take this server's word for anything. The installer
pins team `UG9Z22779U` and rejects a binary signed by anyone else, even when
that signature is otherwise valid.

To check by hand, `latest.json` names the current version and each target's
archive:

```sh
curl -fsSL https://download.bisque.today/bisque-voice/latest.json
```

Every archive has a `.sha256` beside it. The release workflow also writes
`v{VERSION}/SHASUMS256.txt` and `checksums`/`signing` blocks into
`latest.json`, but no release has published them yet — 0.1.20 was the first to
carry the change and its build failed, so 0.1.19 is still current and has
neither.

```sh
V=0.1.19
curl -fsSL -O "https://download.bisque.today/bisque-voice/v$V/bisque-voice-aarch64-apple-darwin.tar.gz"
curl -fsSL -O "https://download.bisque.today/bisque-voice/v$V/bisque-voice-aarch64-apple-darwin.tar.gz.sha256"
shasum -a 256 -c bisque-voice-aarch64-apple-darwin.tar.gz.sha256

tar -xzf bisque-voice-aarch64-apple-darwin.tar.gz
codesign --verify --strict bisque-voice          # signature intact
codesign -dv --verbose=4 bisque-voice 2>&1 | grep TeamIdentifier   # must read UG9Z22779U
spctl -a -t install bisque-voice                 # "source=Notarized Developer ID"
```

A bare Mach-O cannot be stapled, so the notarization ticket stays with Apple
rather than travelling in the file. `spctl` also reports this Mac's Gatekeeper
policy, not only our file's status. The two `codesign` checks are offline and
cryptographic; those are the ones the installer fails on.

Model packs are verified the same way one layer down: each engine's pack
checksum is compiled into the binary at build time, so a verified binary will
not download an unexpected model.

On Windows, extract the **whole** archive — `bisque-voice.exe` loads
`DirectML.dll` from its own directory and will not start without it.

| Platform             | Build                                                   | Engines                    |
| -------------------- | ------------------------------------------------------- | -------------------------- |
| macOS, Apple Silicon | `aarch64-apple-darwin`                                  | all                        |
| macOS, Intel         | `x86_64-apple-darwin` (macOS 13+)                       | voice-cloning engines only |
| Linux                | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` | all                        |
| Windows x64          | `x86_64-pc-windows-msvc`                                | all                        |
| Windows ARM          | the x64 build, under Windows' x64 emulation             | all                        |

**Platform support is per-engine.** Kokoro and the forced aligners run on ONNX
Runtime, which dropped Intel macOS in 1.28 — so the `x86_64-apple-darwin`
build is compiled without them (`--no-default-features`; see the `kokoro` and
`onnx` features in [`Cargo.toml`](Cargo.toml)) and ships the ggml-backed
voice-cloning engines, which build everywhere. On that binary, `engines` and
`aligners` still list everything and say exactly why the ONNX-backed entries
cannot install; `install kokoro` refuses with the same sentence. Rosetta is
not a workaround — it runs Intel binaries on Apple Silicon, and it is the
Apple Silicon build that exists. 32-bit Windows is not supported at all.

## Choosing an engine

Which model to run is **the user's decision**, made from the real trade-offs,
and more than one can be installed at a time. Nothing in here picks for them:
with no engine installed, synthesis fails and says so.

```sh
bisque-voice engines            # the catalogue, for onboarding to render
bisque-voice engines --json     # same, machine-readable
bisque-voice install kokoro     # install one — or several at once
```

```
kokoro  Kokoro 82M
    Small and fast. Nine languages, exact word timings. Runs comfortably on a
    laptop CPU.
    82M params · ~337 MB download · 55 voices
    languages: English (US), English (UK), Spanish, French, Hindi, Italian,
               Portuguese (Brazil), Chinese, Japanese
```

The voice decides the language — `af_heart` is American English, `zf_xiaobei`
Chinese, `jm_kumo` Japanese — so there is no language flag to get wrong.

New models get added to the registry in [`src/engine.rs`](src/engine.rs) and
old ones keep working. Adding one is a registry entry plus an `Engine`
implementation — deliberately the whole cost.

## Cloning a voice

Two engines — `qwen3-clone-06b` and `qwen3-clone-17b` — speak in a voice you
give them as a recording. This is the one thing Kokoro cannot do at any
download size.

Cloning is a **separate, once-per-voice step**. It runs the speaker encoder and
the codec encoder over the clip, stores about 13 KB of latents, and never needs
the recording again; narration reuses those. Paying that per narration would
cost more than the narration it is for, which is why it is its own command.

```sh
bisque-voice install qwen3-clone-06b

bisque-voice clone --name narrator \
  --reference clip.wav \
  --transcript "exactly what the clip says"

bisque-voice list                      # what has been cloned
bisque-voice --text "…" --voice qwen3-clone-06b:narrator --out out.mp3
```

Give it **10–30 seconds of clean speech from one person**, and the transcript of
exactly what is said in it. The clone is conditioned on that text, so an
approximate transcript produces an approximate voice — which is why the
transcript is required rather than optional.

**Hygiene is `clone`'s job, not the caller's.** A clip that opens with a
click and a long pause teaches the clone to begin narrations the same way,
so `clone` trims leading non-speech (clicks, breaths, AGC-boosted room tone,
dead air) and trailing silence itself, with a threshold measured from the
clip's own noise and speech levels. A clip with no audible speech is refused
with the reason. No client is expected to pre-clean audio.

`--source-id <opaque>` records which recording the reference was (an upload
timestamp, say) in the voice manifest; `list --json` returns it as
`sourceId`, so a caller can tell whether a voice came from the recording it
currently cares about without keeping state of its own.

The reference may be any WAV: 16/24/32-bit PCM or float, mono or stereo, at any
sample rate. Anything else, convert it first:

```sh
ffmpeg -i clip.m4a -ar 24000 -ac 1 clip.wav
```

`--voice <engine>:default` narrates in the model's own voice, with no reference
at all.

**A voice belongs to the engine that made it.** The speaker embedding's width
_is_ the talker's hidden size — 1024 for the 0.6B, 2048 for the 1.7B — so a
voice cloned with one cannot load into the other. `bisque-voice` records which
engine made each voice and refuses the mismatch, because the alternative is
handing C a buffer of the wrong length. Re-clone from the original recording to
move a voice between sizes.

**Which size.** Both share the codec, and the codec dominates synthesis, so
0.6B is only about 10% faster end to end (RTF 1.01 against 1.09, measured on an
M2 Air). What the smaller one actually buys is **1.1 GB less download and
~0.8 GB less memory** while running.

**Languages.** Ten, and every one was checked by synthesizing it and
transcribing it back: English, Spanish, German, French, Italian, Portuguese,
Russian, Chinese, Japanese, Korean. In the model's own voice the result is
verbatim or near-verbatim in all ten.

Cloning **across** languages costs some accuracy, though. A voice cloned from an
English recording and then asked to speak Japanese, Korean or Chinese picks up
occasional word-level errors that the same engine's default voice does not make;
Italian and Russian showed none. Cloning and narrating in the _same_ language is
verbatim. For plain multilingual narration with no cloning, Kokoro is the better
tool — exact timings, a seventh of the download.

Word timings from these engines are **approximate** — an autoregressive codec
model emits audio frames, not word boundaries. Install a forced aligner to
measure them against the waveform instead.

## Choosing an aligner

A **forced aligner** measures where each word actually falls in the synthesized
waveform, instead of inferring it from the engine's internals. It is what gives
an engine with no native timings — Qwen3-TTS has none — real per-word start and
end times, and it works on any engine's audio.

It is a second registry, on the same rules: aligners are listed with their real
trade-offs, more than one can be installed, and **there is no default**. Unlike
an engine, having none installed is not an error — timings fall back to the
engine's own and are reported as approximate.

The Qwen3 engines are the exception to "optional": their timings drift and
their output can open with stochastic non-speech that only measured timings
let synthesis trim (with an aligner, audio is cut to the aligned span and
timings shift with it). Each declares `qwen3-q4` as its **companion**
(`companionAligner` in `engines --json`), and `install <engine>` resolves it
automatically; synthesizing one unaligned prints a warning unless
`--align none` opts out.

The aligner also catches a garbled Studio take. Qwen3 samples its speech, so
about one chunk in a few dozen comes out as fluent nonsense that is clean on
the next take. Each chunk is aligned against its own text as it is
synthesized; when its words stack at zero length, the chunk is synthesized
again, up to 3 takes, and then narration fails naming the chunk with nothing
written. Each accepted chunk is also cut to its aligned words, so a chuckle or
"hm" before a mid-slide chunk does not survive. When the aligner has given the
chunk's first word to that sound instead ("Ugh" heard as "A"), the cut would
keep it, so a gap after the first word is checked by aligning again from the
gap; a first word found later in it means the chunk is synthesized again,
within the same 3 takes, and never fails narration. The aligner also says how
sure it was of each word's place; a chunk whose least sure ten words average
under 0.43 has babble or a made-up word in it ("A skill is Naxmourning
instructions"), and is synthesized again the same way, keeping the surest take
if none is clean. A one-syllable insertion can still pass, and Korean is not
judged this way: its clean takes read as unsure as a defect. A take can also
stop before its last words ("statt sie" where the text ends "statt sie liegen
zu lassen"); the aligner then piles two or more words onto the take's last
80 ms or past it, and the chunk is synthesized again the same way, in every
language, keeping the take missing the fewest words if none is whole. A take
missing only its last word can still pass. Thresholds and the measured takes
behind them: `src/quality.rs`, `fixtures/quality/takes.json`, `leads.json`,
`sureness.json` and `tails.json`. The fixtures keep every measured timing,
sureness and label, with the words replaced by neutral ones of the same script:
the gate reads only where each word sits, how sure the aligner was, and whether
a token holds a letter or Hangul.

Both aligners are ONNX exports, so on Intel Macs — where that runtime has no
build — neither is available; the catalogue says so, and timings come from the
engine itself.

```sh
bisque-voice aligners             # the catalogue
bisque-voice aligners --json      # same, machine-readable
bisque-voice install qwen3-q4     # install and uninstall take aligner ids too
```

```
qwen3-fp32  Qwen3 forced aligner (fp32)
    Full-precision graph, 3.67 GB. Validated against PyTorch: max_abs_diff 9.06e-05.
    ~3670 MB download

qwen3-q4  Qwen3 forced aligner (q4)
    4-bit quantized graph, 1.05 GB. Smoke-tested against PyTorch: max_abs_diff 7.80.
    ~1050 MB download
```

Those figures are deviations in **raw logits**. This path consumes only the
`argmax` over those logits, so a larger deviation does not straightforwardly
mean worse timestamps, and the measurement that would settle it — how often the
two graphs pick a different timestamp over real narration — has not been made.
`qwen3-q4` is _unmeasured_, not worse; a third of the download at unquantified
risk is a reasonable call, and so is the validated graph.

Resolution mirrors `--engine`: `--align <id>` wins; exactly one installed is
the choice already made, not a default; several installed with none named is an
error asking which. `--align none` uses the engine's own timings even with one
installed.

## Synthesizing

```sh
bisque-voice --text "Revenue grew 3.5x to \$4.2M in Q3." \
           --voice kokoro:af_heart --out narration.mp3
```

```json
{
  "words": [ { "word": "Revenue", "start": 0.3, "end": 0.68 }, … ],
  "durationMs": 5125,
  "exact": true,
  "contentType": "audio/mpeg",
  "size": 82176,
  "hash": "sha256:f6336913…",
  "voice": "kokoro:af_heart",
  "speed": 1.0,
  "chunks": 1,
  "aligner": "qwen3-q4"
}
```

`aligner` names the forced aligner that measured `words`, and is **absent** when
the timings came from the engine itself — `exact` says the mapping was exact,
not how it was obtained, so the two answer different questions.

Audio goes to `--out`; the JSON goes to stdout, so it pipes. Progress and
errors go to stderr and never block a non-interactive agent run. **stdout
carries exactly one JSON document and nothing else** — that is contract, not
convention; callers parse it programmatically, and anything that would spray
text onto stdout (Apple's CoreML runtime once did) is a bug here, not in the
caller.

`contentType`, `size` and `hash` are named and typed to drop **verbatim** into
a `presentation.json` v2 audio asset, and `durationMs` into
`slides[].audioDurationMs`. Two consequences worth knowing, because both are
load-bearing:

- `hash` carries its `sha256:` prefix. The publish endpoint rejects a bare
  digest, so the prefix lives in the value rather than in every caller's memory.
- `durationMs` is an **integer**. The manifest reader parses it with `as_u64()`,
  which rejects a float outright — a fractional millisecond would read back as
  absent, silently producing a slide with no duration.

| Flag                    |                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------- |
| `--text`                | Text to speak; `-` reads stdin                                                          |
| `--engine`              | Which installed engine; optional when only one is installed or the voice is qualified   |
| `--align`               | Which installed aligner, or `none`; optional when only one is installed. **No default** |
| `--voice`               | `af_heart`, or `kokoro:af_heart` to qualify it. **No default**                          |
| `--speed`               | Rate multiplier, default `1.0`                                                          |
| `--match-macos`         | Speak at the same pace as the Bisque Mac app (see below)                                |
| `--out`                 | Destination. `.wav` writes WAV; anything else writes MP3                                |
| `--threads`             | ONNX Runtime intra-op threads                                                           |
| `--debug-tokens`        | Dump the engine's pronunciation stream to stderr                                        |
| `engines`               | The engine catalogue — sizes, voices, languages                                         |
| `aligners`              | The aligner catalogue — sizes and reported parity                                       |
| `install` / `uninstall` | Add or remove engines **and** aligners                                                  |
| `clone`                 | Clone a voice from a recording, once — see above                                        |
| `list`                  | List cloned voices                                                                      |
| `forget`                | Delete a cloned voice; the recording is untouched                                       |
| `doctor`                | Where models live and which are installed                                               |

## One voice model per machine

Narrations run one at a time per machine, whoever starts them: a second run
prints `waiting for another voice model to finish (held by pid …)` on stderr
and starts when the first exits. A Qwen 1.7B model uses about 5.7 GB of
memory, a 0.6B about 4.9 GB, and a running Kokoro about 1.8 GB, so a few at
once can fill a 16 GB machine. The queue is a file lock on
`/tmp/bisque-voice-heavy.lock`, which the system releases when its holder
exits for any reason, including a crash or a kill. Every engine takes it;
`engines`, `install`, `list` and `doctor` load no model and never wait.

The 1.7B engines also refuse to load when the startup disk has less than
20 GiB free. macOS grows swap on that disk, and when it cannot, running out of
memory freezes or restarts the whole machine instead of stopping one program.
The check runs on macOS only; Linux and Windows size swap ahead of time.

| Variable                       |                                                                |
| ------------------------------ | -------------------------------------------------------------- |
| `BISQUE_VOICE_HEAVY_WAIT_SECS` | How long a queued run waits before failing; default 1800       |
| `BISQUE_VOICE_MIN_FREE_GIB`    | The 1.7B free-disk floor in GiB; default 20, `0` turns it off  |
| `BISQUE_VOICE_HEAVY_LOCK`      | Where the lock lives. Every process sharing a queue must agree |

Only `bisque-voice` 0.1.32 and later takes the lock, and only 0.1.33 and later
takes it for Kokoro. An older binary, or a different program such as a Python
model, runs outside the queue.

## Audio format

**MP3 is the pipeline's contract, not a preference.**
`ManifestAudioAsset.contentType` is literally `"audio/mpeg"`, and all three
existing producers — ElevenLabs, the Cloud Run Kokoro runner, the create worker
— hand it MP3. An engine that emitted only WAV could not publish at all, and
widening the contract for one producer is the wrong side of that trade: WAV is
~3× the bytes for a viewer to download. LAME is compiled in, so there is no
ffmpeg to find on a stranger's machine — which was the whole point of a single
static binary. `--out foo.wav` still gives you WAV for inspection.

## The word-timing guarantee

`words` is **1:1 with the source's whitespace-delimited tokens**, in order,
always. Cue markers index that array (`marker.wordIndex` in
`packages/presentation-format`), so a mismatch would desynchronize every cue in
a presentation with no error anywhere. The engine therefore reconciles Kokoro's
tokens back onto source words and **fails closed** if it cannot.

`exact` reports _how_ the mapping was made, not whether it is safe:

- `true` — the timings' characters matched the source's, so every word has a
  real start and end.
- `false` — the text was expanded ("$4.2M" → "four point two million dollars"),
  so per-token mapping is impossible and the utterance was distributed across
  source words by length. Times are approximate; **the count is still exact**,
  so cues stay on the right words.

A forced aligner is always the exact path: it is handed the source's own
whitespace tokens and returns one measured `[start, end]` for each, so there is
nothing to reconcile away. If that ever failed to map, synthesis fails rather
than reporting `exact: true` over timings that were not measured.

This mirrors `packages/tts-core/src/word-timing.ts`, which serves the same
guarantee server-side. Keep the two in lockstep.

## Pace, and the macOS engine

The macOS app synthesizes through MLX; this binary uses ONNX Runtime. With
phonemes, voice pack and speed held identical the two runtimes **disagree on
duration by a uniform ~1.30×** — not quantization, not the voice data (the
packs are byte-identical), not indexing. The measurements are in the design
doc.

Word _indices_ are unaffected, because timings reconcile onto source tokens.
Only pace moves, and since Kokoro's `speed` input divides the predicted
durations, one constant reconciles them: `--match-macos` selects the Mac app's
pace, and the default is the ONNX reference pace.

## Pronunciation

Kokoro does not read letters. It reads **phonemes**, and it was trained on the
ones [Misaki](https://github.com/hexgrad/misaki) produces — so the phonemizer is
not an implementation detail that can be approximated. Every language here is a
port of Misaki's exact pipeline, checked character-for-character against output
generated by running Misaki itself (`fixtures/g2p/`, regenerated with
`tools/gen-g2p-fixtures.py`). Misaki ships no tests; those fixtures are the
substitute.

| Language                                    | How it is phonemized                                                                                                                                                                                                                                   |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| English (US/UK)                             | The Apache-2.0 Misaki lexicons the macOS engine reads — 90k gold plus 93k silver entries, the same stemming, the same number-to-words logic, and the same 3 MB BART fallback for what is left (~1.8% of real narration, mostly technical identifiers). |
| Spanish, French, Italian, Portuguese, Hindi | eSpeak NG's letter-to-sound rules, through the same `phonemizer` configuration Misaki uses.                                                                                                                                                            |
| Chinese                                     | jieba segmentation, then pypinyin — including its phrase dictionary, without which 银行 is read _yín xíng_ rather than _yín háng_.                                                                                                                     |
| Japanese                                    | UniDic morphological analysis, because kanji carry no pronunciation on their face.                                                                                                                                                                     |

Each non-English language's data is a **separate download**, fetched the first
time a voice of that language is used:

```sh
bisque-voice install g2p-espeak   # Spanish, French, Italian, Portuguese, Hindi — 0.7 MB
bisque-voice install g2p-zh       # Chinese — 0.8 MB
bisque-voice install g2p-ja       # Japanese — 52 MB
```

They are split because the sizes are wildly uneven: UniDic alone is 204 MB on
disk, and someone narrating English slides should not carry a Japanese
dictionary to do it. `bisque-voice doctor` lists what is installed.

eSpeak NG is linked for those five languages because their
phonemes _are_ its rules — a substitute would not be the sequence Kokoro was
trained on — and only its data files travel in a pack. English, Chinese and
Japanese do not touch it.

Where the Swift and Rust English implementations diverge, it is in
part-of-speech tagging: macOS uses Apple's `NLTagger`, and there is nothing to
link against elsewhere, so `g2p/tagger.rs` is rule-based. For an ordinary word
it abstains rather than guess — an unknown tag selects a lexicon entry's
`DEFAULT`, which is the common reading, so abstaining is strictly better than
a wrong guess. For the ~790 POS-conditioned heteronyms ("present", "record",
"live", "read"), abstaining always picked the common reading, so
`tag_sequence` resolves those — and only those — from one word of sentence
context ("will **\_" is a verb, "their \_** of" is a noun), the same signal
`NLTagger` reads on macOS. Where no rule fires, a heteronym still falls to
`DEFAULT`.

A polyphone standing **alone** falls back to its most common reading. The
phrase dictionary resolves a character by the word it sits in, so 行 is _háng_
in 银行 and _xíng_ in 行走 — but in 银行的行 ("the 行 of 银行"), that last 行 is
its own token with no word around it, and comes out _xíng_ where a reader would
say _háng_. Resolving it means following a reference back through the sentence,
which is semantics rather than a lookup; Misaki does the same thing, and this is
a limit of the approach rather than a defect in it.

Chinese and Japanese have no spaces, so a whole sentence is one source word and
its cue timings are reported as approximate. That is the reconciler being
honest, not a failure: the word _count_ still matches the source exactly, so
cues never land on the wrong one.

### Reported mispronunciations

After a Kokoro narration, `bisque-voice` checks what the voice was given for
the three mistakes listeners have reported in published audio: "plugin" with a
soft g, TDD's letters run together, and an IPA marker with its stress mark
before the first consonant (`/ˈɡistˌæk/`, heard as "Uggy Stack"). Each match
prints a warning with the marker that fixes it and is listed in
`pronunciationWarnings` in the JSON. The audio is still written: the check
reads phonemes, not sound, and anything outside those three rules passes
unremarked. Other engines are not checked. The real-audio regression is
`tools/verify-regression.ts`.

## Model packs

Each engine and each aligner is one hash-verified archive holding everything it
needs — the same download path, the same verification, one implementation. Packs
install under `~/.bisque/models/<pack-version>/` — override the root with
`BISQUE_VOICE_MODEL_DIR`. A pack version is immutable: bump it when contents
change, never edit in place, because installed clients resolve a pack by that
name and verify its checksum.

Three properties, because this repo has been burned before:

1. **Verified before use.** A download race once shipped garbled audio, so the
   archive's SHA-256 is checked before a byte is extracted, and a build with no
   pinned checksum refuses to download at all.
2. **Safe under concurrent invocation.** Two agent runs starting at once is
   normal. A lock file serializes them; the loser waits and finds the pack
   already there.
3. **It never wedges.** Progress is byte-interval, not timer-driven, so a
   non-TTY run gets a bounded number of lines and there is no clock to hang on.

Extraction stages into a temporary directory and renames, and the pack only
counts as installed once a completion marker exists — an interrupted extract is
never mistaken for a usable install.

### Building a pack

```sh
bun run tools/build-model-pack.ts --variant fp32
BISQUE_VOICE_PACK_SHA256_KOKORO=<printed hash> cargo build --release

bun run tools/build-qwen-clone-pack.ts
bun run tools/build-aligner-pack.ts --variant fp32   # or --variant q4
bun run tools/build-g2p-packs.ts                      # per-language pronunciation data
```

The variable is **per pack** — `BISQUE_VOICE_PACK_SHA256_<ID>`, uppercased with
`-` replaced by `_`, one per `engine::REGISTRY` and `align::REGISTRY` entry
(see `build.rs`), so the `qwen3-fp32` aligner reads
`BISQUE_VOICE_PACK_SHA256_QWEN3_FP32`. An unsuffixed
`BISQUE_VOICE_PACK_SHA256` is not read by anything: the build would succeed with
an empty checksum and the resulting binary would then refuse to download any
model at all, which is the intended fail-closed behavior but an easy thing to
mistake for a broken download.

`tools/promote-duration.ts` is the step that makes any of this possible: the
stock Kokoro ONNX export publishes only `waveform`, but the duration
predictor's result sits in the graph as `/encoder/Clip_output_0`. Promoting it
to a graph output is a 59-byte protobuf append — no ONNX toolchain, no
re-export. The engine verifies on every synthesis that
`frame_sum × 600 == waveform.len()`, and refuses to emit timings if it ever
stops holding.

**fp32 is the shipping tier.** fp16 shipped first — indistinguishable in the
original measurement at half the download — and then turned out to overflow
its vocoder to NaN for some (voice, text) pairs, deterministically, which
reached real output as silent slides. fp32 synthesizes every
known-failing pair, is the reference precision, and measured the same speed on
CPU; the extra download is the whole cost. q8 also synthesizes the failing
pairs but is 2.4× slower on ARM _and_ shifts durations.

## Chunking

Kokoro's 510-token limit is hit by ordinary slide narration — an 82-word slide
fails outright on the macOS engine. This binary splits on sentence boundaries,
synthesizes each chunk, and concatenates with offset timings, so a chunked
narration is indistinguishable from a single-pass one. `chunks` in the output
says how many passes it took.

## Building from source

```sh
sh vendor/fetch.sh          # the two vendored C++ engines, at their pinned commits
cargo build --release
```

It needs a Rust toolchain, CMake, a C++17 compiler and libclang.
[`CONTRIBUTING.md`](./CONTRIBUTING.md) has the rest.

## Tests

```sh
cargo test                                                    # default (all engines)
cargo check --no-default-features                             # the studio-only build
cargo test --no-default-features                              # …and its test suite
```

Covers the reconciliation guarantee (including the deliberately hard cases:
`3.5x`, `$4.2M`, `state-of-the-art`), the timestamp walk, chunk boundaries,
number-to-words, and the BART arithmetic.

The forced aligner's logic is tested **without any model files**: the
monotonicity repair against hand-worked examples from the reference
implementation, the prompt layout, the timestamp-slot selection, the mel
filterbank, and the 24→16 kHz resampler (including that an out-of-band tone is
rejected rather than aliased into the speech).

## License

bisque-voice is [MIT](./LICENSE).
