# Installing bisque-voice

Written for a coding agent setting `bisque-voice` up for a person. Each step
has the exact command and a check that tells you it worked. Two steps are
decisions for the person, not for you: which voice model to install and which
voice to use. Ask before you install anything large.

## 1. Install the program

macOS and Linux:

```sh
curl -fsSL https://download.bisque.today/bisque-voice/install.sh | sh
```

Windows, in PowerShell 5.1 or 7:

```powershell
irm https://download.bisque.today/bisque-voice/install.ps1 | iex
```

Both install to `~/.bisque/bin` (`%USERPROFILE%\.bisque\bin` on Windows). They
check the download before anything reaches that directory and refuse to install
if a check fails. They do not edit the shell profile; they print the line that
adds the directory to `PATH`, so show the person that line or add it with their
consent. Running the installer again upgrades in place.

| Variable               | Effect                                                |
| ---------------------- | ----------------------------------------------------- |
| `BISQUE_BIN_DIR`       | Install somewhere other than `~/.bisque/bin`          |
| `BISQUE_VOICE_VERSION` | Install that version instead of the latest (`0.1.38`) |

**Check:** `bisque-voice --version` prints a version, and `bisque-voice doctor`
says where models live and that none are installed yet.

### Platforms

| Platform             | Build                                                   | Voice models   |
| -------------------- | ------------------------------------------------------- | -------------- |
| macOS, Apple Silicon | `aarch64-apple-darwin`                                  | all            |
| macOS, Intel         | `x86_64-apple-darwin` (macOS 13 or later)               | Qwen3-TTS only |
| Linux                | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` | all            |
| Windows x64          | `x86_64-pc-windows-msvc`                                | all            |
| Windows on ARM       | the x64 build, under Windows' x64 emulation             | all            |

Kokoro and the forced aligners need ONNX Runtime, which no longer ships for
Intel Macs, so the Intel build leaves them out. On that build, `engines` and
`aligners` still list them with the reason, and `install kokoro` refuses with
the same sentence. 32-bit Windows is not supported.

On Windows, keep `bisque-voice.exe` beside `DirectML.dll`; it does not start
without it.

### Checking a download by hand

The installer already does this. To repeat it, read the current version and
each archive's SHA-256 from `latest.json`:

```sh
curl -fsSL https://download.bisque.today/bisque-voice/latest.json
```

Then, on a Mac, with `V` set to that version:

```sh
curl -fsSL -O "https://download.bisque.today/bisque-voice/v$V/bisque-voice-aarch64-apple-darwin.tar.gz"
curl -fsSL -O "https://download.bisque.today/bisque-voice/v$V/bisque-voice-aarch64-apple-darwin.tar.gz.sha256"
shasum -a 256 -c bisque-voice-aarch64-apple-darwin.tar.gz.sha256

tar -xzf bisque-voice-aarch64-apple-darwin.tar.gz
codesign --verify --strict bisque-voice                            # signature intact
codesign -dv --verbose=4 bisque-voice 2>&1 | grep TeamIdentifier   # must read UG9Z22779U
spctl -a -t install bisque-voice                                   # "source=Notarized Developer ID"
```

The checksum proves the file arrived intact. The signature proves who built
it: someone who could replace the archive could replace its checksum too, but
not sign as team `UG9Z22779U`. Every release also lists all its checksums in
`v{VERSION}/SHASUMS256.txt`.

### Checking it was built from this repository

Every release since 0.1.38 is built by this repository's
[release workflow](./.github/workflows/release.yml) on GitHub's own runners,
and each archive and each file inside it carries a signed provenance record
naming the commit and the run that produced it. With the GitHub CLI:

```sh
gh attestation verify ~/.bisque/bin/bisque-voice --repo bisque-cloud/bisque-voice
```

This works on the installed binary as well as on a downloaded archive. It
passes only for a file built here; the output names the tag and the workflow
run.

## 2. Install a voice model

Show the person the catalogue and let them choose. Nothing is installed by
default, and synthesis fails until a model is.

```sh
bisque-voice engines          # sizes, voices and languages
bisque-voice engines --json   # the same, for you to read
bisque-voice install kokoro   # one id, or several at once
```

| Engine id                              | Best for                                                       | Download      |
| -------------------------------------- | -------------------------------------------------------------- | ------------- |
| `kokoro`                               | Fast narration with exact word timings; 55 voices, 9 languages | about 340 MB  |
| `qwen3-clone-06b`, `qwen3-clone-17b`   | Speaking in a voice the person records; 10 languages           | 1.3 or 2.4 GB |
| `qwen3-voices-06b`, `qwen3-voices-17b` | Nine ready-made voices, no recording; 10 languages             | 1.3 or 2.3 GB |

The 0.6B models are only about 10% faster than the 1.7B ones; what they save
is download and memory. A 1.7B model uses about 5.7 GB of memory while
running, and on macOS it refuses to start with less than 20 GiB free on the
startup disk.

Installing a Qwen3 model also installs its forced aligner, `qwen3-q4`, which
measures word timings from the audio and re-records garbled passages (see
step 5).

Kokoro's non-English voices need a small pronunciation pack, fetched the first
time one is used. To install them ahead of time:

```sh
bisque-voice install g2p-espeak   # Spanish, French, Italian, Portuguese, Hindi — 0.7 MB
bisque-voice install g2p-zh       # Chinese — 0.8 MB
bisque-voice install g2p-ja       # Japanese — 52 MB
```

Models live under `~/.bisque/models/`; set `BISQUE_VOICE_MODEL_DIR` to move
them. Every model's checksum is built into the program, and a model is checked
before it is unpacked.

**Check:** `bisque-voice doctor` lists the model as installed.

## 3. Pick a voice

The voice decides the language: `af_heart` is American English, `zf_xiaobei`
Chinese, `jm_kumo` Japanese. There is no language flag. `engines` marks the
recommended voices and lists them first; `af_heart` is Kokoro's best-rated
voice. [VOICE-CURATION.md](./VOICE-CURATION.md) says how voices are graded.

`--voice <engine>:default` uses a Qwen3 model's own voice.

## 4. Clone the person's voice (optional)

Only with the `qwen3-clone-*` models, and only with a recording the person
gives you of their own voice, or one they have the right to use. Cloning runs
once per voice and keeps about 13 KB; the recording is not needed afterwards.

```sh
bisque-voice clone --name narrator \
  --reference clip.wav \
  --transcript "exactly what the clip says"

bisque-voice list                     # cloned voices
bisque-voice forget narrator          # delete one; the recording is untouched
```

- **The clip:** 10–30 seconds of clean speech from one person. Any WAV works
  (16, 24 or 32-bit, mono or stereo, any sample rate); convert anything else
  first with `ffmpeg -i clip.m4a -ar 24000 -ac 1 clip.wav`. `clone` trims
  clicks, breaths and silence itself, and refuses a clip with no speech.
- **The transcript:** exactly what is said in the clip. An approximate
  transcript gives an approximate voice.
- **One model per voice:** a voice cloned with the 0.6B model does not load
  into the 1.7B one. Clone again from the recording to switch.
- **Languages:** a voice speaks its own language best. A voice cloned in
  English makes occasional word errors in Japanese, Korean or Chinese.
- `--source-id <id>` stores which recording a voice came from; `list --json`
  returns it as `sourceId`.

**Check:** `bisque-voice list` shows the voice, and it narrates as
`--voice qwen3-clone-06b:narrator`.

## 5. Forced aligners

An aligner measures where each word falls in the finished audio. Kokoro does
not need one. The Qwen3 models do: their own timings are approximate, and the
aligner is also what catches a garbled passage and records it again, up to 3
times, before narration fails with nothing written.

```sh
bisque-voice aligners          # the catalogue
bisque-voice install qwen3-q4  # 1.05 GB; qwen3-fp32 is the 3.67 GB full-precision graph
```

With one aligner installed it is used automatically; with several, name one
with `--align <id>`. `--align none` uses the model's own timings. Synthesizing
with a Qwen3 model and no aligner prints a warning. The aligners are not
available on Intel Macs.

## 6. Synthesize

```sh
bisque-voice --text "Revenue grew 3.5x to \$4.2M in Q3." \
             --voice kokoro:af_heart --out narration.mp3
```

The audio goes to `--out`: `.wav` writes WAV, anything else writes MP3. Exactly
one JSON document goes to stdout, and progress and errors go to stderr, so the
output always parses.

```json
{
  "words": [{ "word": "Revenue", "start": 0.3, "end": 0.68 }, …],
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

- `words` has exactly one entry per whitespace-separated word of `--text`, in
  order. If that cannot be guaranteed, synthesis fails instead.
- `exact` is `true` when every word has its own measured time, and `false`
  when the text was expanded ("$4.2M") and times were spread across the words
  it came from. The count is right either way.
- Chinese and Japanese have no spaces, so a sentence is one word with one
  approximate time, unless an aligner is installed.
- `aligner` names the aligner that measured `words`, and is absent when the
  timings came from the model.
- `durationMs` is an integer and `hash` keeps its `sha256:` prefix.
- `pronunciationWarnings` appears when Kokoro was given a pronunciation known
  to come out wrong ("plugin" with a soft g, "TDD" run together, or a stress
  mark before an IPA marker's first consonant). The audio is still written.

| Option                    |                                                                             |
| ------------------------- | --------------------------------------------------------------------------- |
| `--text`                  | Text to speak; `-` reads stdin                                              |
| `--voice`                 | `af_heart`, or `kokoro:af_heart` to name the model too. No default          |
| `--engine`                | Which installed model; optional when one is installed or the voice names it |
| `--align`                 | Which installed aligner, or `none`; optional when one is installed          |
| `--speed`                 | Rate multiplier; default `1.0`                                              |
| `--match-macos`           | Kokoro about 1.3× faster, matching its MLX build                            |
| `--out`                   | Where the audio goes                                                        |
| `--threads`               | ONNX Runtime threads                                                        |
| `--debug-tokens`          | Print the pronunciation sent to the model, on stderr                        |
| `engines`, `aligners`     | The catalogues                                                              |
| `install`, `uninstall`    | Add or remove models and aligners; `install` also takes pronunciation packs |
| `clone`, `list`, `forget` | Cloned voices                                                               |
| `doctor`                  | Where models live and which are installed                                   |

Long text is split at sentence boundaries and joined back with continuous
timings; `chunks` says how many pieces it took.

**Check:** the command exits 0, the JSON has a `words` array, and the file at
`--out` plays.

## Running several at once

One voice model runs at a time per machine. A second run prints `waiting for
another voice model to finish (held by pid …)` on stderr and starts when the
first one exits, crash included. `engines`, `install`, `list` and `doctor` load
no model and never wait.

| Variable                       |                                                                    |
| ------------------------------ | ------------------------------------------------------------------ |
| `BISQUE_VOICE_HEAVY_WAIT_SECS` | How long a queued run waits before failing; default 1800 seconds   |
| `BISQUE_VOICE_MIN_FREE_GIB`    | The 1.7B models' free-disk floor; default 20 GiB, `0` turns it off |
| `BISQUE_VOICE_HEAVY_LOCK`      | The lock file; every process sharing a queue must use the same one |
