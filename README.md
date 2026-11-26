# bisque-voice

Turn text into speech on your own computer, and get the start and end time of
every word with it. For developers and agents on macOS, Linux and Windows.

```sh
bisque-voice --text "Revenue grew 3.5x to \$4.2M in Q3." \
             --voice kokoro:af_heart --out narration.mp3
```

```json
{
  "words": [{ "word": "Revenue", "start": 0.3, "end": 0.68 }, …],
  "durationMs": 5125,
  "contentType": "audio/mpeg",
  "hash": "sha256:f6336913…"
}
```

## Why it's interesting

- **Free, and nothing leaves your machine.** The voice models run locally, so
  you need no API key, pay nothing per character, and your text stays on your
  computer.
- **A timing for every word you wrote.** You get one start and end time for
  each word of your input, in order, every time, so captions and highlighting
  stay in step with the audio. "$4.2M" is spoken as "four point two million
  dollars" and still lands on the one word you typed.
- **Your own voice from a 10–30 second clip.** Record yourself once, and
  every narration after that speaks in your voice.
- **Narration needs no Python or ffmpeg.** Repairing timings on an existing
  MP3 uses ffmpeg to decode it; WAV decoding is built in.
- **You can check what you're running.** Every release is built
  from this repository on GitHub's runners, with a signed record of the commit
  it came from, and the macOS builds are notarized by Apple.
- **Easy for an agent to drive.** It never stops to ask a question, prints
  exactly one JSON document, and keeps its progress messages on a separate
  stream.

## How it works

```
  text  ──►  pronunciation  ──►  voice model  ──►  audio + word timings
```

1. **Pronunciation.** Your text becomes the sounds the model was trained on.
   English, Spanish, French, Italian, Portuguese, Hindi, Chinese and Japanese
   each follow [Misaki](https://github.com/hexgrad/misaki)'s rules, checked
   character for character against Misaki's own output.
2. **Voice model.** Pick the one that fits. More than one can be installed:

   | Engine                        | What it's for                                    | Download      |
   | ----------------------------- | ------------------------------------------------ | ------------- |
   | Kokoro 82M                    | Fast, 55 voices, 9 languages, exact word timings | about 340 MB  |
   | Qwen3-TTS 0.6B or 1.7B clone  | Speaks in a voice you record, 10 languages       | 1.3 or 2.4 GB |
   | Qwen3-TTS 0.6B or 1.7B voices | Nine ready-made voices, 10 languages             | 1.3 or 2.3 GB |

3. **Word timings.** Kokoro reports where each word falls. For the Qwen3
   models, a forced aligner listens to the finished audio and measures it, and
   it re-records any passage that comes out garbled.

Each model downloads when you install it, and its checksum is built into the
program, so it installs only the file it expects.

## Align an existing recording

```sh
bisque-voice align --audio narration.mp3 --text - --aligner qwen3-q4 < script.txt
```

This prints word timings and the original file's hash and size as JSON. It
does not synthesize, trim, change speed, encode, or write the recording. Times
refer to the decoded file's playback clock, including its leading silence.
MP3 decoding needs ffmpeg on PATH; WAV decoding does not.

Use the original script, including `[Qwen3](kwen three)` pronunciation
overrides. The aligner hears the spoken form and returns the display words in
source order, so cue indices stay valid. For a multiword display whose spoken
form has a different word count, use one override per display word; ambiguous
mappings are rejected. `exact: true` means source-token mapping, not a claim
that every acoustic boundary is correct. Silent recordings, collapsed or
low-confidence alignments, and invalid spans fail without emitting metadata.

Choose an installed aligner from `bisque-voice aligners`. Intel macOS builds
cannot run the bundled ONNX aligners; they refuse this command rather than
substitute estimated timings. Existing Intel narration remains approximate.

## Get started

Paste this into your coding agent:

```
Install bisque-voice for me by following
https://github.com/bisque-cloud/bisque-voice/blob/main/INSTALL.md
```

Or install it yourself on macOS or Linux:

```sh
curl -fsSL https://download.bisque.today/bisque-voice/install.sh | sh
```

On Windows, in PowerShell:

```powershell
irm https://download.bisque.today/bisque-voice/install.ps1 | iex
```

[INSTALL.md](./INSTALL.md) has every command and option.
[CONTRIBUTING.md](./CONTRIBUTING.md) covers building from source and how it's
made inside, and [GLOSSARY.md](./GLOSSARY.md) explains every term. Every
release is built, signed (macOS), attested and published by this repository's
[release workflow](./.github/workflows/release.yml).

## License

[MIT](./LICENSE)
