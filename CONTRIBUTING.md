# Contributing

Thanks for wanting to make `bisque-voice` better.

- **Found a bug?** [Open an issue](https://github.com/bisque-cloud/bisque-voice/issues)
  with the command you ran, what you heard or saw, `bisque-voice --version`,
  and your OS. If a word was mispronounced, the sentence and the voice are the
  single most useful thing you can give us.
- **Have a fix or improvement?** Open a pull request. Small and focused beats
  large and sweeping, and for anything substantial an issue first saves you
  from building something we can't take.

## Building

```sh
sh vendor/fetch.sh          # the two vendored C++ engines, at their pinned commits
cargo build --release
cargo test
cargo test --no-default-features   # the build without Kokoro and the aligners (Intel Macs)
```

You need a Rust toolchain, CMake, a C++17 compiler and libclang (`bindgen`
uses it). [`GLOSSARY.md`](./GLOSSARY.md) says what each dependency is for.

The Spanish, French, Italian, Portuguese, Hindi and Japanese fixture tests read
the per-language pronunciation data. Without it they fail and say where to put
it: set `BISQUE_VOICE_ESPEAK_DATA` and `BISQUE_VOICE_JA_DATA` to the unpacked
`kokoro-g2p-espeak` and `kokoro-g2p-ja` packs from
`https://download.bisque.today/bisque-voice/models/`.

A local build has no model checksums compiled in, so it refuses to download a
model pack until you set one; see "Building a model pack" below.

## How it works inside

### One timing per word, or a failure

`words` is one-to-one with the input's whitespace-separated words, in order,
always: callers index that array to place cues, so a mismatch would move every
cue with no error anywhere. The engine reconciles the model's tokens back onto
the input's words and fails if it cannot. `exact: false` means the text was
expanded ("$4.2M" becomes "four point two million dollars") and the time was
spread across the words it came from by length; the count is still exact. A
forced aligner is given the input's own words and measures each one, so it is
always the exact path. The tests cover the hard cases (`3.5x`, `$4.2M`,
`state-of-the-art`), the timestamp walk, chunk boundaries and number-to-words.

### Pronunciation

Kokoro reads phonemes, not letters, and it was trained on the ones
[Misaki](https://github.com/hexgrad/misaki) produces, so every language here is
a port of Misaki's pipeline. `fixtures/g2p/` holds Misaki's own output, and the
tests compare against it character for character; `tools/gen-g2p-fixtures.py`
regenerates the fixtures by running Misaki. Misaki ships no tests of its own.

| Language                                    | How it is phonemized                                                                                                      |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| English (US and UK)                         | Misaki's Apache-2.0 lexicons (90k gold and 93k silver entries), its stemming and number rules, and its 3 MB BART fallback |
| Spanish, French, Italian, Portuguese, Hindi | eSpeak NG's letter-to-sound rules, through the `phonemizer` options Misaki uses                                           |
| Chinese                                     | jieba segmentation, then pypinyin with its phrase dictionary, so 银行 is _yín háng_                                       |
| Japanese                                    | UniDic morphological analysis                                                                                             |

English part-of-speech tagging is rule-based (`src/g2p/tagger.rs`). For an
ordinary word it abstains, which selects the lexicon's common reading. For the
roughly 790 heteronyms whose reading depends on part of speech ("present",
"record", "live", "read"), it decides from one word of context ("will \_" is a
verb, "their \_ of" is a noun).

A Chinese polyphone standing alone gets its most common reading: 行 is _háng_
in 银行, but the last 行 in 银行的行 is its own token and comes out _xíng_.
Misaki does the same.

`tools/verify-regression.ts` narrates the mispronunciations listeners have
reported, as they shipped and as fixed, through the built binary.

### Model packs

Each model and aligner is one archive holding everything it needs, installed
under `~/.bisque/models/<pack-version>/`. A pack version never changes once
published: bump it when the contents change, because installed clients find a
pack by that name and check its checksum.

- **Checked before use.** The archive's SHA-256 is checked before anything is
  extracted, and a build with no checksum for a pack refuses to download it.
- **Safe to start twice.** A lock file serializes concurrent installs; the
  second one waits and finds the pack in place.
- **Never half-installed.** Extraction goes to a temporary directory that is
  renamed into place, and a pack counts as installed only once its completion
  marker exists.

### Building a model pack

```sh
bun run tools/build-model-pack.ts --variant fp32
BISQUE_VOICE_PACK_SHA256_KOKORO=<printed hash> cargo build --release

bun run tools/build-qwen-clone-pack.ts
bun run tools/build-aligner-pack.ts --variant fp32   # or --variant q4
bun run tools/build-g2p-packs.ts                      # per-language pronunciation data
```

The checksum variable is per pack: `BISQUE_VOICE_PACK_SHA256_<ID>`, uppercased
with `-` replaced by `_`, one for each entry in `engine::REGISTRY` and
`align::REGISTRY` (see `build.rs`). The `qwen3-fp32` aligner reads
`BISQUE_VOICE_PACK_SHA256_QWEN3_FP32`. An unsuffixed `BISQUE_VOICE_PACK_SHA256`
is read by nothing, so the build succeeds and the binary then refuses every
download.

The stock Kokoro ONNX export returns only the waveform. Its per-phoneme
durations exist inside the graph as `/encoder/Clip_output_0`, and
`tools/promote-duration.ts` makes that tensor an output with a 59-byte protobuf
append. The engine checks `frame_sum × 600 == waveform.len()` on every
synthesis and refuses to emit timings if it fails.

Kokoro ships as fp32. The fp16 export's vocoder overflows to NaN for some
(voice, text) pairs, which came out as silence; fp32 synthesizes every known
failing pair at the same speed on a CPU. q8 also works but is 2.4× slower on
ARM and shifts durations.

### Audio

MP3 is the default because it is a third the size of WAV for whoever downloads
the narration, and LAME is compiled in, so there is no ffmpeg to find. Long
text is split at sentence boundaries, because Kokoro accepts at most 510 tokens
and an ordinary 82-word paragraph exceeds that, and the pieces are joined with
continuous timings.

## How changes land

We review every pull request. Changes reach this repository through our
release process, so your pull request may be closed with your change committed
and credited rather than merged in place. Either way, you'll see your change,
with your name on it, here and in the next release.

By contributing you agree your contributions are licensed under the
[MIT License](./LICENSE).

## Security

Please report a security problem privately, through
[GitHub's private vulnerability reporting](https://github.com/bisque-cloud/bisque-voice/security/advisories/new),
rather than in a public issue.
