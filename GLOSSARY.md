# Glossary and dependency list

Everything `bisque-voice` depends on, what it does, and why it is here. Written
to be readable without prior exposure to speech synthesis.

## What the program does

`bisque-voice` turns written text into a spoken audio file, entirely on the user's
own machine — no server, no API key. It also returns **the start and end time of
every word**, which is what lets a presentation highlight words as they are
spoken.

## The pipeline

Almost every term below belongs to one of these five steps.

```
  text  ──►  G2P  ──►  phonemes  ──►  neural model  ──►  audio + word timings
             (1)          (2)              (3)                 (4)  (5)
```

1. **G2P** decides how the text is pronounced.
2. **Phonemes** are that pronunciation, written down.
3. The **model** turns phonemes into sound.
4. **Audio** comes out as WAV or MP3.
5. **Word timings** come out alongside it.

The interesting problem is step 1, and it is where all nine languages differ.

---

## Core terminology

| Term                                      | Meaning                                                                                                                                                                                         |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phoneme**                               | The smallest unit of speech sound that can change a word's meaning. "cat" is three phonemes: _k_, _æ_, _t_.                                                                                     |
| **Grapheme**                              | A written character. The letter `c`.                                                                                                                                                            |
| **G2P** (grapheme-to-phoneme)             | Working out pronunciation from spelling. Hard because spelling lies: "knight" is pronounced _naɪt_ — the k and gh are silent.                                                                   |
| **IPA** (International Phonetic Alphabet) | The standard alphabet for writing sounds, one symbol per sound. `ˈola` is Spanish "hola". Most phoneme strings in this codebase are IPA.                                                        |
| **Lexicon**                               | A pronunciation dictionary: word in, phonemes out. English uses one; most other languages here do not have one and use rules instead.                                                           |
| **Phonemization**                         | Running G2P over a whole text.                                                                                                                                                                  |
| **Vocabulary / tokenizer**                | The model's lookup table from a single phoneme character to an integer id. The model only ever sees integers.                                                                                   |
| **Word timings**                          | `{word, start, end}` for each word, in seconds. The published guarantee of this tool.                                                                                                           |
| **Cue**                                   | A marker in a presentation that fires at a given word. Cues index into word timings, so a timing that drifts moves the wrong thing on screen.                                                   |
| **Frame**                                 | The model's unit of time: 600 audio samples at 24 kHz, so 1/40 of a second.                                                                                                                     |
| **Duration predictor**                    | The part of the model that decides how many frames each phoneme lasts. Reading it is what gives exact word timings rather than guessed ones.                                                    |
| **Forced aligner**                        | A separate tool that listens to finished audio and works out when each word was said. Used for engines that cannot report their own timings. Less accurate than reading the duration predictor. |
| **Model pack**                            | A single compressed archive holding everything an engine needs, downloaded on first use and checked against a SHA-256 hash before anything is unpacked.                                         |
| **Golden fixture**                        | A recorded known-correct output, committed to the repo, that tests compare against character by character.                                                                                      |

## Models and the reference implementation

| Name          | What it is                                                                                                                                         | Why we need it                                                                                                                                                                 |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Kokoro**    | The neural text-to-speech model this tool runs. 82 million parameters, small enough for a laptop CPU. Its pack holds 55 voices across 9 languages. | It is the engine. Critically, **it does not read letters** — it only accepts phonemes, which is why G2P is the whole problem.                                                  |
| **Misaki**    | The G2P library written by Kokoro's author. Python.                                                                                                | It is **the specification, not a reference to consult**. Kokoro was trained on Misaki's exact output, so any deviation is a mispronunciation. We ported it; we do not ship it. |
| **Qwen3-TTS** | A different, larger speech model, used for voice cloning.                                                                                          | Unrelated to the 9-language work; it shares the binary.                                                                                                                        |

## The eSpeak languages: Spanish, French, Italian, Portuguese, Hindi

| Name                       | What it is                                                                                                      | Why we need it                                                                                                                                                                                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **eSpeak NG**              | A very old, very small rule-based speech synthesizer. Rules, not machine learning.                              | Misaki has no lexicon for these five languages, so it asks eSpeak for the pronunciation. **We use none of eSpeak's audio** — only its text→phoneme rules. Those rules _are_ the pronunciation Kokoro learned.                                                                            |
| **phonemizer**             | A Python library that wraps eSpeak and adds things eSpeak lacks.                                                | The trap: Misaki configures phonemizer with four options, and **three of them are implemented in phonemizer, not eSpeak** — preserving punctuation, stripping language-switch markers, and tie characters. Calling eSpeak directly gives different output, so this had to be ported too. |
| **Language switch marker** | When eSpeak meets a foreign word (French text containing "football") it flags the switch inline as `(en)…(fr)`. | Misaki strips these. Left in, they would become nonsense phonemes.                                                                                                                                                                                                                       |
| **Tie character**          | A mark joining two symbols that are one sound — `a͡ɪ` is the single vowel in "price", not _a_ then _ɪ_.          | Misaki asks for ties so it can map those pairs to Kokoro's single-character equivalents. Without ties they cannot be told apart from two separate sounds.                                                                                                                                |
| **Stress mark**            | `ˈ` before the emphasized syllable.                                                                             | Kokoro's vocabulary includes them; dropping them flattens the speech.                                                                                                                                                                                                                    |

## Chinese

| Name                          | What it is                                                                                                                       | Why we need it                                                                                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Segmentation**              | Chinese is written with no spaces, so "we work at the bank" arrives as one run of characters. Segmentation splits it into words. | Pronunciation depends on the split, and so does where pauses land.                                                                                    |
| **jieba**                     | The standard Chinese word segmenter. `jieba-rs` is the Rust port.                                                                | Misaki segments with it. Verified: the Rust port matches Python exactly.                                                                              |
| **HMM** (hidden Markov model) | A statistical method for guessing structure in a sequence.                                                                       | jieba uses it to segment words that are not in its dictionary. It is **on by default in Python**, which is easy to miss and changes the output.       |
| **Pinyin**                    | The standard way of writing Mandarin pronunciation in Latin letters, with tone numbers or accents. `yín háng` = "bank".          | The intermediate step: characters → pinyin → IPA.                                                                                                     |
| **Tone**                      | Mandarin's pitch contour, which distinguishes otherwise identical syllables. Four tones plus a neutral one.                      | Meaning-bearing. In the phoneme string they appear as arrows: `→ ↗ ↓ ↘`.                                                                              |
| **Polyphone**                 | A character with more than one reading, chosen by context. 行 is _háng_ in 银行 (bank) but _xíng_ in 行走 (to walk).             | Getting these wrong is an audible error. Only a **phrase dictionary** can resolve them.                                                               |
| **pypinyin**                  | The Python library that maps characters to pinyin, including a 47,000-entry phrase dictionary.                                   | The Rust `pinyin` crate is character-only with no phrase dictionary, so it cannot resolve polyphones. We port the algorithm and ship pypinyin's data. |
| **Initials and finals**       | A Mandarin syllable splits into an initial consonant and a final (vowel plus any ending). `bāng` = initial _b_ + final _ang_.    | Misaki's character-to-IPA tables are keyed on this split.                                                                                             |
| **cn2an**                     | A Python library converting between Arabic numerals and Chinese numerals.                                                        | Misaki rewrites "2026" into Chinese characters before phonemizing, so the model says "two thousand twenty-six" rather than spelling out digits.       |

## Japanese

| Name                        | What it is                                                                                                           | Why we need it                                                                                                                                                                |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Kanji**                   | Characters borrowed from Chinese. 東京 = Tokyo.                                                                      | They carry **no pronunciation information on their face** — the same character is read differently in different words. This is why Japanese needs a dictionary and not rules. |
| **Hiragana / Katakana**     | Japan's two phonetic scripts, ~46 symbols each. Hiragana for native grammar, katakana mostly for loanwords.          | Unlike kanji, these map directly to sounds.                                                                                                                                   |
| **Romaji**                  | Japanese written in Latin letters.                                                                                   | The intermediate step before IPA.                                                                                                                                             |
| **Hepburn**                 | The standard romaji scheme.                                                                                          | Misaki uses a modified Hepburn table, 164 entries, mapping kana to IPA.                                                                                                       |
| **Morphological analyzer**  | A tool that splits Japanese into words _and_ reports each word's pronunciation.                                      | The only way to know how a kanji is read here.                                                                                                                                |
| **MeCab**                   | The standard Japanese morphological analyzer. **fugashi** is its Python wrapper; **lindera** is the Rust equivalent. | Misaki uses fugashi; we use lindera.                                                                                                                                          |
| **UniDic**                  | The dictionary MeCab reads: every word form and its pronunciation. 248 MB.                                           | The single largest cost of the nine languages, and why Japanese data is a separate download rather than bundled for everyone.                                                 |
| **Full-width / half-width** | CJK text has two Unicode forms of ordinary ASCII: `Ａ` and `A`.                                                      | Misaki normalizes between them; **mojimoji** is the Python library that does it.                                                                                              |
| **NFKC**                    | A Unicode normalization scheme that folds such variants together.                                                    | Part of Misaki's Japanese text cleanup.                                                                                                                                       |
| **jaconv**                  | Python library for converting between kana scripts.                                                                  | Misaki converts katakana readings to hiragana before mapping to sounds.                                                                                                       |

## What ships inside the binary

Libraries compiled into the released executable. Mostly Rust crates; the C/C++
ones are linked statically, so there is still exactly one file to ship.

| Crate                  | What it does                                                                                                                        | Why                                                                                                                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `qwentts.cpp` / `ggml` | A C++ implementation of Qwen3-TTS, and the tensor-maths library under it — the same one behind `llama.cpp`.                         | Voice cloning. Vendored as a pinned submodule, built by `build.rs`, linked statically. On a Mac it also brings in **Metal** (GPU) and **Accelerate** (Apple's CPU maths library). |
| `ort`                  | Bindings to **ONNX Runtime**, Microsoft's engine for running neural models. **ONNX** is a portable file format for a trained model. | Runs Kokoro. Also selects hardware acceleration: **CoreML** on Mac, **DirectML** on Windows, **CUDA** on NVIDIA Linux.                                                            |
| `half`                 | 16-bit floating point numbers.                                                                                                      | Kokoro shipped in **fp16** (half precision) until its vocoder was found to overflow to NaN for some voice/text pairs — silent audio. It ships in fp32 now.                        |
| `safetensors`          | A file format for model weights that cannot execute code when loaded.                                                               | Holds the small neural network that guesses English pronunciation for words missing from the lexicon.                                                                             |
| `espeak-rs-sys`        | Raw bindings to the eSpeak NG C library.                                                                                            | The five eSpeak languages. Its higher-level sibling `espeak-rs` was unusable: no tie support, and its own incompatible flag handling.                                             |
| `jieba-rs`             | Chinese word segmentation.                                                                                                          | Chinese.                                                                                                                                                                          |
| `lindera`              | Japanese morphological analysis.                                                                                                    | Japanese.                                                                                                                                                                         |
| `tokenizers`           | HuggingFace's text tokenizer, here using **BPE** (byte-pair encoding — a scheme that splits text into frequent sub-word chunks).    | Qwen3's text input.                                                                                                                                                               |
| `fancy-regex`          | A regex engine supporting **lookahead** and **lookbehind** — conditions like "a period _not_ preceded by a digit".                  | The standard `regex` crate deliberately omits these. Misaki's punctuation rule needs one: `.` is punctuation, except between two digits, where it is a decimal point.             |
| `regex`                | The standard, faster regex engine, without lookaround.                                                                              | Everything else.                                                                                                                                                                  |
| `rustfft`              | **FFT** (fast Fourier transform) — converts a sound wave into the frequencies it contains.                                          | The forced aligner's audio front end. Pure Rust, so it adds no C toolchain.                                                                                                       |
| `mp3lame-encoder`      | MP3 encoding.                                                                                                                       | Output format.                                                                                                                                                                    |
| `ureq`                 | HTTP client.                                                                                                                        | Downloads model packs. Uses rustls rather than OpenSSL so there is no system library to find on a stranger's machine.                                                             |
| `flate2`, `tar`        | Decompression and archive unpacking.                                                                                                | Model packs are `.tar.gz`.                                                                                                                                                        |
| `sha2`                 | SHA-256 hashing.                                                                                                                    | Verifies a pack before unpacking it. A corrupted download once shipped garbled audio.                                                                                             |
| `fs4`                  | File locking.                                                                                                                       | Two runs starting at once must not race on the same download.                                                                                                                     |
| `dirs`                 | Locates the home directory across platforms.                                                                                        | Where packs are stored.                                                                                                                                                           |
| `serde`, `serde_json`  | Converting Rust data to and from JSON.                                                                                              | Config, fixtures, and the JSON output contract.                                                                                                                                   |
| `clap`                 | Command-line argument parsing.                                                                                                      | The CLI.                                                                                                                                                                          |
| `anyhow`               | Error handling with context.                                                                                                        | Readable failures.                                                                                                                                                                |

## Build-time only

Not present in the shipped binary; required on the machine doing the compiling.

| Tool                                 | What it is                                                                   | Why                                                                                                                                                           |
| ------------------------------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CMake**                            | A build system for C and C++ projects.                                       | eSpeak NG and qwentts.cpp are C/C++ and build with it.                                                                                                        |
| **bindgen**                          | Reads a C header file and writes the matching Rust declarations.             | Hand-transcribing a C interface is a silent-corruption bug waiting to happen.                                                                                 |
| **libclang**                         | The C/C++ parser (part of the Clang compiler) that bindgen uses.             | Needed because bindgen is. **CI must set `LIBCLANG_PATH`** on machines where Clang is not on the default search path.                                         |
| **FFI** (foreign function interface) | The mechanism for calling C code from Rust.                                  | How Rust reaches eSpeak and qwentts.cpp.                                                                                                                      |
| **git submodule**                    | A pin to an exact commit of another repository, checked out inside this one. | How `vendor/qwentts.cpp` and `vendor/vibevoice.cpp` are carried. `sh vendor/fetch.sh` checks them out at their pinned commits, or there is no C++ to compile. |
| **Python**                           | Only for regenerating golden fixtures, via `tools/gen-g2p-fixtures.py`.      | The fixtures are committed. Running Misaki is the only way to know a port of Misaki is faithful. Nothing at runtime touches Python.                           |

## Data that ships separately

Kokoro's weights speak all nine languages, but the _data_ needed to work out
pronunciation is per-language and very unevenly sized. Bundling it all would
make someone narrating English slides download a quarter-gigabyte Japanese
dictionary, so each language's data is a separate on-demand download.

| Pack         | Contents                                            | Size              |
| ------------ | --------------------------------------------------- | ----------------- |
| Engine       | Kokoro, all 55 voices, English lexicons             | ~171 MB           |
| eSpeak G2P   | eSpeak's phoneme tables and 5 language dictionaries | ~3 MB             |
| Chinese G2P  | pypinyin's character and phrase dictionaries        | ~3 MB             |
| Japanese G2P | UniDic                                              | ~49 MB compressed |

## Why there is C in a Rust program

The crate was deliberately pure Rust so that release builds cross-compile
without hunting for a toolchain. Two dependencies break that, both knowingly.

**eSpeak**, for the five languages it covers: they have no pronunciation
dictionary, and their phonemes _are_ eSpeak's rules. A hand-written substitute
would not produce the sequences Kokoro was trained on, so it would be a
worse-sounding program that merely built more easily.

**qwentts.cpp**, for voice cloning. Every mature way to run Qwen3-TTS is
Python — the official package is ~5.7 GB installed — except this one, and
upstream publishes a versioned C99 ABI naming Rust as a supported consumer. The
alternative was not "a pure-Rust cloner"; it was no local cloning at all. It
costs the build nothing eSpeak had not already cost, since CMake and libclang
were required either way.
