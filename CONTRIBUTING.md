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
```

You need a Rust toolchain, CMake, a C++17 compiler and libclang (`bindgen`
uses it). [`GLOSSARY.md`](./GLOSSARY.md) says what each dependency is for.

The Spanish, French, Italian, Portuguese, Hindi and Japanese fixture tests read
the per-language pronunciation data. Without it they fail and say where to put
it: set `BISQUE_VOICE_ESPEAK_DATA` and `BISQUE_VOICE_JA_DATA` to the unpacked
`kokoro-g2p-espeak` and `kokoro-g2p-ja` packs from
`https://download.bisque.today/bisque-voice/models/`.

A local build has no model checksums compiled in, so it refuses to download a
model pack until you set one; see "Building a pack" in the README.

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
