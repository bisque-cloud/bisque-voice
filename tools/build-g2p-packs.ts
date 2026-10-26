#!/usr/bin/env bun
/**
 * Build the per-language G2P data packs.
 *
 * Kokoro's weights speak all nine languages, but the *data* needed to work out
 * pronunciation is per-language and very unevenly sized — eSpeak's tables are
 * about 3 MB, pypinyin's dictionaries about 3 MB, and Japanese needs a 248 MB
 * morphological dictionary. Bundling them all would make someone narrating
 * English slides download a quarter-gigabyte Japanese dictionary, so each is a
 * separate pack installed on demand. See `src/g2p/pack.rs`.
 *
 *   bun run tools/build-g2p-packs.ts [--out dist] [--only espeak|zh|ja]
 *
 * Prints each archive's path and SHA-256. Feed those to the build as
 * BISQUE_VOICE_PACK_SHA256_G2P_ESPEAK / _G2P_ZH / _G2P_JA and publish them as
 * download.bisque.today/bisque-voice/models/<PACK_VERSION>.tar.gz
 *
 * Every input is pinned by version and, where the source offers one, by hash.
 * A pack rebuilt from the same pins must be byte-identical in content, because
 * the checksum baked into a release has to keep matching.
 */

import { mkdir, rm, writeFile, readdir, copyFile } from "node:fs/promises";
import { join } from "node:path";

/** Must match `pack_version` on the matching entry in `src/g2p/pack.rs`. */
const PACKS = {
  espeak: "kokoro-g2p-espeak-1",
  zh: "kokoro-g2p-zh-1",
  ja: "kokoro-g2p-ja-1",
} as const;

/**
 * eSpeak NG's version is pinned to the one `espeak-rs-sys` vendors. The data
 * files and the linked library share a binary format; a mismatch does not fail
 * loudly, it mispronounces.
 */
const ESPEAK_VERSION = "1.52.0";

/**
 * The five languages Misaki phonemizes with eSpeak, as Kokoro asks for them
 * (`LANG_CODES` in kokoro/pipeline.py). Note `fr` and `pt` rather than `fr-fr`
 * and `pt-br`: the *dictionary* files are per base language, while the voice
 * names carry the region.
 *
 * `en` is here for a reason that is not obvious and that a trimmed pack proved
 * the hard way. When eSpeak meets a foreign word it *switches voices* to
 * pronounce it — French "J'aime le football" is phonemized with the English
 * voice for that one word, which is what the `(en)` markers Misaki strips are
 * announcing. Without `en_dict` the switch still happens but has no dictionary
 * to switch to, and eSpeak falls back to spelling the word out one letter at a
 * time: "football" became "ˈɛf ˈo ˈo tˈe bˈe a ˈɛl ˈɛl". It does not error.
 *
 * English covers essentially every loanword in practice, at 168 KB. Shipping
 * all 117 of eSpeak's dictionaries would cost 22 MB to also cover, say, a
 * Vietnamese word inside a Spanish sentence — where the same silent
 * degradation still applies, and is the accepted limit of this pack.
 */
const ESPEAK_DICTS = ["es", "fr", "it", "pt", "hi", "en"];

/**
 * Files every eSpeak voice needs regardless of language: the phoneme
 * definitions, the index, and the intonation tables.
 */
const ESPEAK_SHARED = [
  "phondata",
  "phonindex",
  "phontab",
  "phondata-manifest",
  "intonations",
];

/**
 * UniDic's MeCab source, pinned by URL and MD5. This is the same artifact
 * `lindera-unidic`'s build script fetches.
 */
const UNIDIC_DIR = "unidic-mecab-2.1.2";
const UNIDIC_TARBALL = `${UNIDIC_DIR}.tar.gz`;
const UNIDIC_URL = `https://Lindera.dev/${UNIDIC_TARBALL}`;
const UNIDIC_MD5 = "f4502a563e1da44747f61dcd2b269e35";
/** Must match the `lindera` version in Cargo.toml: the compiled dictionary
 * format is not guaranteed stable across versions. */
const LINDERA_VERSION = "4.0.1";
/** The Misaki release these fixtures and this port were taken from. */
const MISAKI_VERSION = "0.7.4";

/** pypinyin ships its dictionaries as JSON inside the wheel, already parsed. */
const PYPINYIN_VERSION = "0.55.0";
const PYPINYIN_WHEEL = `pypinyin-${PYPINYIN_VERSION}-py2.py3-none-any.whl`;
const PYPINYIN_URL = `https://files.pythonhosted.org/packages/py2.py3/p/pypinyin/${PYPINYIN_WHEEL}`;

const args = process.argv.slice(2);
const flag = (name: string, fallback: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const outDir = flag("out", "dist");
const only = flag("only", "");

async function run(cmd: string[], cwd?: string) {
  const p = Bun.spawnSync(cmd, { cwd, stdout: "inherit", stderr: "inherit" });
  if (p.exitCode !== 0)
    throw new Error(`${cmd.join(" ")} failed with ${p.exitCode}`);
}

async function archive(stage: string, packVersion: string) {
  const path = join(outDir, `${packVersion}.tar.gz`);
  await run(["tar", "-czf", path, "-C", stage, "."]);
  const hash = new Bun.CryptoHasher("sha256");
  hash.update(new Uint8Array(await Bun.file(path).arrayBuffer()));
  const sha = hash.digest("hex");
  const bytes = (await Bun.file(path).stat()).size;
  return { path, sha, bytes };
}

const built: {
  id: string;
  env: string;
  path: string;
  sha: string;
  bytes: number;
}[] = [];

// --- eSpeak -----------------------------------------------------------------
//
// The data is NOT built from the vendored source. `espeak-rs-sys` can compile
// it (its `compile-espeak-intonations` feature), but that step runs the
// freshly-built espeak-ng binary and traps on arm64 macOS — and it cannot run
// at all when cross-compiling, which the release matrix does. So the data comes
// from an installed eSpeak NG of the pinned version instead, and the version is
// checked rather than trusted.
async function buildEspeak() {
  const stage = join(outDir, PACKS.espeak);
  await rm(stage, { recursive: true, force: true });
  await mkdir(join(stage, "espeak-ng-data"), { recursive: true });

  const which = Bun.spawnSync(["espeak-ng", "--version"]);
  if (which.exitCode !== 0)
    throw new Error(
      "espeak-ng is not installed. `brew install espeak-ng` (macOS) or " +
        "`apt install espeak-ng-data` (Debian/Ubuntu), then re-run.",
    );
  const banner = new TextDecoder().decode(which.stdout);
  if (!banner.includes(ESPEAK_VERSION))
    throw new Error(
      `eSpeak NG ${ESPEAK_VERSION} is required to match the vendored library, ` +
        `but the installed one reports: ${banner.trim()}`,
    );
  // `espeak-ng --version` prints "... Data at: <dir>".
  const dataDir = banner.match(/Data at:\s*(\S+)/)?.[1];
  if (!dataDir)
    throw new Error(`could not parse the data path from: ${banner}`);

  const dest = join(stage, "espeak-ng-data");
  for (const name of ESPEAK_SHARED) {
    await copyFile(join(dataDir, name), join(dest, name));
  }
  for (const lang of ESPEAK_DICTS) {
    await copyFile(join(dataDir, `${lang}_dict`), join(dest, `${lang}_dict`));
  }
  // `lang/` declares which voices exist and `voices/!v` holds the variants;
  // espeak_ListVoices reads both, and language selection fails without them.
  await run(["cp", "-R", join(dataDir, "lang"), dest]);
  await run(["cp", "-R", join(dataDir, "voices"), dest]);

  await writeFile(
    join(stage, "PACK.md"),
    [
      `# ${PACKS.espeak}`,
      "",
      `eSpeak NG ${ESPEAK_VERSION} phoneme data, trimmed to the five languages`,
      "Misaki phonemizes with it: Spanish, French, Italian, Portuguese, Hindi.",
      "",
      "eSpeak NG is GPL-3.0. Only its data files are here; the library is linked",
      "into the binary from the vendored source.",
      "",
    ].join("\n"),
  );

  const a = await archive(stage, PACKS.espeak);
  built.push({ id: "g2p-espeak", env: "G2P_ESPEAK", ...a });
}

// --- Chinese ----------------------------------------------------------------
//
// pypinyin's two dictionaries, taken straight from the published wheel so the
// build does not need a Python environment. `pinyin_dict.json` is one reading
// per character; `phrases_dict.json` is the 47,000-entry phrase table that
// resolves polyphones (银行 is "yín háng", not "yín xíng") and is the reason
// this data is carried rather than approximated with a Rust crate.
async function buildChinese() {
  const stage = join(outDir, PACKS.zh);
  await rm(stage, { recursive: true, force: true });
  await mkdir(stage, { recursive: true });

  const wheel = join(outDir, PYPINYIN_WHEEL);
  if (!(await Bun.file(wheel).exists())) {
    console.log(`→ ${PYPINYIN_URL}`);
    const res = await fetch(PYPINYIN_URL);
    if (!res.ok) throw new Error(`GET ${PYPINYIN_URL} → ${res.status}`);
    await Bun.write(wheel, await res.arrayBuffer());
  }

  const unpacked = join(outDir, "pypinyin-unpacked");
  await rm(unpacked, { recursive: true, force: true });
  await mkdir(unpacked, { recursive: true });
  // A wheel is a zip.
  await run(["unzip", "-q", wheel, "-d", unpacked]);

  for (const name of ["pinyin_dict.json", "phrases_dict.json"]) {
    const src = join(unpacked, "pypinyin", name);
    if (!(await Bun.file(src).exists()))
      throw new Error(`${name} is missing from ${PYPINYIN_WHEEL}`);
    await copyFile(src, join(stage, name));
  }
  await rm(unpacked, { recursive: true, force: true });

  await writeFile(
    join(stage, "PACK.md"),
    [
      `# ${PACKS.zh}`,
      "",
      `pypinyin ${PYPINYIN_VERSION} pronunciation dictionaries, MIT.`,
      "",
      "- `pinyin_dict.json` — one reading per character.",
      "- `phrases_dict.json` — per-word readings, which is what makes",
      "  polyphones come out right.",
      "",
      "jieba's segmentation dictionary is not here: `jieba-rs` embeds it.",
      "",
    ].join("\n"),
  );

  const a = await archive(stage, PACKS.zh);
  built.push({ id: "g2p-zh", env: "G2P_ZH", ...a });
}

// --- Japanese ---------------------------------------------------------------
//
// UniDic, the morphological dictionary MeCab reads. Kanji carry no
// pronunciation on their face, so this is not optional and there is no smaller
// substitute that produces the same readings Misaki does.
//
// Built rather than downloaded prebuilt: lindera compiles the MeCab CSV source
// into its own format, and the compiled form is what `load_fs_dictionary`
// reads. Both inputs are pinned — the CSV source by URL and MD5, the build
// metadata by crate version.
async function buildJapanese() {
  const stage = join(outDir, PACKS.ja);
  await rm(stage, { recursive: true, force: true });
  await mkdir(stage, { recursive: true });

  const lindera = Bun.spawnSync(["lindera", "--version"]);
  if (lindera.exitCode !== 0)
    throw new Error(
      `lindera-cli is required to compile UniDic. Install the matching ` +
        `version with:\n  cargo install lindera-cli --version ${LINDERA_VERSION}`,
    );

  // The MeCab CSV source.
  const tarball = join(outDir, UNIDIC_TARBALL);
  if (!(await Bun.file(tarball).exists())) {
    console.log(`→ ${UNIDIC_URL} (140 MB)`);
    const res = await fetch(UNIDIC_URL);
    if (!res.ok) throw new Error(`GET ${UNIDIC_URL} → ${res.status}`);
    await Bun.write(tarball, await res.arrayBuffer());
  }
  const md5 = Bun.spawnSync(["md5", "-q", tarball]);
  const digest = new TextDecoder().decode(md5.stdout).trim();
  if (md5.exitCode === 0 && digest && digest !== UNIDIC_MD5)
    throw new Error(
      `${UNIDIC_TARBALL} hashes to ${digest}, not the pinned ${UNIDIC_MD5}`,
    );

  const src = join(outDir, UNIDIC_DIR);
  if (!(await Bun.file(join(src, "char.def")).exists())) {
    console.log("→ extracting UniDic");
    await run(["tar", "-xzf", tarball, "-C", outDir]);
  }

  // lindera's build metadata describes UniDic's column layout. It ships in the
  // lindera-unidic crate, which is 19 KB — the dictionary itself is fetched
  // separately, above.
  const metaCrate = join(outDir, `lindera-unidic-${LINDERA_VERSION}.crate`);
  if (!(await Bun.file(metaCrate).exists())) {
    const url = `https://static.crates.io/crates/lindera-unidic/lindera-unidic-${LINDERA_VERSION}.crate`;
    console.log(`→ ${url}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GET ${url} → ${res.status}`);
    await Bun.write(metaCrate, await res.arrayBuffer());
  }
  await run(["tar", "-xzf", metaCrate, "-C", outDir]);
  const metadata = join(
    outDir,
    `lindera-unidic-${LINDERA_VERSION}`,
    "metadata.json",
  );
  if (!(await Bun.file(metadata).exists()))
    throw new Error(`metadata.json is missing from ${metaCrate}`);

  console.log("→ compiling UniDic (this takes a few minutes)");
  await run([
    "lindera",
    "build",
    "--src",
    src,
    "--dest",
    join(stage, "lindera-unidic"),
    "--metadata",
    metadata,
  ]);

  // Misaki's word list, used to rejoin compounds the analyzer split.
  const misaki = join(outDir, `misaki-${MISAKI_VERSION}.tar.gz`);
  if (!(await Bun.file(misaki).exists())) {
    const url = `https://files.pythonhosted.org/packages/source/m/misaki/misaki-${MISAKI_VERSION}.tar.gz`;
    console.log(`→ ${url}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GET ${url} → ${res.status}`);
    await Bun.write(misaki, await res.arrayBuffer());
  }
  await run([
    "tar",
    "-xzf",
    misaki,
    "-C",
    outDir,
    `misaki-${MISAKI_VERSION}/misaki/data/ja_words.txt`,
  ]);
  await copyFile(
    join(outDir, `misaki-${MISAKI_VERSION}`, "misaki", "data", "ja_words.txt"),
    join(stage, "ja_words.txt"),
  );

  await writeFile(
    join(stage, "PACK.md"),
    [
      `# ${PACKS.ja}`,
      "",
      "Japanese pronunciation data.",
      "",
      `- \`lindera-unidic/\` — UniDic ${UNIDIC_DIR.replace("unidic-mecab-", "")},`,
      "  compiled to lindera's format. BSD/LGPL/GPL tri-license.",
      `- \`ja_words.txt\` — Misaki ${MISAKI_VERSION}'s word list, Apache-2.0.`,
      "",
    ].join("\n"),
  );

  const a = await archive(stage, PACKS.ja);
  built.push({ id: "g2p-ja", env: "G2P_JA", ...a });
}

await mkdir(outDir, { recursive: true });
if (!only || only === "espeak") await buildEspeak();
if (!only || only === "zh") await buildChinese();
if (!only || only === "ja") await buildJapanese();

console.log("");
for (const b of built) {
  console.log(`${b.id}`);
  console.log(`  archive : ${b.path}`);
  console.log(`  size    : ${(b.bytes / 1e6).toFixed(1)} MB`);
  console.log(`  sha256  : ${b.sha}`);
}
console.log("");
console.log("Build the binary against them with:");
console.log(
  "  " +
    built
      .map((b) => `BISQUE_VOICE_PACK_SHA256_${b.env}=${b.sha}`)
      .join(" \\\n  "),
);
