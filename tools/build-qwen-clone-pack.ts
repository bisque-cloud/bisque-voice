#!/usr/bin/env bun
/**
 * Build a Qwen3-TTS model pack for `bisque-voice` — voice cloning (base) or
 * named speakers (customvoice).
 *
 * Same contract as the other pack builders: one hash-verified archive holding
 * everything the engine needs, so there is no partial state to reason about.
 * Two GGUFs load together —
 *
 *   qwen-talker-{size}-{mode}-Q8_0.gguf   LM + code predictor (+ speaker
 *                                         encoder on base, speaker table on
 *                                         customvoice)
 *   qwen-tokenizer-12hz-Q8_0.gguf         the shared 12.5 Hz codec
 *
 * **The mode is the checkpoint.** Cloning from a reference clip is a Base
 * capability; `customvoice` is the named-speaker checkpoint (serena, vivian,
 * uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan) and `voicedesign` is
 * the `--instruct` one. The upstream README, the model card and
 * `qwen-tts --help` all agree — `--ref-wav` is "(Base only)", `--speaker` is
 * "(CustomVoice only)".
 *
 *   bun run tools/build-qwen-clone-pack.ts --size 0.6b [--mode base] [--out dist]
 *   bun run tools/build-qwen-clone-pack.ts --size 1.7b --mode customvoice
 *
 * `--local <dir>` reuses already-downloaded GGUFs from a directory instead of
 * re-pulling gigabytes from Hugging Face; each file is still verified against
 * the upstream content-length and its GGUF magic, so a truncated local copy
 * fails the same way a truncated download does.
 *
 * Prints the archive path and its SHA-256. Feed that hash to the build as
 * BISQUE_VOICE_PACK_SHA256_QWEN3_CLONE_06B / _17B (base) or
 * BISQUE_VOICE_PACK_SHA256_QWEN3_VOICES_06B / _17B (customvoice) and publish
 * the archive as
 * download.bisque.today/bisque-voice/models/<PACK_VERSION>.tar.gz
 */

import { copyFile, mkdir, rm, writeFile } from "node:fs/promises";
import { constants, existsSync } from "node:fs";
import { join } from "node:path";

/** GGUF conversions of the upstream checkpoints. Apache-2.0 weights. */
const GGUF_REPO =
  "https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main";

/** Shared by every talker size and mode — the codec is one model. */
const CODEC = "qwen-tokenizer-12hz-Q8_0.gguf";

/**
 * Each (mode, size)'s talker and the registry entry it belongs to.
 * `packVersion` must match `pack_version` on that entry in src/engine.rs, and
 * `shaVar` must match the id in build.rs's ENGINES list.
 */
const MODELS = {
  "base/0.6b": {
    talker: "qwen-talker-0.6b-base-Q8_0.gguf",
    packVersion: "qwen3-clone-06b-q8-1",
    shaVar: "BISQUE_VOICE_PACK_SHA256_QWEN3_CLONE_06B",
    engine: "qwen3-clone-06b",
    /** Speaker-embedding width, which is the talker's hidden size. A voice
     *  cloned on one size will not load into the other, and the engine checks
     *  this — see src/voices.rs. Base only: customvoice does not clone. */
    spkDim: 1024,
  },
  "base/1.7b": {
    talker: "qwen-talker-1.7b-base-Q8_0.gguf",
    packVersion: "qwen3-clone-17b-q8-1",
    shaVar: "BISQUE_VOICE_PACK_SHA256_QWEN3_CLONE_17B",
    engine: "qwen3-clone-17b",
    spkDim: 2048,
  },
  "customvoice/0.6b": {
    talker: "qwen-talker-0.6b-customvoice-Q8_0.gguf",
    packVersion: "qwen3-voices-06b-q8-1",
    shaVar: "BISQUE_VOICE_PACK_SHA256_QWEN3_VOICES_06B",
    engine: "qwen3-voices-06b",
    spkDim: null,
  },
  "customvoice/1.7b": {
    talker: "qwen-talker-1.7b-customvoice-Q8_0.gguf",
    packVersion: "qwen3-voices-17b-q8-1",
    shaVar: "BISQUE_VOICE_PACK_SHA256_QWEN3_VOICES_17B",
    engine: "qwen3-voices-17b",
    spkDim: null,
  },
} as const;

const args = process.argv.slice(2);
const flag = (name: string, fallback: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const outDir = flag("out", "dist");
const size = flag("size", "");
const mode = flag("mode", "base");
const localDir = flag("local", "");

const key = `${mode}/${size}`;
if (!(key in MODELS)) {
  const sizes = [...new Set(Object.keys(MODELS).map((k) => k.split("/")[1]))];
  const modes = [...new Set(Object.keys(MODELS).map((k) => k.split("/")[0]))];
  throw new Error(
    `--size is required (${sizes.join("|")}); --mode is ${modes.join("|")}`,
  );
}
const spec = MODELS[key as keyof typeof MODELS];

const stage = join(outDir, spec.packVersion);
await rm(stage, { recursive: true, force: true });
await mkdir(stage, { recursive: true });

/**
 * The expected size, from a HEAD against the resolved URL.
 *
 * Read separately rather than off the GET, because the GET's response arrives
 * without a `content-length` — so a check against that header silently never
 * runs, which is worse than not having one.
 */
async function expectedSize(url: string): Promise<number> {
  const res = await fetch(url, { method: "HEAD", redirect: "follow" });
  if (!res.ok) throw new Error(`HEAD ${url} → ${res.status}`);
  return Number(res.headers.get("content-length") ?? 0);
}

/**
 * Download to disk via curl.
 *
 * Not `fetch`: these are 0.7–2.1 GB files, and Bun's fetch stalled part-way
 * through one with no error and no bytes written. A stalled release job that
 * never fails is the worst outcome here, so this uses the tool that already
 * retries, resumes and times out on a dead connection. curl is present on
 * every runner this job runs on.
 */
async function fetchTo(url: string, dest: string, file: string) {
  const expected = await expectedSize(url);
  // A truncated model is the failure that produced the garbled-audio incident.
  // The archive's own SHA covers the user's download; this covers ours — for
  // a local copy exactly as for a fresh transfer.
  if (!expected) {
    throw new Error(`${url}: no content-length, cannot verify the download`);
  }

  const local = localDir ? join(localDir, file) : "";
  if (local && existsSync(local)) {
    console.log(`  (reusing ${local})`);
    // Clone-on-write where the filesystem supports it (APFS): reusing a
    // multi-GB local GGUF should not cost its size in disk a second time.
    await copyFile(local, dest, constants.COPYFILE_FICLONE);
  } else {
    const proc = Bun.spawnSync(
      [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "3",
        "--retry-delay",
        "5",
        // Abort rather than hang forever if the transfer dies mid-stream.
        "--speed-limit",
        "1024",
        "--speed-time",
        "60",
        "--output",
        dest,
        url,
      ],
      { stdout: "inherit", stderr: "inherit" },
    );
    if (proc.exitCode !== 0) {
      throw new Error(`curl failed for ${url} (exit ${proc.exitCode})`);
    }
  }

  const got = (await Bun.file(dest).stat()).size;
  if (got !== expected) {
    throw new Error(`${dest}: expected ${expected} bytes, got ${got}`);
  }
  return got;
}

let total = 0;
for (const file of [spec.talker, CODEC]) {
  console.log(`→ ${file}`);
  total += await fetchTo(`${GGUF_REPO}/${file}`, join(stage, file), file);
}

// Fail here rather than on a user's machine: a pack that extracts but whose
// GGUFs are not GGUFs fails at load time, far from the cause.
for (const file of [spec.talker, CODEC]) {
  const magic = await Bun.file(join(stage, file)).slice(0, 4).text();
  if (magic !== "GGUF") {
    throw new Error(`${file} is not a GGUF (magic ${JSON.stringify(magic)})`);
  }
}

const modeNotes =
  mode === "base"
    ? [
        "Base checkpoint, which is the one that clones from a reference clip.",
        "`customvoice` (named speakers) and `voicedesign` (`--instruct`) are",
        "different checkpoints and are not in this pack.",
        "",
        `Speaker embeddings from this pack are ${spec.spkDim}-dimensional. A voice`,
        "cloned with it will not load into the other size, and `bisque-voice` refuses",
        "that rather than reading a buffer of the wrong width.",
      ]
    : [
        "CustomVoice checkpoint — nine named speakers baked into the talker",
        "(serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan;",
        "eric speaks Sichuan-dialect and dylan Beijing-dialect Chinese). It does",
        "not clone: `base` (reference-clip cloning) and `voicedesign`",
        "(`--instruct`) are different checkpoints and are not in this pack.",
      ];

await writeFile(
  join(stage, "PACK.md"),
  [
    `# ${spec.packVersion}`,
    "",
    `Model pack for \`bisque-voice\`, engine \`${spec.engine}\`.`,
    "",
    `- \`${spec.talker}\` — talker LM and code predictor.`,
    `- \`${CODEC}\` — the shared 12.5 Hz codec.`,
    "",
    "Weights: Qwen3-TTS (Apache-2.0), quantised to Q8_0 and published as",
    "Serveurperso/Qwen3-TTS-GGUF. Run by the vendored qwentts.cpp (MIT).",
    "",
    ...modeNotes,
    "",
    "End-to-end synthesis, so there is no phonemizer and no eSpeak NG licensing",
    "question here.",
    "",
  ].join("\n"),
);

console.log("→ archiving");
const archive = join(outDir, `${spec.packVersion}.tar.gz`);
const tar = Bun.spawnSync(["tar", "-czf", archive, "-C", stage, "."], {
  stdout: "inherit",
  stderr: "inherit",
});
if (tar.exitCode !== 0) throw new Error("tar failed");

const hash = new Bun.CryptoHasher("sha256");
hash.update(new Uint8Array(await Bun.file(archive).arrayBuffer()));
const sha = hash.digest("hex");
const bytes = (await Bun.file(archive).stat()).size;

console.log("");
console.log(`archive  : ${archive}`);
console.log(`raw      : ${(total / 1e6).toFixed(1)} MB`);
console.log(`size     : ${(bytes / 1e6).toFixed(1)} MB`);
console.log(`sha256   : ${sha}`);
console.log("");
console.log("Build the binary against it with:");
console.log(`  ${spec.shaVar}=${sha} cargo build --release`);
