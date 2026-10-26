#!/usr/bin/env bun
/**
 * Build the `bisque-voice` model pack — the single hash-verified archive the
 * binary downloads on first run.
 *
 * Everything the engine needs travels together, so there is no partial state to
 * reason about: either the pack verified and extracted, or it didn't.
 *
 *   kokoro.onnx            fp32 export with the duration tensor promoted
 *   tokenizer.json         phoneme → id vocabulary
 *   voices/<id>.bin        Kokoro voice packs
 *   us_gold.json           Misaki lexicons (Apache-2.0), from packages/kokoro-swift
 *   us_silver.json
 *   us_bart.safetensors    neural fallback for out-of-vocabulary words
 *   us_bart_config.json
 *
 * Not included: eSpeak NG. It is compiled into the binary, and
 * its language data travels in the separate g2p-espeak pack
 * (`tools/build-g2p-packs.ts`).
 *
 *   bun run tools/build-model-pack.ts [--out dist] [--variant fp32]
 *
 * Prints the archive path and its SHA-256. Feed that hash to the build as
 * BISQUE_VOICE_PACK_SHA256_KOKORO — the variable is per engine, uppercased, and
 * an unsuffixed BISQUE_VOICE_PACK_SHA256 is read by nothing — and publish the
 * archive as download.bisque.today/bisque-voice/models/<PACK_VERSION>.tar.gz
 */

import { mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";

const HF =
  "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main";
const MISAKI = "packages/kokoro-swift/Sources/MisakiSwift/Resources";

/**
 * One pack version per variant — a pack version is immutable, so a variant
 * change is a new name, never an in-place edit.
 *
 * The **fp32** entry must match the Kokoro registry entry's `pack_version` in
 * src/engine.rs: fp32 is what ships. The fp16 export is the one that
 * deterministically NaN'd its vocoder for some (voice, text) pairs and
 * shipped as digital silence; it and q8 remain buildable
 * for experiments only.
 */
const VERSION_BY_VARIANT: Record<string, string> = {
  fp32: "kokoro-v1.0-fp32-1",
  fp16: "kokoro-v1.0-fp16-2",
  q8: "kokoro-v1.0-q8-1",
};

/**
 * Every voice Kokoro publishes, across all nine of its languages.
 *
 * This used to be fourteen English voices, chosen to match what the Presenter
 * app offered. The model file was always multilingual — the pack just wasn't.
 * Voice ids are `<language><gender>_<name>`: a/b English (US/UK), e Spanish,
 * f French, h Hindi, i Italian, j Japanese, p Portuguese, z Chinese.
 */
const VOICES = [
  // a — American English
  "af",
  "af_alloy",
  "af_aoede",
  "af_bella",
  "af_heart",
  "af_jessica",
  "af_kore",
  "af_nicole",
  "af_nova",
  "af_river",
  "af_sarah",
  "af_sky",
  "am_adam",
  "am_echo",
  "am_eric",
  "am_fenrir",
  "am_liam",
  "am_michael",
  "am_onyx",
  "am_puck",
  "am_santa",
  // b — British English
  "bf_alice",
  "bf_emma",
  "bf_isabella",
  "bf_lily",
  "bm_daniel",
  "bm_fable",
  "bm_george",
  "bm_lewis",
  // e — Spanish
  "ef_dora",
  "em_alex",
  "em_santa",
  // f — French
  "ff_siwis",
  // h — Hindi
  "hf_alpha",
  "hf_beta",
  "hm_omega",
  "hm_psi",
  // i — Italian
  "if_sara",
  "im_nicola",
  // j — Japanese
  "jf_alpha",
  "jf_gongitsune",
  "jf_nezumi",
  "jf_tebukuro",
  "jm_kumo",
  // p — Portuguese
  "pf_dora",
  "pm_alex",
  "pm_santa",
  // z — Chinese
  "zf_xiaobei",
  "zf_xiaoni",
  "zf_xiaoxiao",
  "zf_xiaoyi",
  "zm_yunjian",
  "zm_yunxi",
  "zm_yunxia",
  "zm_yunyang",
];

const ONNX_BY_VARIANT: Record<string, { file: string; elemType: number }> = {
  fp16: { file: "onnx/model_fp16.onnx", elemType: 10 },
  fp32: { file: "onnx/model.onnx", elemType: 1 },
  q8: { file: "onnx/model_quantized.onnx", elemType: 1 },
};

const args = process.argv.slice(2);
const flag = (name: string, fallback: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const outDir = flag("out", "dist");
const variant = flag("variant", "fp32");
const repoRoot = flag(
  "repo-root",
  process.cwd().replace(/\/apps\/bisque-voice$/, ""),
);

const onnx = ONNX_BY_VARIANT[variant];
if (!onnx)
  throw new Error(
    `unknown variant ${variant}; expected ${Object.keys(ONNX_BY_VARIANT).join("|")}`,
  );
const PACK_VERSION = VERSION_BY_VARIANT[variant];
if (!PACK_VERSION) throw new Error(`no pack version for variant ${variant}`);

const stage = join(outDir, PACK_VERSION);
await rm(stage, { recursive: true, force: true });
await mkdir(join(stage, "voices"), { recursive: true });

async function fetchTo(url: string, dest: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} → ${res.status}`);
  await Bun.write(dest, await res.arrayBuffer());
}

console.log(`→ ${onnx.file}`);
const rawModel = join(outDir, `raw-${variant}.onnx`);
await fetchTo(`${HF}/${onnx.file}`, rawModel);

// Promote the duration tensor. Without it the export emits only `waveform`,
// and there are no word timings — the whole cue system depends on this step.
console.log("→ promoting the duration tensor");
const promote = Bun.spawnSync(
  [
    "bun",
    "run",
    join(import.meta.dir, "promote-duration.ts"),
    rawModel,
    join(stage, "kokoro.onnx"),
  ],
  {
    env: { ...process.env, ELEM_TYPE: String(onnx.elemType) },
    stdout: "inherit",
    stderr: "inherit",
  },
);
if (promote.exitCode !== 0) throw new Error("promote-duration failed");
await rm(rawModel, { force: true });

console.log("→ tokenizer");
await fetchTo(`${HF}/tokenizer.json`, join(stage, "tokenizer.json"));

console.log(`→ ${VOICES.length} voices`);
for (const v of VOICES) {
  await fetchTo(`${HF}/voices/${v}.bin`, join(stage, "voices", `${v}.bin`));
}

// The lexicons and the fallback network come from the vendored MisakiSwift
// tree, so macOS and every other platform read the exact same data.
console.log("→ Misaki lexicons + BART fallback (from packages/kokoro-swift)");
for (const f of [
  "us_gold.json",
  "us_silver.json",
  "us_bart.safetensors",
  "us_bart_config.json",
  // British English has its own lexicon and its own fallback network. They
  // were always in the repo and never in the pack, so bf_/bm_ voices read
  // American pronunciations in a British accent.
  "gb_gold.json",
  "gb_silver.json",
  "gb_bart.safetensors",
  "gb_bart_config.json",
]) {
  const src = Bun.file(join(repoRoot, MISAKI, f));
  if (!(await src.exists()))
    throw new Error(`missing ${join(repoRoot, MISAKI, f)}`);
  await Bun.write(join(stage, f), src);
}

await writeFile(
  join(stage, "PACK.md"),
  [
    `# ${PACK_VERSION}`,
    "",
    "Model pack for `bisque-voice`.",
    "",
    `- \`kokoro.onnx\` — onnx-community/Kokoro-82M-v1.0-ONNX (${variant}), Apache-2.0,`,
    "  with the duration predictor's output promoted to a graph output so the",
    "  engine can emit native word timings.",
    "- `us_gold.json` / `us_silver.json` / `us_bart.*` — MisakiSwift, Apache-2.0.",
    "- `voices/*.bin` — Kokoro voice packs, Apache-2.0.",
    "",
    "No eSpeak NG data is included; it ships in the separate g2p-espeak pack.",
    "",
  ].join("\n"),
);

console.log("→ archiving");
const archive = join(outDir, `${PACK_VERSION}.tar.gz`);
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
console.log(`archive : ${archive}`);
console.log(`size    : ${(bytes / 1e6).toFixed(1)} MB`);
console.log(`sha256  : ${sha}`);
console.log("");
console.log("Build the binary against it with:");
console.log(`  BISQUE_VOICE_PACK_SHA256_KOKORO=${sha} cargo build --release`);
