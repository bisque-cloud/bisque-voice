#!/usr/bin/env bun
/**
 * Build a Qwen3 forced-aligner pack for `bisque-voice`.
 *
 * Same contract as the other pack builders: one hash-verified archive holding
 * everything `src/align.rs` needs, so there is no partial state to reason
 * about — either the pack verified and extracted, or it didn't.
 *
 *   model.onnx                the single alignment graph
 *   model.onnx_data           fp32 only — its external weights
 *   config.json               timestamp_token_id, timestamp_segment_time,
 *                             audio_config.n_window
 *   tokenizer_config.json     added_tokens_decoder — the four special ids
 *   vocab.json / merges.txt   the Qwen byte-level BPE
 *   preprocessor_config.json  the Whisper feature extractor's parameters
 *
 *   bun run tools/build-aligner-pack.ts --variant fp32|q4 [--out dist]
 *
 * Two variants, two packs, two registry entries. They are exports of the same
 * 0.6B checkpoint at different numeric precision; the registry states what is
 * known about each and does not rank them.
 *
 * Note that these exports are third-party — neither `Qwen` nor
 * `onnx-community` publishes one. Rebuilding the pack here and pinning its
 * SHA-256 is what makes what ships verified; the upstream conversion is still
 * someone else's.
 *
 * Prints the archive path and its SHA-256. Feed that hash to the build as
 * BISQUE_VOICE_PACK_SHA256_QWEN3_FP32 / _QWEN3_Q4 — the variable is per pack,
 * uppercased with `-` replaced by `_` — and publish the archive as
 * download.bisque.today/bisque-voice/models/<PACK_VERSION>.tar.gz
 */

import { mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";

/** The ONNX export. Apache-2.0, same `ort` runtime as every other graph here. */
const ONNX_REPO =
  "https://huggingface.co/valoomba/Qwen3-ForcedAligner-0.6B-ONNX/resolve/main";

/**
 * Per variant: the pack name (must match the `pack_version` of the matching
 * entry in `align::REGISTRY`), the graph files to fetch, and the reported
 * parity, quoted rather than characterized.
 */
const VARIANTS = {
  fp32: {
    packVersion: "qwen3-aligner-fp32-1",
    shaVar: "BISQUE_VOICE_PACK_SHA256_QWEN3_FP32",
    // The fp32 graph carries its weights in a sidecar. ONNX resolves that by
    // the relative name recorded *inside* the graph, so the data file must
    // travel under exactly this name or loading fails.
    graphs: [
      { from: "onnx/model.onnx", to: "model.onnx" },
      { from: "onnx/model.onnx_data", to: "model.onnx_data" },
    ],
    parity: "full validation, max_abs_diff 9.06e-05, mean_abs_diff 9.77e-06",
  },
  q4: {
    packVersion: "qwen3-aligner-q4-1",
    shaVar: "BISQUE_VOICE_PACK_SHA256_QWEN3_Q4",
    // Self-contained, so it is renamed to the one filename the reader opens.
    graphs: [{ from: "onnx/model_q4.onnx", to: "model.onnx" }],
    parity: "smoke test, max_abs_diff 7.80, mean_abs_diff 0.79",
  },
} as const;

/** Config and tokenizer, from the same export. */
const SUPPORT_FILES = [
  "config.json",
  "tokenizer_config.json",
  "vocab.json",
  "merges.txt",
  "preprocessor_config.json",
];

const args = process.argv.slice(2);
const flag = (name: string, fallback: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const outDir = flag("out", "dist");
const variant = flag("variant", "");

if (!(variant in VARIANTS)) {
  throw new Error(
    `--variant is required; expected ${Object.keys(VARIANTS).join("|")}`,
  );
}
const spec = VARIANTS[variant as keyof typeof VARIANTS];

const stage = join(outDir, spec.packVersion);
await rm(stage, { recursive: true, force: true });
await mkdir(stage, { recursive: true });

async function fetchTo(url: string, dest: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} → ${res.status}`);
  await Bun.write(dest, await res.arrayBuffer());
}

for (const g of spec.graphs) {
  console.log(`→ ${g.from}`);
  await fetchTo(`${ONNX_REPO}/${g.from}`, join(stage, g.to));
}
for (const f of SUPPORT_FILES) {
  console.log(`→ ${f}`);
  await fetchTo(`${ONNX_REPO}/${f}`, join(stage, f));
}

// Everything below fails loudly here rather than at alignment time on a user's
// machine: a pack that extracts but cannot be read is the worst kind of broken,
// because the failure surfaces far from the cause. Each check mirrors one the
// Rust side makes, so a pack that builds is a pack that loads.
const cfg = JSON.parse(await Bun.file(join(stage, "config.json")).text());
for (const key of ["timestamp_token_id", "timestamp_segment_time"]) {
  if (cfg[key] === undefined) throw new Error(`config.json is missing ${key}`);
}
// The upstream export nests audio_config under thinker_config; align.rs
// accepts both locations, so this check must too.
const audio = cfg.audio_config ?? cfg.thinker_config?.audio_config ?? {};
if (audio.n_window === undefined) {
  // Without it there is no way to derive how many <|audio_pad|> tokens the
  // prompt needs, and src/align.rs refuses to guess.
  throw new Error("config.json is missing audio_config.n_window");
}
if (audio.num_mel_bins !== 128) {
  throw new Error(
    `config.json wants ${audio.num_mel_bins} mel bins; the reader extracts 128`,
  );
}

const tok = JSON.parse(
  await Bun.file(join(stage, "tokenizer_config.json")).text(),
);
const decoder = tok.added_tokens_decoder ?? {};
const byContent = new Map<string, number>();
for (const [id, entry] of Object.entries<{ content?: string }>(decoder)) {
  if (entry?.content) byContent.set(entry.content, Number(id));
}
for (const token of [
  "<|audio_start|>",
  "<|audio_pad|>",
  "<|audio_end|>",
  "<timestamp>",
]) {
  if (!byContent.has(token)) {
    throw new Error(`tokenizer_config.json declares no id for ${token}`);
  }
}
if (byContent.get("<timestamp>") !== cfg.timestamp_token_id) {
  throw new Error(
    `the export disagrees with itself about <timestamp>: config.json says ` +
      `${cfg.timestamp_token_id}, tokenizer_config.json says ` +
      `${byContent.get("<timestamp>")}`,
  );
}

const pre = JSON.parse(
  await Bun.file(join(stage, "preprocessor_config.json")).text(),
);
for (const [key, want] of [
  ["feature_size", 128],
  ["n_fft", 400],
  ["hop_length", 160],
  ["sampling_rate", 16000],
  ["dither", 0],
] as const) {
  if (pre[key] !== want) {
    throw new Error(
      `preprocessor_config.json declares ${key} = ${pre[key]}, but the ` +
        `reader's Whisper front end is fixed at ${want}`,
    );
  }
}
console.log(
  `   timestamp id ${cfg.timestamp_token_id} · ${cfg.timestamp_segment_time} ms per unit · ` +
    `${audio.n_window * 2}-frame encoder chunks`,
);

await writeFile(
  join(stage, "PACK.md"),
  [
    `# ${spec.packVersion}`,
    "",
    "Forced-aligner pack for `bisque-voice`. Measures word start/end times against",
    "a synthesized waveform, so timings are measured rather than inferred.",
    "",
    `- ONNX graph — valoomba/Qwen3-ForcedAligner-0.6B-ONNX (${variant}).`,
    `  Reported against PyTorch: ${spec.parity}.`,
    "- `config.json` / `tokenizer_config.json` / `vocab.json` / `merges.txt` /",
    "  `preprocessor_config.json` — from the same export.",
    "",
    "Upstream checkpoint Qwen/Qwen3-ForcedAligner-0.6B, Apache-2.0. The ONNX",
    "conversion is third-party; this pack is rebuilt here and its SHA-256 pinned",
    "into the binary, so what ships is verified even though the conversion is not",
    "ours.",
    "",
    variant === "q4"
      ? [
          "The reported deviation is over raw logits. This path consumes only the",
          "argmax of those logits, and how often that argmax differs from the fp32",
          "graph's has not been measured.",
          "",
        ].join("\n")
      : "",
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
console.log(`archive : ${archive}`);
console.log(`size    : ${(bytes / 1e6).toFixed(1)} MB`);
console.log(`sha256  : ${sha}`);
console.log("");
console.log("Build the binary against it with:");
console.log(`  ${spec.shaVar}=${sha} cargo build --release`);
