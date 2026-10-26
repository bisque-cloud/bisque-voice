#!/usr/bin/env bun
/**
 * run-test-suite.ts — cross-environment launch-verification suite for
 * bisque-voice. Runs on any supported machine (macOS ARM/Intel, Linux
 * x64/ARM, Windows x64/ARM) and proves, for every engine + voice + language
 * that is INSTALLED, that synthesis produces AUDIBLE audio with SANE word
 * timings — not merely exit 0. It reuses the same gate as assert-audio.ts
 * (`checkSynth`), so a leg here and a leg in CI assert identically.
 *
 * WHY A DRIVER AND NOT JUST assert-audio.ts
 *   The failure modes that bite are per-engine and
 *   per-platform: Kokoro returned NaN-silent audio while reporting success;
 *   CoreML fails Kokoro's graph on macOS; the Studio engines and every
 *   non-English language were never asserted non-silent in automation at all.
 *   This driver enumerates what the machine actually has and refuses to pass
 *   unless each one makes real sound.
 *
 * USAGE
 *   bun tools/run-test-suite.ts            # test what's installed
 *   bun tools/run-test-suite.ts --cargo    # also run `cargo test`
 *   BISQUE_VOICE_BIN=/path/to/bisque-voice bun …             # override the binary
 *   --keep      leave the synthesized wavs on disk
 *   --bin P     path to the bisque-voice binary
 *
 * Exit code is the number of failed legs (0 = everything installed is audible).
 */

import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { checkSynth, type SynthDoc } from "./assert-audio.ts";

// ── args ────────────────────────────────────────────────────────────────
const argv = process.argv.slice(2);
const has = (f: string) => argv.includes(f);
const val = (f: string) => {
  const i = argv.indexOf(f);
  return i !== -1 ? argv[i + 1] : undefined;
};

const HOME = os.homedir();
const TOOLS_DIR = import.meta.dir;
const CARGO_TOML = path.resolve(TOOLS_DIR, "../Cargo.toml");
const VOICES_DIR = path.join(HOME, ".bisque", "models", "voices");

function resolveBin(): string {
  const candidates = [
    val("--bin"),
    process.env.BISQUE_VOICE_BIN,
    path.join(HOME, ".bisque", "bin", "bisque-voice"),
    path.join(HOME, ".bisque", "bin", "bisque-voice.exe"),
    path.resolve(TOOLS_DIR, "../target/release/bisque-voice"),
    path.resolve(TOOLS_DIR, "../target/release/bisque-voice.exe"),
  ].filter(Boolean) as string[];
  for (const c of candidates) if (existsSync(c)) return c;
  console.error(
    "could not find a bisque-voice binary. Install it or pass --bin.",
  );
  process.exit(2);
}

const BIN = resolveBin();

interface Run {
  status: number | null;
  stdout: string;
  stderr: string;
}
function run(args: string[]): Run {
  const r = spawnSync(BIN, args, {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    timeout: 240_000, // a wedged engine must not hang the suite
  });
  return { status: r.status, stdout: r.stdout ?? "", stderr: r.stderr ?? "" };
}

// ── environment banner ────────────────────────────────────────────────────
const platform = `${process.platform}/${process.arch}`;
const version = run(["--version"]).stdout.trim();
let headVersion = "";
try {
  headVersion =
    (readFileSync(CARGO_TOML, "utf8").match(/^version\s*=\s*"([^"]+)"/m) ??
      [])[1] ?? "";
} catch {
  /* Cargo.toml not beside the binary — fine when testing an installed copy. */
}
console.log(`bisque-voice test suite`);
console.log(`  platform : ${platform} (${os.release()})`);
console.log(`  binary   : ${BIN}`);
console.log(
  `  version  : ${version}${headVersion && !version.includes(headVersion) ? `  ⚠ repo HEAD is ${headVersion} — reinstall to test what ships` : ""}`,
);

// ── discover what is installed ────────────────────────────────────────────
interface EngineInfo {
  id: string;
  installed?: boolean;
  unsupported?: unknown;
}
function installedEngines(): EngineInfo[] {
  const r = run(["engines", "--json"]);
  if (r.status !== 0) {
    console.error("could not list engines:\n" + r.stderr);
    return [];
  }
  try {
    const parsed = JSON.parse(r.stdout);
    const list: EngineInfo[] = Array.isArray(parsed)
      ? parsed
      : (parsed.engines ?? []);
    return list.filter((e) => e.installed && !e.unsupported);
  } catch (err) {
    console.error("could not parse engines --json: " + (err as Error).message);
    return [];
  }
}
function clonedVoices(): string[] {
  try {
    return readdirSync(VOICES_DIR).filter((n) => !n.startsWith("."));
  } catch {
    return [];
  }
}

const engines = installedEngines();
const clones = clonedVoices();
console.log(
  `  engines  : ${engines.map((e) => e.id).join(", ") || "(none installed)"}`,
);
if (clones.length) console.log(`  clones   : ${clones.join(", ")}`);
console.log("");

// ── build the leg matrix from what's actually installed ────────────────────
interface Leg {
  label: string;
  voice: string; // engine:voice
  text: string;
  language?: string;
  checkTokens: boolean; // whitespace-token count == word count (skip for CJK)
}
const EN = "Bisque narrates on your own machine.";
const RU = "Привет, это тест синтеза речи.";
const ZH = "你好，这是语音合成测试。";

const legs: Leg[] = [];
for (const e of engines) {
  const id = e.id;
  if (id === "kokoro") {
    legs.push({
      label: `kokoro EN`,
      voice: "kokoro:af_heart",
      text: EN,
      checkTokens: true,
    });
    // ZH exercises a non-English g2p path; skip the token check (no spaces).
    legs.push({
      label: `kokoro ZH`,
      voice: "kokoro:zf_xiaobei",
      text: ZH,
      language: "chinese",
      checkTokens: false,
    });
  } else if (id.startsWith("qwen3-clone")) {
    const voice = clones[0];
    if (!voice) {
      console.log(
        `skip ${id}: no cloned voice on disk (run \`bisque-voice clone …\`)`,
      );
      continue;
    }
    legs.push({
      label: `${id} EN`,
      voice: `${id}:${voice}`,
      text: EN,
      language: "english",
      checkTokens: true,
    });
    legs.push({
      label: `${id} RU`,
      voice: `${id}:${voice}`,
      text: RU,
      language: "russian",
      checkTokens: true,
    });
  } else if (id.startsWith("qwen3-voices")) {
    legs.push({
      label: `${id} EN`,
      voice: `${id}:serena`,
      text: EN,
      language: "english",
      checkTokens: true,
    });
    legs.push({
      label: `${id} RU`,
      voice: `${id}:serena`,
      text: RU,
      language: "russian",
      checkTokens: true,
    });
  }
}

if (legs.length === 0) {
  console.error(
    "No installed engines to test. Install one, e.g. `bisque-voice install kokoro`.",
  );
  process.exit(2);
}

// ── run the legs ──────────────────────────────────────────────────────────
const workDir = mkdtempSync(path.join(os.tmpdir(), "bv-suite-"));
interface Row {
  label: string;
  ok: boolean;
  skip?: boolean; // an uninstalled optional engine/pack is uncovered, not failed
  detail: string;
}
const rows: Row[] = [];

for (const leg of legs) {
  const wav = path.join(
    workDir,
    leg.label.replace(/[^a-z0-9]+/gi, "_") + ".wav",
  );
  const args = ["--text", leg.text, "--voice", leg.voice, "--out", wav];
  if (leg.language) args.push("--language", leg.language);
  const r = run(args);
  if (r.status !== 0) {
    // "not installed" (a missing optional engine/voice/g2p pack) is uncovered,
    // not a failure. A non-zero exit for any other reason IS a failure.
    const missing =
      /install it with|not installed|no such (engine|voice|aligner|pack)/i.test(
        r.stderr,
      );
    const lastLine = r.stderr.trim().split("\n").pop() ?? "";
    rows.push({
      label: leg.label,
      ok: !missing,
      skip: missing,
      detail:
        (missing ? "not installed — " : `synth exit ${r.status}: `) + lastLine,
    });
    continue;
  }
  let doc: SynthDoc;
  try {
    doc = JSON.parse(r.stdout);
  } catch {
    rows.push({
      label: leg.label,
      ok: false,
      detail: "stdout was not one JSON document",
    });
    continue;
  }
  const res = checkSynth(doc, readFileSync(wav), {
    text: leg.checkTokens ? leg.text : undefined,
  });
  rows.push({
    label: leg.label,
    ok: res.pass,
    detail: res.pass ? res.summary : res.failures.join("; "),
  });
}

// ── CoreML / accelerator guard (macOS + Kokoro only) ───────────────────────
// The known macOS failure: CoreML accepts Kokoro's graph and then fails it at
// run time. `--device gpu` MUST fail loudly; `--device auto` MUST fall back to
// CPU and succeed. Anything else is a silent-audio hazard.
if (process.platform === "darwin" && engines.some((e) => e.id === "kokoro")) {
  const gpu = run([
    "--text",
    "hi",
    "--voice",
    "kokoro:af_heart",
    "--device",
    "gpu",
    "--out",
    path.join(workDir, "gpu.wav"),
  ]);
  const gpuFailedAsItShould = gpu.status !== 0;
  rows.push({
    label: "kokoro --device gpu guard",
    ok: gpuFailedAsItShould,
    detail: gpuFailedAsItShould
      ? `correctly refused the accelerator (exit ${gpu.status})`
      : "DANGER: --device gpu returned 0 on macOS — CoreML silent-audio hazard",
  });
}

if (!has("--keep")) rmSync(workDir, { recursive: true, force: true });

// ── optional: the Rust unit suite ─────────────────────────────────────────
let cargoFailed = false;
if (has("--cargo")) {
  console.log("\nRunning `cargo test` …");
  const env = { ...process.env };
  if (process.platform === "darwin" && !env.LIBCLANG_PATH) {
    // bindgen needs libclang for the qwen C++ core; the Xcode CLT copy works.
    for (const p of [
      "/Library/Developer/CommandLineTools/usr/lib",
      "/opt/homebrew/opt/llvm/lib",
    ]) {
      if (existsSync(path.join(p, "libclang.dylib"))) {
        env.LIBCLANG_PATH = p;
        break;
      }
    }
  }
  const cargo = spawnSync("cargo", ["test"], {
    cwd: path.resolve(TOOLS_DIR, ".."),
    stdio: "inherit",
    env,
  });
  cargoFailed = cargo.status !== 0;
}

// ── report ─────────────────────────────────────────────────────────────────
console.log("\n─── results ───────────────────────────────────────────────");
for (const row of rows) {
  const tag = row.skip ? "SKIP" : row.ok ? "PASS" : "FAIL";
  console.log(`  ${tag}  ${row.label.padEnd(28)} ${row.detail}`);
}
const failed = rows.filter((r) => !r.ok && !r.skip).length;
const skipped = rows.filter((r) => r.skip).length;
const passed = rows.length - failed - skipped;
console.log("───────────────────────────────────────────────────────────");
console.log(
  `${passed}/${rows.length - skipped} legs passed on ${platform}` +
    (skipped ? ` (${skipped} skipped — not installed)` : "") +
    (has("--cargo") ? `; cargo test ${cargoFailed ? "FAILED" : "passed"}` : ""),
);
if (failed || cargoFailed) console.log("Some checks failed.");

process.exit(failed + (cargoFailed ? 1 : 0));
