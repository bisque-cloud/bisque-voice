#!/usr/bin/env bun
/**
 * verify-regression.ts — the reported Kokoro mispronunciations, narrated for
 * real.
 *
 * WHY IT EXISTS
 *   The unit tests in `src/verify.rs` feed the check phoneme strings copied
 *   from the engine. This drives the built binary with the installed Kokoro
 *   model over each sentence as it shipped and as fixed, so a change to the
 *   G2P, the lexicon or the rules that breaks either side shows up here:
 *
 *     - each original is narrated and warns with its rule;
 *     - each fix is narrated with no warning;
 *     - a control sentence holding initialisms nobody reported warns nothing.
 *
 *   `--asr` transcribes every clip with whisper base.en (via `uv`, about
 *   140 MB on first run) so the originals can be compared with their fixes.
 *   It is evidence, not a gate: a recognizer hears through context.
 *
 * USAGE
 *   bun tools/verify-regression.ts [--bin P] [--keep] [--asr]
 *
 * Needs Kokoro installed. Exit code is the number of failed cases.
 */

import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";

const argv = process.argv.slice(2);
const val = (f: string) => {
  const i = argv.indexOf(f);
  return i !== -1 ? argv[i + 1] : undefined;
};

const bin =
  val("--bin") ??
  process.env.BISQUE_VOICE_BIN ??
  path.resolve(import.meta.dir, "../target/release/bisque-voice");
if (!existsSync(bin)) {
  console.error(`no bisque-voice at ${bin} — build it or pass --bin`);
  process.exit(1);
}

/** Each reported mistake as it shipped, and as fixed. `rule` is the warning
 *  the original must raise; `null` is a control that must raise none. */
const CASES = [
  {
    name: "plugin",
    rule: "plugin-soft-g",
    // mattpocock/skills 1.3, heard as "plooogin".
    original: "Three skills join the Claude Code plugin, and one leaves it.",
    fixed:
      "Three skills join the Claude Code [plugin](/plˈʌɡɪn/), and one leaves it.",
  },
  {
    name: "tdd",
    rule: "tdd-run-together",
    // mattpocock/skills 1.3, the marker the episode carried.
    original:
      "It builds its ticket with the [tdd](/tˌidˌiˈdi/) skill one failing test at a time.",
    fixed:
      "It builds its ticket with the [tdd](/tˈi dˈi dˈi/) skill one failing test at a time.",
  },
  {
    name: "gstack",
    rule: "stress-before-onset",
    // #116, heard as "Uggy Stack" six times in a rendered video.
    original: "Install [gstack](/ˈɡistˌæk/) before the first run.",
    fixed: "Install [gstack](/ɡˈistˌæk/) before the first run.",
  },
  {
    name: "control",
    rule: null,
    original: "The API serves CSS and HTML over HTTP.",
    fixed: null,
  },
] as const;

type Warning = { rule: string; fix: string };

function narrate(text: string, out: string) {
  const res = spawnSync(
    bin,
    ["--text", "-", "--voice", "kokoro:af_heart", "--out", out],
    { input: text, encoding: "utf8" },
  );
  let warnings: Warning[] | undefined;
  try {
    warnings = JSON.parse(res.stdout).pronunciationWarnings;
  } catch {}
  return { status: res.status, stderr: res.stderr, warnings };
}

const dir = mkdtempSync(path.join(os.tmpdir(), "verify-regression-"));
let failed = 0;
const clips: string[] = [];
const check = (ok: boolean, line: string, detail = "") => {
  if (!ok) failed += 1;
  console.log(`${ok ? "ok  " : "FAIL"} ${line}${ok ? "" : `\n${detail}`}`);
};

for (const c of CASES) {
  const original = path.join(dir, `${c.name}-original.wav`);
  const o = narrate(c.original, original);
  const rules = (o.warnings ?? []).map((w) => w.rule);
  if (c.rule) {
    check(
      o.status === 0 && existsSync(original) && rules.includes(c.rule),
      `${c.name}: original narrated and warned ${c.rule} — fix ${o.warnings?.find((w) => w.rule === c.rule)?.fix}`,
      `exit ${o.status}, warnings ${JSON.stringify(o.warnings)}\n${o.stderr}`,
    );
  } else {
    check(
      o.status === 0 && Array.isArray(o.warnings) && rules.length === 0,
      `${c.name}: narrated with no warning`,
      `exit ${o.status}, warnings ${JSON.stringify(o.warnings)}\n${o.stderr}`,
    );
  }
  clips.push(original);
  if (!c.fixed) continue;

  const fixed = path.join(dir, `${c.name}-fixed.wav`);
  const f = narrate(c.fixed, fixed);
  check(
    f.status === 0 && Array.isArray(f.warnings) && f.warnings.length === 0,
    `${c.name}: fix narrated with no warning`,
    `exit ${f.status}, warnings ${JSON.stringify(f.warnings)}\n${f.stderr}`,
  );
  clips.push(fixed);
}

if (argv.includes("--asr")) {
  const script = [
    "import sys, wave, numpy as np",
    "from faster_whisper import WhisperModel",
    "m = WhisperModel('base.en', device='cpu', compute_type='int8')",
    "for p in sys.argv[1:]:",
    "    w = wave.open(p); a = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768",
    "    a = np.interp(np.arange(0, len(a), w.getframerate() / 16000), np.arange(len(a)), a).astype(np.float32)",
    "    segs, _ = m.transcribe(a, beam_size=5, condition_on_previous_text=False)",
    "    print(p.rsplit('/', 1)[-1], '|', ' '.join(s.text.strip() for s in segs))",
  ].join("\n");
  const asr = spawnSync(
    "uv",
    [
      "run",
      "--quiet",
      "--with",
      "faster-whisper",
      "--with",
      "numpy",
      "python",
      "-c",
      script,
      ...clips.filter(existsSync),
    ],
    { encoding: "utf8" },
  );
  console.log(asr.status === 0 ? asr.stdout : `asr failed:\n${asr.stderr}`);
}

if (argv.includes("--keep")) console.log(`audio kept in ${dir}`);
else rmSync(dir, { recursive: true, force: true });
process.exit(failed);
