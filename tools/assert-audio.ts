#!/usr/bin/env bun
/**
 * assert-audio.ts — the single per-leg gate for a bisque-voice synthesis.
 *
 * WHY IT EXISTS
 *   Exit code 0 is not enough. bisque-voice's own −60 dBFS silence floor
 *   (`src/main.rs`, `src/audio.rs`) catches *digital* silence — all-zero or
 *   NaN samples — but it passes audio that is quiet-but-wrong, and it asserts
 *   nothing about the word timings. Every cross-environment test leg (macOS
 *   ARM/Intel, Linux x64/ARM, Windows x64/ARM) must prove the SAME three
 *   things, so they live here in one place instead of drifting between shells:
 *
 *     1. The synthesis returned exactly one parseable JSON document with a
 *        positive integer duration and a non-empty word list.
 *     2. The audio is actually audible — RMS and peak well above silence, and
 *        not clipping. (Encoded audio is peak-normalized to ~0.89 full scale,
 *        so real speech sits far above the floor; the known GPU/NaN failure
 *        mode sits at ~0.) This is stricter than the binary's own floor.
 *     3. The word timings are sane — start ≥ 0, monotonic and non-overlapping,
 *        the last word ends within the reported duration, and (when the source
 *        text is given) one timing per whitespace token, since cue markers
 *        index that array by position.
 *
 * USAGE
 *   bisque-voice --text "Hello there." --voice kokoro:af_heart --out o.wav > o.json
 *   bun tools/assert-audio.ts --json o.json --wav o.wav --text "Hello there."
 *
 *   --json <path|->   the JSON bisque-voice wrote to stdout ('-' reads stdin)
 *   --wav  <path>     the audio it wrote to --out (must be a .wav for RMS)
 *   --text <string>   optional; enables the one-timing-per-token check
 *   --min-rms <n>     RMS floor as a fraction of full scale (default 0.005)
 *   --min-peak <n>    peak floor as a fraction of full scale (default 0.05)
 *   --label <string>  a name for this leg, printed in the report
 *
 * Exits 0 if every check passes, 1 otherwise, printing each failure.
 */

import { readFileSync } from "node:fs";

export interface Word {
  word: string;
  start: number;
  end: number;
}
export interface SynthDoc {
  words: Word[];
  durationMs: number;
  exact?: boolean;
  contentType?: string;
  voice?: string;
  aligner?: string;
}

const EPS = 0.02; // 20 ms of float slack for timing comparisons.

function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const eq = a.indexOf("=");
    if (eq !== -1) {
      out[a.slice(2, eq)] = a.slice(eq + 1);
      continue;
    }
    const name = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) {
      out[name] = "true";
    } else {
      out[name] = next;
      i += 1;
    }
  }
  return out;
}

/** RMS and peak of a 16-bit PCM WAV, as fractions of full scale. Walks the
 *  chunk list to the `data` chunk rather than assuming a 44-byte header —
 *  identical to `wavRms` in the present skill, extended to also return peak. */
function wavStats(buf: Buffer): { rms: number; peak: number; samples: number } {
  let off = 12;
  while (off + 8 <= buf.length) {
    const id = buf.toString("ascii", off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    if (id === "data") {
      const end = Math.min(off + 8 + size, buf.length);
      let sum = 0;
      let peak = 0;
      let n = 0;
      for (let i = off + 8; i + 1 < end; i += 2) {
        const s = buf.readInt16LE(i) / 32768;
        sum += s * s;
        const abs = Math.abs(s);
        if (abs > peak) peak = abs;
        n += 1;
      }
      return { rms: n > 0 ? Math.sqrt(sum / n) : 0, peak, samples: n };
    }
    off += 8 + size + (size % 2);
  }
  return { rms: 0, peak: 0, samples: 0 };
}

/** Whitespace token count, matching how the server counts narration tokens. */
function tokenCount(text: string): number {
  const trimmed = text.trim();
  return trimmed === "" ? 0 : trimmed.split(/\s+/).length;
}

export interface CheckOpts {
  minRms?: number;
  minPeak?: number;
  /** Source narration; enables the one-timing-per-whitespace-token check. */
  text?: string;
}

export interface CheckResult {
  pass: boolean;
  failures: string[];
  notes: string[];
  summary: string;
  stats: { rms: number; peak: number; samples: number };
}

/** The gate itself — pure, so the suite driver can reuse it in-process. */
export function checkSynth(
  doc: SynthDoc,
  wavBuf: Buffer,
  opts: CheckOpts = {},
): CheckResult {
  const minRms = opts.minRms ?? 0.005;
  const minPeak = opts.minPeak ?? 0.05;
  const failures: string[] = [];
  const notes: string[] = [];

  // ── 1. the JSON document ────────────────────────────────────────────────
  if (!Array.isArray(doc.words) || doc.words.length === 0) {
    failures.push("word list is missing or empty");
  }
  if (!Number.isInteger(doc.durationMs) || doc.durationMs <= 0) {
    failures.push(`durationMs is not a positive integer (${doc.durationMs})`);
  }
  if (doc.contentType && doc.contentType !== "audio/wav") {
    notes.push(`contentType is ${doc.contentType} (asserting RMS on the wav)`);
  }

  // ── 2. the audio is audible ─────────────────────────────────────────────
  const stats = wavStats(wavBuf);
  if (stats.samples === 0) {
    failures.push("wav has no PCM samples (no data chunk / not 16-bit PCM?)");
  } else {
    if (stats.rms < minRms) {
      failures.push(
        `SILENT: RMS ${stats.rms.toFixed(5)} < ${minRms} — the "reports success but no sound" failure mode`,
      );
    }
    if (stats.peak < minPeak) {
      failures.push(`near-silent: peak ${stats.peak.toFixed(5)} < ${minPeak}`);
    }
    if (stats.peak > 1.0) {
      failures.push(`clipping: peak ${stats.peak.toFixed(5)} > 1.0`);
    }
  }

  // ── 3. the timings are sane ─────────────────────────────────────────────
  const words = Array.isArray(doc.words) ? doc.words : [];
  const durSec = doc.durationMs / 1000;
  for (let i = 0; i < words.length; i += 1) {
    const w = words[i];
    if (typeof w.start !== "number" || typeof w.end !== "number") {
      failures.push(`word ${i} (${w.word}) has non-numeric start/end`);
      continue;
    }
    if (w.start < 0)
      failures.push(`word ${i} (${w.word}) starts before 0 (${w.start})`);
    if (w.end < w.start)
      failures.push(
        `word ${i} (${w.word}) ends before it starts (${w.start}→${w.end})`,
      );
    if (i > 0 && w.start + EPS < words[i - 1].end) {
      failures.push(
        `word ${i} (${w.word}) overlaps the previous word (${words[i - 1].end} → ${w.start})`,
      );
    }
  }
  if (words.length > 0) {
    const last = words[words.length - 1];
    if (typeof last.end === "number" && last.end > durSec + EPS) {
      failures.push(
        `last word ends (${last.end}s) after the reported duration (${durSec}s)`,
      );
    }
  }

  // one timing per whitespace token (only meaningful for whitespace-delimited
  // languages — skip for CJK, where the caller omits `text`)
  if (opts.text !== undefined) {
    const want = tokenCount(opts.text);
    if (want !== words.length) {
      failures.push(
        `timing count ${words.length} ≠ source token count ${want} — cue markers index this array by position`,
      );
    }
  }

  const aligner = doc.aligner
    ? `, aligner=${doc.aligner}${doc.exact ? " (exact)" : ""}`
    : "";
  const summary =
    `voice=${doc.voice ?? "?"}, ${words.length} words, ${doc.durationMs}ms, ` +
    `RMS=${stats.rms.toFixed(4)}, peak=${stats.peak.toFixed(4)}${aligner}`;

  return { pass: failures.length === 0, failures, notes, summary, stats };
}

function main(): number {
  const args = parseArgs(process.argv.slice(2));
  const label = args.label ? `[${args.label}] ` : "";

  if (!args.json || !args.wav) {
    console.error(
      "usage: assert-audio.ts --json <path|-> --wav <path> [--text T]",
    );
    return 2;
  }

  let doc: SynthDoc;
  try {
    const raw =
      args.json === "-"
        ? readFileSync(0, "utf8")
        : readFileSync(args.json, "utf8");
    doc = JSON.parse(raw);
  } catch (err) {
    console.error(
      `${label}FAIL: could not read/parse JSON (${(err as Error).message})`,
    );
    return 1;
  }

  let wavBuf: Buffer;
  try {
    wavBuf = readFileSync(args.wav);
  } catch (err) {
    console.error(
      `${label}FAIL: could not read wav (${(err as Error).message})`,
    );
    return 1;
  }

  const r = checkSynth(doc, wavBuf, {
    minRms: args["min-rms"] ? Number(args["min-rms"]) : undefined,
    minPeak: args["min-peak"] ? Number(args["min-peak"]) : undefined,
    text:
      args.text !== undefined && args.text !== "true" ? args.text : undefined,
  });

  for (const n of r.notes) console.log(`${label}note: ${n}`);
  if (!r.pass) {
    console.error(`${label}FAIL — ${r.summary}`);
    for (const f of r.failures) console.error(`  ✗ ${f}`);
    return 1;
  }
  console.log(`${label}PASS — ${r.summary}`);
  return 0;
}

if (import.meta.main) process.exit(main());
