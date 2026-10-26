//! Opt-in per-graph timing, for answering "where does the time go".
//!
//! This engine's whole open question is speed, and a total wall clock cannot
//! answer it: Qwen3 runs six graphs, one of them fifteen times per output
//! frame, and the fix for "the talker is slow" and the fix for "the code
//! predictor is slow" have nothing in common. So the breakdown is a permanent
//! feature rather than a scratch patch — the next person to pick this up
//! should not have to re-derive it.
//!
//! Off unless `BISQUE_VOICE_PROFILE` is set, and the check is a cached atomic, so
//! a normal run pays one relaxed load per graph call and nothing else. Output
//! goes to stderr; stdout is the JSON contract.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::Instant;

static ENABLED: AtomicU8 = AtomicU8::new(2); // 2 = not yet resolved
static TOTALS: Mutex<Option<BTreeMap<&'static str, Entry>>> = Mutex::new(None);

#[derive(Default, Clone, Copy)]
struct Entry {
    nanos: u128,
    calls: u64,
}

pub fn enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = std::env::var_os("BISQUE_VOICE_PROFILE").is_some_and(|v| v != "0");
            ENABLED.store(u8::from(on), Ordering::Relaxed);
            on
        }
    }
}

/// Time `body`, filing the result under `name`. A no-op wrapper when profiling
/// is off, so call sites read the same either way.
pub fn span<T>(name: &'static str, body: impl FnOnce() -> T) -> T {
    if !enabled() {
        return body();
    }
    let start = Instant::now();
    let out = body();
    let elapsed = start.elapsed().as_nanos();
    if let Ok(mut guard) = TOTALS.lock() {
        let entry = guard.get_or_insert_with(BTreeMap::new).entry(name).or_default();
        entry.nanos += elapsed;
        entry.calls += 1;
    }
    out
}

/// Write the breakdown to stderr and reset. Called once, after synthesis.
///
/// `audio_secs` turns each row into a realtime factor, which is the number that
/// decides whether this engine is shippable — a millisecond total is not
/// meaningful on its own.
pub fn dump(audio_secs: f64) {
    if !enabled() {
        return;
    }
    let Ok(mut guard) = TOTALS.lock() else { return };
    let Some(totals) = guard.take() else { return };
    if totals.is_empty() {
        return;
    }

    let grand: u128 = totals.values().map(|e| e.nanos).sum();
    let ms = |n: u128| n as f64 / 1e6;

    eprintln!();
    eprintln!(
        "bisque-voice profile — {:.2} s of audio, {:.2} s measured",
        audio_secs,
        ms(grand) / 1000.0
    );
    eprintln!(
        "  {:<18} {:>10} {:>8} {:>11} {:>7}",
        "graph", "total ms", "calls", "ms/call", "share"
    );
    let mut rows: Vec<_> = totals.iter().collect();
    rows.sort_by(|a, b| b.1.nanos.cmp(&a.1.nanos));
    for (name, entry) in rows {
        eprintln!(
            "  {:<18} {:>10.1} {:>8} {:>11.3} {:>6.1}%",
            name,
            ms(entry.nanos),
            entry.calls,
            ms(entry.nanos) / entry.calls.max(1) as f64,
            100.0 * entry.nanos as f64 / grand.max(1) as f64,
        );
    }
    if audio_secs > 0.0 {
        eprintln!(
            "  realtime factor: {:.2}x slower than realtime",
            (ms(grand) / 1000.0) / audio_secs
        );
    }
    eprintln!();
}
