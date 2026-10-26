//! One voice model per machine, and never a 1.7B on a disk too full to swap.
//!
//! A voice model is not a per-process problem. The Qwen 1.7B models peak at
//! about 5.7 GB resident and the 0.6B at about 4.9 GB, and
//! a running Kokoro holds about 1.8 GB on the Mac mini, so a few runs started
//! by different callers — the mini's TTS worker, a terminal, an agent
//! publishing three presentations at once, a trial script — fill a 16 GB
//! machine between them, and nothing inside any one process can see the
//! others. When that happened on a disk too full for macOS to grow swap, the
//! kernel found nothing to page out, userspace stalled for 90 seconds and the
//! watchdog rebooted the mini, twice on 2026-10-05: a Qwen trial plus three
//! Kokoro narrations the second time. With free disk, the same overcommit
//! ends in an ordinary out-of-memory kill.
//!
//! So two checks run before any weights are loaded:
//!
//! - **A host-wide lock.** Every engine takes the same `flock` on one fixed
//!   path, outside any model directory, so a binary pointed at a different
//!   `BISQUE_VOICE_MODEL_DIR` still queues behind the others. The kernel
//!   releases the lock when its holder exits for any reason, a `SIGKILL` from
//!   the worker's timeout included, so a crashed run cannot leave a stale lock
//!   behind. The lock is the holder's lease: it lasts as long as that process
//!   does. A waiter waits a bounded time, then fails naming the holder, so a
//!   wedged holder delays work and never deadlocks it.
//! - **A free-disk floor for the 1.7B engines.** Refuse to load when the
//!   volume macOS grows swap on has less than the floor free, and say what to
//!   do about it.
//!
//! The lock shape is cargo's (`src/cargo/util/flock.rs`): try first, announce
//! the wait only when contended, and carry on unlocked on a filesystem that
//! does not implement locking at all rather than refuse to run.

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use fs4::fs_std::FileExt;

/// Overrides where the host lock lives. Every process that should queue
/// together must agree on it, so leave it unset outside tests.
pub const LOCK_PATH_ENV: &str = "BISQUE_VOICE_HEAVY_LOCK";

/// How long to wait for another heavy run before giving up, in seconds.
pub const WAIT_ENV: &str = "BISQUE_VOICE_HEAVY_WAIT_SECS";

/// The free-disk floor for the 1.7B engines, in GiB. `0` turns the check off.
pub const MIN_FREE_ENV: &str = "BISQUE_VOICE_MIN_FREE_GIB";

/// Thirty minutes: longer than a long narration on the 1.7B model, short
/// enough that a wedged holder surfaces as an error the same hour.
const DEFAULT_WAIT: Duration = Duration::from_secs(30 * 60);

/// 20 GiB. A 1.7B run can push a 16 GB machine several GB into swap, and swap
/// files are allocated a gigabyte at a time from the same disk everything else
/// is writing to; on 2026-10-05 the mini panicked at 3.8 GiB and at 176 MiB
/// free. The floor leaves room for a second heavy process the guard cannot
/// see, such as a Python model, plus whatever is filling the disk meanwhile.
const DEFAULT_MIN_FREE_GIB: u64 = 20;

const GIB: u64 = 1024 * 1024 * 1024;

const POLL: Duration = Duration::from_millis(250);

/// Whether an engine takes the host lock: every engine, Kokoro included. Its
/// model file is 337 MB, but a running Kokoro holds about 1.8 GB resident on
/// the Mac mini (three of them were in the 2026-10-05 panic alongside a Qwen
/// trial), so one voice model at a time is the rule, whichever engine it is.
pub fn takes_host_lock(_engine_id: &str) -> bool {
    true
}

/// The free-disk floor for an engine, in bytes, before any override.
pub fn default_floor(engine_id: &str) -> Option<u64> {
    matches!(engine_id, "qwen3-clone-17b" | "qwen3-voices-17b")
        .then_some(DEFAULT_MIN_FREE_GIB * GIB)
}

/// Held for as long as the engine is loaded. Dropping it, or the process
/// ending, releases the lock.
#[derive(Debug)]
pub struct HostGuard {
    _file: Option<File>,
}

/// Wait for the host lock, then check the disk. Called before the weights
/// are mapped, so a refusal costs nothing.
pub fn admit(engine_id: &str, progress: &mut dyn Write) -> Result<HostGuard> {
    if !takes_host_lock(engine_id) {
        return Ok(HostGuard { _file: None });
    }
    let wait = std::env::var(WAIT_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(DEFAULT_WAIT, Duration::from_secs);
    let guard = acquire(&lock_path(), engine_id, wait, progress)?;
    // After the wait, not before: the disk can fill while this run queues.
    let floor = floor_override(std::env::var(MIN_FREE_ENV).ok().as_deref())
        .unwrap_or_else(|| default_floor(engine_id));
    if let Some(floor) = floor {
        if let Some((volume, available)) = swap_volume_available() {
            check_floor(engine_id, &volume, available, floor)?;
        }
    }
    Ok(guard)
}

/// One fixed path per machine. Not `std::env::temp_dir()` on Unix: macOS
/// gives each user their own `$TMPDIR`, which would split the queue.
pub fn lock_path() -> PathBuf {
    if let Some(p) = std::env::var_os(LOCK_PATH_ENV) {
        return PathBuf::from(p);
    }
    if cfg!(unix) {
        PathBuf::from("/tmp/bisque-voice-heavy.lock")
    } else {
        std::env::temp_dir().join("bisque-voice-heavy.lock")
    }
}

/// Take the lock at `path`, waiting at most `wait` for its holder.
pub fn acquire(
    path: &Path,
    engine_id: &str,
    wait: Duration,
    progress: &mut dyn Write,
) -> Result<HostGuard> {
    let file = open_lock_file(path)?;
    let started = Instant::now();
    let mut announced = false;
    loop {
        match FileExt::try_lock_exclusive(&file) {
            Ok(()) => break,
            Err(e) if is_contended(&e) => {}
            Err(e) if is_unsupported(&e) => {
                writeln!(
                    progress,
                    "bisque-voice: warning — {} does not support file locks, so \
                     this {engine_id} run cannot queue behind other runs.",
                    path.display()
                )
                .ok();
                return Ok(HostGuard { _file: None });
            }
            Err(e) => {
                return Err(e).with_context(|| format!("locking {}", path.display()));
            }
        }
        let holder = read_holder(path);
        if started.elapsed() >= wait {
            bail!(
                "another voice model is still running on this machine after \
                 {} of waiting ({holder}). One voice model runs at a time, \
                 because two can exhaust memory. Wait for it to finish, or \
                 stop it if it is stuck, then run again. {WAIT_ENV} sets how \
                 long to wait.",
                human(wait)
            );
        }
        if !announced {
            writeln!(
                progress,
                "bisque-voice: waiting for another voice model to finish \
                 ({holder}) — this machine runs one at a time"
            )
            .ok();
            announced = true;
        }
        std::thread::sleep(POLL.min(wait.saturating_sub(started.elapsed())));
    }

    // Record who holds it, for the next waiter's message. Best effort: a file
    // another user created may be read-only to this one, and the lock itself
    // does not depend on the record. It lives beside the lock, not in it,
    // because a Windows lock is mandatory: while held, nothing else can read
    // the locked file, so a record inside it never reaches a waiter.
    let record = format!(
        "pid {} · {engine_id} · since {}\n",
        std::process::id(),
        unix_now()
    );
    let _ = std::fs::write(holder_path(path), record);
    Ok(HostGuard { _file: Some(file) })
}

/// Where the holder record for the lock at `path` lives.
fn holder_path(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".holder");
    PathBuf::from(p)
}

fn open_lock_file(path: &Path) -> Result<File> {
    // `flock` needs only an open descriptor, not a writable one, so a file
    // another user created still serializes this one.
    // Rust opens with O_CLOEXEC, so no child process inherits the lock.
    match OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
    {
        Ok(f) => Ok(f),
        Err(e) if e.kind() == ErrorKind::PermissionDenied => {
            File::open(path).with_context(|| format!("opening the host lock {}", path.display()))
        }
        Err(e) => Err(e).with_context(|| format!("opening the host lock {}", path.display())),
    }
}

fn is_contended(e: &std::io::Error) -> bool {
    e.kind() == ErrorKind::WouldBlock
        || e.raw_os_error() == fs4::lock_contended_error().raw_os_error()
}

fn is_unsupported(e: &std::io::Error) -> bool {
    #[cfg(unix)]
    {
        // ENOTSUP, EOPNOTSUPP, ENOSYS — the set cargo treats as "this
        // filesystem has no locks". The numbers differ by OS.
        #[cfg(target_os = "linux")]
        const CODES: &[i32] = &[95, 38];
        #[cfg(not(target_os = "linux"))]
        const CODES: &[i32] = &[45, 102, 78];
        if e.raw_os_error().is_some_and(|c| CODES.contains(&c)) {
            return true;
        }
    }
    e.kind() == ErrorKind::Unsupported
}

fn read_holder(path: &Path) -> String {
    let mut s = String::new();
    let _ = File::open(holder_path(path)).and_then(|mut f| f.read_to_string(&mut s));
    let s = s.trim();
    if s.is_empty() {
        "holder unknown".to_string()
    } else {
        format!("held by {s}")
    }
}

/// `None` when the variable is unset or unreadable, `Some(None)` when it is
/// `0` (the check is off), `Some(Some(bytes))` otherwise.
pub fn floor_override(raw: Option<&str>) -> Option<Option<u64>> {
    let gib = raw?
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|g| g.is_finite() && *g >= 0.0)?;
    Some((gib > 0.0).then_some((gib * GIB as f64) as u64))
}

/// Refuse when `available` bytes on `volume` is under `floor`.
pub fn check_floor(engine_id: &str, volume: &Path, available: u64, floor: u64) -> Result<()> {
    if available >= floor {
        return Ok(());
    }
    bail!(
        "not loading {engine_id}: {} free on {}, and this model needs at least \
         {} free. It uses about 5.7 GB of memory, and when the disk is too full \
         for macOS to grow swap, running out of memory freezes or restarts the \
         whole machine instead of stopping one program. Free up disk space and \
         run again, or narrate with kokoro, which needs a small fraction of the \
         memory. \
         {MIN_FREE_ENV} sets the floor in GiB.",
        gib(available),
        volume.display(),
        gib(floor)
    )
}

/// The volume macOS creates swap files on, and its free space. macOS grows
/// swap on demand, so this is the disk that decides whether an overcommit is
/// survivable. On Linux and Windows swap is sized ahead of time, so there is
/// nothing to measure and the floor does not apply.
fn swap_volume_available() -> Option<(PathBuf, u64)> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let volume = ["/System/Volumes/VM", "/private/var/vm", "/"]
        .into_iter()
        .map(PathBuf::from)
        .find(|p| p.is_dir())?;
    let available = fs4::available_space(&volume).ok()?;
    Some((volume, available))
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB as f64)
}

fn human(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 60 {
        format!("{} min", s / 60)
    } else {
        format!("{s} s")
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufRead;
    use std::process::{Command, Stdio};

    const CHILD_ENV: &str = "BISQUE_VOICE_TEST_HOLD_LOCK";

    /// Run inside a child process by `a_second_process_waits_then_gets_the_lock_when_the_first_dies`.
    /// It takes the lock, says so, and holds it until killed. A no-op in an
    /// ordinary test run.
    #[test]
    fn lock_holder_child() {
        let Some(path) = std::env::var_os(CHILD_ENV) else {
            return;
        };
        let guard = acquire(
            Path::new(&path),
            "qwen3-voices-17b",
            Duration::ZERO,
            &mut std::io::sink(),
        )
        .expect("the child takes a free lock");
        println!("HELD");
        std::io::stdout().flush().unwrap();
        std::thread::sleep(Duration::from_secs(120));
        drop(guard);
    }

    fn spawn_holder(path: &Path) -> std::process::Child {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "host_guard::tests::lock_holder_child",
                "--nocapture",
            ])
            .env(CHILD_ENV, path)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut lines = std::io::BufReader::new(child.stdout.take().unwrap()).lines();
        assert!(
            lines.any(|l| l.map(|l| l.contains("HELD")).unwrap_or(false)),
            "the child never reported holding the lock"
        );
        child
    }

    /// The failure this module exists for: two separate processes. A second
    /// run waits, times out naming the holder, and then — once the holder is
    /// killed the way the worker's timeout kills it — gets the lock at once,
    /// because the kernel released it. No stale lock, no cleanup step.
    #[test]
    fn a_second_process_waits_then_gets_the_lock_when_the_first_dies() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("heavy.lock");
        let mut holder = spawn_holder(&path);

        let mut log = Vec::new();
        let started = Instant::now();
        let err = acquire(
            &path,
            "qwen3-clone-17b",
            Duration::from_millis(600),
            &mut log,
        )
        .expect_err("a held lock must not be taken");
        assert!(started.elapsed() >= Duration::from_millis(600));
        let msg = format!("{err:#}");
        assert!(msg.contains(&format!("pid {}", holder.id())), "{msg}");
        assert!(msg.contains("qwen3-voices-17b"), "{msg}");
        let log = String::from_utf8(log).unwrap();
        assert_eq!(log.matches("waiting for another voice model").count(), 1, "{log}");

        holder.kill().unwrap();
        holder.wait().unwrap();
        let guard = acquire(
            &path,
            "qwen3-clone-17b",
            Duration::from_secs(5),
            &mut std::io::sink(),
        )
        .expect("a dead holder's lock is released by the kernel");
        drop(guard);
    }

    /// Within one process, a waiter proceeds as soon as the holder is dropped,
    /// and leaves its own record for the next waiter.
    #[test]
    fn a_waiter_proceeds_when_the_holder_finishes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("heavy.lock");
        let first = acquire(
            &path,
            "qwen3-voices-06b",
            Duration::ZERO,
            &mut std::io::sink(),
        )
        .unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            drop(first);
        });
        let second = acquire(
            &path,
            "qwen3-voices-17b",
            Duration::from_secs(10),
            &mut std::io::sink(),
        )
        .expect("the lock frees when the first guard drops");
        release.join().unwrap();
        let record = std::fs::read_to_string(holder_path(&path)).unwrap();
        assert!(record.contains("qwen3-voices-17b"), "{record}");
        assert!(
            record.contains(&format!("pid {}", std::process::id())),
            "{record}"
        );
        drop(second);
    }

    /// A lock file another user made, readable but not writable, still locks.
    #[cfg(unix)]
    #[test]
    fn a_read_only_lock_file_still_serializes() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("heavy.lock");
        std::fs::write(&path, "").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();
        let first = acquire(
            &path,
            "qwen3-voices-17b",
            Duration::ZERO,
            &mut std::io::sink(),
        )
        .unwrap();
        assert!(
            first._file.is_some(),
            "a read-only descriptor must still take the lock"
        );
        assert!(acquire(
            &path,
            "qwen3-voices-17b",
            Duration::ZERO,
            &mut std::io::sink()
        )
        .is_err());
    }

    #[test]
    fn low_disk_refuses_the_17b_engines_with_the_numbers_and_a_way_out() {
        let floor = default_floor("qwen3-voices-17b").unwrap();
        assert_eq!(floor, 20 * GIB);
        assert_eq!(default_floor("qwen3-clone-17b"), Some(floor));
        // The two Oct 5 panics: 3.8 GiB and 176 MiB free.
        for available in [38 * GIB / 10, 176 * 1024 * 1024] {
            let err = check_floor(
                "qwen3-voices-17b",
                Path::new("/System/Volumes/VM"),
                available,
                floor,
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("20.0 GiB"), "{err}");
            assert!(err.contains("/System/Volumes/VM"), "{err}");
            assert!(err.contains(MIN_FREE_ENV), "{err}");
            assert!(err.contains("Free up disk space"), "{err}");
        }
        assert!(check_floor("qwen3-voices-17b", Path::new("/"), floor, floor).is_ok());
    }

    #[test]
    fn every_engine_takes_the_lock_and_only_the_17b_has_a_floor() {
        assert!(takes_host_lock("kokoro"));
        assert!(takes_host_lock("qwen3-clone-06b"));
        assert!(takes_host_lock("qwen3-voices-17b"));
        assert_eq!(default_floor("kokoro"), None);
        assert_eq!(default_floor("qwen3-voices-06b"), None);
        assert_eq!(default_floor("qwen3-clone-06b"), None);
    }

    #[test]
    fn a_second_kokoro_queues_behind_the_first() {
        let dir =
            std::env::temp_dir().join(format!("bisque-voice-kokoro-lock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("heavy.lock");

        let first = acquire(
            &path,
            "kokoro",
            Duration::from_secs(5),
            &mut std::io::sink(),
        )
        .unwrap();
        assert!(first._file.is_some());

        // A second handle sees the lock as held, as another process would,
        // waits the bounded time, then fails naming the holder.
        let mut said = Vec::new();
        let err = acquire(&path, "kokoro", Duration::from_millis(300), &mut said)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("another voice model is still running"),
            "{err}"
        );
        assert!(
            err.contains(&format!("pid {}", std::process::id())),
            "{err}"
        );
        let said = String::from_utf8(said).unwrap();
        assert!(said.contains("waiting for another voice model"), "{said}");

        drop(first);
        let second = acquire(
            &path,
            "kokoro",
            Duration::from_secs(5),
            &mut std::io::sink(),
        )
        .unwrap();
        assert!(second._file.is_some());
        drop(second);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_floor_override_parses_gib_and_zero_turns_it_off() {
        assert_eq!(floor_override(None), None);
        assert_eq!(floor_override(Some("nonsense")), None);
        assert_eq!(floor_override(Some("-3")), None);
        assert_eq!(floor_override(Some("0")), Some(None));
        assert_eq!(floor_override(Some(" 30 ")), Some(Some(30 * GIB)));
        assert_eq!(floor_override(Some("0.5")), Some(Some(GIB / 2)));
    }
}
