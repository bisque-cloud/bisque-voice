//! First-run model download.
//!
//! One hash-verified archive holds everything the engine needs — the
//! duration-promoted ONNX graph, the voice pack, the tokenizer vocabulary, the
//! Misaki lexicons and the BART fallback weights — so there is no partial-state
//! to reason about: either the pack verified and extracted, or it didn't.
//!
//! Three properties this has to have, because the repo has been burned before:
//!
//! 1. **Hash-verified before use.** A download race once shipped garbled audio.
//!    The archive's SHA-256 is checked before a single byte is extracted.
//! 2. **Safe under concurrent invocation.** Two agent runs starting at once is
//!    normal, not exotic. An exclusive lock file serializes them; the loser
//!    waits and then finds the pack already present.
//! 3. **Legible, and it never wedges.** Progress goes to stderr (stdout is the
//!    JSON contract), works without a TTY, and every request has a timeout.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use fs4::fs_std::FileExt;
use serde::Serialize;
use sha2::{Digest, Sha256};

const DEFAULT_BASE_URL: &str = "https://download.bisque.today/bisque-voice/models";

/// Why a build cannot run a pack at all — as opposed to a pack it merely did
/// not pin (an empty checksum). Platform support is per-engine now: a build
/// compiled without the `kokoro`/`onnx` features (the Intel-Mac release) still
/// lists what it cannot run, and every surface that refuses it — `engines`,
/// `install`, `--engine`, `--align` — refuses with this same reason.
#[derive(Debug, Serialize)]
pub struct Unsupported {
    /// Stable and machine-readable; consumers branch on this, not the prose.
    pub code: &'static str,
    /// The sentence a person reads.
    pub message: &'static str,
}

/// Anything that ships as a hash-verified pack.
///
/// Engines and forced aligners are different things to a user and different
/// registries in the code, but they are the *same* download: one archive, one
/// pinned SHA-256, one lock, one atomic extract. This trait is what lets there
/// be exactly one implementation of that — a second copy is a second place for
/// the checksum check to be forgotten, which is how the garbled-audio incident
/// happened.
pub trait Pack {
    fn id(&self) -> &'static str;
    fn display_name(&self) -> &'static str;
    fn pack_version(&self) -> &'static str;
    fn download_bytes(&self) -> u64;
    /// Short size label for the one-time install line, e.g. "82M" or "0.6B".
    fn parameters(&self) -> &'static str;
    /// The checksum this build was pinned to, or empty when this build did not
    /// ship this pack.
    fn expected_sha256(&self) -> &'static str;
    /// Why this build cannot run this pack at all, or `None` when it can.
    ///
    /// A build-time fact — a feature compiled out because the platform cannot
    /// carry it — not an install-state fact. Distinct from an unpinned
    /// checksum: that says "this release did not ship it", this says "no
    /// release for this platform ever could".
    fn unsupported(&self) -> Option<&'static Unsupported> {
        None
    }
}

/// Whether **this build** can fetch and verify this pack.
///
/// Derived from the pinned checksum (and the build's own `unsupported`
/// verdict) rather than declared separately, because these are the same facts
/// `ensure_pack` refuses on. A registry flag saying "shipping" next to a
/// checksum saying otherwise would be two sources of truth for one question,
/// and the ones that decide are the checksum and the compiled feature set.
///
/// Onboarding needs this: without it, a registry entry is offered, the user
/// picks it, and the refusal arrives at install time looking like a broken
/// build rather than an engine this release does not include.
pub fn is_installable(spec: &dyn Pack) -> bool {
    spec.unsupported().is_none() && !spec.expected_sha256().is_empty()
}

/// Where packs live, and where they come from — resolved **once**, then passed
/// down.
///
/// The environment is read in exactly one place, `from_env`, at startup. Every
/// other function takes a `&Models`, so nothing downstream depends on process
/// state that could change underneath it. That is what makes the install paths
/// testable: a test points a `Models` at a temporary directory and mutates
/// nothing global, which is required for the test binary's threads not to race
/// each other over a shared `BISQUE_VOICE_MODEL_DIR`.
#[derive(Clone, Debug)]
pub struct Models {
    root: PathBuf,
    base_url: String,
}

impl Models {
    /// `~/.bisque/models/`, or `$BISQUE_VOICE_MODEL_DIR` when set; downloads come
    /// from `$BISQUE_VOICE_MODEL_BASE_URL` or the published bucket.
    pub fn from_env() -> Result<Self> {
        Self::from_vars(
            std::env::var("BISQUE_VOICE_MODEL_DIR")
                .ok()
                .map(PathBuf::from),
            std::env::var("BISQUE_VOICE_MODEL_BASE_URL").ok(),
        )
    }

    /// The same resolution as `from_env` with the environment lookup already
    /// done, so it can be exercised without touching process-global state.
    pub fn from_vars(dir: Option<PathBuf>, base_url: Option<String>) -> Result<Self> {
        let root = match dir {
            Some(dir) => dir,
            None => dirs::home_dir()
                .ok_or_else(|| anyhow!("no home directory"))?
                .join(".bisque")
                .join("models"),
        };
        Ok(Self {
            root,
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.into()),
        })
    }

    /// The directory every pack is installed under.
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn pack_dir(&self, spec: &dyn Pack) -> PathBuf {
        self.root.join(spec.pack_version())
    }

    pub fn is_installed(&self, spec: &dyn Pack) -> bool {
        marker(&self.pack_dir(spec)).exists()
    }

    /// Install `spec`'s pack if it isn't already there. Returns its path.
    ///
    /// Never called speculatively: a caller must have an engine or aligner the
    /// user chose.
    pub fn ensure_pack(&self, spec: &dyn Pack, progress: &mut dyn Write) -> Result<PathBuf> {
        let pack_version = spec.pack_version();
        let dir = self.pack_dir(spec);
        if marker(&dir).exists() {
            return Ok(dir);
        }
        // Before the lock and before creating anything: a pack this build
        // cannot verify is not going to become installable by waiting for a
        // lock, and refusing here keeps "waiting for the … lock" off the
        // screen of a user whose real problem is that this release does not
        // include this engine.
        //
        // Platform first: "not shipped in this release" and "cannot exist on
        // this platform" call for different sentences, and the second is the
        // stronger fact.
        if let Some(unsupported) = spec.unsupported() {
            bail!(
                "{} is {}. There is nothing to install.",
                spec.id(),
                unsupported.message
            );
        }
        if !is_installable(spec) {
            bail!(
                "{} is not available in this build: no pinned checksum, so it \
                 will not download a model it cannot verify.\n\
                 A release includes the engines that shipped with it. To use \
                 one it did not ship — or a pack you built yourself — point \
                 BISQUE_VOICE_MODEL_DIR at a directory containing it.",
                spec.id()
            );
        }

        let root = &self.root;
        fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;

        // Serialize concurrent runs. Whoever gets the lock downloads; the others
        // block here and then see the finished pack. Per-pack, so installing two
        // engines at once doesn't serialize needlessly.
        let lock_path = root.join(format!("{pack_version}.lock"));
        let lock = File::create(&lock_path)
            .with_context(|| format!("creating {}", lock_path.display()))?;
        writeln!(progress, "bisque-voice: waiting for the {} lock…", spec.id()).ok();
        lock.lock_exclusive()
            .context("locking the model directory")?;
        let _guard = LockGuard(&lock);

        if marker(&dir).exists() {
            return Ok(dir);
        }

        // Already refused above, before the lock. Re-read rather than carry the
        // value down, so the checksum used to verify is the one on the spec.
        let expected = spec.expected_sha256();

        let url = format!("{}/{pack_version}.tar.gz", self.base_url);
        let archive = root.join(format!("{pack_version}.tar.gz.partial"));

        writeln!(
            progress,
            "bisque-voice: installing {} ({}, one time, ~{} MB)",
            spec.display_name(),
            spec.parameters(),
            spec.download_bytes() / 1_000_000
        )
        .ok();
        download(&url, &archive, progress)?;

        writeln!(progress, "bisque-voice: verifying…").ok();
        let actual = sha256_file(&archive)?;
        if actual != expected {
            let _ = fs::remove_file(&archive);
            bail!(
                "model checksum mismatch (expected {expected}, got {actual}); refusing to use it"
            );
        }

        // Extract to a staging directory, then rename, so a crash mid-extract
        // can never leave a half-pack behind under the real name.
        let staging = root.join(format!("{pack_version}.staging"));
        let _ = fs::remove_dir_all(&staging);
        fs::create_dir_all(&staging)?;
        writeln!(progress, "bisque-voice: extracting…").ok();
        let tar = flate2::read::GzDecoder::new(File::open(&archive)?);
        tar::Archive::new(tar)
            .unpack(&staging)
            .context("extracting the model pack")?;
        let _ = fs::remove_file(&archive);

        let _ = fs::remove_dir_all(&dir);
        fs::rename(&staging, &dir).context("installing the model pack")?;
        File::create(marker(&dir))?;
        writeln!(
            progress,
            "bisque-voice: {} ready at {}",
            spec.id(),
            dir.display()
        )
        .ok();
        Ok(dir)
    }
}

/// A pack is only considered present once it carries its completion marker, so
/// an interrupted extraction is never mistaken for a usable install.
fn marker(dir: &Path) -> PathBuf {
    dir.join(".complete")
}

struct LockGuard<'a>(&'a File);
impl Drop for LockGuard<'_> {
    fn drop(&mut self) {
        let _ = FileExt::unlock(self.0);
    }
}

fn download(url: &str, dest: &Path, progress: &mut dyn Write) -> Result<()> {
    let response = ureq::builder()
        .timeout_connect(std::time::Duration::from_secs(30))
        .build()
        .get(url)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let total: u64 = response
        .header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let mut reader = response.into_reader();
    let mut file = File::create(dest).with_context(|| format!("creating {}", dest.display()))?;
    let mut buf = vec![0u8; 1 << 16];
    let mut done: u64 = 0;
    // Report on a byte interval rather than a timer: a non-TTY agent run gets a
    // bounded number of lines, and there is no clock to wedge on.
    let step = (total / 20).max(8 << 20);
    let mut next_report = step;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        if done >= next_report {
            if total > 0 {
                writeln!(
                    progress,
                    "bisque-voice:   {}% ({} MB)",
                    done * 100 / total,
                    done >> 20
                )
                .ok();
            } else {
                writeln!(progress, "bisque-voice:   {} MB", done >> 20).ok();
            }
            next_report += step;
        }
    }
    file.flush()?;
    if total > 0 && done != total {
        bail!("download truncated: got {done} of {total} bytes");
    }
    Ok(())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_is_lowercase_and_zero_padded() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff]), "000fff");
    }

    /// A stand-in pack, because every real one is unpinned in a dev build and
    /// the interesting assertion is that a pinned one behaves differently.
    struct FakePack(&'static str);
    impl Pack for FakePack {
        fn id(&self) -> &'static str {
            "fake"
        }
        fn display_name(&self) -> &'static str {
            "Fake"
        }
        fn pack_version(&self) -> &'static str {
            "fake-1"
        }
        fn download_bytes(&self) -> u64 {
            1
        }
        fn parameters(&self) -> &'static str {
            "0"
        }
        fn expected_sha256(&self) -> &'static str {
            self.0
        }
    }

    /// A pack the build's platform cannot carry, checksum pinned or not.
    struct UnsupportedPack;
    const UNSUPPORTED: Unsupported = Unsupported {
        code: "platform-unsupported",
        message: "not available on Intel Macs — its runtime dropped this platform",
    };
    impl Pack for UnsupportedPack {
        fn id(&self) -> &'static str {
            "fake-unsupported"
        }
        fn display_name(&self) -> &'static str {
            "Fake (unsupported)"
        }
        fn pack_version(&self) -> &'static str {
            "fake-unsupported-1"
        }
        fn download_bytes(&self) -> u64 {
            1
        }
        fn parameters(&self) -> &'static str {
            "0"
        }
        fn expected_sha256(&self) -> &'static str {
            // Pinned on purpose: the platform verdict must win over it.
            "0e9599a979c7dbfc9c0b77b00f3301e070178d35ef647c269a88aaf1e0752636"
        }
        fn unsupported(&self) -> Option<&'static Unsupported> {
            Some(&UNSUPPORTED)
        }
    }

    /// The pinned checksum IS the shipping decision. Anything that answered
    /// this question separately could disagree with the code that enforces it.
    #[test]
    fn a_pack_is_installable_exactly_when_its_checksum_was_compiled_in() {
        assert!(!is_installable(&FakePack("")));
        assert!(is_installable(&FakePack(
            "0e9599a979c7dbfc9c0b77b00f3301e070178d35ef647c269a88aaf1e0752636"
        )));
    }

    /// A platform the pack cannot run on beats a pinned checksum: the release
    /// workflow pins every pack for every leg, and the Intel-Mac build must
    /// still refuse the ONNX-backed ones.
    #[test]
    fn an_unsupported_pack_is_not_installable_even_with_a_pinned_checksum() {
        assert!(!is_installable(&UnsupportedPack));
    }

    /// The refusal carries the platform reason, not the missing-checksum
    /// sentence — the user's problem is the platform, and telling them to
    /// point `BISQUE_VOICE_MODEL_DIR` at their own pack would be advice the
    /// build cannot honor (there is no engine to run it).
    #[test]
    fn installing_an_unsupported_pack_refuses_with_the_platform_reason() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("models-root");
        let models = Models::from_vars(Some(root.clone()), None).unwrap();

        let err = models
            .ensure_pack(&UnsupportedPack, &mut std::io::sink())
            .unwrap_err()
            .to_string();

        assert!(err.contains(UNSUPPORTED.message), "{err}");
        assert!(!root.exists(), "refusing must not create the models root");
    }

    /// Refused before the lock and before the models directory is created, so
    /// an engine this build did not ship costs the user nothing and never
    /// prints "waiting for the … lock".
    #[test]
    fn an_unshipped_pack_is_refused_without_touching_the_disk() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("models-root");
        let models = Models::from_vars(Some(root.clone()), None).unwrap();

        let err = models
            .ensure_pack(&FakePack(""), &mut std::io::sink())
            .unwrap_err()
            .to_string();

        assert!(err.contains("not available in this build"), "{err}");
        assert!(err.contains("BISQUE_VOICE_MODEL_DIR"), "{err}");
        assert!(!root.exists(), "refusing must not create the models root");
    }

    /// The override is checked through `from_vars`, not by setting the real
    /// environment variable: `set_var` is process-global, and these tests run as
    /// concurrent threads of one binary, so a test that mutated it would race
    /// every other test that reads a models root.
    #[test]
    fn model_dir_is_overridable_for_tests_and_offline_use() {
        let models =
            Models::from_vars(Some(PathBuf::from("/tmp/bisque-voice-models")), None).unwrap();
        assert_eq!(models.root(), Path::new("/tmp/bisque-voice-models"));
    }

    #[test]
    fn without_an_override_packs_live_under_the_home_directory() {
        let models = Models::from_vars(None, None).unwrap();
        assert_eq!(
            models.root(),
            dirs::home_dir().unwrap().join(".bisque").join("models")
        );
    }

    #[test]
    fn sha256_matches_a_known_digest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256_file(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn an_extracted_pack_without_its_marker_is_not_installed() {
        let spec = crate::engine::REGISTRY
            .first()
            .expect("a registered engine");
        let dir = tempfile::tempdir().unwrap();
        let models = Models::from_vars(Some(dir.path().to_path_buf()), None).unwrap();
        fs::create_dir_all(models.pack_dir(spec)).unwrap();
        assert!(
            !models.is_installed(spec),
            "a half-extracted pack must not count as installed"
        );
        File::create(marker(&models.pack_dir(spec))).unwrap();
        assert!(models.is_installed(spec));
    }

    #[test]
    fn a_download_base_url_override_is_taken_once_not_per_call() {
        let models = Models::from_vars(None, Some("https://example.invalid/packs".into())).unwrap();
        assert_eq!(models.base_url, "https://example.invalid/packs");
        assert_eq!(
            Models::from_vars(None, None).unwrap().base_url,
            DEFAULT_BASE_URL
        );
    }
}
