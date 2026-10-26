//! The cloned-voice library.
//!
//! Cloning a voice is not synthesis. `qt_extract_voice_ref` runs the speaker
//! encoder and the codec encoder over a reference clip **once** and returns two
//! latents; every later narration in that voice reuses them and never touches
//! the clip again. For a 17-second reference that is 8 KB of speaker embedding
//! plus 4.7 KB of RVQ codes — small enough to keep beside the models, sync, or
//! hand to a server, which is why this is a library on disk and not a flag.
//!
//! **A voice belongs to the engine that extracted it.** The ABI requires
//! `ref_spk_dim` to equal the talker's hidden size (2048 for the 1.7B, smaller
//! for the 0.6B), so a voice cloned on one variant is not loadable on the other.
//! That is recorded per voice and checked on load, because the alternative —
//! handing a mismatched buffer to the C side — is a segfault, not an error
//! message.
//!
//! The on-disk form is deliberately dumb: a JSON manifest plus two raw
//! little-endian arrays. `.rvq` upstream is an 11-bit packed stream, and
//! reimplementing that packing here would buy interop with a CLI this tool does
//! not ship, at the price of a bit-twiddling bug that would surface as a
//! corrupted voice rather than a failure.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// What a voice records about itself. `engine` and `spk_dim` are the
/// compatibility check; `transcript` is required by ICL cloning and is stored
/// with the voice so the caller never has to supply it twice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceManifest {
    pub name: String,
    /// The engine id that extracted this voice, e.g. `qwen3-clone-17b`.
    pub engine: String,
    #[serde(rename = "spkDim")]
    pub spk_dim: usize,
    #[serde(rename = "numCodebooks")]
    pub num_codebooks: usize,
    /// Reference frame count; `codes` is `[num_codebooks, ref_t]` row-major.
    #[serde(rename = "refFrames")]
    pub ref_t: usize,
    /// The reference clip's transcript. ICL cloning conditions on it, and a
    /// wrong one degrades the clone, so it is captured at extraction time.
    pub transcript: String,
    /// Where the reference came from, for `voices` to show. Not load-bearing.
    #[serde(rename = "sourceClip", skip_serializing_if = "Option::is_none")]
    pub source_clip: Option<String>,
    /// Opaque caller-supplied provenance for the reference recording
    /// (`clone --source-id`, e.g. an upload timestamp). Returned by
    /// `list --json`, so a caller can tell whether this voice came from
    /// the recording it currently cares about without keeping state of
    /// its own. Absent on voices cloned before it existed.
    #[serde(rename = "sourceId", default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
}

/// A voice with its latents resident.
#[derive(Debug)]
pub struct ClonedVoice {
    pub manifest: VoiceManifest,
    pub spk: Vec<f32>,
    pub codes: Vec<i32>,
}

pub struct VoiceLibrary {
    root: PathBuf,
}

impl VoiceLibrary {
    /// Voices live beside the model packs rather than in the pack directory —
    /// a pack is content-addressed and reinstalling one must not delete work
    /// the user cannot regenerate without the original clip.
    pub fn new(models_root: &Path) -> Self {
        Self {
            root: models_root.join("voices"),
        }
    }

    fn voice_dir(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// Reject anything that is not a plain name before it reaches the
    /// filesystem: this string arrives from `--name` and is joined onto a path.
    pub fn validate_name(name: &str) -> Result<()> {
        if name.is_empty() {
            bail!("a voice name cannot be empty");
        }
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            bail!(
                "voice name {name:?} may use only letters, digits, '-' and '_' — \
                 it names a directory."
            );
        }
        Ok(())
    }

    /// Every installed voice, by name. A directory that fails to parse is
    /// skipped rather than fatal: one corrupt voice should not make `voices`
    /// unusable for the rest.
    pub fn list(&self) -> Vec<VoiceManifest> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let mut out: Vec<VoiceManifest> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let text = std::fs::read_to_string(e.path().join("voice.json")).ok()?;
                serde_json::from_str::<VoiceManifest>(&text).ok()
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    pub fn save(&self, voice: &ClonedVoice) -> Result<PathBuf> {
        Self::validate_name(&voice.manifest.name)?;
        let expected = voice.manifest.num_codebooks * voice.manifest.ref_t;
        if voice.codes.len() != expected {
            bail!(
                "voice {:?} declares {} codebooks x {} frames = {expected} codes \
                 but carries {}",
                voice.manifest.name,
                voice.manifest.num_codebooks,
                voice.manifest.ref_t,
                voice.codes.len()
            );
        }
        if voice.spk.len() != voice.manifest.spk_dim {
            bail!(
                "voice {:?} declares a {}-dim speaker embedding but carries {}",
                voice.manifest.name,
                voice.manifest.spk_dim,
                voice.spk.len()
            );
        }

        let dir = self.voice_dir(&voice.manifest.name);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        write_le_f32(&dir.join("speaker.f32"), &voice.spk)?;
        write_le_i32(&dir.join("codes.i32"), &voice.codes)?;
        // The manifest is written last: its presence is what `exists` treats as
        // a complete voice, so a run interrupted mid-write leaves an incomplete
        // directory rather than a voice that loads to garbage.
        std::fs::write(
            dir.join("voice.json"),
            serde_json::to_string_pretty(&voice.manifest)?,
        )
        .with_context(|| format!("writing {}", dir.join("voice.json").display()))?;
        Ok(dir)
    }

    pub fn load(&self, name: &str, engine: &str) -> Result<ClonedVoice> {
        Self::validate_name(name)?;
        let dir = self.voice_dir(name);
        let manifest_path = dir.join("voice.json");
        if !manifest_path.is_file() {
            let known = self.list();
            if known.is_empty() {
                bail!(
                    "no voice named {name:?}, and no voices have been cloned yet.\n\
                     Create one with: bisque-voice clone --name {name} \
                     --reference <clip.wav> --transcript <what it says>"
                );
            }
            bail!(
                "no voice named {name:?}. Cloned voices: {}",
                known
                    .iter()
                    .map(|v| v.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let manifest: VoiceManifest =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)
                .with_context(|| format!("reading {}", manifest_path.display()))?;

        // The speaker embedding's width is fixed by the talker that produced
        // it. Loading it into a different one would hand the C side a buffer of
        // the wrong length.
        if manifest.engine != engine {
            bail!(
                "voice {name:?} was cloned with {:?} and cannot be used with \
                 {engine:?} — their speaker embeddings are different widths. \
                 Re-clone it from the original clip with --engine {engine}.",
                manifest.engine
            );
        }

        let spk = read_le_f32(&dir.join("speaker.f32"))?;
        let codes = read_le_i32(&dir.join("codes.i32"))?;
        if spk.len() != manifest.spk_dim || codes.len() != manifest.num_codebooks * manifest.ref_t {
            bail!(
                "voice {name:?} is corrupt: its manifest and its latents disagree \
                 on size. Re-clone it from the original clip."
            );
        }
        Ok(ClonedVoice {
            manifest,
            spk,
            codes,
        })
    }

    pub fn remove(&self, name: &str) -> Result<bool> {
        Self::validate_name(name)?;
        let dir = self.voice_dir(name);
        if !dir.exists() {
            return Ok(false);
        }
        std::fs::remove_dir_all(&dir).with_context(|| format!("removing {}", dir.display()))?;
        Ok(true)
    }
}

fn write_le_f32(path: &Path, values: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

fn write_le_i32(path: &Path, values: &[i32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

fn read_le_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("{} is not a whole number of f32 values", path.display());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_le_i32(path: &Path) -> Result<Vec<i32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("{} is not a whole number of i32 values", path.display());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice(name: &str, engine: &str) -> ClonedVoice {
        ClonedVoice {
            manifest: VoiceManifest {
                name: name.to_string(),
                engine: engine.to_string(),
                spk_dim: 4,
                num_codebooks: 2,
                ref_t: 3,
                transcript: "hello there".to_string(),
                source_clip: None,
                source_id: None,
            },
            spk: vec![0.5, -0.25, 1.0, 0.0],
            codes: vec![1, 2, 3, 4, 5, 6],
        }
    }

    #[test]
    fn a_saved_voice_round_trips_bit_for_bit() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let lib = VoiceLibrary::new(tmp.path());
        let original = voice("narrator", "qwen3-clone-17b");
        lib.save(&original).expect("save");

        let loaded = lib.load("narrator", "qwen3-clone-17b").expect("load");
        assert_eq!(loaded.spk, original.spk);
        assert_eq!(loaded.codes, original.codes);
        assert_eq!(loaded.manifest.transcript, "hello there");
    }

    /// The speaker embedding's width is the talker's hidden size, so this is
    /// the check standing between a mismatched voice and a segfault in C.
    #[test]
    fn a_voice_refuses_to_load_into_a_different_engine() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let lib = VoiceLibrary::new(tmp.path());
        lib.save(&voice("narrator", "qwen3-clone-06b"))
            .expect("save");

        let err = lib
            .load("narrator", "qwen3-clone-17b")
            .expect_err("must refuse a cross-engine load")
            .to_string();
        assert!(err.contains("qwen3-clone-06b"), "got {err}");
        assert!(
            err.contains("re-clone") || err.contains("Re-clone"),
            "got {err}"
        );
    }

    #[test]
    fn a_name_that_would_escape_the_library_is_refused() {
        for bad in ["../evil", "a/b", "", "with space"] {
            assert!(
                VoiceLibrary::validate_name(bad).is_err(),
                "{bad:?} should be rejected"
            );
        }
        assert!(VoiceLibrary::validate_name("my_voice-2").is_ok());
    }

    #[test]
    fn declared_dimensions_must_match_the_latents() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let lib = VoiceLibrary::new(tmp.path());
        let mut v = voice("narrator", "qwen3-clone-17b");
        v.manifest.ref_t = 99;
        assert!(lib.save(&v).is_err());
    }

    #[test]
    fn a_missing_voice_names_the_ones_that_exist() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let lib = VoiceLibrary::new(tmp.path());
        lib.save(&voice("narrator", "qwen3-clone-17b"))
            .expect("save");
        let err = lib
            .load("nope", "qwen3-clone-17b")
            .expect_err("missing voice")
            .to_string();
        assert!(err.contains("narrator"), "got {err}");
    }

    #[test]
    fn listing_an_empty_library_is_not_an_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(VoiceLibrary::new(tmp.path()).list().is_empty());
    }
}
