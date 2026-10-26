//! Pick the G2P for a voice, and load its data on demand.
//!
//! Kokoro's voice ids encode their language in the first character —
//! `af_heart` is American English, `zf_xiaobei` is Chinese, `jm_kumo` is
//! Japanese. That letter is Kokoro's own language code (`LANG_CODES` in
//! `kokoro/pipeline.py`), and it is the only thing that decides which G2P runs.
//!
//! Each language's data is a separate download (see `pack`), so a G2P is built
//! the first time a voice of that language is narrated and kept afterwards.
//! Nothing is loaded for a language the user never asks for: UniDic alone is
//! 204 MB on disk, and someone narrating English should never pay for it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use super::{espeak, ja, pack, zh, Token};

/// The nine languages, as Kokoro's voice ids spell them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    /// `a` — American English.
    AmericanEnglish,
    /// `b` — British English.
    BritishEnglish,
    /// `e` `f` `i` `p` `h` — the eSpeak languages.
    Espeak(char),
    /// `z` — Chinese.
    Chinese,
    /// `j` — Japanese.
    Japanese,
}

impl Language {
    /// The language a Kokoro voice id speaks.
    ///
    /// Deliberately strict: an unrecognized prefix is an error rather than a
    /// silent fall back to English, which would narrate Japanese text with an
    /// English lexicon and produce confident nonsense.
    pub fn of_voice(voice: &str) -> Result<Language> {
        let (_, voice) = crate::engine::split_voice(voice);
        let Some(code) = voice.chars().next() else {
            bail!("empty voice id");
        };
        Language::of_code(code).ok_or_else(|| {
            anyhow::anyhow!(
                "voice {voice:?} starts with {code:?}, which is not one of \
                 Kokoro's language codes (a b e f h i j p z)"
            )
        })
    }

    pub fn of_code(code: char) -> Option<Language> {
        Some(match code {
            'a' => Language::AmericanEnglish,
            'b' => Language::BritishEnglish,
            'z' => Language::Chinese,
            'j' => Language::Japanese,
            c if espeak::voice_for(c).is_some() => Language::Espeak(c),
            _ => return None,
        })
    }

    pub fn code(self) -> char {
        match self {
            Language::AmericanEnglish => 'a',
            Language::BritishEnglish => 'b',
            Language::Espeak(c) => c,
            Language::Chinese => 'z',
            Language::Japanese => 'j',
        }
    }

    /// The extra data pack this language needs, or `None` for English, whose
    /// lexicons ship inside the engine pack.
    pub fn pack(self) -> Option<&'static pack::G2pPack> {
        pack::for_language(self.code())
    }
}

/// A G2P for one language, built once and reused.
enum Loaded {
    English(super::EnglishG2P),
    Espeak(espeak::EspeakG2P),
    Chinese(zh::ChineseG2P),
    Japanese(ja::JapaneseG2P),
}

/// Holds the engine pack path and every G2P built so far.
pub struct Router {
    engine_pack: PathBuf,
    models_root: PathBuf,
    loaded: HashMap<Language, Loaded>,
}

impl Router {
    pub fn new(engine_pack: &Path, models_root: &Path) -> Self {
        Router {
            engine_pack: engine_pack.to_path_buf(),
            models_root: models_root.to_path_buf(),
            loaded: HashMap::new(),
        }
    }

    /// Where a language's data pack is installed.
    fn data_dir(&self, language: Language) -> Result<PathBuf> {
        let Some(pack) = language.pack() else {
            return Ok(self.engine_pack.clone());
        };
        let dir = self.models_root.join(pack.pack_version);
        if !dir.join(".complete").exists() {
            bail!(
                "{} needs the {} data pack, which is not installed yet.\n\
                 Install it with: bisque-voice install {}",
                pack.display_name,
                pack.id,
                pack.id
            );
        }
        Ok(dir)
    }

    fn build(&self, language: Language) -> Result<Loaded> {
        Ok(match language {
            Language::AmericanEnglish | Language::BritishEnglish => {
                let british = language == Language::BritishEnglish;
                Loaded::English(load_english(&self.engine_pack, british)?)
            }
            Language::Espeak(code) => {
                let voice = espeak::voice_for(code)
                    .with_context(|| format!("no eSpeak voice for {code:?}"))?;
                Loaded::Espeak(espeak::EspeakG2P::new(voice, &self.data_dir(language)?)?)
            }
            Language::Chinese => Loaded::Chinese(zh::ChineseG2P::new(&self.data_dir(language)?)?),
            Language::Japanese => {
                Loaded::Japanese(ja::JapaneseG2P::new(&self.data_dir(language)?)?)
            }
        })
    }

    /// Every word, with what the engine will say for it and what the lexicon
    /// knows about it. Facts only — no filtering and no risk classification;
    /// judging which words are wrong takes knowing how they should sound,
    /// which is the caller's job.
    ///
    /// English only: the other lexicons do not carry the two signals this
    /// leans on. An engine with no G2P at all has nothing to report.
    pub fn report(&mut self, voice: &str, text: &str) -> Result<crate::check::Report> {
        let language = Language::of_voice(voice)?;
        if !self.loaded.contains_key(&language) {
            let built = self.build(language)?;
            self.loaded.insert(language, built);
        }
        let Loaded::English(g2p) = &self.loaded[&language] else {
            // Not a finding-free text — a text nothing looked at. The other
            // lexicons carry neither the part-of-speech readings nor the
            // known-word signal this leans on.
            return Ok(crate::check::Report::Unavailable(format!(
                "the pronunciation report reads the English lexicon, and {voice} \
                 speaks another language"
            )));
        };
        Ok(crate::check::Report::Checked(
            g2p.phonemize(text)
                .into_iter()
                .filter_map(|t| {
                    // `facts` is None only for a token with no word in it.
                    let (in_lexicon, readings) = g2p.facts(&t.text)?;
                    let phonemes = t.phonemes.unwrap_or_default();
                    Some(crate::check::Finding {
                        respelled: crate::check::respell(&phonemes),
                        word: t.text,
                        phonemes,
                        in_lexicon,
                        readings,
                    })
                })
                .collect(),
        ))
    }

    pub fn phonemize(&mut self, voice: &str, text: &str) -> Result<Vec<Token>> {
        let language = Language::of_voice(voice)?;
        if !self.loaded.contains_key(&language) {
            let built = self.build(language)?;
            self.loaded.insert(language, built);
        }
        Ok(match &self.loaded[&language] {
            Loaded::English(g2p) => g2p.phonemize(text),
            Loaded::Espeak(g2p) => g2p.phonemize(text)?,
            Loaded::Chinese(g2p) => g2p.phonemize(text),
            Loaded::Japanese(g2p) => g2p.phonemize(text)?,
        })
    }
}

/// The English G2P, against the American or British lexicon.
///
/// The British half was shipped but never selected: `KokoroEngine` loaded
/// `us_gold`/`us_silver` unconditionally, so `bf_emma` and `bm_george` read
/// American pronunciations in a British voice. The `gb_*` files were already
/// in the pack.
fn load_english(pack: &Path, british: bool) -> Result<super::EnglishG2P> {
    let prefix = if british { "gb" } else { "us" };
    let gold = pack.join(format!("{prefix}_gold.json"));
    let silver = pack.join(format!("{prefix}_silver.json"));
    let lexicon = super::lexicon::Lexicon::new(
        serde_json::from_slice(&std::fs::read(&gold).with_context(|| {
            format!("reading {} — is the engine pack up to date?", gold.display())
        })?)?,
        serde_json::from_slice(&std::fs::read(&silver)?)?,
        british,
    );
    // 3 MB, and it only fires on the ~1.8% of narration the lexicon misses.
    // Missing weights are not fatal: `narrate` refuses per-word instead, so a
    // word is never quietly voiced as silence.
    let config = pack.join(format!("{prefix}_bart_config.json"));
    let weights = pack.join(format!("{prefix}_bart.safetensors"));
    let fallback = if config.exists() && weights.exists() {
        Some(super::fallback::Fallback::load(
            serde_json::from_slice(&std::fs::read(&config)?)
                .with_context(|| format!("parsing {}", config.display()))?,
            &std::fs::read(&weights)?,
        )?)
    } else {
        None
    };
    Ok(super::EnglishG2P::new(lexicon, fallback))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_voice_id_names_its_language() {
        assert_eq!(
            Language::of_voice("af_heart").unwrap(),
            Language::AmericanEnglish
        );
        assert_eq!(
            Language::of_voice("bm_george").unwrap(),
            Language::BritishEnglish
        );
        assert_eq!(Language::of_voice("zf_xiaobei").unwrap(), Language::Chinese);
        assert_eq!(Language::of_voice("jm_kumo").unwrap(), Language::Japanese);
        assert_eq!(Language::of_voice("ef_dora").unwrap(), Language::Espeak('e'));
        assert_eq!(Language::of_voice("ff_siwis").unwrap(), Language::Espeak('f'));
        assert_eq!(Language::of_voice("hf_alpha").unwrap(), Language::Espeak('h'));
        assert_eq!(Language::of_voice("if_sara").unwrap(), Language::Espeak('i'));
        assert_eq!(Language::of_voice("pm_alex").unwrap(), Language::Espeak('p'));
    }

    #[test]
    fn an_engine_qualified_voice_still_names_its_language() {
        assert_eq!(
            Language::of_voice("kokoro:zf_xiaoxiao").unwrap(),
            Language::Chinese
        );
    }

    /// Falling back to English for an unknown prefix would narrate with the
    /// wrong lexicon and sound confident about it.
    #[test]
    fn an_unknown_prefix_is_an_error_not_a_fallback() {
        let err = Language::of_voice("qq_nobody").unwrap_err().to_string();
        assert!(err.contains("not one of Kokoro's language codes"), "{err}");
        assert!(Language::of_voice("").is_err());
    }

    #[test]
    fn every_language_round_trips_through_its_code() {
        for code in ['a', 'b', 'e', 'f', 'h', 'i', 'j', 'p', 'z'] {
            let language = Language::of_code(code)
                .unwrap_or_else(|| panic!("{code} is not a language"));
            assert_eq!(language.code(), code);
        }
        assert_eq!(Language::of_code('q'), None);
    }

    /// English resolves from the engine pack; the other seven need a download.
    #[test]
    fn only_english_needs_no_extra_pack() {
        assert!(Language::AmericanEnglish.pack().is_none());
        assert!(Language::BritishEnglish.pack().is_none());
        for language in [
            Language::Chinese,
            Language::Japanese,
            Language::Espeak('e'),
            Language::Espeak('f'),
            Language::Espeak('h'),
            Language::Espeak('i'),
            Language::Espeak('p'),
        ] {
            assert!(
                language.pack().is_some(),
                "{language:?} has no pack, so its voices cannot speak"
            );
        }
    }

    /// A missing pack must say which command installs it, not fail obscurely
    /// inside a file read.
    #[test]
    fn a_missing_data_pack_names_the_command_that_installs_it() {
        let dir = tempfile::tempdir().unwrap();
        let router = Router::new(dir.path(), dir.path());
        let err = router.data_dir(Language::Japanese).unwrap_err().to_string();
        assert!(err.contains("bisque-voice install g2p-ja"), "{err}");
        assert!(err.contains("Japanese"), "{err}");
    }
}
