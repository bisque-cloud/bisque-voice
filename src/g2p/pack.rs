//! G2P data packs, one per language family.
//!
//! Kokoro's weights are 171 MB and speak all nine languages, but the *data*
//! needed to turn text into phonemes is per-language and wildly uneven: eSpeak
//! NG's tables for Spanish, French, Italian, Portuguese and Hindi come to about
//! 3 MB together, pypinyin's dictionaries are 3 MB, and UniDic — the
//! morphological dictionary MeCab needs to read Japanese — is 248 MB on disk.
//!
//! Folding all of that into the engine pack would make someone narrating an
//! English slide deck download and store a quarter-gigabyte Japanese
//! dictionary. So the engine pack stays what it was, and each language's G2P
//! data is a separate pack installed the first time a voice of that language is
//! used.
//!
//! These are `fetch::Pack`s, which is the whole point of that trait: one
//! hash-verified archive, one lock, one atomic extract, one implementation. A
//! second copy of that logic is a second place to forget the checksum, which is
//! how the garbled-audio incident happened.

use crate::fetch::Pack;

/// A G2P data pack and the Kokoro language letters it serves.
#[derive(Debug, Clone)]
pub struct G2pPack {
    pub id: &'static str,
    pub display_name: &'static str,
    pub pack_version: &'static str,
    pub download_bytes: u64,
    /// What a person choosing sees — never a file listing.
    pub summary: &'static str,
    /// Kokoro voice-id prefixes this pack makes speakable.
    pub languages: &'static [char],
}

/// eSpeak NG's phoneme and dictionary data, trimmed to the languages Misaki
/// routes through it.
///
/// Deliberately not the whole 25 MB distribution: `phondata`/`phontab`/
/// `phonindex`/`intonations` are shared, and only the five languages' own
/// dictionaries are carried on top — plus `en_dict`, because eSpeak switches
/// voices to pronounce a loanword and spells it out letter by letter if the
/// dictionary it switches to is absent. See `tools/build-g2p-packs.ts`.
pub const ESPEAK: G2pPack = G2pPack {
    id: "g2p-espeak",
    display_name: "Spanish, French, Italian, Portuguese, Hindi",
    pack_version: "kokoro-g2p-espeak-1",
    download_bytes: 722_398,
    summary: "Pronunciation data for the five languages Kokoro phonemizes with eSpeak NG.",
    languages: &['e', 'f', 'i', 'p', 'h'],
};

/// pypinyin's character and phrase dictionaries.
///
/// The phrase dictionary is what makes polyphones come out right — 银行 is
/// `háng`, not `xíng` — and no Rust pinyin crate ships one, so the data is
/// carried rather than approximated.
pub const CHINESE: G2pPack = G2pPack {
    id: "g2p-zh",
    display_name: "Chinese",
    pack_version: "kokoro-g2p-zh-1",
    download_bytes: 778_501,
    summary: "Pinyin dictionaries for Chinese, including the phrase table that disambiguates polyphones.",
    languages: &['z'],
};

/// UniDic, plus Misaki's word list.
///
/// By far the largest of the three: reading Japanese needs a full morphological
/// dictionary to recover pronunciations from kanji, and there is no smaller
/// substitute that produces the same readings Misaki does.
pub const JAPANESE: G2pPack = G2pPack {
    id: "g2p-ja",
    display_name: "Japanese",
    pack_version: "kokoro-g2p-ja-1",
    download_bytes: 49_000_000,
    summary: "The UniDic morphological dictionary, which Japanese needs to read kanji.",
    languages: &['j'],
};

/// Every G2P pack. American and British English are absent on purpose: their
/// lexicons ship inside the engine pack, because English is the one language
/// Misaki resolves from a word list rather than an external toolchain.
pub const REGISTRY: &[G2pPack] = &[ESPEAK, CHINESE, JAPANESE];

/// The pack that makes `lang` speakable, or `None` when nothing extra is
/// needed (English).
pub fn for_language(lang: char) -> Option<&'static G2pPack> {
    REGISTRY.iter().find(|p| p.languages.contains(&lang))
}

pub fn by_id(id: &str) -> Option<&'static G2pPack> {
    REGISTRY.iter().find(|p| p.id == id)
}

impl Pack for G2pPack {
    fn id(&self) -> &'static str {
        self.id
    }
    fn display_name(&self) -> &'static str {
        self.display_name
    }
    fn pack_version(&self) -> &'static str {
        self.pack_version
    }
    fn download_bytes(&self) -> u64 {
        self.download_bytes
    }
    fn parameters(&self) -> &'static str {
        // Not a model, so there is no parameter count. `fetch` prints this in
        // the one-time install line; "data" is the honest answer.
        "data"
    }
    fn expected_sha256(&self) -> &'static str {
        pack_sha256(self.id)
    }
}

/// The compiled-in SHA-256 for a G2P pack, or empty in a dev build.
/// `build.rs` reads `BISQUE_VOICE_PACK_SHA256_<ID>` for each entry here.
pub fn pack_sha256(id: &str) -> &'static str {
    match id {
        "g2p-espeak" => env!("BISQUE_VOICE_PACK_SHA256_G2P_ESPEAK"),
        "g2p-zh" => env!("BISQUE_VOICE_PACK_SHA256_G2P_ZH"),
        "g2p-ja" => env!("BISQUE_VOICE_PACK_SHA256_G2P_JA"),
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kokoro_language_is_either_english_or_has_a_pack() {
        // The nine languages Kokoro's voices cover. English (a/b) resolves from
        // lexicons inside the engine pack; the other seven need data.
        for lang in ['e', 'f', 'i', 'p', 'h', 'z', 'j'] {
            assert!(
                for_language(lang).is_some(),
                "language {lang} has no G2P pack, so its voices cannot speak"
            );
        }
        assert!(for_language('a').is_none(), "American English needs no pack");
        assert!(for_language('b').is_none(), "British English needs no pack");
    }

    #[test]
    fn no_language_is_claimed_by_two_packs() {
        let mut seen: Vec<char> = REGISTRY.iter().flat_map(|p| p.languages).copied().collect();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), before, "a language is served by two packs");
    }

    #[test]
    fn pack_versions_are_unique_so_installs_cannot_collide() {
        let mut versions: Vec<&str> = REGISTRY.iter().map(|p| p.pack_version).collect();
        versions.sort_unstable();
        let before = versions.len();
        versions.dedup();
        assert_eq!(versions.len(), before, "two G2P packs share a pack version");
    }

    /// A pack whose id is not in `pack_sha256` would silently never be
    /// installable, which reads as a broken build rather than a missing pack.
    #[test]
    fn every_pack_id_is_wired_to_a_checksum_slot() {
        for pack in REGISTRY {
            assert!(
                by_id(pack.id).is_some(),
                "{} is not resolvable by id",
                pack.id
            );
            // In a dev build every checksum is empty; what is asserted here is
            // that the id reaches a real arm rather than the `_ => ""` catch-all.
            let wired = matches!(pack.id, "g2p-espeak" | "g2p-zh" | "g2p-ja");
            assert!(wired, "{} has no arm in pack_sha256", pack.id);
        }
    }

    #[test]
    fn every_pack_is_described_for_a_person_choosing() {
        for pack in REGISTRY {
            assert!(!pack.summary.is_empty(), "{} has no summary", pack.id);
            assert!(!pack.languages.is_empty(), "{} serves no language", pack.id);
            assert!(pack.download_bytes > 0, "{} has no size", pack.id);
        }
    }
}
