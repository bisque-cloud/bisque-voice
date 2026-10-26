//! FFI to the vendored `vibevoice.cpp`.
//!
//! No `Engine` implementation yet — this is the linkage seam. It exists so the
//! vibevoice archive is *referenced*, which is what makes the link real: the
//! release profile links with `-dead_strip`, so an archive nothing calls is
//! dropped whole and the build passes while carrying no vibevoice at all.
//! `version()` and the test below are what prove otherwise.
//!
//! **One ggml, shared with qwentts.** This is compiled by `build.rs` with our
//! fork's `VIBEVOICE_GGML_EXTERNAL` option, so it builds no ggml of its own and
//! resolves every `ggml_*` symbol against the archives `build_qwentts`
//! produces. Both trees are pinned to the same ggml commit. Letting it build
//! its own would not fail the link — 534 symbols are defined in both, and the
//! linker picks by search order — it would silently run one engine on the
//! other's ggml.

use std::ffi::CStr;

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code
)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/vibevoice_ffi.rs"));
}

/// The vendored engine's build string, e.g. `"vibevoice.cpp 0.0.1"`.
#[allow(dead_code)]
pub fn version() -> String {
    // SAFETY: `vv_version` returns a pointer to a static string owned by the
    // library and never frees it, so the borrow is valid for the copy below.
    let raw = unsafe { ffi::vv_version() };
    if raw.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(raw) }.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The linkage proof. Calling into the archive is the only thing that keeps
    /// `-dead_strip` from discarding it, so a passing build means nothing
    /// without this — see the module docs.
    #[test]
    fn the_vendored_engine_is_linked_in() {
        let v = version();
        assert!(!v.is_empty(), "vv_version returned nothing");
        assert!(
            v.contains("vibevoice"),
            "unexpected version string: {v:?}"
        );
    }

}
