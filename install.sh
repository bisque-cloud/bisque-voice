#!/usr/bin/env sh
# Install `bisque-voice` — the local voice engine behind /present.
#
#   curl -fsSL https://download.bisque.today/bisque-voice/install.sh | sh
#
# Installs to ~/.bisque/bin. Deliberately POSIX sh with no jq/python
# dependency: this runs on a stranger's machine, and the whole point of a
# static binary is not needing a runtime to get it.
#
# Windows has its own installer, install.ps1, published beside this one; it
# handles the DLLs that ship next to bisque-voice.exe.
#
# Environment:
#   BISQUE_VOICE_BASE_URL   release root. curl speaks file://, so pointing this
#                         at a local directory exercises the whole
#                         resolve → download → verify → extract path offline.
#   BISQUE_BIN_DIR        install directory
#   BISQUE_VOICE_VERSION    pin a version instead of resolving latest.json
set -eu

BASE="${BISQUE_VOICE_BASE_URL:-https://download.bisque.today/bisque-voice}"
BIN_DIR="${BISQUE_BIN_DIR:-$HOME/.bisque/bin}"

case "$(uname -s)" in
  Darwin) os="apple-darwin" ;;
  Linux)  os="unknown-linux-gnu" ;;
  MINGW* | MSYS* | CYGWIN*)
    # Git Bash and friends are Windows. The MSVC build is the right one, but
    # it needs its DLLs placed beside the .exe — install.ps1 does that.
    echo "bisque-voice: this is Windows — install with PowerShell:" >&2
    echo "  irm $BASE/install.ps1 | iex" >&2
    exit 1
    ;;
  *)
    echo "bisque-voice: unsupported OS $(uname -s)." >&2
    echo "  On Windows:  irm $BASE/install.ps1 | iex" >&2
    echo "  Builds exist for macOS (Apple Silicon and Intel), Linux (x86_64, aarch64) and Windows (x86_64)." >&2
    exit 1
    ;;
esac

case "$(uname -m)" in
  x86_64 | amd64)  arch="x86_64" ;;
  arm64 | aarch64) arch="aarch64" ;;
  *) echo "bisque-voice: unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

# A Rosetta-translated shell reports x86_64 on an Apple Silicon Mac. The arm64
# build is the correct one there, so check before installing the Intel one.
intel_mac=0
if [ "$os" = "apple-darwin" ] && [ "$arch" = "x86_64" ]; then
  if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
    arch="aarch64"
  else
    # Platform support is per-engine on Intel Macs: the voice-clone engines
    # are supported; Kokoro (the small quick engine) is not — its runtime
    # dropped this platform, so that engine cannot exist in this binary and
    # `bisque-voice engines` will say so.
    intel_mac=1
    echo "bisque-voice: Intel Mac — the voice-clone engine is supported here." >&2
    echo "  Kokoro is not: its runtime dropped Intel macOS, so this build ships without it." >&2
  fi
fi

target="${arch}-${os}"

version="${BISQUE_VOICE_VERSION:-}"
if [ -z "$version" ]; then
  # latest.json is small and regular; a grep beats requiring jq to be installed.
  version="$(curl -fsSL "$BASE/latest.json" | tr ',' '\n' | sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
fi
version="${version#v}"
if [ -z "$version" ]; then
  echo "bisque-voice: could not resolve the latest version from $BASE/latest.json" >&2
  exit 1
fi

url="$BASE/v$version/bisque-voice-$target.tar.gz"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "bisque-voice $version ($target)"
curl -fsSL "$url" -o "$tmp/bisque-voice.tar.gz" || {
  echo "bisque-voice: download failed: $url" >&2
  exit 1
}

# Verify before installing, and fail rather than skip when verification is not
# possible. Downloading a binary over HTTPS and running it unchecked is exactly
# the habit this project does not want to teach — and a silent skip is how you
# end up believing you checked.
curl -fsSL "$url.sha256" -o "$tmp/expected.sha256" || {
  echo "bisque-voice: no checksum published at $url.sha256 — refusing to install an unverified binary." >&2
  exit 1
}
expected="$(awk '{print $1}' "$tmp/expected.sha256")"
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$tmp/bisque-voice.tar.gz" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  actual="$(shasum -a 256 "$tmp/bisque-voice.tar.gz" | awk '{print $1}')"
else
  echo "bisque-voice: neither sha256sum nor shasum is available — cannot verify the download, refusing to install." >&2
  exit 1
fi
if [ -z "$expected" ] || [ "$actual" != "$expected" ]; then
  echo "bisque-voice: CHECKSUM MISMATCH — refusing to install." >&2
  echo "  expected ${expected:-<empty>}" >&2
  echo "  actual   $actual" >&2
  exit 1
fi

# Extract to a scratch directory rather than straight into $BIN_DIR, so a
# binary that fails the signature check below is never written where the user
# would find it and run it.
mkdir -p "$tmp/extract"
tar -xzf "$tmp/bisque-voice.tar.gz" -C "$tmp/extract"
chmod +x "$tmp/extract/bisque-voice"

# The checksum above came from the same server as the tarball, so by itself it
# only proves the download arrived intact: whoever can replace one file can
# replace both. Apple's Developer ID signature is the check that does not trust
# this server — it chains to Apple's root CA, and forging it takes our signing
# key, not write access to a bucket. So on macOS, verify the signature and
# verify it is OURS.
if [ "$os" = "apple-darwin" ]; then
  team_id="UG9Z22779U" # Purr and Meow, LLC — matches TEAM_ID in .github/workflows/release.yml
  if ! command -v codesign >/dev/null 2>&1; then
    echo "bisque-voice: codesign is missing — install the Xcode command line tools" >&2
    echo "  (xcode-select --install). Refusing to install an unverified binary." >&2
    exit 1
  fi
  if ! codesign --verify --strict "$tmp/extract/bisque-voice" >/dev/null 2>&1; then
    echo "bisque-voice: SIGNATURE INVALID — this binary is not signed the way it was built." >&2
    echo "  Refusing to install. Do not run the file." >&2
    exit 1
  fi
  # An ad-hoc signature passes --verify but carries no team, and codesign
  # spells that "not set" — report it as what it means rather than echoing it.
  signer="$(codesign -dv --verbose=4 "$tmp/extract/bisque-voice" 2>&1 | sed -n 's/^TeamIdentifier=//p' | head -1)"
  if [ "$signer" != "$team_id" ]; then
    case "$signer" in
      "" | "not set") signer="none — the binary is ad-hoc signed or unsigned" ;;
    esac
    echo "bisque-voice: WRONG SIGNER — refusing to install." >&2
    echo "  expected team $team_id" >&2
    echo "  actual   team $signer" >&2
    exit 1
  fi
  # Notarization is worth reporting and not worth failing on: `spctl` answers
  # with the machine's Gatekeeper policy as much as with the file's status, so
  # a user who turned assessment off would be blocked by a check that proves
  # nothing the two above did not already prove cryptographically.
  if spctl -a -t install "$tmp/extract/bisque-voice" >/dev/null 2>&1; then
    echo "verified: sha256 ok, Developer ID $team_id, notarized by Apple"
  else
    echo "verified: sha256 ok, Developer ID $team_id (this Mac did not confirm notarization)"
  fi
fi

# Every member: on Windows the .exe loads DLLs from its own directory. Each
# one is written beside its destination and renamed into place, never copied
# over the existing file. A narration may be running from the old binary at
# this moment, and writing into a file that is executing makes macOS mark its
# code signature invalid, so the new binary is killed on every launch until
# the file is recreated (that is how the 0.1.34 upgrade on the Mac mini
# failed). A rename gives the new file its own inode and leaves the running
# process on the old one — the same move Bun's and rustup's installers make.
mkdir -p "$BIN_DIR"
for member in "$tmp"/extract/*; do
  name="$(basename "$member")"
  cp -R "$member" "$BIN_DIR/.$name.new"
  mv -f "$BIN_DIR/.$name.new" "$BIN_DIR/$name"
done
chmod +x "$BIN_DIR/bisque-voice"

echo "installed: $BIN_DIR/bisque-voice"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "add it to your PATH:  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
# An upgrade keeps the models already on disk, so only point at a download when
# the new binary reports none it can run. `engines` labels an engine
# `[installed]` only when this build can open it; the JSON `installed` field is
# just "a pack is on disk", which an Intel Mac upgrading from a build that ran
# Kokoro still has.
if ! "$BIN_DIR/bisque-voice" engines 2>/dev/null | grep -q '\[installed\]'; then
  echo
  echo "no voice model is installed yet — pick one:"
  echo "  bisque-voice engines          # the catalogue, with sizes and trade-offs"
  if [ "$intel_mac" = "1" ]; then
    echo "  bisque-voice install qwen3-clone-06b   # the voice-clone engine"
  else
    echo "  bisque-voice install kokoro   # small, fast, English"
  fi
fi
