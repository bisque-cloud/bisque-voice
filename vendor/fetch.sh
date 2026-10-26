#!/bin/sh
# Check out the two vendored engines at the commits in vendor/PINS.
#
#   sh vendor/fetch.sh
#
# build.rs compiles qwentts.cpp and vibevoice.cpp from vendor/, and both bring
# their own ggml as a git submodule, so each is a full clone at its pinned
# commit followed by `git submodule update --init --recursive`. A directory
# that already has a checkout is moved to its pin rather than cloned again.
set -eu

cd "$(dirname "$0")"

grep -v '^[[:space:]]*#' PINS | while read -r dir url rev; do
  [ -n "${dir:-}" ] || continue
  if [ ! -e "$dir/.git" ]; then
    rm -rf "$dir"
    git clone --quiet "$url" "$dir"
  fi
  git -C "$dir" fetch --quiet origin "$rev" 2>/dev/null || git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet --detach "$rev"
  git -C "$dir" submodule update --quiet --init --recursive
  echo "vendor/$dir at $(git -C "$dir" rev-parse --short HEAD)"
done
