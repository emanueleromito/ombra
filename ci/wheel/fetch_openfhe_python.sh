#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# Downloads the sources of the pinned openfhe-python tag into DIR (for ombra-openfhe).
#   fetch_openfhe_python.sh DIR
set -euo pipefail
# v1.6.1.0, pinned by commit, and the SHA-256 of the archive GitHub makes for that commit. If GitHub
# ever changes how it builds archives the check fails; recompute it from the same commit then.
COMMIT="${OPENFHE_PYTHON_COMMIT:-a016560c6a03e13f09ba59e07cb131597202039a}"
SHA256="${OPENFHE_PYTHON_SHA256:-27464e784fe2a69d9d94da21f19a054703165b1a1d0b3b30b8bfb3c3c93d1927}"
DIR="${1:?usage: fetch_openfhe_python.sh DIR}"
rm -rf "$DIR"
mkdir -p "$DIR"
TMP="$(mktemp)"
curl -fsSL -o "$TMP" "https://codeload.github.com/openfheorg/openfhe-python/tar.gz/$COMMIT"
if command -v sha256sum >/dev/null 2>&1; then
  echo "$SHA256  $TMP" | sha256sum -c -
else
  echo "$SHA256  $TMP" | shasum -a 256 -c -
fi
tar xz -C "$DIR" --strip-components=1 -f "$TMP"
rm -f "$TMP"
test -f "$DIR/src/lib/bindings.cpp"
