#!/usr/bin/env bash
# Downloads the sources of the pinned openfhe-python tag into DIR (for ombra-openfhe).
#   fetch_openfhe_python.sh DIR
set -euo pipefail
TAG="${OPENFHE_PYTHON_TAG:-v1.6.1.0}"
DIR="${1:?usage: fetch_openfhe_python.sh DIR}"
rm -rf "$DIR"
mkdir -p "$DIR"
curl -fsSL "https://codeload.github.com/openfheorg/openfhe-python/tar.gz/refs/tags/$TAG" \
  | tar xz -C "$DIR" --strip-components=1
test -f "$DIR/src/lib/bindings.cpp"
