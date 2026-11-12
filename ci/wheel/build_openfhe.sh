#!/usr/bin/env bash
# Builds and installs OpenFHE (C++) for the client wheels. Run once per platform by cibuildwheel
# (before-all). Same tag and flags as ci/Dockerfile and modal_app.py; change the version in all.
#
#   build_openfhe.sh PREFIX
#
# Linux (manylinux_2_28): GCC's OpenMP (libgomp), which auditwheel copies into the wheel.
# macOS: Homebrew's libomp (Apple clang has no OpenMP). OpenFHE and the extension use OpenMP for
# the expansion tree, so leaving it out would make local mode single-threaded. delocate copies
# libomp into the wheel.
set -euo pipefail

OPENFHE_TAG="${OPENFHE_TAG:-v1.6.1}"
PREFIX="${1:?usage: build_openfhe.sh PREFIX}"
SRC="$(mktemp -d)/openfhe-src"
BUILD="$(dirname "$SRC")/openfhe-build"
JOBS="$( (nproc || sysctl -n hw.ncpu) 2>/dev/null || echo 2)"

if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
  # manylinux images keep their Pythons in /opt/python; the system python3 there is too old.
  PY="$(ls /opt/python/cp312-*/bin/python 2>/dev/null | head -n1 || true)"
  PY="${PY:-python3}"
  "$PY" -m pip install --quiet cmake ninja
  export PATH="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):$PATH"
fi

extra=()
if [ "$(uname -s)" = "Darwin" ]; then
  brew list libomp >/dev/null 2>&1 || brew install libomp
  OMP="$(brew --prefix libomp)"
  extra+=("-DOpenMP_C_FLAGS=-Xpreprocessor -fopenmp -I${OMP}/include"
    "-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp -I${OMP}/include"
    -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp
    "-DOpenMP_omp_LIBRARY=${OMP}/lib/libomp.dylib")
fi

git clone --depth 1 --branch "$OPENFHE_TAG" https://github.com/openfheorg/openfhe-development "$SRC"
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib "${extra[@]}"
cmake --build "$BUILD" -j"$JOBS"
cmake --install "$BUILD"
rm -rf "$(dirname "$SRC")"
