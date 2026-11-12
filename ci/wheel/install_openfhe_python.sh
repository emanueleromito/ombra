#!/usr/bin/env bash
# Installs openfhe-python 1.6.1 into the active (test) Python, built from the pinned tag against the
# OpenFHE that build_openfhe.sh installed. Used by cibuildwheel (before-test) only: the tests need
# it, the wheel does not. Same tag as ci/Dockerfile.
set -euo pipefail

TAG="${OPENFHE_PYTHON_TAG:-v1.6.1.0}"
PREFIX="${OMBRA_OPENFHE_PREFIX:-/usr/local}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

python -m pip install --quiet pybind11 numpy cmake ninja
git clone --depth 1 --branch "$TAG" https://github.com/openfheorg/openfhe-python "$WORK/src"
extra=()
if [ "$(uname -s)" = "Darwin" ]; then
  OMP="$(brew --prefix libomp)"
  extra+=("-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp -I${OMP}/include"
    -DOpenMP_CXX_LIB_NAMES=omp "-DOpenMP_omp_LIBRARY=${OMP}/lib/libomp.dylib")
fi
cmake -S "$WORK/src" -B "$WORK/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PREFIX" -DOpenFHE_DIR="$PREFIX/lib/OpenFHE" \
  -DPYTHON_EXECUTABLE_PATH="$(python -c 'import sys; print(sys.executable)')" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" "${extra[@]}"
cmake --build "$WORK/build" -j"$( (nproc || sysctl -n hw.ncpu) 2>/dev/null || echo 2)"
cmake --install "$WORK/build"
# The module finds OpenFHE through "$ORIGIN/lib" (@loader_path/lib), next to itself, as in the
# openfhe wheel. Copy the libraries there instead of putting the OpenFHE prefix on the library
# path, which would also hide the libraries bundled in the ombra wheel under test.
LIBDIR="$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')/lib"
mkdir -p "$LIBDIR"
cp -a "$PREFIX"/lib/libOPENFHE* "$LIBDIR"/
if [ "$(uname -s)" = "Darwin" ]; then cp -a "$OMP/lib/libomp.dylib" "$LIBDIR"/; fi
python -c "import openfhe; print('openfhe-python', openfhe.__version__)"
