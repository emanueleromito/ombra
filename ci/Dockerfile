# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# CI image: OpenFHE 1.6.1, openfhe-python 1.6.1.0, the C++ toolchain, clang-format and Python 3.12
# with the test packages. No CUDA: GPU tests run on Modal (see .github/workflows/gpu.yml).
#
# Built by .github/workflows/ci-image.yml and pushed to ghcr.io/erailabs/ombra-ci, tagged with
# a hash of this file and ci/requirements.txt. Change either file and the next run builds a new tag.
#
# The OpenFHE and openfhe-python steps are the ones in modal_app.py. Change the versions in both.
FROM ubuntu:24.04

LABEL org.opencontainers.image.source="https://github.com/erailabs/ombra"

ARG OPENFHE_TAG=v1.6.1
ARG OPENFHE_PYTHON_TAG=v1.6.1.0

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_GENERATOR=Ninja \
    LD_LIBRARY_PATH=/usr/local/lib \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git build-essential cmake ninja-build libomp-dev pkg-config \
        python3 python3-venv python3-dev \
        protobuf-compiler libssl-dev openssl \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

COPY ci/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# OpenFHE (C++), installed to /usr/local. Same flags as modal_app.py. ldconfig puts its libraries in
# the linker cache, so processes started with a minimal environment (no LD_LIBRARY_PATH) find them.
RUN git clone --depth 1 --branch ${OPENFHE_TAG} \
        https://github.com/openfheorg/openfhe-development /opt/openfhe-src \
    && cmake -S /opt/openfhe-src -B /opt/openfhe-build \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
    && cmake --build /opt/openfhe-build -j \
    && cmake --install /opt/openfhe-build \
    && ldconfig \
    && rm -rf /opt/openfhe-src /opt/openfhe-build

# openfhe-python, built against the OpenFHE above and installed into /opt/venv.
RUN git clone --depth 1 --branch ${OPENFHE_PYTHON_TAG} \
        https://github.com/openfheorg/openfhe-python /opt/openfhe-python-src \
    && cmake -S /opt/openfhe-python-src -B /opt/openfhe-python-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE_PATH=/opt/venv/bin/python \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build /opt/openfhe-python-build -j \
    && cmake --install /opt/openfhe-python-build \
    && rm -rf /opt/openfhe-python-src /opt/openfhe-python-build \
    && python -c "import openfhe; print('openfhe-python', openfhe.__version__)"

WORKDIR /work
