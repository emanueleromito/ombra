# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# CI image: OpenFHE 1.6.1, openfhe-python 1.6.1.0, the C++ toolchain, clang-format and Python 3.12
# with the test packages. No CUDA: GPU tests run on Modal (see .github/workflows/gpu.yml).
#
# Built by .github/workflows/ci-image.yml and pushed to ghcr.io/erailabs/ombra-ci, tagged with
# a hash of this file and ci/requirements.txt. Change either file and the next run builds a new tag.
#
# The OpenFHE and openfhe-python steps are the ones in modal_app.py. Change the versions in both.
FROM ubuntu:24.04@sha256:2198e4c65076e9cb0b3d695219df667c0e3b0072da64ca3c2648aca17dcecaf5

LABEL org.opencontainers.image.source="https://github.com/erailabs/ombra"
LABEL org.opencontainers.image.description="Build and test environment for Ombra. Contains OpenFHE (BSD-2-Clause) and no Ombra code."
LABEL org.opencontainers.image.licenses="BSD-2-Clause"

# v1.6.1 and v1.6.1.0, pinned by commit (OpenFHE is the oracle of the kernel tests).
ARG OPENFHE_COMMIT=b59e1f291965f14fbee5172cd83a20f5b856204f
ARG OPENFHE_PYTHON_COMMIT=a016560c6a03e13f09ba59e07cb131597202039a
ARG OPENFHE_PYTHON_SHA256=27464e784fe2a69d9d94da21f19a054703165b1a1d0b3b30b8bfb3c3c93d1927

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_GENERATOR=Ninja \
    LD_LIBRARY_PATH=/usr/local/lib \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential cmake ninja-build libomp-dev pkg-config \
        python3 python3-venv python3-dev \
        protobuf-compiler libssl-dev openssl \
        libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

COPY ci/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# OpenFHE (C++), installed to /usr/local. Same flags as modal_app.py. ldconfig puts its libraries in
# the linker cache, so processes started with a minimal environment (no LD_LIBRARY_PATH) find them.
RUN git init --quiet /opt/openfhe-src \
    && git -C /opt/openfhe-src fetch --quiet --depth 1 \
        https://github.com/openfheorg/openfhe-development ${OPENFHE_COMMIT} \
    && git -C /opt/openfhe-src checkout --quiet --detach FETCH_HEAD \
    && test "$(git -C /opt/openfhe-src rev-parse HEAD)" = "${OPENFHE_COMMIT}" \
    && cmake -S /opt/openfhe-src -B /opt/openfhe-build \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
    && cmake --build /opt/openfhe-build -j \
    && cmake --install /opt/openfhe-build \
    && ldconfig \
    && rm -rf /opt/openfhe-src /opt/openfhe-build

# openfhe-python, built against the OpenFHE above and installed into /opt/venv.
RUN curl -sSfL -o /tmp/ofp.tgz https://codeload.github.com/openfheorg/openfhe-python/tar.gz/${OPENFHE_PYTHON_COMMIT} \
    && echo "${OPENFHE_PYTHON_SHA256}  /tmp/ofp.tgz" | sha256sum -c - \
    && mkdir /opt/openfhe-python-src && tar xzf /tmp/ofp.tgz -C /opt/openfhe-python-src --strip-components=1 \
    && rm /tmp/ofp.tgz \
    && cmake -S /opt/openfhe-python-src -B /opt/openfhe-python-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE_PATH=/opt/venv/bin/python \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build /opt/openfhe-python-build -j \
    && cmake --install /opt/openfhe-python-build \
    && rm -rf /opt/openfhe-python-src /opt/openfhe-python-build \
    && python -c "import openfhe; print('openfhe-python', openfhe.__version__)"

COPY licenses /licenses

WORKDIR /work
