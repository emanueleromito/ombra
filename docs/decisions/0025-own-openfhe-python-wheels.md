# 0025. We build and publish our own openfhe-python wheels (ombra-openfhe)

- **Status:** proposed (the user chose option A on 2026-11-12; waiting for acceptance)
- **Date:** 2026-11-12

## Context

The client needs openfhe-python for keys, encryption and decryption. The `openfhe` 1.6.1.0.24.4 package on PyPI is one `py3-none-any` wheel with `Requires-Python >=3.12` that holds a single CPython 3.12 x86-64 Linux binary. On Python 3.11 or 3.13, on aarch64 and on macOS it cannot work, so wheels of `ombra` for those platforms would still need a user-built openfhe-python.

## Options considered

- **A. Our own package `ombra-openfhe`.** Built from the pinned openfhe-python v1.6.1.0 tag against the same OpenFHE 1.6.1 as `ombra._native`, for the same platform matrix, with OpenFHE and the OpenMP runtime bundled. Import name stays `openfhe`. Works everywhere `ombra` does. Costs one more package to build, publish and keep in step with OpenFHE releases, and it conflicts with the PyPI `openfhe` package (same import name).
- **B. Bundle openfhe-python inside the `ombra` wheel.** One package, but it would install a top-level `openfhe` module from `ombra`, which collides with the PyPI package in a worse way (two owners of one module with no dependency between them), and the module would be copied into every `ombra` wheel.
- **C. Support only CPython 3.12 on x86-64 Linux.** Nothing to build, but it drops macOS, aarch64, 3.11 and 3.13, which T4.1 exists to cover.

## Decision

Option A. `packaging/ombra-openfhe/` builds the wheels, `wheels.yml` builds and tests them with the `ombra` wheels, and the `local` and `grpc` extras of `ombra` depend on `ombra-openfhe==1.6.1.0` on every platform. The release workflow publishes both.

## Consequences

- Users install only `ombra-openfhe`, never `openfhe` next to it (same import name); docs/local-mode.md says so.
- The OpenFHE version stays pinned to 1.6.1 in one place per use: `ci/wheel/build_openfhe.sh`, `ci/Dockerfile`, `modal_app.py`.
- openfhe-python is BSD 2-Clause: both license files ship in the wheel and NOTICE lists it.
- A rebuild of the same upstream version is published as `1.6.1.0.post1`; the `ombra` extras then need the pin loosened to `~=1.6.1.0`.
- CI builds OpenFHE once per package and platform (about twice the build time of one package, in parallel jobs).
