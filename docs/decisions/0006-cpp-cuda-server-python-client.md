# 0006. C++/CUDA server and core, Python client, one repository

- **Status:** accepted
- **Date:** 2026-11-06

## Context

We need a language for the server, the GPU kernels and the client library.

## Options considered

- **A. C++/CUDA server and core, Python client.** One language for the whole performance path; OpenFHE used directly.
- **B. Rust server calling C++/CUDA.** Safer server code, but two languages and a bridge.
- **C. Go server calling C++/CUDA.** Easy servers, awkward C++ calls.
- **D. Python server with a C++/CUDA extension.** Fast to build, bottleneck in production.

## Decision

A. One repository with `core/`, `server/`, `client/` and `integrations/`. The client uses openfhe-python.

## Consequences

Memory-safety risk in C++ is reduced with sanitizers, fuzz testing and modern C++ practices. Rust is the option to revisit if the server or team grows.
