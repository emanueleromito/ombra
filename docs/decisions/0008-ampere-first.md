# 0008. Target NVIDIA Ampere only for the first version

- **Status:** accepted
- **Date:** 2026-11-07

## Context

Supporting every GPU generation from day one slows builds and testing. We develop and benchmark on Modal.

## Decision

Compile only for `sm_80` (A100) and `sm_86` (A10G). Develop on A10G, benchmark on A100. Ada, Hopper and Blackwell come later.

## Consequences

Builds stay fast and Ampere-specific features can be used freely. Adding generations later means extending `CMAKE_CUDA_ARCHITECTURES` and re-running the GPU test suite on each.
