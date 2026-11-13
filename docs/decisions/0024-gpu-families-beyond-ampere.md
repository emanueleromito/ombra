# 0024. GPU families beyond Ampere (update to 0008)

- **Status:** proposed
- **Date:** 2026-11-13

## Context

Decision 0008 compiles for `sm_80` and `sm_86` only and says Ada, Hopper and Blackwell come later. Research experiment 0008 (`docs/research/0008-gpu-families/`) built the tree once for `sm_75`, `sm_80`, `sm_86`, `sm_89` and `sm_90` with the pinned CUDA 12.6.3 image and ran the same binaries on a T4, an A10, an A100, an L4, an L40S and an H200. Every oracle test passed on every card (146 passed, 0 failed each) with no source change. Expansion at d = 768: T4 177 ms, L4 114 ms, A10 74 ms, L40S 38 ms, A100 36 ms, H200 19 ms. Per dollar of GPU time the six cards are within 1.4x of each other.

## Options considered

- **A. Stay Ampere only.** Nothing to maintain, but users on Ada or Hopper clouds cannot run the released binaries, and the fastest card measured (Hopper, 1.9x the A100) is excluded.
- **B. Add Ada and Hopper (`sm_89`, `sm_90`) to the default build.** Covers L4, L40S, RTX 40, H100 and H200. Same source, same image. Costs build time for two more architectures and a GPU test run on one Ada and one Hopper card before a release.
- **C. B plus Turing (`sm_75`).** Also covers the T4, the cheapest card on most clouds. It passes every test, but expansion takes 351 ms at d = 1536 and it has 16 GB of memory: usable for trials and small collections, not for the latency targets.
- **D. C plus Blackwell (`sm_100`, `sm_120`).** Needs CUDA 12.8 or later: a new pinned image and an OpenFHE rebuild, and nothing was measured. Not now.

## Decision

Proposed: **C.** Build `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90` by default, with two support levels:

| Level | Cards | Meaning |
| --- | --- | --- |
| Reference | A10 (tests), A100 (benchmarks) | Development and published numbers, as today |
| Supported | L40S, H100/H200, L4 | Full GPU test suite on one Ada and one Hopper card before each release |
| Works, not a target | T4 | Built and tested before a release; no latency target |

Until this is accepted the defaults stay Ampere only; the wider build is available through `make prebuild` and `make on-gpu GPU=...` (`modal_app.py`, `ALL_CUDA_ARCHS`).

## Consequences

- Hard rule 7 in `CLAUDE.md` and the "Ampere only" rule in `core/CLAUDE.md` would read "the five architectures of 0024"; "Ampere features are fine to use" becomes "use nothing a listed architecture lacks" (today's kernels already comply: 16 KB of shared memory per block, no architecture-specific intrinsics).
- A kernel that wants an Ampere-or-later feature needs a fallback for `sm_75`, or Turing leaves the list.
- Release testing grows by three short GPU runs (about 3 minutes each on prebuilt binaries; about USD 0.50 in total at today's Modal prices).
- `CMAKE_CUDA_ARCHITECTURES` default, `CUDA_ARCHS` in the Makefile and in `modal_app.py` change from `80;86` to `75;80;86;89;90` when accepted.
- Blackwell stays open; it needs its own decision with the image change.
