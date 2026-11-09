# 0015. GPU encoder: tolerance only for the FFT and rounding

- **Status:** proposed
- **Date:** 2026-11-09

## Context

The GPU CKKS encoder (`core/kernels/encode.cu`) repeats OpenFHE's floating-point FFT, rounds to integers, reduces them into every RNS tower and runs the NTT. Floating-point code cannot be promised word-identical to OpenFHE's, so the oracle in `ombra_plain_search` allowed 2 units of coefficient difference. That tolerance also covered the integer steps, and the NTT words were not compared at all once one coefficient differed (finding F1 of `docs/plan/results/m1-crypto-review.md`). Hard rule 3 and `core/CLAUDE.md` ask for exact matches on integer RNS arithmetic.

## Options considered

- **A. Keep one tolerance for the whole encoder.** Simple, but a wrong reduction or NTT word of one unit passes.
- **B. Split the oracle.** Exact comparison for reduction and NTT on OpenFHE's own rounded integers, and for the GPU evaluation form against OpenFHE's NTT of the GPU's own coefficients; a tolerance only for the FFT and rounding.
- **C. Require word-identical output everywhere.** Not achievable: at `m0_depth2` a few coefficients differ by one unit.

## Decision

Option B. The only tolerance is `COEF_TOLERANCE = 2` units on the centered difference between GPU and OpenFHE coefficients (1 unit = 1/scale in slot space), and it applies to the FFT and rounding only. Measured on the A10 with 16 synthetic columns: `m0_depth2` max difference 1 unit (27 to 30 of 786432 coefficient words differ, f64, fp16 and int8); `m0_compact_n8192` max difference 0 (all 393216 words identical). So the margin is 1 unit at `m0_depth2` and 2 units at `m0_compact_n8192`. Reduction, both NTT comparisons and a check that the GPU's coefficient words are one integer in every tower have no tolerance. A test-only fault injection in the driver (one wrong reduction word, one wrong NTT word beside a one-unit coefficient difference, or one tower off by one unit) checks that the oracle fails, for both presets and all three formats.

## Consequences

- A defect in tower reduction or the NTT fails the test even when coefficients are within tolerance.
- The tolerance can be lowered to 1 if no preset ever shows 2; it was not, because only these two presets and one data seed were measured.
- A wider tolerance needs a new decision.
