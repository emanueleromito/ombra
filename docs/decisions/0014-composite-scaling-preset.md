# 0014. Composite scaling preset with 32-bit limbs

- **Status:** proposed
- **Date:** 2026-11-09

## Context

GPUs multiply 32-bit integers natively, and OpenFHE 1.6 has a composite scaling mode that replaces each 50-bit tower by two 29-bit ones. T1.5 built the candidate `m1_composite_n16384` (N = 16384, depth 2, `COMPOSITESCALINGMANUAL`, composite degree 2, register word size 32, 128-bit classic security, log2(QP) = 301.99 against the limit of 438) and compared it with `m0_depth2` on an A100. Numbers and limits are in `docs/plan/results/2026-11-09-t1.5-composite.md`. Every GPU step matches OpenFHE word for word, and `m0_depth2` was not changed.

## Options considered

- **A. Adopt the candidate as the default preset.** Score error is 107x to 218x lower (4.6e-11 to 6.6e-11 against 6.7e-9 to 1.4e-8). But search throughput is the same within noise (2.355e7 against 2.150e7 vectors/s at d = 384, 1.194e7 against 1.144e7 at 768, 5.815e6 against 5.744e6 at 1536), because the kernel is limited by memory bandwidth and both presets read the same stored bytes. Expansion is 2.09x to 2.13x slower (38.28, 75.92, 151.06 ms against 18.33, 35.78, 70.83 ms) and automorphism keys are 2.0x larger (62.9 MB against 31.5 MB at d = 1536). Storage per vector is equal (147,456 B at d = 1536) only if the server keeps 4-byte words; OpenFHE's file format doubles it.
- **B. Reject; keep `m0_depth2`.** No gain in the measure that dominates the search, a loss in expansion and key size, and an API cost: under this mode the public `Rescale` does nothing, so the Python API cannot rescale explicitly.
- **C. Reject for now, revisit after a native 32-bit expansion path.** Expansion and key switching were left on 64-bit words in T1.5. Running them on 32-bit words would speed them up, but the 6-tower work is twice as large, so it would not be expected to beat `m0_depth2` by much. Not tested.

## Decision

Proposed: B, do not adopt. Keep `m0_depth2` as the default. Leave `m1_composite_n16384` and its GPU path (exact against OpenFHE, tested) in the tree as a candidate, not offered to users; no recall check has been run for it.

## Consequences

`m0_depth2` stays the baseline for T1.8. The T1.5 code stays: composite rescale in `GpuCtOps`, 32-bit search kernel, `tests/test_composite_feasibility.py`, the preset fields `composite_degree` and `register_word_size`. Anyone using the preset must encrypt the mask at level 2 and rescale through the scheme's internal call, not `CryptoContext::Rescale`. If a future step makes the search kernel compute-bound again (for example a much larger d per ciphertext), or if a lower-error preset is wanted, the numbers here are the starting point. A recall check (decision 0012) was not run for the candidate.
