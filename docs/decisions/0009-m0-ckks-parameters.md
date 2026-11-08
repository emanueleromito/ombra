# 0009. CKKS parameter preset for the M0 spike

- **Status:** accepted
- **Date:** 2026-11-08

## Context

Every M0 task needs one fixed CKKS parameter set, at no less than 128-bit security (hard rule 1), with two multiplication levels for the dot product and filter mask (decision 0002). The preset also has to leave enough slots for column packing and for rotating queries of up to 1536 dimensions (decision 0003).

## Options considered

- **A. Ring dimension 8192.** Cheaper, but the 128-bit limit on log2(QP) is 218, which leaves little room at depth 2 once the key-switching primes are counted, and only 4096 slots per ciphertext.
- **B. Ring dimension 16384.** Twice the work per ciphertext, 8192 slots, and a 438-bit limit that leaves a wide margin.
- **C. Ring dimension 32768.** More slots, but twice the cost again with no need at depth 2.

Scaling: FIXEDMANUAL (we rescale explicitly and control the ciphertext level, which our kernels must match) versus OpenFHE's automatic modes (ciphertext state changes implicitly, which is harder to reproduce on the GPU).

## Decision

B: preset `m0_depth2` in `core/params/ckks_presets.json`, using HEStd_128_classic, ring dimension 16384, multiplicative depth 2, scaling modulus 50 bits, first modulus 60 bits, batch size 8192, FIXEDMANUAL scaling and HYBRID key switching.

## Consequences

The measured log2(QP) is 280, below the 438-bit limit, and a test asserts this on every `make test-openfhe` run. FIXEDMANUAL means callers must rescale explicitly. The product presets ("fast", "balanced", "precise") are still to be defined and may differ.
