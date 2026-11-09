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

## Addendum (2026-11-09): the 2^L expansion factor and the level budget

The expansion tree of decision 0010 multiplies every value by 2^L, L = ceil(log2(2d)). The client removes that factor by multiplying q by 2^-L before encoding. The server does no multiplication and no scale change during expansion. Two alternatives were rejected.

**Adjusting the ciphertext scale does not work under FIXEDMANUAL.** Multiplying the ciphertext's scaling factor by 2^L only changes metadata. On decryption under FIXEDMANUAL, `CKKSPackedEncoding::Decode` (OpenFHE v1.6.1, `ckkspackedencoding.cpp:315-361`) divides by powers of 2^p fixed by the preset and by `noiseScaleDeg`; it reads the ciphertext's scaling factor only in the FLEXIBLEAUTO, FLEXIBLEAUTOEXT and composite-scaling branches. The leaves would still decrypt to 2^L * q_j.

**A server-side EvalMult by 2^-L costs the mask's level.** `EvalMult` by a double raises `noiseScaleDeg` by one (`ckksrns-leveledshe.cpp:855`), so a leaf leaves expansion at degree 2 and must be rescaled, which uses one of the preset's two levels. Query x stored then uses the other, and nothing remains for the filter mask (decision 0005) without moving to depth 3, which T0.6 forbids.

**Level budget at `m0_depth2`:**

| Step | Levels used |
| --- | --- |
| Query expansion | 0 |
| Query x stored (column-packed sum, then rescale) | 1 |
| Filter mask (then rescale) | 1 |
| **Total** | **2** |

The expansion driver (`core/src/ombra_expand.cpp`) refuses any leaf not at level 0 and degree 1, and `tests/test_query_expansion.py` asserts that the masked score has used exactly 2 levels. Measured errors are in `docs/plan/results/`.
