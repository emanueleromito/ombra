# 0010. Query expansion via automorphisms

- **Status:** accepted; amends 0003
- **Date:** 2026-11-08

## Context

Decision 0003 expands one query ciphertext into d per-dimension ciphertexts on the server. Rotations alone cannot isolate one query value, and the usual 0/1 slot mask uses a multiplication level. The `m0_depth2` preset (0009) has two levels, both already used by the dot product and the filter mask, and T0.6 requires no parameter change.

## Options considered

- **A. Coefficient-packed expansion with automorphisms.** The query is packed in polynomial coefficients and split by substitutions, as in SealPIR's oblivious expansion. Uses no level and keeps the preset. More engineering, and the client encoding must be built on top of OpenFHE's plaintext and encryption calls.
- **B. Slot masking with a deeper preset.** Conventional, but needs depth 3, which changes the ring dimension and modulus budget and invalidates the T0.2 and T0.4 numbers.
- **C. Blocked packing.** The fallback in 0003. Gives up rotation-free search.

## Decision

A. The client builds q(X) = sum_j q_j X^j, computes its canonical embedding with numpy, encodes that with OpenFHE's `MakeCKKSPackedPlaintext` and encrypts it with OpenFHE, so hard rule 3 holds.

The server expands with a tree of substitutions X -> X^(N/2^k + 1) plus key switching, pruned at d outputs, using automorphism keys from OpenFHE's `EvalAutomorphismKeyGen`. Each output is a constant polynomial, so q_j lands in every slot.

The tree multiplies each value by 2^l. That factor is handled by adjusting the ciphertext scale, not by dividing the query on the client.

## Consequences

Expansion costs no multiplication level. It needs one automorphism key per tree level and adds key-switching noise, which the spike must measure.

Expanded ciphertexts carry a scale that differs from the stored ciphertexts by 2^l, so the search path must accept that difference; the T0.4 kernel wrapper currently rejects it.

When relinearization is added, relinearize and rescale lazily, once per output ciphertext after the d-term sum.

Blocked packing (0003, option D) stays the fallback if expansion time is too high.
