# 0010. Query expansion via automorphisms

- **Status:** accepted; amends 0003
- **Date:** 2026-11-08

## Context

Decision 0003 expands one query ciphertext into d per-dimension ciphertexts on the server. Rotations alone cannot isolate one query value, and the usual 0/1 slot mask uses a multiplication level. The `m0_depth2` preset (0009) has two levels, both already used by the dot product and the filter mask, and T0.6 requires no parameter change.

## Options considered

- **A. Coefficient-packed expansion with automorphisms.** The query is packed in polynomial coefficients and split by substitutions, as in SealPIR's oblivious expansion. Uses no level and keeps the preset. More engineering, and the client encoding must be built on top of OpenFHE's plaintext and encryption calls.
- **B. Slot masking with a deeper preset.** Conventional, but needs depth 3, which changes the ring dimension and modulus budget and invalidates the T0.2 and T0.4 numbers.
- **C. Blocked packing.** The fallback in 0003. Gives up rotation-free search.

Complex slots, which would allow an unsymmetric query, were also rejected. OpenFHE needs `CKKSDataType::COMPLEX` for them, which changes the preset and disables noise flooding on decryption, the mitigation against key recovery from decryption results (`gen-cryptocontext-params.h`, "Noise flooding is only enabled for real values").

## Decision

A, with a symmetric real query. In OpenFHE's default REAL mode the encoded polynomial is real-slotted: it keeps Re(q(z)), which is (q(X) + q(X^-1))/2. So the client encodes p(X) = q_0 + sum_{j>=1} q_j (X^j + X^-j), where X^-j = -X^(N-j): coefficient j is q_j and coefficient N-j is -q_j. The client computes the real canonical embedding of p with numpy, encodes it with OpenFHE's `MakeCKKSPackedPlaintext` and encrypts with OpenFHE, so hard rule 3 holds. The preset stays unchanged.

The server expands with a tree of substitutions X -> X^(N/2^k + 1) plus key switching, of depth L = ceil(log2(2d)), pruned to the d leaves for j = 0..d-1, using automorphism keys from OpenFHE's `EvalAutomorphismKeyGen`. Each leaf is a constant polynomial, so q_j lands in every slot. L needs 2^L >= 2d, one level more than an unsymmetric query, because the tree must separate j from N-j.

The tree multiplies each value by 2^L. The client removes that factor by multiplying q by 2^-L before encoding; the reasoning and the level budget are in decision 0009 (addendum).

## Consequences

Expansion costs no multiplication level. It needs one automorphism key per tree level, one more level than an unsymmetric query, and adds key-switching noise, which the spike must measure.

Expanded ciphertexts have the same scale, level and degree as a fresh query, so the T0.4 search path takes them unchanged.

When relinearization is added, relinearize and rescale lazily, once per output ciphertext after the d-term sum.

Blocked packing (0003, option D) stays the fallback if expansion time is too high.
