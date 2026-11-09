# 0017. Position on decryption noise (IND-CPA-D)

- **Status:** accepted
- **Date:** 2026-11-09

## Context

Decision 0010 calls "noise flooding on decryption" the mitigation against key recovery from decryption results, but the presets do not enable it (finding F9, `docs/plan/results/m1-crypto-review.md`). OpenFHE's default, `FIXED_NOISE_DECRYPT`, is what runs: `Decode` adds Gaussian noise sized from an estimate, and OpenFHE's own comment says this needs at least 128, and in practice closer to 10,000, decryption queries to average out, before a further attack. The threat model already lets the server see which clusters and document IDs the client picks, and those choices are functions of decrypted scores. That leak is ordinal only. The review knows of no practical attack from it, and nobody has analysed it. The behaviour also relies on `ckksDataType` staying at its default, `REAL`, which nothing pins.

## Options considered

- **A. Keep `FIXED_NOISE_DECRYPT`, state its limits, pin it.** No cost. The presets pin `REAL` and the decryption mode, a test shows that two decryptions of one ciphertext differ, and a rule says decrypted values never leave the client. The ordinal leak stays, stated and unanalysed.
- **B. Adopt `NOISE_FLOODING_DECRYPT`.** The stronger mitigation, but it runs computations twice to estimate the noise, and it adds noise whose effect on latency and recall has not been measured here.
- **C. Hold M2 until the ordinal leak has been analysed externally.** The most cautious, and it blocks work on a leak with no known attack.

## Decision

A, and correct 0010's wording, since flooding is not what runs. Ask for an outside analysis of the ordinal leak before the product is offered where a client's choices can be observed by an active attacker. Revisit B if that analysis finds a practical attack.

## Consequences

Accepted 2026-11-09 (option A). Not implemented yet: T1.17 still needs the two preset pins (`m0_depth2` and `m0_compact_n8192`, with the same values they use by default today), the two-decryptions test, the rule in `CLAUDE.md`, and the 0010 correction. The product must not claim IND-CPA-D security.
