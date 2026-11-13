# 0023. Seeded stored ciphertexts

- **Status:** accepted
- **Date:** 2026-11-13

## Context

A stored column ciphertext is a pair `(c0, c1)` of polynomials and costs 96 B per vector per column on disk and on upload (T0.9). The half `c1` carries no information by itself: under RLWE it is indistinguishable from uniform. If it is generated from a short public seed, only `c0` and the seed need to be stored and sent, which halves both. T0.11 described the idea and stopped at two obstacles: OpenFHE has no seeded ciphertext form, and hard rule 3 says encryption uses OpenFHE only. The user asked for seeded ciphertexts on 2026-11-13 and chose the options marked below.

## Options considered

**Who builds the seeded half**

- **A. Ombra replaces the random half on the client (chosen by the user).** OpenFHE encodes, samples the noise and encrypts as today, giving `(c0, c1)`. The client draws a fresh 256-bit seed, expands it to a uniform polynomial `u`, and outputs `c0' = c0 + (c1 - u) * s` with the seed, where `s` is the secret key. Then `c0' + u * s = c0 + c1 * s`: the pair `(c0', u)` decrypts to exactly the same words as OpenFHE's ciphertext. The message, the noise and the key are OpenFHE's; Ombra adds one polynomial multiplication with the secret key, done with OpenFHE's polynomial type.
- **B. A seeded random-generator plug-in for OpenFHE.** Keeps rule 3 to the letter, but OpenFHE draws the noise and the random half from one generator. With a public seed the noise would be derivable from stored data, which breaks the encryption; avoiding that depends on OpenFHE's internal call order. Rejected.

**Where the server regenerates `u`**

- **A. On the CPU when a block is loaded (chosen).** Disk and upload halve. GPU memory, every kernel and every oracle test are unchanged.
- **B. On the GPU inside the search.** GPU memory would also halve, at the price of a hash kernel on the hot path. Left for a later decision, after A is measured.

## Decision

Option A for both.

**Exception to hard rule 3, stated narrowly.** Key generation, encoding, noise sampling, encryption and decryption stay OpenFHE's. Ombra performs one step itself, on the client only: the replacement of `c1` by a seeded uniform polynomial with the matching correction of `c0`. The test for that step is exact, not statistical: for the same secret key, `c0' + u * s` equals `c0 + c1 * s` word for word, and OpenFHE decrypts the expanded pair.

**Noise form.** OpenFHE encrypts with the public key, so the noise of the stored pair is `e0 + v*e_pk + e1*s`, where `v`, `e0`, `e1` are fresh samples and `e_pk` is the noise of the public key. It depends on the secret key `s`. This is not the textbook seeded form, in which the noise is sampled independently of the key (SEAL encrypts with the secret key directly). The argument below covers the uniform mask and the seed. It does not by itself cover this noise form. No attack is known, since the noise is never seen apart from the message. The form is in the scope of the external cryptographic review.

**Why it is safe.** `(u, c0')` with `c0' = m + e - u * s` is a secret-key RLWE sample whose mask is uniform and whose noise `e` is the one OpenFHE sampled; publishing the seed of a uniform mask is the standard seeded form (SEAL's seeded ciphertexts, the public matrix of Kyber). It needs the generator to be indistinguishable from random and **a seed never used twice**: two ciphertexts with the same `u` under one key reveal the difference of their plaintexts up to noise. Seeds come from the operating system's generator, 256 bits, one per ciphertext. The original `c1` is discarded and never leaves the client.

**Generator.** BLAKE2xb as shipped inside OpenFHE (`blake2xb`, already linked by client and server; no new dependency). For tower `t` with modulus `q` of bit length `n`, block `c = 0, 1, ...` is `BLAKE2xb(out = 4096 bytes, in = "ombra-seeded-v1" || seed || u32 t || u64 c)`, read as little-endian 64-bit words; a word masked to `n` bits is kept if it is below `q` (rejection sampling), and the first N kept words are the tower, in evaluation form. A fixture pins this byte for byte.

**Format.** The envelope (0016) gets **version 2**: version 1 plus kind 4, "seeded ciphertext". Kinds 1 to 3 are still written as version 1, so existing files do not change. Payload of kind 4, little-endian:

| Size | Field |
| --- | --- |
| 2 | payload version, u16 (1) |
| 2 | generator, u16 (1 = BLAKE2xb as above) |
| 32 | seed |
| 4 | ring dimension N, u32 |
| 2 | towers T, u16 |
| 4 | level, u32 |
| 4 | noise scale degree, u32 |
| 4 | slots, u32 |
| 8 | scaling factor, IEEE double |
| 2 + K | key tag length K, u16, and the key tag |
| 8 * T | moduli, u64 each |
| 8 * T * N | `c0'`, [tower][coefficient], evaluation form, reduced words |

Readers refuse an unknown payload version or generator, a size that does not match N and T, an unreduced word, and trailing bytes. The server checks ring dimension, moduli, level, noise degree, scale and key tag against the collection, as it does for an OpenFHE column today. A collection may hold both kinds.

**Who writes it.** The step needs polynomial arithmetic with the secret key, which openfhe-python does not expose, so it lives in the compiled client module (`ombra._native`). A client without the module keeps writing kind 2, and refuses to read kind 4 with a message that names the module.

## Consequences

- Disk and upload per stored vector halve (about 49 KB to 25 KB at d = 384); to be measured.
- Loading a block no longer runs OpenFHE's deserializer per column (3.3 ms each at `m0_depth2`, experiment 0005); it runs BLAKE2xb instead. To be measured.
- GPU memory, search latency and scores are unchanged: the device holds the same words as before.
- The client's compiled module handles the secret key (it already runs in the client's process; the server still never sees it, hard rule 2).
- Rule 3 in `CLAUDE.md` names this exception (added when the user accepted the decision on 2026-11-13).
- Seed reuse is a break, so the seed source and its test are part of the crypto review surface.
- Follow-ups not decided here: regenerating on the GPU (option B above), and packing the 50 to 60-bit words of `c0'` instead of storing 64 bits each (about 17% more).
