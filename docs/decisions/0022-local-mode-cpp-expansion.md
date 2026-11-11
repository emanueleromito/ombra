# 0022. Local mode uses the server's C++ expansion

- **Status:** proposed
- **Date:** 2026-11-11

## Context

Local mode (T3.7) runs the v1 service on the CPU, inside a child process of the client. The C++ server cannot search on the CPU, and openfhe-python cannot multiply a ciphertext by X^-k exactly, which the server's expansion tree (decision 0010) needs.

## Options considered

- **A. Python expansion: plaintext product and subgroup trace on the same keys.** No compiled code beyond the openfhe wheel. Uses 2 of the preset's 2 levels instead of 1 (filter headroom at level 2 still holds: 0019 cliffs 2^21 and 2^25 against an allowed 64 and 2,048) and is about 5x slower than the tree.
- **B. Compile the C++ OpenFHE expansion (`core/`) into the client.** Exact and the same level use as the server, but needs our own wheels.

## Decision

B, chosen by the user on 2026-11-11 after A measured 27.4 s per search at d = 768 (Modal, 4 vCPU, compact preset, 12 vectors: expansion alone), against a 5 s target. Local mode compiles the C++ OpenFHE expansion into the client (Linux x86_64, Python 3.12 for M3; the full wheel matrix is T4.1), defaults to `m0_compact_n8192`, pins OpenFHE 1.6.1 and refuses envelopes of another version, and is checked against the server by a parity test. Target: under 5 s per search at 10k vectors, d = 768, 4 CPUs; if missed, the user decides. There is no plaintext mode.

## Consequences

The server, the presets and their level budget are unchanged, and local mode uses the same levels as the server. Installing `[local]` needs our wheel (or a compiler with OpenFHE 1.6.1) until T4.1. The client requires Python 3.12 until T4.1.
