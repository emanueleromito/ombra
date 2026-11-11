# 0022. Local mode expands queries in Python with one extra level

- **Status:** proposed
- **Date:** 2026-11-11

## Context

Local mode (T3.7) runs the v1 service on the CPU, inside a child process of the client. The C++ server cannot search on the CPU, and openfhe-python cannot multiply a ciphertext by X^-k exactly, which the server's expansion tree (decision 0010) needs.

## Options considered

- **A. Python expansion: plaintext product and subgroup trace on the same keys.** No compiled code beyond the openfhe wheel. Uses 2 of the preset's 2 levels instead of 1 (filter headroom at level 2 still holds: 0019 cliffs 2^21 and 2^25 against an allowed 64 and 2,048) and is about 5x slower than the tree.
- **B. Compile the C++ OpenFHE expansion (`core/`) into the client.** Exact and the same level use as the server, but needs our own wheels.

## Decision

A for M3, local mode only, under three conditions set by the user: local query time at 10k vectors, d = 384 and 768, is measured (above 5 s at d = 768 means B now); the openfhe package is pinned to the server's OpenFHE 1.6.1 and local mode refuses envelopes of another OpenFHE version; a parity test shows local mode and the server give the same top-k on the same data. B is T4.1.

## Consequences

The server, the presets and their level budget are unchanged. Local mode results are checked against the server, not assumed. The client requires Python 3.12 until T4.1 (the openfhe wheel), and local-mode latency is not representative of the server.
