# 0002. Hybrid OpenFHE + CUDA stack on CKKS

- **Status:** accepted
- **Date:** 2026-10-31

## Context

We need trustworthy cryptography and fast GPU search. Our search is shallow (dot product, then mask: about two multiplication levels), so bootstrapping is not needed.

## Options considered

- **A. OpenFHE only.** Trusted and full-featured, but CPU-only.
- **B. Microsoft SEAL.** Simple, but CPU-only and less active.
- **C. An existing GPU library (e.g. Phantom).** Fast, but smaller projects; license and maintenance risk.
- **D. Lattigo.** Go; poor fit with CUDA and Python.
- **E. Our own full CKKS in CUDA.** Full control, but we own correctness and security of everything.
- **F. Hybrid.** OpenFHE for key generation, encryption and decryption; our own CUDA kernels for the server hot path; OpenFHE CPU as reference and fallback.

Scheme: CKKS (works on floats, approximate) vs BFV/BGV (exact, needs quantized embeddings).

## Decision

F with CKKS.

## Consequences

Our kernels must read and write OpenFHE's ciphertext format exactly, and every kernel is tested against OpenFHE CPU results. BFV with quantized embeddings may be tested in the spike as an alternative.
