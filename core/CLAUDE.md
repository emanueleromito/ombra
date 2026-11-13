# core/ rules

C++20 and CUDA. The GPU hot path: kernels plus code that reads and writes OpenFHE's ciphertext format. Everything here is performance-critical and correctness-critical.

## Layout (as it grows)

```
core/
  include/ombra/core/   public headers
  src/                  host code (format I/O, kernel wrappers)
  kernels/              CUDA kernels, one family per .cu file
  params/               CKKS parameter presets (the only place parameters are defined)
  tests/                C++ tests (GoogleTest), each compared with OpenFHE CPU
  bench/                benchmarks, run only via `make bench`
```

## Rules

- **Every kernel has an oracle test.** Same inputs through OpenFHE on CPU and through our kernel; results must match exactly for integer RNS arithmetic, and within a stated tolerance after decryption. No kernel without one.
- **Match OpenFHE's representation.** Same ring dimension, RNS moduli order, NTT form and scaling. When in doubt, read OpenFHE's source; do not guess.
- **No secret key here.** Core code never touches the secret key. Tests may create keys through OpenFHE to produce test ciphertexts, and must not pass them into core APIs.
- **Parameters only from `params/`.** Never hard-code ring sizes or moduli in kernels or tests.
- **Architectures (decision 0024):** `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`. Use nothing a listed architecture lacks; a kernel that wants an Ampere-or-later feature needs a fallback for `sm_75`.
- **Check every CUDA call** with the project error macro; check kernel launch errors in debug builds.
- **Document each kernel:** a comment above it with what it computes, the expected grid/block shape and memory assumptions.
- **Performance changes need numbers.** A PR that claims a speedup includes a before/after `make bench` result.

## Testing

`make test-gpu` runs these tests on a Modal A10. Keep default test sizes small (one or a few ciphertexts) so a run takes seconds, not minutes.
