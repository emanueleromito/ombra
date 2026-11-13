---
name: new-kernel
description: Add or change a CUDA kernel in core/ for Ombra, together with its OpenFHE oracle test and benchmark. Use whenever writing a new GPU kernel or changing an existing one.
---

# Add a CUDA kernel

Every kernel ships with three things: the kernel, an oracle test against OpenFHE, and a benchmark entry. Read `core/CLAUDE.md` first.

## Steps

1. **Define the operation in OpenFHE terms.** Write down which OpenFHE call(s) produce the same result on CPU (e.g. `EvalMult`, `EvalAdd`, `EvalRotate`). That is the oracle.
2. **Write the test first** in `core/tests/`:
   - parameters from `core/params/` only (at least 128-bit security);
   - small inputs (one or a few ciphertexts) so the test runs in seconds;
   - run the OpenFHE CPU operation and our GPU kernel on the same input ciphertexts;
   - compare the raw RNS polynomials exactly, and the decrypted values within the stated tolerance.
3. **Write the kernel** in `core/kernels/<family>.cu` with a host wrapper in `core/src/`. Above the kernel, document what it computes, the expected grid and block shape, and memory assumptions. Check every CUDA call with the error macro.
4. **Build for Ampere only** (`sm_80`, `sm_86`); do not change the architecture list.
5. **Run** `make test` then `make test-gpu` (see the run-on-gpu skill) until the test passes.
6. **Add a benchmark case** in `core/bench/` for the supported sizes (d = 384, 768, 1536) and run `make bench TASK=<task id>` on A100.
7. **Record results** in `docs/plan/results/` and update the task's status in `docs/plan/` (local maintainer folders; skip if absent).

## Never

- Merge a kernel without its oracle test.
- Loosen a comparison tolerance to make a test pass without explaining why in the PR.
- Touch the secret key in core code (tests may create keys through OpenFHE only to make inputs).
