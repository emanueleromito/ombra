# M0: Technical spike

**Goal:** prove the hybrid OpenFHE + CUDA design works on Ampere, and get real numbers for query expansion, search throughput, batching, accuracy and sizes. Spike code may be rough, but every number must come from a reproducible command.

**GPUs:** A10 for development (`make test-gpu`), A100 for benchmarks (`make bench`).
**Vector dimensions:** 384, 768, 1536.
**Collection sizes:** 100k, 1M (10M only if 1M looks good).
**Datasets:** synthetic normalized random vectors first; then a real embedding set (e.g. a public text-embedding benchmark set) for recall.

**Order and parallel work:**

```
T0.1 -> T0.2 -> T0.3 -> T0.4 + T0.5 (in parallel)
                        T0.4 done      -> T0.6 + T0.7 (in parallel)
                        T0.4, T0.5 done -> T0.8 + T0.9 (in parallel)
                        all done        -> T0.10
```

---

### T0.1 Modal environment

**Why:** everything else runs on it.
**Depends on:** none
**Steps:** build the Modal image from `modal_app.py` (CUDA devel base, OpenFHE and openfhe-python from pinned tags); run `nvidia-smi`, `nvcc --version` and an openfhe-python import on A10.
**Done when:** `make gpu-info` prints the A10, the CUDA version and the OpenFHE version, and the image builds from scratch without manual steps.
**Status:** done

### T0.2 CKKS parameters and CPU baseline

**Why:** fixes the parameter set every later task uses, and gives the CPU reference numbers.
**Depends on:** T0.1
**Steps:** pick a CKKS parameter set (ring dimension, scaling, depth 2) that gives at least 128-bit security under OpenFHE's security tables; in Python, encrypt, multiply-add, decrypt; time each step on CPU.
**Done when:** a test asserts the security level is at least 128 bits, a CPU round trip matches plaintext within a stated tolerance, and CPU timings are written to `results/`.
**Status:** done

### T0.3 Ciphertext round trip through CUDA

**Why:** proves our kernels can read and write OpenFHE's format, the base of the hybrid design.
**Depends on:** T0.2
**Steps:** serialize an OpenFHE ciphertext in Python; load it in C++; copy its polynomials to the GPU; run a trivial kernel (ciphertext + ciphertext, then ciphertext × plaintext); copy back; decrypt in Python.
**Done when:** decrypted GPU results match OpenFHE CPU results on the same inputs within tolerance, in an automated test run by `make test-gpu`.
**Status:** done

### T0.4 Column-packed search kernel

**Why:** the hot path of every query.
**Depends on:** T0.3
**Steps:** pack N vectors column-wise (one ciphertext per dimension, many vectors per ciphertext); implement multiply-and-add over d dimensions on the GPU with a pre-expanded query; compare with OpenFHE CPU.
**Done when:** scores match OpenFHE CPU within tolerance, and throughput (vectors scored per second) on A100 is recorded for d = 384, 768, 1536.
**Status:** done

### T0.5 Query expansion

**Why:** the main open risk of decision 0003.
**Depends on:** T0.3
**Steps:** per decision 0010: encode the query as coefficients on the client through OpenFHE; on the GPU, expand one query ciphertext into d per-dimension ciphertexts with a tree of substitutions X -> X^(N/2^k + 1) and key switching, pruned at d outputs, using keys from `EvalAutomorphismKeyGen`; adjust the ciphertext scale for the 2^l factor; time it and measure the automorphism-key size.
**Done when:** expansion matches OpenFHE CPU, and expansion time and rotation-key size are recorded for d = 384, 768, 1536 on A100.
**Status:** todo

### T0.6 Filter mask

**Why:** confirms masks fit in the depth budget.
**Depends on:** T0.4
**Steps:** multiply search scores by an encrypted 0/1 mask; check the result still decrypts correctly at depth 2.
**Done when:** masked scores match plaintext expectations within tolerance, with no parameter change from T0.2.
**Status:** todo

### T0.7 Query batching

**Why:** batching is the main throughput lever (decision 0007).
**Depends on:** T0.4
**Steps:** run 1, 8, 32 and 128 queries in one pass over the same stored ciphertexts.
**Done when:** throughput per batch size is recorded on A100, showing the speedup over one query at a time.
**Status:** todo

### T0.8 Accuracy

**Why:** CKKS noise and clustering both cost recall.
**Depends on:** T0.4, T0.5
**Steps:** on a real embedding set, compare encrypted top-k with plaintext top-k: full scan, then with k-means clusters (faiss) and different numbers of clusters searched, with and without client re-ranking.
**Done when:** recall@10 for each setting is recorded in `results/`.
**Status:** todo

### T0.9 Sizes

**Why:** feeds the sizing guide and the pitch.
**Depends on:** T0.4, T0.5
**Steps:** measure storage per million vectors, query upload size, score download size and evaluation-key size for each dimension.
**Done when:** a size table is recorded in `results/`.
**Status:** todo

### T0.10 Spike report

**Why:** decides whether the architecture stands as designed.
**Depends on:** T0.1–T0.9
**Steps:** summarize all results; compare with the blocked-packing fallback if expansion is slow; list any design changes.
**Done when:** `results/spike-report.md` exists with the numbers, a go / change decision for each architecture choice, and any new decision records added to `docs/decisions/`.
**Status:** todo
