# M0: Technical spike

**Goal:** prove the hybrid OpenFHE + CUDA design works on Ampere, and get real numbers for query expansion, search throughput, batching, accuracy and sizes. Spike code may be rough, but every number must come from a reproducible command.

**GPUs:** A10 for development (`make test-gpu`), A100 for benchmarks (`make bench`).
**Vector dimensions:** 384, 768, 1536.
**Collection sizes:** 100k, 1M (10M only if 1M looks good).
**Datasets:** synthetic normalized random vectors first; then a real embedding set (e.g. a public text-embedding benchmark set) for recall.

**Order and parallel work:**

```
T0.1 -> T0.2 -> T0.3 -> T0.4 + T0.5 (in parallel)
                        T0.4 done       -> T0.6 + T0.7 (in parallel)
                        T0.5 done       -> T0.5b
                        T0.4, T0.5 done -> T0.8 + T0.9 (in parallel)
                        T0.9 done       -> T0.11
                        all done        -> T0.10
```

T0.5b and T0.7 both touch the expansion code: run them one after the other, not in parallel.

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
**Steps:** per decision 0010: encode the query as the symmetric real polynomial q_0 + sum_{j>=1} q_j (X^j + X^-j) on the client through OpenFHE; run the expansion tree of substitutions X -> X^(N/2^k + 1) and key switching, depth L = ceil(log2(2d)), pruned to the d leaves for j = 0..d-1, first on CPU with OpenFHE and then on the GPU, using keys from `EvalAutomorphismKeyGen`; the client multiplies q by 2^-L before encoding, to cancel the tree's 2^L factor (decision 0009, addendum); time it and measure the automorphism-key count and size and the decrypted error.
**Done when:** expansion matches OpenFHE CPU, and expansion time and rotation-key size are recorded for d = 384, 768, 1536 on A100.
**Status:** done

### T0.6 Filter mask

**Why:** confirms masks fit in the depth budget.
**Depends on:** T0.4
**Steps:** multiply search scores by an encrypted 0/1 mask; check the result still decrypts correctly at depth 2.
**Done when:** masked scores match plaintext expectations within tolerance, with no parameter change from T0.2.
**Status:** done

### T0.7 Query batching

**Why:** batching is the main throughput lever (decision 0007).
**Depends on:** T0.4, T0.5 (and after T0.5b, which touches the same expansion code)
**Steps:**
- Search: run 1, 8, 32 and 128 queries in one pass over the same stored ciphertexts.
- Expansion: run the expansion trees of 1, 8, 32 and 128 queries together, batching their splits so the GPU stays busy. Latency per query may not drop, but throughput should.
- Report the achieved memory bandwidth of the search kernel against the A100's peak, to show how much headroom is left.
**Done when:** search throughput and expansion throughput (queries per second) per batch size, and the search kernel's achieved bandwidth, are recorded on A100, showing the speedup over one query at a time.
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
**Steps:** for d = 384, 768 and 1536, and for both modes (plaintext DB and encrypted DB), measure storage per million vectors, bytes per dimension per vector, query upload size, score download size and evaluation-key size. Compare with a plain float32 vector DB (4 bytes per dimension). The expected figure at `m0_depth2` is about 96 bytes per dimension per vector for encrypted DB (2 polynomials x 3 towers x 8 bytes over N/2 slots); confirm or correct it.
**Done when:** a size table is recorded in `results/`, including the ratio to float32 and how many vectors fit in A100 memory per mode and dimension.
**Status:** todo

### T0.5b Faster query expansion

**Why:** at d = 1536 one expansion takes 0.63 s on A100, as much as scanning about 2.5M vectors (T0.4), so it dominates per-query latency. Splits currently run one at a time.
**Depends on:** T0.5
**Steps:** run all splits of the same tree level in one batched launch (nodes at a level are independent); fuse the automorphism with the key switch where possible; profile one split to find the bottleneck (NTT, key-switch inner product, ModDown). Keep the CPU tree as the exact RNS-level oracle.
**Done when:** the oracle still matches exactly, and expansion time on A100 is recorded for d = 384, 768, 1536, with a target under 100 ms at d = 1536. If the target is missed, the results file says why and what limits it.
**Status:** todo

### T0.11 Compact configurations

**Why:** storage is set by the number of RNS towers (each stored as a 64-bit word), not by bit sizes, so `m0_depth2` costs about 24x a float32 vector DB. This task measures the options that make Ombra smaller or faster, without changing `m0_depth2`, which stays the baseline for every other task.
**Depends on:** T0.5, T0.9
**Steps:**
1. **Smaller ring:** add a second preset at ring dimension 8192 with a ~40-bit scale, still depth 2. Check log2(QP) against the HomomorphicEncryption.org 128-bit classic table (218 bits at N = 8192) in a test. Measure expansion time, key size and end-to-end score error (expansion, search, mask) against the 1e-6 bound.
2. **Compact plaintext storage (plaintext-DB mode):** store column-packed plaintext vectors as fp16 and as int8, and encode them on the GPU (FFT and NTT into RNS form) just before the multiply. Measure storage per vector, search throughput against T0.4 and score error.
3. **Seeded encryption (encrypted-DB mode), write-up only:** with secret-key encryption, half of each ciphertext is randomness that could be regenerated from a seed, halving storage. Find out whether OpenFHE supports seeded or compressed ciphertexts, and what reproducing its random generator on the GPU would take. No implementation: it touches hard rule 3.
**Done when:** results for steps 1 and 2 are recorded in `results/`, step 3 is a short note in `results/`, and any preset that passes is proposed as a new decision record (not adopted).
**Status:** todo

### T0.10 Spike report

**Why:** decides whether the architecture stands as designed.
**Depends on:** T0.1–T0.9, T0.5b, T0.11
**Steps:** summarize all results; compare with the blocked-packing fallback if expansion is slow; recommend the product presets based on T0.11; list any design changes.
**Done when:** `results/spike-report.md` exists with the numbers, a go / change decision for each architecture choice, and any new decision records added to `docs/decisions/`.
**Status:** todo
