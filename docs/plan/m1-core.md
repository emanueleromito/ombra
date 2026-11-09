# M1: Core library

**Goal:** turn the spike into a reviewed, tested `core/` library that runs a full encrypted query on the GPU, end to end, and measure its real latency. Then make it faster and decide the product presets.

**Starting point:** the M0 spike report (`results/spike-report.md`). Baseline preset stays `m0_depth2` (decision 0009) unless a task here proposes a change through a decision record.
**GPUs:** A10 for tests and oracles, A100 for benchmarks.
**Vector dimensions:** 384, 768, 1536.

**Order and parallel work:**

```
T1.1 (crypto review) ----------------------------------------------+
T1.2 -> T1.3 (end-to-end latency)                                  |
T1.4 (profile search) -> T1.5 (composite scaling preset)           +-> T1.8 (M1 report)
T1.6 (recall-based check) -> T1.7 (compact preset recall)          |
```

T1.1, T1.2, T1.4 and T1.6 can start in parallel. T1.2 and T1.4 both touch the search path: keep T1.4 to profiling and measurement until T1.2 is merged, so they don't edit the same kernels at once.

---

### T1.1 Independent crypto review

**Why:** M0 merged 18 PRs in a day. Expansion, key switching, encoding and parameter handling are where a subtle bug would break security or correctness, and M1 builds on them.
**Depends on:** none
**Steps:** a reviewer with no part in writing the M0 code (a fresh agent session or subagent, started without the M0 conversation) reads all of `core/`, `client/` and the tests against the hard rules in `CLAUDE.md`. Check in particular: parameters only from `core/params/` and at least 128-bit; no secret key outside the client; every kernel has an exact OpenFHE oracle test, and the oracle really compares RNS words rather than only decrypted values; no plaintext logged or written outside synthetic tests; format version headers; correct handling of OpenFHE's level, noise-scale degree and scaling factor metadata. The reviewer writes findings only and changes no code.
**Done when:** `results/m1-crypto-review.md` lists every finding with file, line, severity and a suggested fix, and each finding of severity high or medium has an open follow-up task or a fix PR.
**Status:** todo

### T1.2 GPU relinearize, rescale and ciphertext mask

**Why:** these steps still run on the CPU through OpenFHE. They are the largest gap on the search path (spike report, "Next work for M1").
**Depends on:** none
**Steps:** GPU kernels for relinearization (reusing the key-switching code from T0.5b), rescale, and ciphertext x ciphertext multiply for the filter mask. Relinearize and rescale lazily, once per output ciphertext after the d-term sum. Each kernel gets an exact OpenFHE oracle test at RNS level, including level, noise-scale degree and scaling factor.
**Done when:** all three kernels match OpenFHE exactly for d = 384, 768, 1536 on the A10, and a full search (multiply-and-add, relinearize, rescale, mask, rescale) runs on the GPU only, with its score error recorded against the T0.6 results.
**Status:** todo

### T1.3 End-to-end query latency

**Why:** the spike measured pieces, never a full query. This is the number the product pitch and sizing depend on.
**Depends on:** T1.2
**Steps:** time one query on the A100 through every step: client encode and encrypt, upload size, expansion, stage 1 (cluster centers), stage 2 (chosen clusters, with mask), relinearize and rescale, download size, client decrypt and top-k. Use 100k and 1M vectors at d = 384, 768, 1536, with the cluster settings from T0.8 (about 5% and 20% of the collection searched). Exclude network time but report the bytes sent each way. Report queries per second per A100 for a single query stream.
**Done when:** a results file has a per-step latency breakdown and total latency for each size and dimension, with the share of time per step.
**Status:** todo

### T1.4 Profile the search kernel

**Why:** the search kernel runs at about 1.5e11 modular multiplications per second and 18% to 21% of the A100's memory bandwidth. That is probably the cost of emulated 64-bit modular multiplication, not a hardware limit.
**Depends on:** none (profiling only until T1.2 is merged)
**Steps:** profile the T0.4 kernel with Nsight Compute on Modal if possible (otherwise use instruction counts and timing experiments); report achieved instruction throughput against the A100 peak, the instruction mix of one modular multiplication, occupancy and stall reasons. Try low-risk optimizations inside the 64-bit design (Barrett vs Montgomery, lazy reduction across the d-term sum, wider loads), each kept only if the exact oracle still passes.
**Done when:** a results file explains what limits the kernel with profiler or experiment evidence, and records throughput before and after any kept optimization.
**Status:** todo

### T1.5 Composite scaling preset with 32-bit limbs

**Why:** GPUs multiply 32-bit integers natively. OpenFHE 1.6 has a composite scaling mode that uses more, smaller RNS towers, which could make every modular multiplication much cheaper at similar storage.
**Depends on:** T1.4
**Steps:** define a candidate preset using OpenFHE's composite scaling with ~30-bit primes, still depth 2 and at least 128-bit security (checked against the HomomorphicEncryption.org table in a test). Check that the kernels' assumptions (level, scale and rescale handling, metadata) hold under composite scaling, adapt the search kernel to 32-bit limbs, and keep the exact OpenFHE oracle. Measure search throughput, expansion time, storage per vector and score error against `m0_depth2`.
**Done when:** the results file compares the candidate with `m0_depth2` on all four measures, the oracle passes, and a decision record is written as **proposed** (adopt or reject, with the numbers). `m0_depth2` is not changed.
**Status:** todo

### T1.6 Recall-based acceptance check

**Why:** the 1e-6 score-error bound was a guess. T0.8 showed CKKS noise never came close to affecting recall, and the bound alone rejected the faster N = 8192 preset.
**Depends on:** none
**Steps:** add a reusable recall check to `core/bench/recall.py`: a preset passes if recall@10 under encryption is within 0.005 of plaintext recall at the same cluster settings, on at least two public embedding datasets, with at least 1,000 queries. Keep the score-error measurement as a reported safety margin, not as the pass/fail rule. Write a decision record changing the acceptance rule for presets.
**Done when:** the check runs with one command on `m0_depth2` and passes, results for both datasets are recorded, and the decision record is **proposed**.
**Status:** done

### T1.7 Recall at the compact preset

**Why:** `m0_compact_n8192` expands 2.1x faster with 40% smaller keys, but failed the old 1e-6 bound (3.5e-6 to 6.3e-6). It is the candidate for the "fast" preset.
**Depends on:** T1.6
**Steps:** run the T1.6 recall check on `m0_compact_n8192` for d = 384, 768, 1536.
**Done when:** results are recorded, and if the preset passes, a decision record proposes it as the "fast" product preset.
**Status:** todo

### T1.8 M1 report

**Why:** decides the product presets and the priorities for M2.
**Depends on:** T1.1 to T1.7
**Steps:** summarize the review findings and their status, the end-to-end latency, the speedups, and the preset results; recommend "fast", "balanced" and "precise".
**Done when:** `results/m1-report.md` exists with the numbers and a recommendation for each preset and each proposed decision.
**Status:** todo

---

## Not in M1

- **Seeded encryption:** deferred until after M2. It would halve encrypted-DB storage but has no OpenFHE support, so it breaks hard rule 3 and needs its own audit (`results/2026-11-09-t0.11-seeded-encryption.md`).
- **Server, storage, router, sharding and cluster paging:** M2.
- **Compact fp16/int8 plaintext storage in the product:** measured in T0.11; its recall needs checking with the T1.6 tool before M2 decides whether to offer it.
