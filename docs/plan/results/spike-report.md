# M0 spike report

**Verdict: the architecture stands.** The hybrid OpenFHE + CUDA design, column packing with server-side query expansion, and the depth-2 preset all work and meet their targets. One premise must change: batching is not the main throughput lever (proposed decision 0011). Three things remain open and are listed at the end.

Every number comes from a results file in this folder, which records the command, commit, GPU and parameter set. Preset `m0_depth2` was never changed: HEStd_128_classic, ring dimension 16384, depth 2, 50-bit scale, 8192 slots, FIXEDMANUAL, HYBRID.

## Results at a glance

| Question | Result | Source |
| --- | --- | --- |
| Security | 128-bit classic; log2(QP) = 280, limit 438 | T0.2 |
| Do GPU kernels match OpenFHE? | Yes: exact in every RNS word, for every kernel, up to d = 1536 for expansion | T0.3 to T0.7 |
| Query expansion, A100 | 18.5 / 36.1 / **71.6 ms** at d = 384 / 768 / 1536; target under 100 ms | T0.5b |
| Level budget | Expansion 0, query x stored 1, mask 1: total 2, as designed | T0.5, T0.6 |
| Worst score error, depth 2 | 1.5e-8 at d = 1536, bound 1e-6 | T0.6 |
| Search, A100 | 17.7M / 8.5M / 3.9M vectors per second at d = 384 / 768 / 1536 | T0.4 |
| Batching gain | Search 1.2x to 1.5x; expansion 1.01x to 1.05x | T0.7 |
| Recall@10, real embeddings | 1.000 on full scan; CKKS noise costs no recall | T0.8 |
| Storage against float32 | Encrypted DB 24x, plaintext DB 12x | T0.9 |
| A100 capacity at d = 1536 | 270k encrypted or 541k plaintext vectors | T0.9 |

## What the numbers say

**Expansion is cheap enough.** At d = 1536 one expansion costs 71.6 ms, the time to scan about 280k vectors with the T0.4 kernel. It is fixed per query and does not grow with the database. T0.5b's batching took it from 632 ms, an 8.8x gain, through level-wise launches rather than arithmetic changes.

**Accuracy has a wide margin.** CKKS noise reaches 2.6e-8 on real ada-002 embeddings against a bound of 1e-6, and encrypted recall matched plaintext recall at every setting measured, to the 3 decimal places 100 queries resolve. Any recall loss comes from clustering, not encryption. Client re-ranking changed nothing on this dataset.

**Clustering is the recall dial.** On 100,000 vectors, with share of the database searched in brackets:

| nlist | Recall@10, narrower search | Recall@10, wider search |
| --- | --- | --- |
| 100 | 0.897 (5.5%) | 0.981 (21.5%) |
| 316 | 0.942 (6.5%) | 0.973 (16.5%) |
| 1000 | 0.945 (5.2%) | 0.974 (10.4%) |

**Search and expansion are compute-bound.** The search kernel sustains about 1.5e11 modular multiplications per second at every batch size, at only 18% to 21% of the A100's published memory bandwidth when batching. So batching queries gives little: 0.93x to 1.03x against the single-query T0.4 kernel. Memory, not throughput, is what limits batch size: at a 34 GiB budget only 25 to 97 queries fit per search pass and 11 to 46 per expansion pass.

**Storage is the main cost.** Encrypted DB takes 96 bytes per dimension per vector, 24x float32, confirmed by measurement: 148.6 GB per million vectors at d = 1536. That does not fit one A100 (42.4 GB), so a million encrypted vectors at d = 1536 needs several GPUs or paging clusters from storage. Query upload is 787 KB. Score download is 263 KB per 8192 scores. Evaluation keys are 29 to 34 MB.

## Decisions: go or change

| Decision | Verdict | Evidence and notes |
| --- | --- | --- |
| 0002 Hybrid OpenFHE + CUDA, CKKS | **Go** | Kernels read and write OpenFHE's format exactly; the oracle holds throughout. |
| 0003 Column packing, server expansion | **Go** | Expansion meets its target. The blocked-packing fallback is not needed, so it was not built or measured. |
| 0010 Expansion by automorphisms | **Go** | Costs no level; client pre-scaling by 2^-L works. |
| 0009 `m0_depth2` preset | **Go** as the baseline | All bounds met with a 66x margin on score error. |
| 0004 Two-stage cluster search | **Go** | Recall curve above; center scoring is accurate under encryption. Fast and Decoy leakage were not part of the spike. |
| 0005 Homomorphic filtering | **Go** on depth; **work to do** on speed | An encrypted mask fits the budget (T0.6), but relinearize, rescale and the mask multiply still run on the CPU through OpenFHE. They need GPU kernels. |
| 0007 Router and sharded nodes | **Change the premise** | The sharding design was not tested. Its stated reason, that batching is the main speed lever, did not hold. See proposed decision 0011. |
| 0006 C++/CUDA, Python client | No new evidence | Keep. |
| 0008 Ampere only | **Go** | All work ran on A10 and A100. |

## Recommended product presets

The product defines "fast", "balanced" and "precise". The spike validates only one parameter set, so:

- **"balanced": use `m0_depth2`.** It is the only preset that passes every bound.
- **"fast": none yet.** `m0_compact_n8192` expands 2.1x faster with 40% smaller keys, but misses the 1e-6 bound (3.5e-6 to 6.3e-6). It saves no storage per vector, since both presets have 3 towers. It could still be acceptable, because the real test is recall and noise was never close to mattering in T0.8. That is measurable, and it would need a looser bound for that preset, so it is your decision.
- **"precise": same as balanced for now.** Nothing in the spike shows a need for more precision at `m0_depth2`.

For plaintext-DB storage, fp16 and int8 reach 2d and d bytes per vector, below float32's 4d. Search is 8.7x to 9.9x slower with GPU encoding on every pass, but encoding is a cost per load, so a cluster could be stored compactly and held encoded on the GPU while hot. Quantization error is 2e-5 to 4e-5 for fp16 and 1e-3 to 2.5e-3 for int8, so their recall needs measuring before they are offered.

## Limits of this spike

- **Not a full query measurement.** No end-to-end latency including relinearize, rescale, mask and download; those steps ran on the CPU. Search throughput excludes them.
- **Oracle ran on the A10, timings on the A100.** The kernels are integer-only (the encoder is compared within a stated tolerance), so the results should not depend on the GPU model, but the exact comparison itself did not run on the A100.
- **One dataset, 100 queries, 100,000 vectors, one seed** for recall.
- **Single GPU.** Router, sharding, storage and failure handling are unbuilt, as planned.
- **Repeat-run variance** is mostly small, but the single-query search baseline in T0.7 is noisy.

## Open decisions

1. **Proposed decision 0011:** batching is not the main throughput lever. Accept or reject.
2. **A compact preset:** measure recall at `m0_compact_n8192`, and decide whether a looser error bound is acceptable for a "fast" preset.
3. **Seeded encryption:** could halve encrypted-DB storage (96 to about 48 bytes per dimension per vector), but touches hard rule 3 and has no OpenFHE support (`2026-11-09-t0.11-seeded-encryption.md`). Decide whether to pursue it.

## Next work for M1

- GPU kernels for relinearize, rescale and ciphertext x ciphertext mask, each with an exact oracle. This is the largest gap on the search path.
- Reduce modular multiplication cost in the search kernel, now that it is the limit.
- Recall on a second dataset, and at larger N.
- Paging clusters between storage and GPU, given the storage figures above.
