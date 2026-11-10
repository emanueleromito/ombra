# M2: Minimal server

**Goal:** a single-node server that stores encrypted collections, keeps clusters on the GPU and answers two-round encrypted searches over gRPC, with the client-side parts that set its latency made fast.

**Starting point:** the M1 report (`results/m1-report.md`). Baseline preset stays `m0_depth2` (decision 0009). Level budget is unchanged: expansion 0, query x stored 1, mask 1 (depth 2). Files use the 0016 envelope (hard rule 6).
**Scope:** one node. The router, sharding and cross-node cluster placement come later (decision 0007).
**GPUs:** A10 for tests and oracles, A100 for benchmarks.
**Vector dimensions:** 384, 768, 1536.

**Order and parallel work:**

```
T2.1 (parallel decrypt) -----------------------------------------> T2.10
T2.2 (compact latency) ------------------------------------------> T2.10
T2.3 (filter design) ---+--> T2.4 (gRPC API) --> T2.9 (admin) ---> T2.10
                        |         |
                        |         +--> T2.5 (storage) --> T2.6 (GPU cache) --> T2.7 (search service) --> T2.8 (compaction) --> T2.10
                        +---------------------------------------------^ (filters in T2.7)
```

T2.1, T2.2 and T2.3 can start in parallel. T2.5 can start once T2.4 has fixed the message shapes it stores. T2.7 owns the search path in `server/`; do not edit it in parallel with T2.6.

---

### T2.1 Parallel client decrypt and top-k

**Why:** M1 measured client decryption as the largest CPU step in the end-to-end latency, and at 1M vectors the client's stage 2 is 63% to 78% of the total.
**Depends on:** none
**Steps:** profile `decrypt` and top-k on 25 blocks; decrypt blocks across cores (processes or threads, whichever openfhe-python allows without the GIL); merge per-block top-k; keep the results identical to the serial path; record the host CPU model and core count.
**Done when:** on 25 blocks the decrypt and top-k step is at least 4x faster than the serial path on the same host (number and CPU recorded in `results/`), and a test shows identical top-k to the serial path.
**Status:** done (persistent pool: 6x at 8 workers, 10x to 20x at 16 to 32; see `results/2026-11-09-t2.1-parallel-decrypt.md`)

### T2.2 End-to-end latency of `m0_compact_n8192`

**Why:** decision 0013 proposes the compact preset as "fast", but its end-to-end latency has not been measured.
**Depends on:** none
**Steps:** run the T1.3 end-to-end harness for `m0_compact_n8192` on the A100 at d = 384, 768, 1536, with host CPU model and core count; compare with `m0_depth2` from the same run; apply the decision-0012 recall rule if the data are available.
**Done when:** the numbers are in `results/`, and decision 0013 is marked **accepted** or **rejected** with the figures that decided it.
**Status:** done (compact end-to-end latency 0.52x to 0.65x of `m0_depth2`, recall rule passes; decision 0013 accepted; `results/2026-11-09-t2.2-compact-latency.md`)

### T2.3 Filter evaluation design within depth 2

**Why:** filters (decision 0005) must become an encrypted mask on the server without a third multiplication level.
**Depends on:** none
**Steps:** write a decision record with options and a recommendation. Start from this candidate: one-hot field columns are stored encrypted like vector columns; the client sends a one-hot query for the filter value, the server expands it like a vector query (level 0) and multiplies it with the one-hot columns (level 0 to 1), in parallel with the scores, then applies the mask to the score with one multiplication (level 1 to 2). Cover: equality on one field, several fields (AND), small value sets (OR), the cost in columns per field and in stored bytes, what the server learns (the number and name of filter fields), and what is left out (ranges, text).
**Done when:** the decision is **proposed** with at least two options measured or costed, and the level budget of the recommended option is shown to stay within depth 2. **Stop and ask the user before building anything.**
**Status:** done as a decision (0019 proposed; user chose B on 2026-11-10; accepted after the measurements in T2.7)

### T2.4 gRPC API v1

**Why:** the client and the server need a versioned contract before either is built on.
**Depends on:** T2.3 (filter message shape)
**Steps:** write `proto/ombra/v1/*.proto` with these calls: create, get, list and drop a collection; upload evaluation keys; insert; delete; and a search stream with two rounds (round 1: expanded query ciphertexts in, cluster-level scores out; round 2: the client's chosen clusters in, scores and ids out). Put the API version in the package name and every message that is stored. Add error codes, size limits and a compatibility test (generated code builds; an unknown version is refused). No secret key field anywhere (hard rule 2).
**Done when:** the proto compiles for C++ and Python, a test checks that a v1 message round-trips and an unknown version is rejected, and the API is described in `docs/architecture.md`.
**Status:** done (`proto/ombra/v1/ombra.proto`, `client/tests/test_proto_v1.py`, architecture.md; filter shape per decision 0019)

### T2.5 Storage

**Why:** collections must survive restarts and crashes.
**Depends on:** T2.4
**Steps:** segment files (one per cluster block, inside the 0016 envelope); a write-ahead log for insert and delete; a SQLite catalog for collections, clusters, segment lists, key tags and tombstones; recovery that replays the log after a crash; fsync rules written down. No plaintext in logs (hard rule 4).
**Done when:** a crash-recovery test kills the writer at each step of an insert and a delete (before the log record, after it, before the segment, after it) and every restart returns a consistent collection with no lost acknowledged write; readers refuse an unknown envelope version.
**Status:** done (`server/storage/`, ctest `storage_*`, also under ASan/UBSan; the crash matrix kills the writer at 6 insert and 3 delete points; SQLite 3.46.1 amalgamation fetched by CMake with a pinned hash; durability rules in `docs/architecture.md`)

### T2.6 GPU cluster cache

**Why:** a collection is larger than GPU memory, so clusters must move between pinned host memory and the GPU quickly.
**Depends on:** T2.5
**Steps:** pinned host memory pool holding the segments; a GPU cache with an eviction policy (LRU to start); asynchronous paging that overlaps copy and compute; a metric for hit rate that reveals no data. Measure the paging bandwidth on the A100.
**Done when:** a test searches a collection larger than the configured GPU memory and gets the same results as with everything resident; paging throughput and hit rate are in `results/`.
**Status:** done (paged search equals resident on A10 for both presets; A100 paging 22 to 26 GB/s, hit rate 0.32 uniform and 0.58 zipf at a 1/3 budget; overlap gives no gain while paging-bound; `results/2026-11-10-t2.6-gpu-cache.md`)

### T2.7 Search service

**Why:** it turns the kernels into the two-round search the client calls.
**Depends on:** T2.4, T2.5, T2.6, T2.3 (for filters)
**Steps:** one stream per query; expand the query once and reuse it in both stages; stage 1 scores the cluster centers; stage 2 scores the chosen clusters; filters as in decision 0019 (additive penalty: filter one-hots in the query, extra columns in the dot product, no mask multiply); deletes as a plaintext penalty added at the tombstoned slots; selective-filter behaviour per 0019 (more rounds on the same stream, then Strict or an `incomplete` answer); check key tags on every request (the wrong key is refused, as in the T1.12 to T1.14 refusals). Rule 3 applies to anything new on the device.
**Done when:** an end-to-end gRPC test returns the same top-k as the CPU OpenFHE oracle on a small collection with and without a filter and with deleted vectors, the new device code has exact OpenFHE oracle tests, and the 0019 conditions are measured and written into `results/`: score error against the band gap on the compact preset at F = 1, 4 and F_max, the 0012 recall rule with the extra columns, and filtered top-10 recall at selectivity about 1%, 10% and 50% (with the extra rounds needed). Decision 0019 is then marked accepted or rejected.
**Status:** done (engine `server/search`, gRPC `server/grpc`, exactness vs OpenFHE on A10, e2e `tests/test_gpu_e2e_grpc.py` 10 passed on both presets; decision 0019 accepted with measurements in `results/2026-11-10-t2.7-filter-0019.md`)

### T2.8 Compaction of deleted vectors

**Why:** tombstones in the mask keep deleted vectors in the search cost forever.
**Depends on:** T2.5, T2.7
**Approach (decided by the user):** in column packing a deleted vector is one slot in every column, so the server cannot drop slots without rotation, which would change the level budget. So the server drops only blocks whose vectors are all deleted, and offers an atomic ReplaceBlock; the client repacks partly deleted blocks (T3.5). The server never sees a key or plaintext.
**Steps:** a background job that drops fully deleted blocks through the catalog and removes their files; ListBlocks (deleted share per block, counts only); GetBlock and ReplaceBlock for the client repack; every swap is safe against a crash and against concurrent searches.
**Done when:** fully deleted blocks are dropped and their files removed; a test searches while compacting and gets consistent results before, during and after; ReplaceBlock survives a crash test at each step of the swap (nothing lost, no orphan files); after a client repack of a block with scattered deletes the stored bytes fall in proportion to the deleted share.
**Status:** done (PR #38; storage crash matrices and the gRPC e2e tests pass in CI)

### T2.9 REST admin, TLS and API keys

**Why:** an operator needs to know the server is healthy and the API must not be open.
**Depends on:** T2.4
**Steps:** REST endpoints for health and metrics (counts, latencies, GPU memory, cache hit rate; no data, no ids, no scores); TLS on the gRPC and REST ports; API keys, stored hashed, with a per-key scope (admin, read, write).
**Done when:** tests show that a request without a valid key is refused, that the metrics output contains no vector, id or score, and that a plain-text connection is refused when TLS is on.
**Status:** done (PR #39; unit tests and the TLS and auth e2e tests pass in CI)

### T2.10 End-to-end test and benchmark over gRPC

**Why:** it gives the real latency and throughput of the server, which replaces the single-process numbers from M1.
**Depends on:** T2.1, T2.2, T2.7, T2.8, T2.9
**Steps:** an end-to-end test (create, upload keys, insert, search with and without a filter, delete, restart, search again); a benchmark at 100k and 1M vectors on the A100 for the presets T2.2 accepted, with the latency split per step and the recall under encryption by the decision-0012 rule. Record the host CPU model and core count for the client and the server.
**Done when:** the test passes in CI and the benchmark file in `results/` has the numbers, the CPU model and core count, and the recall figures.
**Status:** done (restart test in CI; benchmark at 100k and 1M for both presets in `results/2026-11-10-t2.10-e2e-bench.md`: 100k totals 54 to 94 ms, 1M is disk-bound at 4.5 to 16 s p50 with the small cache budgets; recall gaps 0.000; the CPU model is hidden by Modal, core count recorded)

---

## Not in M2

- **Router, sharding and several nodes:** after M2 (decision 0007).
- **Key rotation, migration tool and seeded encryption:** after M3.
- **Python client features beyond what the tests above need:** M3.
