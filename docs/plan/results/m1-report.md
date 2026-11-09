# M1 report

All numbers are on the A100 with `m0_depth2` unless stated. Sources are in `docs/plan/results/`.

## Recommendation

| Preset name | Recommendation | Reason |
| --- | --- | --- |
| **fast** | `m0_compact_n8192`, if 0012 and 0013 are accepted | Recall gap at most 0.0004 (limit 0.005). Expansion 2.1x to 2.5x faster, keys about 40% smaller. Client encrypt is 1.21x to 1.23x slower and decrypt 1.39x to 1.49x slower, and storage per vector is unchanged. Measure its end-to-end latency before confirming the label. |
| **balanced** | `m0_depth2` | The baseline. Recall gap 0.0000 on 3 datasets; max score error 2.8e-8. |
| **precise** | None | `m0_depth2` already has no measurable recall loss, so there is nothing for a higher-precision preset to win. The composite candidate was rejected (0014). |

## Decisions

| # | Decision | Status | Recommendation |
| --- | --- | --- | --- |
| 0012 | Recall-based preset acceptance | proposed | Accept |
| 0013 | `m0_compact_n8192` as "fast" | proposed | Accept, after its end-to-end latency is measured |
| 0014 | Composite scaling preset | rejected | Done; code removed |
| 0015 | Encoder tolerance only for FFT and rounding | proposed | Accept |
| 0016 | File envelope; driver fixtures stay raw until M2 | proposed | Accept |
| 0017 | Keep default decryption noise, pin it, claim no IND-CPA-D | proposed | Accept option A |
| 0018 | Plaintext-DB mode only for corpora any client may read | proposed | Accept option B |

## Crypto review (T1.1)

1 high, 11 medium and 15 low findings.

| Findings | Tasks | Status |
| --- | --- | --- |
| F1 (high), F7, F12 | T1.9, T1.15, T1.20 | Done, PR #28 |
| F11 | T1.19 | Workflow in PR #28; needs Modal repository secrets and the required-check setting |
| F2 to F6, F8 | T1.10 to T1.14, T1.16 | Done, PR #29 |
| F9, F10 | T1.17, T1.18 | Decision proposed; the rest waits on 0017 and 0018 |
| F13 to F27 (low) | None | Open |

No real bug appeared in the code the new oracles cover. Hard rule 6 stays unmet for test-driver fixtures until 0016 is accepted.

## GPU pipeline and accuracy

- **T1.2:** relinearize, rescale and mask run on the GPU, word-exact against OpenFHE at d = 384, 768, 1536. Score error up to 1.3e-8.
- **T1.10:** exact for every block of 7, at every stage, on both presets.

## Search speed

Kernel, single query (ms):

| d | Before T1.4 | T1.4 "after" | On main, this refresh | Speedup, same run (T1.4) | Speedup, across runs |
| --- | --- | --- | --- | --- | --- |
| 384 | 29.57 | 19.29 | 22.10 | 1.53x | 1.34x |
| 768 | 31.36 | 20.39 | 23.95 | 1.54x | 1.29x |
| 1536 | 33.30 | 22.42 | 22.77 | 1.49x | 1.46x |

Across-run speedups compare this refresh with the original T0.4 file (29.6, 30.8, 33.3 ms). The same kernel varies by up to 26% between runs at d = 384, 17% at d = 768 and 2% at d = 1536. Batched search, Q = 4: 1.75x to 1.78x; ciphertext x plaintext: 1.20x to 1.25x (T1.4, same-run).

## End-to-end latency (`2026-11-09-m1-refresh-a100.md`, commit `aba6ddc`)

Total per query in ms; QPS for one stream including the client. Before: the T1.3 run.

| d | 100k, 5% | 100k, 20% | 1M, 5% | 1M, 20% | QPS range | Total before |
| --- | --- | --- | --- | --- | --- | --- |
| 384 | 67.9 | 91.5 | 134.9 | 329.4 | 3.0 to 14.8 | 47.7 to 201.3 |
| 768 | 86.6 | 108.6 | 152.5 | 364.8 | 2.7 to 11.6 | 69.9 to 236.4 |
| 1536 | 129.1 | 153.4 | 199.4 | 409.0 | 2.4 to 7.7 | 114.9 to 290.2 |

Server only: 7.6 to 26.6 QPS. Upload is 787,423 bytes; download is 526,502 to 6,844,526 bytes.

This run is 1.12x to 1.64x slower than the T1.3 run. GPU steps are flat or faster (Stage 2 GPU median 0.87x). The increase is in CPU steps whose code did not change: client decrypt about 1.8x to 1.9x, reading the query about 2.2x. The cause is probably the host CPU, which was not recorded, so treat the two runs as a range.

## Priorities for M2

1. **Client decrypt and top-k.** 63% to 78% of the total at 1M vectors, 20% searched (about 257 ms); it grows with the blocks chosen.
2. **Expansion.** The largest server step when few blocks are searched: 62% to 74% of server time at 100k vectors, 5% searched. The search kernel is only 3.5% to 8.1% of the total at 1M vectors, 20% searched.
3. **Record the host CPU in benchmarks,** so CPU steps can be compared between runs.
4. **Close the open items:** the decisions above, the T1.19 settings, and the low findings. If the T1.19 check becomes required, the known `m0_compact_n8192` expansion flake (up to 1.47e-6 against 1e-6) can block merges, and needs resolving with 0012.
