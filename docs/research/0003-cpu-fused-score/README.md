# 0003. Local mode: fused score on the CPU

- **Outcome:** adopted (PERF-B)
- **Date:** 2026-11-13
- **Area:** client CPU (local mode)

## What was tried and why

Local mode scores a block with one `EvalMultNoRelin` and one `EvalAdd` per column, which allocates
a 3-component ciphertext per column and reduces every product. The GPU kernel already fuses this.
Two CPU variants of one pass over the raw words: `fused` accumulates 128-bit sums and reduces
(Montgomery) only when the sum would leave the reduction's domain; `karatsuba` also gets the cross
term from `(s0 + s1)(q0 + q1) - s0 q0 - s1 q1`, three wide products per column instead of four.
Lazy accumulation is standard in RNS libraries; the fold bound is `K * 4 q^2 < q * 2^64`.

## Setup

| | |
| --- | --- |
| Commit | `7d83b19` (dev) plus this folder |
| GPU | none used |
| CPU | AMD Ryzen 5 5600H (6 cores, 12 threads), WSL2; threads as stated per row |
| OpenFHE | 1.6.1 (`~/.local/openfhe-1.6.1`), openfhe-python 1.6.1.0, Python 3.12 |
| Presets | `m0_depth2` (N = 16384) and `m0_compact_n8192` (N = 8192), values passed from `core/params/ckks_presets.json` |
| Data | synthetic uniform values in [-1, 1], keys made in-process and never written |
| Seed | 20261113 |

d = 384 and 768. The columns cycle through 16 distinct ciphertexts (encrypting 768 would take
minutes), which favours all three variants' caches equally.

## Command

```
cmake -S docs/research/0003-cpu-fused-score -B /tmp/b0003 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$HOME/.local/openfhe-1.6.1 && cmake --build /tmp/b0003
OMP_NUM_THREADS=4 taskset -c 0-7 /tmp/b0003/fused_score 8192 2 40 55 4096 3 384 5 20261113
```

Arguments: ring, depth, scaling bits, first-modulus bits, batch, digits (0 = default), d, reps,
seed. One JSON line per run; all runs are in `raw/local.jsonl`.
Table: `python docs/research/tools/jsonl_table.py docs/research/0003-cpu-fused-score/raw/local.jsonl --ratio baseline_ms:karatsuba_ms`.

## Exactness

Every word of the 3-component sum is compared with OpenFHE's own loop; the binary exits non-zero
on a difference. All runs: `fused_exact` and `karatsuba_exact` true.

## Results

| ring_dim | d | threads | towers | words_compared | fused_exact | karatsuba_exact | baseline_ms | fused_ms | karatsuba_ms | baseline_ms/fused_ms | baseline_ms/karatsuba_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 384 | 1 | 3 | 73728 | true | true | 137.452 | 38.003 | 26.416 | 3.62x | 5.20x |
| 8192 | 768 | 1 | 3 | 73728 | true | true | 252.286 | 75.195 | 48.703 | 3.36x | 5.18x |
| 16384 | 384 | 1 | 3 | 147456 | true | true | 268.325 | 84.068 | 55.785 | 3.19x | 4.81x |
| 8192 | 384 | 4 | 3 | 73728 | true | true | 59.502 | 13.498 | 10.522 | 4.41x | 5.66x |
| 8192 | 768 | 4 | 3 | 73728 | true | true | 105.138 | 28.344 | 18.247 | 3.71x | 5.76x |
| 16384 | 384 | 4 | 3 | 147456 | true | true | 139.208 | 31.906 | 25.257 | 4.36x | 5.51x |

## Reading

Karatsuba with lazy reduction is 4.8x to 5.8x faster than the current loop. But the score is a
small part of a local search (expansion is 70% to 90%, see 0004), so the search gains a few
percent. Its other value: the same arithmetic is the GPU candidate of 0006.

## Limits

16 distinct ciphertexts, so memory traffic is understated for all three. Relinearize and rescale
are not included (unchanged).
