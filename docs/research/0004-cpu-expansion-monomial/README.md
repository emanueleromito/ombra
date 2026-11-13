# 0004. Local mode: expansion with the monomial in evaluation form

- **Outcome:** adopted (PERF-B), small gain
- **Date:** 2026-11-13
- **Area:** client CPU (local mode)

## What was tried and why

The CPU expansion multiplies the odd child by X^-k in coefficient form: an inverse and a forward
NTT of every tower of both components per split. The GPU path multiplies pointwise by the NTT of
the monomial instead. This applies the same idea on the CPU.

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

d = 384 (767 splits).

## Command

```
cmake -S docs/research/0004-cpu-expansion-monomial -B /tmp/b0004 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$HOME/.local/openfhe-1.6.1 && cmake --build /tmp/b0004
OMP_NUM_THREADS=4 taskset -c 0-7 /tmp/b0004/expand_cpu 8192 2 40 55 4096 3 384 3 20261113
```

Table: `python docs/research/tools/jsonl_table.py docs/research/0004-cpu-expansion-monomial/raw/local.jsonl --ratio baseline_ms:eval_mono_ms`.

## Exactness

Every word and the metadata of every leaf are compared with the reference tree
(`core/src/openfhe_expand_ops.hpp`). All runs exact.

## Results

| ring_dim | d | threads | leaves | exact | metadata_equal | baseline_ms | eval_mono_ms | baseline_ms/eval_mono_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 384 | 1 | 384 | true | true | 4145.92 | 3684.88 | 1.13x |
| 16384 | 384 | 1 | 384 | true | true | 8230.79 | 6683.42 | 1.23x |
| 8192 | 384 | 4 | 384 | true | true | 1369.19 | 1276.4 | 1.07x |
| 16384 | 384 | 4 | 384 | true | true | 3036.25 | 2430.91 | 1.25x |

## Reading

1.07x to 1.25x. The monomial is not where the time is: `EvalAutomorphism` is. One split costs
about 4.8 ms on one thread at N = 8192, several times what its roughly 20 NTTs need, so OpenFHE's
own overhead (allocation, copies, index tables per call) dominates. Four threads give only 2.7x to
3x. A hand-written CPU key switch on raw words, reusing the tables the GPU path already extracts,
is the candidate that could give 2x or more here; it is not built (index: 0010, not run).

## Limits

Three repetitions per row; the gain varied between 1.07x and 1.35x across runs on this machine.
