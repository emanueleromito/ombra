# 0007. Expansion writes its leaves in place

- **Outcome:** adopted (server time 1.09x to 1.14x; d leaves no longer held twice)
- **Date:** 2026-11-13
- **Area:** GPU host, GPU memory

## What was tried and why

After the expansion tree, every leaf was copied device to device into the contiguous buffer the
search kernel reads ("leaf handoff": 3.6 ms at d = 384, 12.2 ms at d = 1536 in the M1 refresh),
and until the copies finished the leaves existed twice (1.2 GB at d = 1536). The last level of the
tree now writes each child directly at its place in that buffer. No kernel changes: the combine
kernel already writes through per-node output pointers.

## Setup

| | |
| --- | --- |
| Commit | `c854ce3` |
| GPU | NVIDIA A100-SXM4-40GB, driver 610.57.04 (Modal); oracle on an NVIDIA A10 |
| CPU | Modal host, model hidden, 20 logical CPUs |
| CUDA / OpenFHE | image `nvidia/cuda:12.6.3-devel-ubuntu22.04`; OpenFHE 1.6.1 |
| Preset | `m0_depth2` |
| Dimensions | d = 384, 768, 1536; 100k and 1M vectors; 5% and 20% searched |
| Data | synthetic unit vectors, as `core/bench/e2e.py` describes |
| Seeds | `core/bench/e2e.py` SEED = 20260301 |

## Command

```
git checkout c854ce3
make prebuild BUILD_TYPE=Release
make on-gpu GPU=a100 CMD="OMBRA_LEAF_HANDOFF=copy OMBRA_COLUMN_SEARCH_KERNEL=mont python core/bench/run.py --task e2e-latency --out /data/results --name-e2e-latency perf-e2e-before-a100.json --e2e-queries 10 ;; python core/bench/run.py --task e2e-latency --out /data/results --name-e2e-latency perf-e2e-after-a100.json --e2e-queries 10"
make results-pull
python docs/research/0007-expansion-leaf-sink/analyze.py docs/research/0007-expansion-leaf-sink/raw/perf-e2e-before-a100.json docs/research/0007-expansion-leaf-sink/raw/perf-e2e-after-a100.json
```

Both files come from one container: first the old path (`OMBRA_LEAF_HANDOFF=copy`), then the new.
10 timed queries per case after 3 warm-ups, medians.

## Exactness

`ombra_expand_gpu` stage 5 fills the buffer with a sentinel, expands into it (whole levels and
3-node chunks) and compares every word of every leaf with OpenFHE's leaves. Passed on the A10 on
both presets (`tests/test_gpu_query_expansion.py`), with the whole GPU suite (152 passed), which
includes the server end-to-end tests that use the new path.

## Results (ms, medians)

| d | N | searched | expansion + handoff before (ms) | after (ms) | stage 2 GPU before | after | server before (ms) | after (ms) | server speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 384 | 100,000 | 5% | 22.42 | 19.39 | 1.04 | 0.96 | 28.42 | 25.04 | 1.13x |
| 384 | 100,000 | 20% | 22.21 | 19.12 | 1.90 | 1.59 | 28.94 | 25.55 | 1.13x |
| 384 | 1,000,000 | 5% | 22.22 | 19.14 | 3.17 | 3.11 | 30.51 | 27.24 | 1.12x |
| 384 | 1,000,000 | 20% | 22.20 | 19.39 | 9.38 | 9.70 | 37.74 | 35.28 | 1.07x |
| 768 | 100,000 | 5% | 42.71 | 36.59 | 1.56 | 1.44 | 49.64 | 43.23 | 1.15x |
| 768 | 100,000 | 20% | 42.83 | 36.57 | 3.30 | 2.67 | 51.60 | 44.50 | 1.16x |
| 768 | 1,000,000 | 5% | 42.95 | 36.60 | 5.64 | 5.53 | 54.46 | 47.67 | 1.14x |
| 768 | 1,000,000 | 20% | 47.40 | 36.88 | 19.19 | 17.99 | 73.68 | 61.70 | 1.19x |
| 1536 | 100,000 | 5% | 84.16 | 71.93 | 2.64 | 2.40 | 93.25 | 80.64 | 1.16x |
| 1536 | 100,000 | 20% | 84.54 | 71.99 | 6.05 | 4.86 | 97.21 | 83.54 | 1.16x |
| 1536 | 1,000,000 | 5% | 84.12 | 71.72 | 10.62 | 10.41 | 101.47 | 88.89 | 1.14x |
| 1536 | 1,000,000 | 20% | 84.16 | 71.74 | 33.23 | 34.50 | 125.54 | 114.34 | 1.10x |

Server = read query, expansion, handoff, both GPU stages and their downloads. The handoff itself
went from 3.08, 6.13 and 12.55 ms (d = 384, 768, 1536) to 0.00; expansion is unchanged (19.3,
36.6, 71.6 to 71.9 ms).

## Reading

- The handoff was 11% to 15% of the server's time and is gone.
- Device memory: the d leaves exist once. At d = 1536 that is 1.2 GB less at the peak
  (1536 x 786,432 B), by construction; not measured with a memory probe.

## Second run, at the final commit

Same command at `3c36b14` (after the lazy kernel of 0006 was removed, so both files use the kept
search kernel and differ only in the handoff): `raw/final-e2e-before-a100.json` and
`raw/final-e2e-after-a100.json`, same A100 model, another host.

| d | N | searched | expansion + handoff before (ms) | after (ms) | stage 2 GPU before | after | server before (ms) | after (ms) | server speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 384 | 100,000 | 5% | 28.86 | 25.05 | 1.28 | 1.24 | 42.33 | 38.15 | 1.11x |
| 384 | 100,000 | 20% | 28.85 | 25.03 | 2.38 | 2.34 | 43.75 | 39.57 | 1.11x |
| 384 | 1,000,000 | 5% | 28.88 | 25.04 | 4.09 | 4.04 | 46.05 | 41.89 | 1.10x |
| 384 | 1,000,000 | 20% | 28.95 | 25.04 | 11.53 | 11.51 | 56.93 | 51.71 | 1.10x |
| 768 | 100,000 | 5% | 45.65 | 38.42 | 1.62 | 1.62 | 59.92 | 52.38 | 1.14x |
| 768 | 100,000 | 20% | 45.23 | 38.34 | 3.36 | 3.33 | 61.66 | 54.17 | 1.14x |
| 768 | 1,000,000 | 5% | 45.73 | 38.85 | 5.74 | 6.41 | 64.85 | 58.40 | 1.11x |
| 768 | 1,000,000 | 20% | 53.64 | 46.44 | 21.22 | 21.24 | 90.53 | 83.38 | 1.09x |
| 1536 | 100,000 | 5% | 86.05 | 73.46 | 2.67 | 2.69 | 102.02 | 89.55 | 1.14x |
| 1536 | 100,000 | 20% | 86.21 | 73.55 | 6.13 | 6.23 | 106.01 | 94.09 | 1.13x |
| 1536 | 1,000,000 | 5% | 86.04 | 73.47 | 10.71 | 10.74 | 110.99 | 99.23 | 1.12x |
| 1536 | 1,000,000 | 20% | 86.07 | 73.48 | 33.33 | 33.35 | 135.99 | 124.34 | 1.09x |

The handoff went from 3.79, 7.16 and 12.52 ms to 0; the server is 1.09x to 1.14x faster. This
host's CPU steps are slower than the first run's (the server totals are higher in both columns),
which is why the two runs are not compared with each other.

## Limits

- In the first run the "after" file used the lazy search kernel of 0006, which was then removed
  (stage 2 differs by -1.2 to +1.3 ms because of it). The second run has no such difference and
  is the one to quote.
- One run. The bench's stage 2 keeps its own query buffer and still receives a copy, outside the
  timed steps; the server has one buffer.
