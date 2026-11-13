# 0006. GPU search kernel with lazy accumulation and a Karatsuba cross term

- **Outcome:** rejected (2% to 10% slower than the current kernel)
- **Date:** 2026-11-13
- **Area:** GPU kernel

## What was tried and why

The column-search kernel does 4 Montgomery products and 3 modular additions per stored block and
dimension. The same lazy arithmetic that made the CPU score 5x faster (0003) was written as a
second kernel: 3 wide products per block and dimension, 128-bit sums, one reduction every 4 to 256
dimensions. Expected: fewer instructions, so a faster kernel if the kernel is compute-bound
(decision 0011 says it is; decision 0019 says it is memory-bound).

## Setup

| | |
| --- | --- |
| Commit | `c854ce3` (the kernel exists at this commit only; it was removed afterwards) |
| GPU | NVIDIA A100-SXM4-40GB, driver 610.57.04 (Modal) |
| CPU | not relevant (kernel time from CUDA events) |
| CUDA / OpenFHE | image `nvidia/cuda:12.6.3-devel-ubuntu22.04`, built for sm_75, 80, 86, 89, 90; OpenFHE 1.6.1 |
| Preset | `m0_depth2` |
| Dimensions | d = 384, 768, 1536; 64, 32, 16 blocks (19.3 GB stored) |
| Data | one encrypted block of seeded random values, copied on the device |
| Seeds | the benchmark's fixed seeds (`ombra_bench_column_search.cpp`) |

## Command

```
git checkout c854ce3
make prebuild BUILD_TYPE=Release
make on-gpu GPU=a100 CMD="python core/bench/run.py --task column-search --out /data/results --name perf-column-search-a100.json"
make results-pull
python docs/research/0006-gpu-lazy-search-kernel/analyze.py docs/research/0006-gpu-lazy-search-kernel/raw/perf-column-search-a100.json
```

The benchmark runs the old kernel then the new one in the same process environment
(`OMBRA_COLUMN_SEARCH_KERNEL=mont|lazy`). Median of 10 after 2 warm-ups. Setup record:
`raw/run-20261113T112517Z-nvidia-a100-sxm4-40gb.json`.

## Exactness

Exact. `tests/test_gpu_column_search.py` compared both kernels with OpenFHE word for word on an
A10 at the same commit (152 GPU tests passed), and a CPU unit test ran the kernel's arithmetic
against a 128-bit reference for 40 to 60-bit moduli and up to 1536 dimensions.

## Results

| d | blocks | kernel | median (ms) | min to max (ms) | vectors/s | GB/s read | lazy / mont |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 384 | 64 | mont | 19.99 | 19.97 to 24.38 | 2.622e+07 | 967 |  |
| 384 | 64 | lazy | 22.11 | 21.00 to 22.94 | 2.371e+07 | 874 | 0.904x |
| 768 | 32 | mont | 20.90 | 20.87 to 25.64 | 1.255e+07 | 925 |  |
| 768 | 32 | lazy | 21.61 | 21.58 to 23.65 | 1.213e+07 | 895 | 0.967x |
| 1536 | 16 | mont | 23.47 | 22.73 to 28.21 | 5.584e+06 | 823 |  |
| 1536 | 16 | lazy | 23.99 | 21.92 to 24.04 | 5.462e+06 | 805 | 0.978x |

"GB/s read" is the stored data divided by the kernel time.

## Reading

The kernel is memory-bound. The current kernel already reads stored ciphertexts at 820 to 970
GB/s, more than half the A100's published peak of 1,555 GB/s, so removing arithmetic cannot make
it faster. The lazy kernel is slower because it needs more registers (three 128-bit sums per
block), which lowers how many threads run at once. This settles the disagreement between
decisions 0011 and 0019 for the single-query kernel: 0019 is right.

What would speed up the search is reading fewer bytes per vector: fewer towers (a depth-1
preset) or packed words. Both change a preset or a format.

## Limits

One run on one A100. The tile size and launch bounds of the lazy kernel were not tuned; with the
kernel at 60% of peak bandwidth the possible gain from tuning is small.
