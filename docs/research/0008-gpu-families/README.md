# 0008. GPU families beyond Ampere

- **Outcome:** adopted as an option (all four new cards pass every oracle test); default build list set by decision 0024 (accepted)
- **Date:** 2026-11-13
- **Area:** GPU families

## What was tried and why

Decision 0008 builds for Ampere only (A10, A100). The most common cards outside it on today's
clouds are the T4 (Turing, the cheapest), the L4 and L40S (Ada) and the H100 or H200 (Hopper).
The kernels use nothing Ampere-specific (64-bit integer arithmetic, 16 KB of shared memory per
block), so they should run unchanged. The tree was built once for sm_75, sm_80, sm_86, sm_89 and
sm_90 and the same binaries were run on six cards.

## Setup

| | |
| --- | --- |
| Commit | `3c36b14` |
| GPUs | Tesla T4 16 GB, NVIDIA A10 24 GB, A100-SXM4-40GB, L4 24 GB, L40S 48 GB, H200 144 GB (Modal served an H200 for the H100 request) |
| CPU | Modal hosts, model hidden; 4 CPUs per container |
| CUDA / OpenFHE | image `nvidia/cuda:12.6.3-devel-ubuntu22.04`, one Release build for the five architectures (`make prebuild`); OpenFHE 1.6.1 |
| Preset | `m0_depth2` for the benchmarks; both presets in the tests |
| Dimensions | expansion d = 384, 768, 1536; search d = 384 and 1536, 4 blocks (fits 16 GB) |
| Data | synthetic, the benchmarks' fixed seeds |

## Command

```
git checkout 3c36b14
make prebuild BUILD_TYPE=Release
for g in t4 a10 a100 l4 l40s h100; do
  make on-gpu GPU=$g CMD="python -m pytest -m gpu -q -rf -p no:cacheprovider --ignore=tests/test_gpu_search_e2e.py --ignore=tests/test_gpu_collection.py --ignore=tests/test_gpu_local_parity.py ;; python core/bench/run.py --task column-search --dims 384,1536 --blocks 4 --out /data/results --name fam-column-search-$g.json ;; python core/bench/run.py --task expansion --out /data/results --name-expansion fam-expansion-$g.json"
done
make results-pull
python docs/research/0008-gpu-families/analyze.py docs/research/0008-gpu-families/raw
```

One setup record per run is in `raw/run-*.json` (commit, card, driver, commands).

## Exactness

On every card: 146 GPU tests passed, 0 failed (1 skipped: the license-header test, which needs
a checkout). They include every kernel's word-for-word comparison with OpenFHE on both presets,
the expansion into a buffer, relinearize and rescale, the filter mask, the gRPC end-to-end tests
and the seeded-column test. The three files left out (`test_gpu_search_e2e.py`,
`test_gpu_collection.py`, `test_gpu_local_parity.py`) spend most of their four minutes on the
CPU; they passed on the A10 at the same commit (149 passed with them).

## Results

| GPU served | Generation | Build | Expansion d = 384 / 768 / 1536 (ms) | vs A10 | Search, vectors/s d = 384 / 1536 | vs A10 | USD/h | Expansions per USD (d = 768) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tesla T4 | Turing | sm_75 | 88.5/176.7/350.8 | 0.42x | 4.02e+06 / 1.18e+06 | 0.37x | 0.59 | 34,505 |
| NVIDIA A10 | Ampere | sm_86 | 37.5/74.2/148.2 | 1.00x | 1.09e+07 / 2.67e+06 | 1.00x | 1.10 | 44,046 |
| NVIDIA A100-SXM4-40GB | Ampere | sm_80 | 18.4/35.8/70.9 | 2.07x | 1.47e+07 / 4.08e+06 | 1.34x | 2.10 | 47,861 |
| NVIDIA L4 | Ada | sm_89 | 55.0/114.4/236.0 | 0.65x | 5.56e+06 / 1.38e+06 | 0.51x | 0.80 | 39,381 |
| NVIDIA L40S | Ada | sm_89 | 18.0/37.5/76.1 | 1.98x | 1.60e+07 / 4.03e+06 | 1.46x | 1.95 | 49,145 |
| NVIDIA H200 | Hopper | sm_90 | 10.0/19.3/38.1 | 3.85x | 2.62e+07 / 6.61e+06 | 2.40x | 3.95 | 47,249 |

Expansion: whole tree, median of 10. Search: the column-search kernel over 4 blocks (a small
launch; with 64 blocks the A100 reaches 2.6e7 vectors/s at d = 384, see 0006). "Expansions per
USD" uses the GPU price only.

## Reading

- **Everything is exact on Turing, Ada and Hopper** with no source change: the build list is the
  only difference.
- **Hopper is the fastest:** expansion 3.9x the A10 and 1.9x the A100.
- **The L40S equals the A100** on both workloads at 93% of its price, with 48 GB of memory.
- **The L4 is slower than the A10** (0.65x expansion, 0.51x search) for 73% of the price; the T4
  is 0.42x and 0.37x for 54%. Per dollar, the six cards are within 1.4x of each other on
  expansion (34,500 to 49,100 per USD at d = 768): the cheap cards are not cheaper per query.
- **Latency is what separates them.** At d = 1536 expansion alone is 351 ms on a T4 and 236 ms
  on an L4, against 148 ms on an A10 and 38 ms on an H200.
- The search kernel scales with memory bandwidth (0006): the H200 is 2.4x the A10, the T4 0.37x.

## Limits

- One run per card; Modal picks the host. The A10 run started 20 minutes after the others.
- The H100 itself was not measured (an H200 was served).
- Blackwell (sm_100, sm_120) is not covered: it needs CUDA 12.8 or later, so a new pinned image
  and an OpenFHE rebuild.
- End-to-end latency was not run per card (its largest case needs 30 GB of device memory).
