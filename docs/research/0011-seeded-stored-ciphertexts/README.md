# 0011. Seeded stored ciphertexts

- **Outcome:** adopted, pending the user's acceptance of decision 0023 (stored bytes about halved)
- **Date:** 2026-11-13
- **Area:** memory (disk and upload), client CPU, server CPU

## What was tried and why

Experiment 0006 showed the search is bound by the bytes it reads, and T0.9 that a stored vector
costs 96 B per column. Half of every stored ciphertext is a uniformly random polynomial. Following
decision 0023 (the idea is the seeded ciphertext of SEAL and of PIR queries; T0.11 described it for
Ombra), the client replaces that half by one generated from a 256-bit seed and stores the seed and
the other half only. The server regenerates the random half on the CPU when it loads a block, so
the GPU holds the same words as before.

## Setup

| | |
| --- | --- |
| Commit | `3c36b14` |
| GPU | NVIDIA A10 (Modal) for the server test; none for the measurements below |
| CPU | AMD Ryzen 5 5600H, WSL2; local mode on logical CPUs 0 to 7; load cost on one thread |
| OpenFHE | 1.6.1; BLAKE2xb as shipped inside it |
| Presets | `m0_compact_n8192` (local mode default) and `m0_depth2` |
| Dimensions | d = 384, 10,000 vectors, 2 clusters |
| Data | synthetic unit vectors (`tests/bench_local_mode.py`, seed 0) |
| Seeds | per ciphertext from the operating system (`getrandom`); fixed seed 1..32 in the load-cost run |

## Commands

```
# bytes, create and search time, plain then seeded (needs the compiled client module)
OMBRA_SEEDED=0 PYTHONPATH=client taskset -c 0-7 python tests/bench_local_mode.py --d 384
OMBRA_SEEDED=1 PYTHONPATH=client taskset -c 0-7 python tests/bench_local_mode.py --d 384
# server load cost per column
cmake -S docs/research/0011-seeded-stored-ciphertexts -B /tmp/b0011 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$HOME/.local/openfhe-1.6.1 && cmake --build /tmp/b0011
OMP_NUM_THREADS=1 /tmp/b0011/seeded_load 16384 2 50 60 8192 0 30
```

Tables: `python docs/research/tools/jsonl_table.py docs/research/0011-seeded-stored-ciphertexts/raw/<file>.jsonl`.

## Exactness

- `ombra_seeded_exact` (`tests/test_openfhe_seeded.py`, both presets): for the same secret key,
  `c0' + u * s` equals `c0 + c1 * s` word for word; every metadata field is preserved; OpenFHE
  decrypts the rebuilt ciphertext to the same values.
- The generator is pinned by a known-answer fixture and checked against an independent Python
  BLAKE2xb.
- A collection stored seeded returns the same top-k as the same data stored plain, with filters,
  deletes, repack and re-cluster, in local mode (`tests/test_openfhe_seeded_local.py`) and through
  the GPU server on an A10 (`tests/test_gpu_seeded.py`; whole GPU suite: 149 passed).

## Results

Local mode, 10,000 vectors, d = 384 (`raw/local-mode-d384.jsonl`):

| d | n | cpus | preset | clusters | seeded_columns | data_bytes | create_s | search_s | search_median_s | top1_is_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 384 | 10000 | 12 | m0_compact_n8192 | 2 | false | 938204534 | 46.0 | [1.72, 2.11, 1.69, 1.47, 1.61] | 1.69 | 5/5 |
| 384 | 10000 | 12 | m0_compact_n8192 | 2 | true | 559044854 | 28.3 | [1.43, 1.41, 1.35, 1.25, 1.29] | 1.35 | 5/5 |

Server load cost per stored column, one thread (`raw/load-cost-local.jsonl`, microseconds):

| ring_dim | towers | plain_bytes | seeded_bytes | plain_load_us | seeded_parse_us | seeded_load_us | seeded_bytes/plain_bytes | plain_load_us/seeded_load_us |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 3 | 787423 | 393336 | 3615.3 | 93.7 | 2134.7 | 0.50x | 1.69x |
| 8192 | 3 | 394207 | 196728 | 1092.0 | 66.8 | 1250.7 | 0.50x | 0.87x |

## Reading

- A stored column is 0.4995 of its former size (393,336 B against 787,423 B at `m0_depth2`).
  Everything the local server wrote for the collection went from 938 MB to 559 MB (0.60); the
  rest is keys, records and cluster centers' overhead, which do not shrink.
- Creating the collection took 28.3 s instead of 46.0 s. Part of that is the seeding workers
  (up to 4 processes) replacing a single-process encryption loop, so it is not all due to the
  smaller files.
- Loading costs about the same CPU as before: regenerating the random half with BLAKE2xb takes
  2.0 ms at N = 16384, which replaces OpenFHE's 3.3 ms context rebuild (0005). At N = 8192 it is
  15% slower. OpenFHE's BLAKE2 is the portable reference code (about 185 MB/s here); a SIMD
  BLAKE2 or another generator would make loads faster, as a new generator id in the format.
- GPU memory, search time and scores do not change: the device holds the same words.

## Limits

- Disk and upload on the GPU server are covered by the A10 test's ratio check (0.45 to 0.60 of
  the column bytes), not by a large run.
- Search times in the first table differ by run-to-run noise, not by the format.
- The client needs the compiled module to write or read seeded columns; without it the old
  format is written.
