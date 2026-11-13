# 0005. What deserializing one ciphertext costs

- **Outcome:** baseline, and one candidate proposed (template reader; index 0009)
- **Date:** 2026-11-13
- **Area:** server CPU, client CPU

## What was tried and why

"Server: read query" is 8.4 ms per query on the A100 runs and deserialization is 5.1 ms of an
11.7 ms client block after 0001. This measures `Serial::Deserialize` against the cost of just
placing the words in a ciphertext, and checks whether a serialized ciphertext is a fixed skeleton
plus the words.

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

## Command

```
cmake -S docs/research/0005-ciphertext-deserialize -B /tmp/b0005 -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH=$HOME/.local/openfhe-1.6.1 && cmake --build /tmp/b0005
OMP_NUM_THREADS=1 /tmp/b0005/deser 16384 2 50 60 8192 0 30
python docs/research/0005-ciphertext-deserialize/skeleton.py
```

Table: `python docs/research/tools/jsonl_table.py docs/research/0005-ciphertext-deserialize/raw/local.jsonl --ratio deserialize_us:words_only_us`.

## Results (microseconds, median of 30, one thread)

| ring_dim | ciphertext | bytes | serialize_us | deserialize_us | memcpy_us | words_only_us | deserialize_us/words_only_us |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | fresh | 787423 | 54.2 | 3321.9 | 20.7 | 57.9 | 57.37x |
| 16384 | served | 525453 | 48.0 | 3224.9 | 13.3 | 34.6 | 93.21x |
| 8192 | fresh | 394207 | 36.6 | 934.9 | 8.8 | 24.1 | 38.79x |
| 8192 | served | 263309 | 24.2 | 943.4 | 5.3 | 15.6 | 60.47x |

Skeleton (`raw/skeleton-local.json`): six ciphertexts of one shape under one key have the same
length and differ only inside the word runs. Everything else is 807 to 1,105 bytes.

## Reading

- Deserializing costs 3.3 ms at N = 16384 and 0.94 ms at N = 8192 whatever the size, 39x to 93x
  the cost of placing the words. Every serialized ciphertext carries its crypto context, and
  OpenFHE rebuilds it (with its precomputed tables) before finding the one already loaded.
- Because the non-word bytes are constant for a given shape and key, a reader can compare them
  with a template serialized once, copy the word runs, and fall back to `Serial::Deserialize` on
  any difference. That needs no format change. Expected saving: about 3.3 ms per query on the
  server and per block on the client at `m0_depth2`.

## Limits

The reader itself is not built here. The Python client can only use it through the compiled
module (`ombra._native`), which exists today for local mode only.
