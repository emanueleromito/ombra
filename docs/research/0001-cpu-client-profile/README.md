# 0001. Where the client's CPU time goes

- **Outcome:** baseline, and one candidate adopted (client drops to one tower before decrypting)
- **Date:** 2026-11-13
- **Area:** client CPU

## What was tried and why

The M1 report says client decrypt and top-k are 63% to 78% of a query at 1M vectors, 20% searched.
This times every OpenFHE call the client makes per query and per result block, to find which call
that is. It then times one candidate: dropping the result to its first RNS tower before `Decrypt`
(`LevelReduce`), either on the server (`drop_to_1_tower`) or on the client
(`client_drop_to_1_tower`).

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
LD_LIBRARY_PATH=~/.local/openfhe-1.6.1/lib OMP_NUM_THREADS=1 \
  python docs/research/0001-cpu-client-profile/run.py \
  --out docs/research/0001-cpu-client-profile/raw/local-omp1.json
python docs/research/0001-cpu-client-profile/analyze.py docs/research/0001-cpu-client-profile/raw/local-omp1.json
```

## Exactness

Decryption is not deterministic (OpenFHE adds noise on decryption, decision 0017), so the check is
the error against the known plaintext product, over 20 decryptions: 1.19e-9 with the drop against
1.19e-9 without (`m0_depth2`), 3.6e-7 against 3.8e-7 (`m0_compact_n8192`). The product test of
PERF-A adds the largest value the client can produce and the top-k comparison.

## Results (ms, median of 30, one thread)

| Step | `m0_depth2` | `m0_compact_n8192` |
| --- | --- | --- |
| Encode a column | 3.48 | 1.88 |
| Encrypt, public key | 7.22 | 3.80 |
| Encrypt, secret key | 5.40 | 2.75 |
| Deserialize a served block (file) | 5.14 | 2.40 |
| **Decrypt a served block (2 towers)** | **16.50** | **8.60** |
| Decrypt after dropping to 1 tower | 2.39 | 1.12 |
| Read the values into NumPy | 1.07 | 0.56 |
| **One block, as today** | **21.86** | **11.23** |
| One block, client drops the tower | 11.68 | 5.09 |
| One block, server drops the tower | 9.01 | 3.88 |

Served block size: 525,453 B with 2 towers, 263,211 B with 1 (`m0_depth2`).

## Reading

- `Decrypt` is 75% of a block. With two towers OpenFHE's CKKS decode interpolates every
  coefficient to a big integer; with one tower it stays in 64-bit words. Dropping the tower first
  makes `Decrypt` 6.9x to 7.7x faster and a whole block 1.9x to 2.2x faster.
- The client can do this alone: no server, preset, level-budget or format change. Adopted (PERF-A).
- After the drop, deserialization is the largest step left (see 0005).
- Doing the drop on the server would also halve the download, but it changes what the server
  returns, so it is listed as "needs approval" in the index and not built.

## Limits

One machine; times vary by about 30% between runs when the machine is busy. Client workers use one
OpenMP thread each (as `topk.py` sets), so one thread is the relevant case.
