# Latency and throughput

!!! note "Some numbers are pending a new benchmark run on the current code."
    Figures from before the one-tower decrypt, in-place query expansion and seeded storage were removed. The A10 figures are TBD until they are rerun.

Synthetic vectors are used for the speed runs. The server and the client run on the same machine, so the network is not in the times unless a line says otherwise. Latency means the whole query as a user sees it: the client encrypts the query, the server searches, the client decrypts and ranks.

Two names first. Balanced is the default preset (ring dimension {{ preset("balanced", "ring_dim") }}, {{ preset("balanced", "batch_size") }} vectors per ciphertext). Fast is the smaller preset (ring dimension {{ preset("fast", "ring_dim") }}, {{ preset("fast", "batch_size") }} vectors per ciphertext). "Searched" below is the share of the collection the second search stage reads.

## One query on an A100

Total time per query on an A100 (40 GB), by preset, dimension, collection size and share searched. Medians of 20 queries, measured on the current code with `core/bench/e2e.py`. The full grid, including 20% searched at 100,000 and 5% at 1,000,000, is in `docs/charts/latency.csv`.

| Dimension | Collection | Searched | Balanced | Fast |
| --- | --- | --- | --- | --- |
| 384 | 100,000 | 5% | 63.2 ms | 31.5 ms |
| 384 | 1,000,000 | 20% | 327.8 ms | 211.4 ms |
| 768 | 100,000 | 5% | 80.4 ms | 39.6 ms |
| 768 | 1,000,000 | 20% | 359.1 ms | 232.8 ms |
| 1536 | 100,000 | 5% | 118.1 ms | 59.0 ms |
| 1536 | 1,000,000 | 20% | 401.9 ms | 262.1 ms |

The client's decrypt and ranking step grows with the number of blocks searched, so choosing a smaller share to search is the biggest lever on latency. Fast answers in 0.49 to 0.65 of Balanced's time across these cells.

![Query latency on an A100 for Balanced and Fast at three dimensions](img/latency.svg)

At 1,000,000 vectors, searching 20% instead of 5% costs more time than going from 384 to 1536 dimensions.

Timing varies between runs. The same search kernel took up to 26% longer in one run than in another on the same card. Read each figure here as one sample, not a guarantee.

## The server's share

Time the server spends on one query on an A100, from the same run (medians of 20 queries). This covers reading the query, expansion, both GPU search stages and their downloads.

| Dimension | Collection | Searched | Balanced | Fast |
| --- | --- | --- | --- | --- |
| 384 | 100,000 | 5% | 32.6 ms | 16.2 ms |
| 384 | 100,000 | 20% | 36.9 ms | 16.7 ms |
| 384 | 1,000,000 | 5% | 39.3 ms | 20.4 ms |
| 384 | 1,000,000 | 20% | 48.5 ms | 30.1 ms |
| 768 | 100,000 | 5% | 49.6 ms | 24.1 ms |
| 768 | 100,000 | 20% | 51.6 ms | 25.9 ms |
| 768 | 1,000,000 | 5% | 54.4 ms | 28.4 ms |
| 768 | 1,000,000 | 20% | 80.1 ms | 50.4 ms |
| 1536 | 100,000 | 5% | 87.2 ms | 43.1 ms |
| 1536 | 100,000 | 20% | 91.0 ms | 46.3 ms |
| 1536 | 1,000,000 | 5% | 96.2 ms | 50.5 ms |
| 1536 | 1,000,000 | 20% | 122.3 ms | 80.6 ms |

Most of this is query expansion, which costs about 19 ms at d = 384, 37 ms at 768 and 72 ms at 1536 and does not depend on the collection size. The search itself takes 1 to 35 ms.

## Over gRPC with a real server

The same query through the gRPC API, with the client encrypting in 16 processes, on an A100 40 GB in a container with 32 CPUs. Medians over 30 queries at d = 384, with p95 in brackets.

| Collection | Filter | Balanced | Fast |
| --- | --- | --- | --- |
| 100,000 | none | 143 ms (170 ms) | 60 ms (62 ms) |
| 100,000 | 2 fields | 143 ms (163 ms) | 59 ms (61 ms) |

A filter is part of the same encrypted search, so it adds no extra round. A query sends 0.79 MB (Balanced) or 0.39 MB (Fast) up and, in this run, gets 3.7 MB (Balanced) or 2.0 MB (Fast) back over both stages. The answer size depends on how many blocks are searched.

These are warm-cache times: the benchmark loads every cluster once before it times queries. The first query that reaches a cluster not yet in memory waits while the server reads that cluster's blocks from disk. On Fast at 100,000 vectors that took 4.4 to 9.1 s per cluster, for about 0.3 GB of columns each.

## What is slow

**A collection larger than the GPU cache.** When the collection does not fit in the GPU and host cache, every probe is read from disk, and the query takes seconds instead of milliseconds. The test below has 1,000,000 vectors at d = 384, about 49 GB on disk, with a 24 GB GPU budget and an 8 GB host budget, so most clusters come from disk on every query. Medians over 30 queries, with p95 in brackets. Both variants probe the 2 best clusters and read the same number of blocks. The filtered queries ran second and found more of their clusters already in the cache, which is why they are faster.

| Collection | Cache | Balanced | Fast |
| --- | --- | --- | --- |
| 100,000 | fits | 143 ms | 60 ms |
| 1,000,000 | does not fit, no filter | 9.2 s (46 s) | 2.0 s (11.5 s) |
| 1,000,000 | does not fit, 2 fields | 4.6 s (31 s) | 98 ms (17 s) |

![Median query time at 100k and 1M vectors on a log scale](img/cold-cache.svg)

Going from a collection that fits the cache to one that does not makes the median query 30 to 60 times slower.

Loading one cluster from disk took 11.1 s at the median on Balanced (5.0 s on Fast), for about 0.6 GB (0.3 GB) of columns: roughly 60 MB/s. That is slow for a local disk. Nobody has profiled the disk path yet, so we cannot say whether disk speed, parsing or copies cost the time. A cold cache with small budgets is the worst case, not what a tuned A100 box would show. This benchmark uploads columns without seeding, so a client with seeding, the default, reads about half the bytes.

The path from host memory to the GPU runs at 22 to 26 GB/s on the A100's PCIe link, and a search over a collection three times the GPU budget spent about 95% of its time moving data and 5% computing (a synthetic test of the cache). Past the GPU's memory, the link sets the speed.

**Local mode on a CPU.** Local mode runs the server on your CPU in a child process. With 10,000 vectors, 2 clusters and the Fast preset, on a Modal container with 8 CPUs: at d = 384, creating the collection took 19.3 s and a search took 0.75 s at the median; at d = 768, 36.1 s and 1.28 s (`tests/bench_local_mode.py`). A laptop is slower: a Ryzen 5 5600H took 28.3 s and 1.35 s at d = 384. Use local mode for development and small collections, not for serving traffic.

**Strict scans.** Strict reads every cluster. We have not timed a Strict query on its own, so there is no figure for it yet.

## Throughput

| Measure | Result |
| --- | --- |
| One query stream, Balanced, A100, client included | 2.5 to 15.8 queries per second (1 / median latency over the grid above) |
| One query stream, Balanced, server alone | 8.2 to 30.7 queries per second (1 / median server time) |
| Ingest, d = 384, A100 server, 16 client processes, 100,000 vectors | 1,132 (Balanced) and 1,755 to 2,010 (Fast, two runs) vectors per second, unseeded |
| Ingest, d = 384, same setup, 1,000,000 vectors | 2,494 (Balanced, 401 s) and 2,228 (Fast, 449 s) vectors per second, unseeded |

Sending several queries together helps the search kernel: four queries at once ran 1.75x to 1.78x faster per query than one at a time (A100, kernel only, which the changes above do not affect).

## On an A10

The A10 numbers on filtered queries over real text embeddings (Fast, d = 384, 10,003 vectors in 6 clusters) were measured before the changes above and are TBD until they are rerun. Recall under encryption matched plaintext recall in those runs. The accuracy pages have the numbers.
