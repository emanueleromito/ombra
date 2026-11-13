# Latency and throughput

!!! note "Numbers pending a new benchmark run on the current code."
    Several changes since the last full run speed up the client and the server: a one-tower decrypt on the client, query expansion that writes in place, and seeded stored ciphertexts. Every latency, throughput and ingest figure from before those changes was removed and shows as TBD. The tables keep their shape so that the new numbers can go in. Server times measured on the current code are given where we have them, with their source.

Synthetic vectors are used for the speed runs. The server and the client run on the same machine, so the network is not in the times unless a line says otherwise. Latency means the whole query as a user sees it: the client encrypts the query, the server searches, the client decrypts and ranks.

Two names first. Balanced is the default preset (ring dimension {{ preset("balanced", "ring_dim") }}, {{ preset("balanced", "batch_size") }} vectors per ciphertext). Fast is the smaller preset (ring dimension {{ preset("fast", "ring_dim") }}, {{ preset("fast", "batch_size") }} vectors per ciphertext). "Searched" below is the share of the collection the second search stage reads.

## One query on an A100

Total time per query on an A100 (40 GB), by preset, dimension, collection size and share searched.

| Dimension | Collection | Searched | Balanced | Fast |
| --- | --- | --- | --- | --- |
| 384 | 100,000 | 5% | TBD | TBD |
| 384 | 1,000,000 | 20% | TBD | TBD |
| 768 | 100,000 | 5% | TBD | TBD |
| 768 | 1,000,000 | 20% | TBD | TBD |
| 1536 | 100,000 | 5% | TBD | TBD |
| 1536 | 1,000,000 | 20% | TBD | TBD |

The client's decrypt and ranking step grows with the number of blocks searched, so choosing a smaller share to search is the biggest lever on latency. Earlier runs showed Fast answering in roughly half to two thirds of Balanced's time; the new run will confirm the ratio.

Timing varies between runs. The same search kernel took up to 26% longer in one run than in another on the same card. Read each figure here as one sample, not a guarantee.

## The server's share

Time the server spends on one query on an A100, Balanced, medians of 10 queries. This covers reading the query, expansion, both GPU search stages and their downloads, and it is measured on the current code.

| Dimension | Collection | Searched | Server time |
| --- | --- | --- | --- |
| 384 | 100,000 | 5% | 25.0 ms |
| 384 | 100,000 | 20% | 25.6 ms |
| 384 | 1,000,000 | 5% | 27.2 ms |
| 384 | 1,000,000 | 20% | 35.3 ms |
| 768 | 100,000 | 5% | 43.2 ms |
| 768 | 100,000 | 20% | 44.5 ms |
| 768 | 1,000,000 | 5% | 47.7 ms |
| 768 | 1,000,000 | 20% | 61.7 ms |
| 1536 | 100,000 | 5% | 80.6 ms |
| 1536 | 100,000 | 20% | 83.5 ms |
| 1536 | 1,000,000 | 5% | 88.9 ms |
| 1536 | 1,000,000 | 20% | 114.3 ms |

Most of this is query expansion, which costs about 19 ms at d = 384, 37 ms at 768 and 72 ms at 1536 and does not depend on the collection size. The search itself takes 1 to 35 ms.

## Over gRPC with a real server

The same query through the gRPC API, with the client encrypting in 16 processes, on an A100 80 GB. Medians over 30 queries at d = 384.

| Collection | Filter | Balanced | Fast |
| --- | --- | --- | --- |
| 100,000 | none | TBD | TBD |
| 100,000 | 2 fields | TBD | TBD |

A filter is part of the same encrypted search, so it adds no extra round. A query sends 0.79 MB (Balanced) or 0.39 MB (Fast) up. The answer size depends on how many blocks are searched.

## What is slow

**A collection larger than the GPU cache.** When the collection does not fit in the GPU and host cache, every probe is read from disk, and the query takes seconds instead of milliseconds. The earlier test, at 1,000,000 vectors with a 24 GB GPU budget and an 8 GB host budget, showed a gap of two orders of magnitude to the cached case. That test ran before seeded storage, which halves the bytes read from disk, so its numbers are removed.

| Collection | Cache | Balanced median | Fast median |
| --- | --- | --- | --- |
| 100,000 | fits | TBD | TBD |
| 1,000,000 | does not fit | TBD | TBD |

A cold cache with small budgets is the worst case, not what a tuned A100 box would show. Nobody has profiled the disk path yet, so we cannot say whether disk speed, parsing or copies cost the time.

The path from host memory to the GPU runs at 22 to 26 GB/s on the A100's PCIe link, and a search over a collection three times the GPU budget spent about 95% of its time moving data and 5% computing (a synthetic test of the cache). Past the GPU's memory, the link sets the speed.

**Local mode on a CPU.** Local mode runs the server on your CPU in a child process. On a Ryzen 5 5600H laptop, with 8 logical CPUs, 10,000 vectors, 2 clusters, d = 384 and the Fast preset, creating the collection took 28.3 s and a search took 1.35 s at the median. Other dimensions are TBD. Use local mode for development and small collections, not for serving traffic.

**Strict scans.** Strict reads every cluster. We have not timed a Strict query on its own, so there is no figure for it yet.

## Throughput

| Measure | Result |
| --- | --- |
| One query stream, Balanced, A100, client included | TBD queries per second |
| One query stream, server alone | TBD queries per second |
| Ingest, d = 384, A100 server, 16 client processes | TBD vectors per second |

Sending several queries together helps the search kernel: four queries at once ran 1.75x to 1.78x faster per query than one at a time (A100, kernel only, which the changes above do not affect).

## On an A10

The A10 numbers on filtered queries over real text embeddings (Fast, d = 384, 10,003 vectors in 6 clusters) were measured before the changes above and are TBD until they are rerun. Recall under encryption matched plaintext recall in those runs. The accuracy pages have the numbers.
