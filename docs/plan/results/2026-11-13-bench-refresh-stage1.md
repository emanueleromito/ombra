# Benchmark refresh, stages 1 and 2 (2026-11-13)

Re-measured on the code after the client one-tower decrypt, in-place expansion and seeded storage. Modal spend about $0.60.

- **Code:** dev at c87fe3a (docs branch of #81); the Balanced gRPC run used `fix/server-message-limit` (07ee61f), because the 256 MiB receive limit from #79 refused its 315 MB insert.
- **Hardware:** A100-SXM4-40GB, CUDA 12.6.3, OpenFHE 1.6.1, openfhe-python 1.6.1.0.24.4. gRPC runs in a 32-CPU container with 16 client workers. Local mode on Modal `cpu_run8` (8 CPUs requested).

## e2e-latency (core/bench/e2e.py, 20 queries, medians; total / server ms)

| d | n | searched | Balanced | Fast |
| --- | --- | --- | --- | --- |
| 384 | 100k | 5% | 63.2 / 32.6 | 31.5 / 16.2 |
| 384 | 100k | 20% | 88.2 / 36.9 | 42.8 / 16.7 |
| 384 | 1M | 5% | 132.3 / 39.3 | 74.6 / 20.4 |
| 384 | 1M | 20% | 327.8 / 48.5 | 211.4 / 30.1 |
| 768 | 100k | 5% | 80.4 / 49.6 | 39.6 / 24.1 |
| 768 | 100k | 20% | 103.0 / 51.6 | 52.0 / 25.9 |
| 768 | 1M | 5% | 147.0 / 54.4 | 82.8 / 28.4 |
| 768 | 1M | 20% | 359.1 / 80.1 | 232.8 / 50.4 |
| 1536 | 100k | 5% | 118.1 / 87.2 | 59.0 / 43.1 |
| 1536 | 100k | 20% | 142.7 / 91.0 | 72.8 / 46.3 |
| 1536 | 1M | 5% | 189.1 / 96.2 | 104.9 / 50.5 |
| 1536 | 1M | 20% | 401.9 / 122.3 | 262.1 / 80.6 |

Server times are about 7 ms higher at small sizes than research 0007 (10 queries); different run and query count.

## gRPC end to end (core/bench/grpc_e2e.py, d = 384, 100k, 30 queries)

| | Balanced | Fast |
| --- | --- | --- |
| no filter p50 / p95 | 143.3 / 170.4 ms | 58.8 / 8097.3 ms |
| 2 fields p50 / p95 | 142.6 / 162.9 ms | 58.7 / 65.8 ms |
| ingest | 88.3 s, 1,132 vectors/s | 49.8 s, 2,010 vectors/s |
| on disk (unseeded upload) | 5.70 GB, 57.0 KB/vector | 4.91 GB, 49.1 KB/vector |
| query up / answer down | 787,464 B / 3.65 MB | 394,255 B / 2.01 MB |

Recall gaps to plaintext 0.000 at 1, 2 and 4 probes (Balanced). Fast's no-filter p95 of 8.1 s in the first run was the cold cache: the warm-up touched only 4 of Fast's 12 clusters (Balanced has 6). With the benchmark now loading every cluster first (same commit range plus that change), Fast gave no filter p50 59.8 / p95 61.9 ms, 2 fields 59.4 / 61.2 ms, ingest 57.0 s (1,755 vectors/s), and first-touch times per cluster of 0.01 s (already cached) or 4.4 to 9.1 s (read from disk, about 0.3 GB each). The docs use the second run for Fast.

## Local mode (tests/bench_local_mode.py, Fast, 10k vectors, 2 clusters)

d = 384: create 19.3 s, search median 0.75 s. d = 768: create 36.1 s, search median 1.28 s. Top-1 5/5.

## Stage 2: gRPC at 1,000,000 vectors (d = 384, 30 queries, 24 GB GPU and 8 GB host budget)

Branch: dev + `fix/server-message-limit` + the benchmark change that loads every cluster once before timing. Modal spend about $3.

| | Balanced | Fast |
| --- | --- | --- |
| clusters | 61 | 122 |
| ingest | 401 s, 2,494 vectors/s | 449 s, 2,228 vectors/s |
| on disk (unseeded) | 48.95 GB, 49.0 KB/vector | 49.01 GB, 49.0 KB/vector |
| no filter p50 / p95 / mean | 9.15 / 46.0 / 11.9 s | 1.96 / 11.5 / 4.03 s |
| 2 fields p50 / p95 / mean | 4.60 / 31.3 / 10.3 s | 0.098 / 17.3 / 4.71 s |
| first touch per cluster, median / p95 / max | 11.1 / 27.5 / 49.4 s | 5.0 / 12.3 / 22.4 s |
| blocks per query | 5.47 | 5.67 |

Recall gaps 0.000 at 1, 2 and 4 probes. The collection is about 1.5x the GPU plus host budget, so nearly every probe pages. The filtered variant ran second and found more clusters cached (same block count). Disk path: about 60 MB/s per cluster load (0.6 GB Balanced, 0.3 GB Fast); not profiled. Earlier run before seeding and the one-tower decrypt (T2.10): 16.2 s and 8.0 s medians.

Throughput figures on the pages are derived: 1 / median latency of the e2e-latency grid (2.5 to 15.8 queries per second, client included) and 1 / median server time (8.2 to 30.7).

## Not run

A10 rows (no ready entry point in `modal_app.py`).
