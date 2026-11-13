# Benchmark refresh, stage 1 (2026-11-13)

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

Recall gaps to plaintext 0.000 at 1, 2 and 4 probes (Balanced). Fast's no-filter p95: stage 2 first message p95 8.05 s for a few queries; not explained.

## Local mode (tests/bench_local_mode.py, Fast, 10k vectors, 2 clusters)

d = 384: create 19.3 s, search median 0.75 s. d = 768: create 36.1 s, search median 1.28 s. Top-1 5/5.

## Not run

A10 rows (no ready entry point), cold cache and throughput at 1M (stage 2).
