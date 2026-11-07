# Technical architecture

_Last updated: 2026-11-07. Diagrams live in the shared design doc; this file is the text version agents read._

## Overview

The Python client holds the secret key and does all encryption, clustering, decryption and ranking. The server, one or more machines behind a router, only ever sees ciphertexts and runs the search on GPUs.

```
Client (your environment)              Server (Docker, one or more machines)
  your app                               router: forwards queries, merges encrypted results
  Python client  <------- gRPC ------->    |
    keys and encryption                  node (one per machine)
    clustering (faiss)                     CUDA kernels: batched search on all GPUs, GPU cache
    decryption, top-k, re-ranking          storage: segment files, write-ahead log, catalog
  key file (secret key never leaves)
```

**First version targets NVIDIA Ampere only** (`sm_80` A100, `sm_86` A10G). Other generations come later.

## FHE library and scheme

We use the CKKS scheme with a hybrid stack. OpenFHE handles everything security-critical on the client: key generation, encryption and decryption. Our own CUDA kernels run the server's hot path: packed dot products, rotations and mask multiplications. OpenFHE on the CPU also serves as the reference implementation and the CPU fallback.

- Security-critical code stays in a widely used, reviewed library; performance work happens in our own kernels and can grow over time.
- Our kernels read and write OpenFHE's ciphertext format (same parameters and representation), and every kernel is tested against OpenFHE's CPU results.
- Search needs only about two levels of multiplication (dot product, then filter mask), so no bootstrapping is required.

## Packing strategy

Vectors are stored with column packing: each ciphertext holds one dimension of many vectors, so ciphertext i holds dimension i of thousands of vectors. The client sends the query as one small ciphertext, and the server expands it once per query into one ciphertext per dimension, with that query value in every slot. Search is then only multiply-and-add, with no rotations, and each slot of the result holds one complete score.

```
query (1 ciphertext) --expand once (rotations)--> [q1 q1 q1 ...], [q2 q2 q2 ...], ...
stored: dim 1 = [a1 b1 c1 ...], dim 2 = [a2 b2 c2 ...], ...
scores = sum_i dim_i * expanded_q_i = [score_a score_b score_c ...]   (no rotations)
```

- Rotations happen only during query expansion, once per query, so the cost of searching more vectors does not include any rotations.
- Scores come out one per slot, aligned with filter masks and cluster blocks, so masks apply directly and downloads stay compact.
- The upload per query stays small: one ciphertext, not one per dimension.
- Fallback: blocked packing (a few dimensions per ciphertext) if query expansion turns out too slow for small collections. The spike measures both.

## Language and code structure

The server, storage and GPU kernels are written in C++ and CUDA in one codebase, using OpenFHE directly with no translation layer. The client is a Python package that uses OpenFHE's official Python bindings (openfhe-python) for all cryptography. Client and server talk over gRPC.

Everything lives in one repository, in four parts:

- **core:** CUDA kernels and the code that reads and writes OpenFHE's ciphertext format
- **server:** API, storage, clustering and GPU scheduling
- **client:** the Python package for keys, encryption, top-k and re-ranking
- **integrations:** LangChain and LlamaIndex adapters

Memory-safety risk in the C++ server code is reduced with modern C++ practices, sanitizers and fuzz testing. A Rust server is the option to revisit if the server code or the team grows.

## Server architecture

The server runs across one or more machines (nodes). A router receives each query, sends it to the nodes that hold the clusters it needs, and merges their encrypted results before returning them to the client.

- **Clusters are the unit of distribution.** Each cluster lives on one node, so adding nodes spreads clusters and storage over more GPUs.
- **One process per node, using all its GPUs.** Incoming queries go into a queue, and a scheduler groups them into batches, so each stored ciphertext is read from memory once and multiplied with many queries.
- **The router never decrypts anything.** It only forwards ciphertexts and merges encrypted scores.
- **A single machine is the same design with one node**, so small deployments stay simple.

## Storage engine

Search reads whole clusters in big sequential chunks and pushes them to the GPU, so storage is built around large files rather than many small records.

- **Segment files.** Each cluster's ciphertexts (vectors, filter fields) are stored in large files that are written once and never modified. New data goes into new segments; compaction merges segments and drops deleted vectors.
- **Write-ahead log.** New writes land here first for crash safety, then are flushed into segments.
- **Catalog database.** A small embedded database (SQLite or RocksDB) holds collection settings, the cluster map, deletion marks, evaluation keys and format versions.
- **GPU memory cache.** The most-used clusters stay in GPU memory, so popular queries skip the disk entirely.
- **Backup.** Backups are plain file copies; backup to object storage (S3, MinIO) comes later, mainly for cloud deployments.

## API and protocol

All data goes over gRPC; a small REST API handles admin and monitoring. Each search runs as one two-way gRPC stream, so the query expanded in the first round stays on the server and is reused in the second round instead of being expanded again.

```
client -> server: encrypted query            (server expands it once)
server -> client: encrypted center scores    (client picks clusters, + decoys)
client -> server: chosen clusters            (server reuses the expanded query)
server -> client: encrypted cluster scores   (client decrypts, top-k, re-ranks)
```

- **gRPC data API:** searches, inserts, deletes and evaluation key uploads. Big uploads (vector batches, evaluation keys) are sent in chunks over streams.
- **REST admin API:** health checks, metrics and collection management, easy to script and monitor.
- **Security:** TLS for all traffic, plus API keys or mutual TLS so only authorized clients connect.
- **Versioning:** the API is versioned (v1, v2, ...) so older clients keep working after upgrades.

## Client library and integrations

The Python client does more than in a normal vector DB, because the secret key lives there: keys, encryption, clustering, decryption, top-k and re-ranking. It is packaged so the core install stays simple and each integration looks native in its framework.

- **Sync and async clients** with the same methods, since LangChain and LlamaIndex both rely on async in production.
- **Core package** with an optional `[local]` extra that bundles the CPU server (OpenFHE on CPU), so everything runs in-process without Docker. Slow, but ideal for trying the product in a notebook.
- **Separate integration packages** following each framework's naming convention (`langchain-ombra`, `llama-index-vector-stores-ombra`), released independently.
- **Clustering on the client** with faiss (k-means) when a collection is created or re-clustered.

## Packaging and deployment

The server ships as two Docker images on a public registry (GitHub Container Registry or Docker Hub), and the client ships as Python wheels on PyPI.

- **GPU image:** one image with compiled code for the supported GPU generations, plus generic code the driver compiles for future GPUs. First version: Ampere only (`sm_80`, `sm_86`); Ada, Hopper and Blackwell are added later.
- **CPU image:** for laptops, testing and evaluation.
- **Single machine:** a `docker compose` file starts the server with one command.
- **Multiple machines:** a Helm chart deploys the router and the nodes on Kubernetes, using the NVIDIA GPU Operator for GPU access.
- **Client wheels:** Linux (x86 and ARM) and macOS, where local mode runs on CPU. Windows comes later.
- **License check:** the server checks the license key at startup, following the licensing decision: free non-commercial keys, paid commercial keys, and offline license files for air-gapped setups.

## Development environment

GPU development and benchmarks run on Modal (`modal_app.py`): A10G for day-to-day tests, A100 for benchmarks. The image is built from an NVIDIA CUDA devel base with OpenFHE and openfhe-python compiled from pinned release tags.
