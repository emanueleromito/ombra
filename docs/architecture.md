# Technical architecture

This file describes how Ombra is built today. What is not built yet is under "Planned" at the end. For user-facing pages, see the documentation site in `docs/site/`.

## Overview

The Python client holds the secret key and does all encryption, clustering, decryption and ranking. The server is one process on one machine. It only sees ciphertexts and runs the search on a GPU.

```
Client (your environment)              Server (Docker, one machine)
  your app
  Python client  <------- gRPC ------->  ombra_server
    keys and encryption                    CUDA kernels: search on the GPU, GPU cache
    clustering (faiss, child process)      storage: segment files, write-ahead log, catalog
    decryption, top-k, re-ranking
  key file (secret key never leaves)
```

The code builds for five NVIDIA architectures: `sm_75` (T4), `sm_80` (A100), `sm_86` (A10), `sm_89` (L4, L40S) and `sm_90` (H100, H200). A10 and A100 are the reference cards. Blackwell is not supported.

The server is single-tenant. An API key has a scope (`read`, `write` or `admin`), not a collection: any read key reads every collection on the server, and any write key can overwrite or delete data in any of them. Run one server per trust domain.

## FHE library and scheme

The scheme is CKKS, with a hybrid stack. OpenFHE does everything security-critical on the client: key generation, encryption and decryption. Our own CUDA kernels run the server's hot path: packed dot products and mask multiplications. OpenFHE on the CPU is the reference implementation and powers local mode.

- Security-critical code stays in a widely used library. Performance work happens in our own kernels.
- Our kernels read and write OpenFHE's ciphertext format (same parameters and representation), and every kernel is tested against OpenFHE's CPU results.
- Search needs about two levels of multiplication (dot product, then filter mask), so no bootstrapping is needed.
- One step on the client is Ombra's own: stored ciphertexts are seeded. The client replaces the uniformly random half of a ciphertext with a 256-bit seed and corrects the other half, so that the pair decrypts to the same values. The noise is the one OpenFHE sampled for its public-key encryption, `e0 + v*e_pk + e1*s`, which depends on the secret key. This is a deliberate difference from the textbook seeded form, in which the noise is independent of the key. No attack is known, and the step is in the scope of the planned external cryptographic review. Everything else in encryption stays OpenFHE's.

## Packing strategy

Vectors are stored with column packing: each ciphertext holds one dimension of many vectors, so ciphertext i holds dimension i of thousands of vectors. The client sends the query as one small ciphertext, and the server expands it once per query into one ciphertext per dimension, with that query value in every slot. Search is then only multiply-and-add, and each slot of the result holds one complete score.

```
query (1 ciphertext) --expand once--> [q1 q1 q1 ...], [q2 q2 q2 ...], ...
stored: dim 1 = [a1 b1 c1 ...], dim 2 = [a2 b2 c2 ...], ...
scores = sum_i dim_i * expanded_q_i = [score_a score_b score_c ...]
```

- Rotations (automorphisms) happen only during query expansion, once per query, so searching more vectors adds none.
- Scores come out one per slot, aligned with filter masks and cluster blocks, so masks apply directly and downloads stay compact.
- The upload per query stays small: one ciphertext, not one per dimension.

## Language and code structure

The server, storage and GPU kernels are C++ and CUDA in one codebase, using OpenFHE directly. The client is a Python package that uses openfhe-python for all cryptography, plus a compiled module for seeded ciphertexts and the CPU query expansion of local mode. Client and server talk over gRPC. Everything lives in one repository:

- **core:** CUDA kernels and the code that reads and writes OpenFHE's ciphertext format
- **server:** API, storage, GPU cache
- **client:** the Python package for keys, encryption, top-k and re-ranking
- **integrations:** the LangChain adapter

## Server architecture

One process serves one data directory. A cache keeps the most used blocks in GPU memory, backed by a second cache in host memory and then the disk. Clusters are the unit of work: a search asks for clusters, and the server scores the blocks of those clusters.

## Storage engine

Search reads whole clusters in big sequential chunks and pushes them to the GPU, so storage is built around large files rather than many small records.

- **Segment files.** Each block of ciphertexts (vectors, filter fields) is a file that is written once and never modified. New data goes into new segments. The server drops blocks whose vectors are all deleted. The client repacks blocks that are partly deleted.
- **Write-ahead log.** New writes land here first for crash safety, then go into segments.
- **Catalog database.** SQLite holds collection settings, the cluster map, deletion marks, evaluation keys and format versions.
- **Blob store.** Record and document blobs are encrypted by the client and stored as opaque bytes in fixed size buckets.
- **Backup.** Backups are plain copies of the data directory.

#### Durability rules

Implemented in `server/storage/`. Layout under the data root: `catalog.db` (SQLite, WAL mode, `synchronous=FULL`), `wal.log`, `segments/*.oseg`, `keys/*.env`, `blobs/*.oblob`. Every file has a version header, and readers refuse an unknown version. Column envelopes keep their own version, checked against the collection's preset. A stored column is either an OpenFHE ciphertext (envelope kind 2) or a seeded ciphertext (envelope version 2, kind 4): a 256-bit seed and one polynomial, about half the bytes. The server regenerates the other polynomial from the seed on the CPU when it loads the block, so the GPU holds the same words for both kinds. A collection may mix them. Clients that have the compiled module write seeded columns.

Insert, in this order; each step is durable before the next starts:

1. Append the insert-intent record (collection, cluster, segment name, ids) to `wal.log` and `fsync` it. From now on recovery knows which segment file may be an orphan.
2. Write the segment to `<name>.tmp`, `fsync` the file, `rename` it to its final name, `fsync` the directory. The file is either absent or complete under its final name.
3. Update the catalog (segment row and one row per id) in one SQLite transaction. A vector is visible to search only through the catalog, so the update is all-or-nothing.
4. Append the insert-commit record and `fsync` the log.
5. Acknowledge to the client. Never earlier.

Delete: append the delete record and `fsync`; tombstone the ids in the catalog (one transaction); acknowledge. A delete is a mark; segments are not touched.

Recovery on open replays `wal.log` in order: a torn last record is truncated; a delete record is re-applied; an intent whose segment is in the catalog is rolled forward; an intent with a commit but no catalog row is applied from the segment file; an intent with neither is discarded. Then any `.tmp` file and any segment file the catalog does not list is removed, and the log is truncated.

Guarantee: an acknowledged write survives a crash or power loss. A write that was not acknowledged is either fully there or fully absent, never partial. Limits: single writer per data directory, and the disk must honour `fsync`. Logs carry sizes, counts and timings only.

## API and protocol

All data goes over gRPC. The server also has a small HTTP port for operators, with `/healthz` and `/metrics` only. There is no REST data API and no collection management over HTTP. Each search runs as one two-way gRPC stream, so the query expanded in the first round stays on the server and is reused in later rounds.

```
client -> server: encrypted query            (server expands it once)
server -> client: encrypted center scores    (client picks clusters, + decoys)
client -> server: chosen clusters            (server reuses the expanded query)
server -> client: encrypted cluster scores   (client decrypts, top-k, re-ranks)
```

- **Contract:** `proto/ombra/v1/ombra.proto` (package `ombra.v1`). The methods and limits are on the [gRPC API](site/reference/grpc-api.md) page. Every stored message carries `api_version = 1`, and the server refuses another value. A breaking change is a new package, `ombra.v2`. Ciphertexts and keys travel inside the versioned envelope. Errors are an `Error{code, message}` in the response and never contain ids or ciphertext bytes. No message has a secret key field.
- **Filters in the API:** there is no filter message. A collection lists `filter_fields` (id and value count, no names). The query ciphertext carries d + sum(V) entries, the filter one-hots scaled by lambda = 4, and the client decodes the penalty from the decrypted scores.
- **Security:** TLS for all traffic and API keys with scopes. The server refuses to start without TLS and an API key unless `--insecure` or `--no-auth` is given, and those only work on loopback addresses.

## Client library and integrations

The Python client does more than in a normal vector DB, because the secret key lives there: keys, encryption, clustering, decryption, top-k and re-ranking.

- **Sync and async clients** with the same methods.
- **Core package** with an optional `[local]` extra that adds the CPU server (OpenFHE on CPU), so everything runs in-process without Docker. It is slow, for trying the product.
- **LangChain adapter** as a separate package, `langchain-ombra`.
- **Clustering on the client** with faiss (k-means), in a short-lived child process, when a collection is created or re-clustered.

## Packaging and deployment

The server ships as a Docker image and the client as Python wheels on PyPI.

- **GPU image:** the server, with compiled code for the five architectures above.
- **CPU image:** the client with local mode, for testing and evaluation. It has no server.
- **Single machine:** a `docker compose` file starts the server.
- **Client wheels:** Python 3.11 to 3.13 on Linux (x86 and ARM) and on macOS. There is no Windows build.

## Development environment

GPU development and benchmarks run on Modal (`modal_app.py`): A10 for day-to-day tests, A100 for benchmarks. The image is built from an NVIDIA CUDA devel base with OpenFHE and openfhe-python compiled from pinned release tags.

## Planned

Not built. None of this exists in the code today. The public version of this list is `docs/site/roadmap.md`.

- **Router and several nodes.** A router would forward each query to the nodes that hold the clusters it needs and merge the encrypted results. Clusters would be the unit of distribution.
- **Batching of queries** into one pass over the stored ciphertexts by a scheduler.
- **Mutual TLS** as an alternative to API keys, and API keys bound to collections.
- **REST admin API** with collection management.
- **Helm chart** for Kubernetes.
- **Backup to object storage** (S3, MinIO).
- **License key check** at server startup.
- **LlamaIndex adapter.**
- **Plaintext-database mode,** where the query is encrypted and the stored vectors are not.
- **Windows client wheels** and **Blackwell GPUs.**
- **Key rotation** by key switching on the server.
- **Embedding model migration tool** that builds a new collection beside the old one.
- **Blocked packing** (a few dimensions per ciphertext) as a fallback for small collections.
- **Seeded ciphertexts regenerated on the GPU,** which would also halve GPU memory.
