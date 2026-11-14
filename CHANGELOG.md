# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- API (`ombra.v1`, not released yet): `Insert` is a client stream of parts like `ReplaceBlock`: one
  `InsertHeader` per block, then one column per message (`InsertRequest`). The server's default
  receive limit falls from 1 GiB to 16 MiB (`--max-message-mb`), and the memory quota default from
  4096 to 2048 MiB. `InsertBlock` is no longer sent by any call.
- `Insert` is idempotent: a block whose ids all exist in one stored block of the same cluster with
  byte-identical columns is a success that stores nothing. The client now retries `Insert`.
- `SearchScores` has `block_sizes`, so each answer describes its own blocks; the client no longer
  calls `ListBlocks` during a search, and a reconnected search stream checks the answers it skips.
- New error code `ERROR_STORAGE_FULL` (the server also sends `RESOURCE_EXHAUSTED`). Search and key
  upload limits are now `RESOURCE_EXHAUSTED` too.

### Added

- `ombra_server`: graceful shutdown with `--shutdown-grace-s` (default 30), `GET /readyz`, and a
  forced exit on a second signal.
- Storage: big files are written without the store lock, so a slow disk no longer blocks reads.

## [0.1.0]

First release. Ombra is a vector database where vectors, queries, filters and scores stay encrypted
with fully homomorphic encryption (CKKS, OpenFHE 1.6.1), at 128-bit security or more. The licence
is PolyForm Noncommercial 1.0.0 or a commercial licence; the project is source-available.

### Added

- Core: CUDA kernels for encrypted search on Ampere GPUs (`sm_80`, `sm_86`): RNS arithmetic, NTT,
  key switching, rescale, encoding, and a column-packed search kernel. Every kernel is checked
  against OpenFHE on the CPU.
- Parameter presets (`fast`, `balanced`, `compact`) with a 128-bit security floor enforced in C++
  and Python, and recall checks on real embeddings.
- Versioned file formats for ciphertexts, keys and collections.
- Server: a gRPC API (`ombra.v1`), segment storage, a GPU cluster cache, a search service,
  compaction of deleted vectors, TLS, API keys with scopes, and a REST admin and metrics port.
- Client (`ombra`): key file protected by a passphrase, parameter derivation from a preset,
  collection create, insert, delete and search, encrypted documents and metadata, parallel
  decryption and top-k, exact re-ranking, and a recall tool for your own data.
- Local mode: `Client.local(...)` runs the server's computation on the CPU in your process, with no
  GPU.
- `langchain-ombra`: a LangChain vector store.
- Wheels for Linux (x86-64, aarch64) and macOS (arm64, x86-64), CPython 3.11 to 3.13, and
  `ombra-openfhe` (openfhe-python 1.6.1.0 with OpenFHE 1.6.1 inside, version 1.6.1.0).
- Docker images: `ombra-server` (CUDA runtime, `sm_80` and `sm_86`), `ombra-cpu` (client with local
  mode), and a `docker-compose.yml` with a persistent volume.
- CI on GitHub runners, a release workflow with trusted publishing, and a quickstart.

### Known limits

- The server needs an NVIDIA Ampere GPU. Without one, use local mode, which is for trying Ombra and
  for small collections.
- Search is cosine similarity on unit vectors; the final scores are exact and computed on your
  machine after decryption.

[Unreleased]: https://github.com/erailabs/ombra/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/erailabs/ombra/releases/tag/v0.1.0
