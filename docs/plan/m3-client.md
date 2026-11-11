# M3: Client and LangChain

**Goal:** a Python client that a user can install, create a collection with, insert into and search in 10 minutes, with the LangChain adapter on top. Keys, documents and metadata stay on the client side.

**Starting point:** the M2 server and its gRPC API v1 (`m2-server.md`), the 0016 file envelope, and the decisions on decryption noise (0017) and the plaintext-DB mode (0018).
**Python:** 3.11 or newer.
**Security levels:** Fast, Decoy and Strict (decision 0004).

**Order and parallel work:**

```
T3.1 (layout) --> T3.2 (key file) --> T3.4 (documents) --+
              \-> T3.3 (presets)  --> T3.5 (collection) -+--> T3.6 (insert, delete, search) --> T3.8 (LangChain) --> T3.10 (quickstart)
                                                         |          |
                                                         |          +--> T3.9 (recall tool)
                                                         |          +--> T3.11 (filtered recall, real data)
                                                         +--> T3.7 (local mode)
```

T3.2 and T3.3 can run in parallel after T3.1. T3.6 needs the server API from M2 (T2.4, T2.7), except where it runs against the local mode (T3.7).

---

### T3.1 Python package layout

**Why:** one package layout and one API for both client styles keeps the code and the docs small.
**Depends on:** none
**Steps:** package `ombra` with `Client` (sync) and `AsyncClient` (async) exposing the same methods; share the logic in a core layer that does no I/O; typed public functions; extras `[local]` and `[langchain]` declared; version number and API version exposed.
**Done when:** both clients import and expose identical method names and signatures (checked by a test), `make test` and `make lint` pass, and the package builds a wheel.
**Status:** done

### T3.2 Key file

**Why:** the secret key must stay on the client and be recoverable without being exposed (hard rule 2).
**Depends on:** T3.1
**Steps:** a key file in the 0016 envelope holding the secret key and the document master key, protected by a passphrase (a memory-hard key derivation and an authenticated cipher, parameters recorded in the file); create, export and import; permissions 0600; never log or print key material (hard rule 4).
**Done when:** tests show a round trip, a wrong passphrase being refused, a changed byte being refused, and an unknown version being refused; no key material appears in logs or error messages.
**Status:** done
### T3.3 Presets and parameter derivation

**Why:** users choose a preset and a dimension, never ring sizes or moduli (hard rule 1).
**Depends on:** T3.1
**Steps:** map preset name and vector dimension to the parameters in `core/params`; derive the column count and block layout; refuse any combination below 128-bit security, with a clear error; the accepted preset names follow decisions 0009 and 0013.
**Done when:** a test covers every supported preset and dimension (384, 768, 1536) and a refusal for an unsupported dimension and for any parameter set below the floor; the derived values match the C++ presets.
**Status:** done
### T3.4 Documents and metadata with envelope encryption

**Why:** the server must store documents and metadata it cannot read, and re-ranking needs exact vectors.
**Depends on:** T3.2, T2.4
**Steps:** a fresh AES-256-GCM key per document, wrapped with the document master key; encrypt document text and metadata; store an AES-GCM-encrypted copy of each vector with its document, for exact re-ranking on the client; bind the document id and collection id as associated data; define the stored layout and its version. Pad every encrypted document blob to one of a few fixed size buckets so the server never sees an unpadded document size. Two storage modes per document: the whole file stored as an encrypted blob on the server, or only an encrypted reference (URI and content hash) to the customer's own storage; the reference mode is the default above a size threshold (large files). Server side: the M2 server has no blob store, so add an encrypted blob store with put, fetch-by-ID and delete (gRPC API, versioned format), under the same storage and crash-safety rules as segments (atomic write, fsync, catalog entry, recovery on restart); it stores opaque bytes only.
**Done when:** tests show a round trip, refusal of a changed ciphertext, refusal of a document moved to another id (associated data), that the stored record holds no plaintext, that stored blob sizes are only bucket sizes, that both storage modes round-trip (reference mode default above the threshold), and that the server blob store survives a crash between write and catalog commit with no partial or orphaned blob visible.
**Status:** done
### T3.5 Collection create

**Why:** clustering and encrypted centers are what make the two-stage search possible.
**Depends on:** T3.2, T3.3, T2.4
**Steps:** cluster the vectors with faiss k-means; encrypt the cluster centers; generate the evaluation keys and upload them once; create the collection on the server; a re-clustering job that builds a new clustering, swaps it in and repacks blocks as it rewrites them, dropping deleted slots via ReplaceBlock (T2.8); a standalone repack (ListBlocks, GetBlock, decrypt, drop deleted slots, re-encrypt, ReplaceBlock, from T2.8) for any block over 30% deleted slots (threshold configurable).
**Done when:** a test on a synthetic corpus creates a collection against the server (or the local mode) and the clusters, centers and keys are present; re-clustering keeps search working and the results correct during and after; a block with 30% or more deleted slots is repacked and the stored bytes fall accordingly, with unchanged results.
**Status:** done

### T3.6 Insert, delete and search

**Why:** it is the product's main path.
**Depends on:** T3.4, T3.5, T2.7
**Steps:** insert (assign to a cluster, encrypt, upload with the encrypted document and vector copy); delete; search with security levels Fast, Decoy and Strict, filters as decided in 0019 (one-hot filter columns folded into the query, lambda = 4, band decode on the client; refuse a schema whose field count is above F_max of the preset; always probe at least the clusters an unfiltered query would, and widen only if fewer than k matches, with the 0019 round, max_probe and Strict-fallback rules), top-k, and re-ranking with the exact vectors; decrypt in parallel (T2.1). The search never sends a decrypted value to the server beyond the cluster and id choices the threat model names.
**Done when:** an end-to-end test shows that top-k with re-ranking matches a plaintext search on the same data at each security level and with a filter, that a schema above F_max is refused, that a filtered query probes at least the unfiltered clusters and widens only below k matches, and that deleted vectors are not returned.
**Status:** done
### T3.7 Local mode

**Why:** a user should be able to try Ombra without a GPU server.
**Depends on:** T3.6
**Steps:** the `[local]` extra runs the CPU (OpenFHE) server in a child process behind the same API, receiving only ciphertexts and public keys; query expansion uses the server's exact C++ OpenFHE method compiled into the client (decision 0022), so local mode uses the same levels as the server; default preset `m0_compact_n8192` ("fast"); OpenFHE pinned to the server's 1.6.1, envelopes of another version refused; client `requires-python` 3.12 until T4.1; the same client code works in both modes; there is no plaintext mode. `ombra.testing.FakeClient` is an in-memory test double with the same API for users' unit tests only (no persistence, no network, never documented as a mode).
**Done when:** the end-to-end test from T3.6 runs in local mode and gives the same results; a parity test gives the same top-k through local mode and the server on the same data; search time at 10k vectors for d = 384 and 768 on 4 and 8 CPUs is in `results/`, under 5 s at d = 768 on 4 CPUs; installing the extra on a machine without a GPU works; FakeClient passes the client API signature test.
**Status:** done (parity and e2e on an A10, `results/2026-11-11-t3.11-a10-confirmation.md`; timings `results/2026-11-11-t3.7-local-mode.md`, local WSL, 2.71 s at d = 768 on 4 CPUs; install checked on a CPU-only machine from source with OpenFHE 1.6.1, wheels are T4.1)
### T3.8 `langchain-ombra`

**Why:** most users reach a vector database through LangChain.
**Depends on:** T3.6
**Steps:** package `langchain-ombra` with a `VectorStore` implementation (add texts, delete, similarity search with and without scores, `from_texts`); an example that runs end to end.
**Done when:** the example runs against the local mode, and the package passes LangChain's own vector store tests that apply.
**Status:** todo

### T3.9 Recall tool on the user's own data

**Why:** users need to see what the encryption costs them in recall, on their data.
**Depends on:** T3.6
**Steps:** build on the T1.6 recall check; a command that takes the user's vectors and a sample of queries, runs plaintext and encrypted search at the chosen settings and prints recall@10 and the gap, with no data written out.
**Done when:** the tool runs on a sample corpus and prints recall@10 for plaintext and encrypted search, and the 0012 rule's pass or fail.
**Status:** todo

### T3.11 Filtered recall on real datasets

**Why:** decision 0019 was measured on synthetic data only (`results/2026-11-10-t2.7-filter-0019.md`).
**Depends on:** T3.6
**Steps:** measure filtered recall@10 on two real public datasets at filter selectivity of about 1%, 10% and 50%, including filters correlated with the clusters, with the probe rule of 0019 (at least the clusters an unfiltered query probes, widening only below k matches); judge each case under the decision-0012 rule; record the extra rounds.
**Done when:** the numbers for both datasets are in `results/`; 0019's recall condition and the 0012 rule hold on real data, or the user is told where they do not. **Due before release.**
**Status:** todo

### T3.10 Quickstart docs

**Why:** the first ten minutes decide whether a user stays.
**Depends on:** T3.7, T3.8
**Steps:** a page that goes from install to first search (local mode first, then a server), using a 384-dimension embedding model, with copy-and-paste code; a short key-handling note (back up the key file); the limits stated in the threat model.
**Done when:** someone who has not seen the project follows the page and runs a first search in 10 minutes, and the code samples run in CI.
**Status:** todo

---

## Not in M3

- **Migration tool, key rotation and LlamaIndex adapter:** after M3.
- **Router, sharding and several nodes:** after M2.
