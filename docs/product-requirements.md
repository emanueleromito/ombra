# Product requirements

_Last updated: 2026-10-31. Diagrams live in the shared design doc; this file is the text version agents read._

## Overview

A vector database where vectors and search queries stay encrypted with fully homomorphic encryption (FHE), so the server never sees the data or what users search for. Users run a server and talk to it through a client library that handles all encryption, so it feels like a normal vector DB.

Target users are developers building RAG and semantic search on sensitive data, mainly in regulated sectors (healthcare, finance, legal, public sector).

## Architecture (user view)

The app computes embeddings locally and hands them to the client library, which encrypts them before anything leaves the user's environment. The server stores and searches only ciphertexts and returns encrypted scores, which only the client can decrypt.

```
Client (user's environment)                 Server (Docker, GPU or CPU)
  app (LangChain / LlamaIndex / custom)       encrypted storage
  embedding model (any, run locally)            vector ciphertexts
  Python client library                         AES-encrypted documents
    holds the secret key      --encrypted-->    public evaluation keys
    encrypts vectors/queries  <--enc scores--  encrypted search engine (GPU)
    decrypts scores, top-k                      never sees plaintext
```

## Supported modes

Both modes are supported, chosen per collection.

| Mode | What stays private | Speed | Typical use |
| --- | --- | --- | --- |
| Encrypted query + plaintext DB | The query and the results | Faster | Private search over a corpus whose owner accepts that every client with query access can read the searched vectors (public data) |
| Encrypted query + encrypted DB | The query, the results and all stored vectors | Slower | A company's own sensitive data hosted on infrastructure it doesn't fully trust |

## Key features

1. **Server as a Docker image.** A GPU image for performance and a CPU-only image for laptops, testing and evaluation. A Helm chart for Kubernetes later.
2. **Python client library.** Installed with pip. It generates and stores keys, encrypts vectors and queries, decrypts results and picks the top-k. The secret key never leaves the client.
3. **User-computed embeddings.** Users bring their own vectors from any embedding model; the product does not depend on a specific model.
4. **LangChain and LlamaIndex integrations.** Usable as a drop-in vector store in both frameworks.
5. **Local mode.** An in-process mode with no server, so someone can try it in a notebook in two minutes.
6. **gRPC API with streaming and batch inserts.** Ciphertexts are large, so REST/JSON is not a good fit.
7. **Encrypted documents and metadata.** The original text stored with each vector is encrypted on the client with standard symmetric encryption (AES).
8. **Core in C++/CUDA with Python bindings.** Other client languages (TypeScript, Java) can be added later without rewriting the crypto.

## Design points

### Metadata filtering

Filter fields are encrypted, and the server evaluates filters under FHE: each filter becomes an encrypted mask applied to the similarity scores. The server learns neither the field values nor which filter was used. Equality filters (customer, category) are cheap; range filters (dates, numbers) need encrypted comparisons and cost much more.

### Key management

The secret key lives in a passphrase-protected file on the client and never leaves it. It can be exported and imported to share it between apps and team members, and it works offline and in air-gapped setups. Losing the key means losing the data, so users must back it up.

The key file also holds a document master key, used to protect the AES keys that encrypt documents. Evaluation keys are public, so they are generated from the secret key and uploaded once when a collection is created, then stored on the server with that collection. They are only re-uploaded after a key rotation.

### Threat model

We protect against an honest-but-curious server: it runs the software correctly but tries to learn what it can. This covers the hosting provider, insiders and attackers after a breach.

- **The server cannot see:** vectors, queries, documents, filter values or scores.
- **The server can still learn:** collection size, number of queries, query timing, which clusters a query searched (except in strict mode), and which document IDs the client fetches after top-k.
- **Fast and Decoy modes:** the server sees which clusters and IDs the client picks, and these choices come from decrypted results. We make no IND-CPA-D claim (decision 0017): decrypted values never leave the client, and the ordinal leak from fixed-noise decryption is stated, not analysed.
- **Selective filters:** a filter that matches few vectors can leave fewer than k matches in the clusters chosen, so the client searches more clusters (extra rounds on the same stream), and in Fast and Decoy modes may fall back to Strict (decision 0019). The server then sees extra rounds and the clusters that follow, so it learns that the filter was selective in the first clusters. A fixed probe budget removes this at the cost of the same latency for every query.
- **Plaintext-DB mode:** it protects the query and the results from the server, not the corpus from clients. A client can recover the stored vectors with one query per dimension, so offer the mode only for corpora any client may read in full (decision 0018).
- **Out of scope:** a server that deliberately returns wrong or incomplete results.

### Search and top-k

Search uses a two-stage clustering index, so the same design works from small to very large collections.

1. **Clustering by the client.** When a collection is created, the client clusters the vectors (e.g. k-means on a sample), since the server can't see them. Cluster centers are encrypted and stored like the vectors.
2. **Stage 1, centers.** The server scores the encrypted query against the encrypted cluster centers and returns encrypted scores; the client decrypts them and picks the best clusters.
3. **Stage 2, chosen clusters.** The server scores only the vectors in those clusters, with filter masks applied, and returns encrypted scores.
4. **Top-k on the client.** The client decrypts the scores and picks the final top-k.

A small collection is a single cluster, which is the same as a full scan. As the collection grows, the number of clusters grows with it, so query cost depends on the clusters searched, not on the whole collection. The number of clusters searched trades speed for recall. Clusters drift as data changes, so the client runs an occasional re-clustering job.

Users choose the security level per collection, and can override it per query:

| Level | What is searched | What the server learns about the query | Cost |
| --- | --- | --- | --- |
| Fast | Only the chosen clusters | Which clusters were searched (a coarse hint about the topic) | Lowest |
| Decoy | Chosen clusters plus random decoy clusters; the number of decoys is configurable | It can't tell real clusters from decoys | Grows with the number of decoys |
| Strict | All clusters (full scan) | Nothing about which vectors matched | Highest |

### Accuracy

CKKS is approximate, so encrypted scores carry a little noise, and clustering can miss a match in a cluster that wasn't searched.

- **Default parameters** are tuned for a balance of speed and accuracy, and we publish recall compared with plaintext search on standard benchmarks.
- **Client re-ranking** is on by default: the client takes a few extra candidates (2–3× k), fetches and decrypts their vectors, and re-scores them exactly. The final ranking is exact at a small extra cost, in both modes.
- **A recall tool** ships with the client: it measures recall on the user's own data and suggests settings (clusters searched, precision, re-ranking).
- **Higher precision** is available as an advanced setting, at the cost of slower search and bigger ciphertexts.
- The client **normalizes vectors** automatically, so cosine similarity works without extra steps.

### Updates and deletes

A delete marks the vector as deleted: its score is masked out during search, reusing the filter mechanism. A background compaction job later rewrites the packed blocks without the deleted vectors, freeing the space. An update is a delete plus a new insert, so the vector also moves to the correct cluster.

### Changing embedding models

The client ships a migration tool that builds a new collection alongside the old one, then switches an alias (such as "production") to it when done. There is no downtime and rollback is easy; storage is doubled during the migration.

- By default, the tool decrypts the documents already stored on the server, re-embeds them with the new model, re-clusters and re-encrypts them.
- Users who don't store documents in the DB can supply them again from their own source.
- The tool works in batches and can resume after an interruption.

### Key rotation

Key rotation uses server-side key switching. The client generates a key-switching key from the old key to the new one and sends it to the server, which converts all ciphertexts to the new key without decrypting anything. The data never leaves the server and rotation runs as a background job. The server holds the switching key only during rotation, which is safe under the honest-but-curious threat model.

Documents use envelope encryption: each document is encrypted with its own AES data key, and the data keys are encrypted (wrapped) with the document master key. To rotate, the client downloads only the small wrapped data keys, re-wraps them with the new master key and uploads them back. The documents themselves are never touched.

### FHE parameter defaults

Users pick a preset in plain words: "fast", "balanced" (the default) or "precise". The client derives the exact FHE parameters from the preset and the vector dimension, so users never touch low-level settings.

- Every preset guarantees at least 128-bit security.
- Manual parameters are available only behind an explicit "expert" flag, with a clear warning.
- The library refuses any setting below 128-bit security, even in expert mode.

### Format versioning

Every ciphertext, key file and collection carries a format version. The server and client read both the current and the previous version, and a migration tool upgrades older data. Upgrades that change FHE parameters need the secret key, so they run through the client migration tool, like an embedding model change.
