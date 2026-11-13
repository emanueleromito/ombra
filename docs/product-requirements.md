# Product requirements

This file says what Ombra is and what its users get. It uses the present tense only for what exists. Everything that is not built is under "Planned" at the end.

## Overview

A vector database where vectors and search queries stay encrypted with fully homomorphic encryption (FHE), so the server never sees the data or what users search for. Users run a server and talk to it through a client library that handles all encryption.

Target users are developers building RAG and semantic search on sensitive data, mainly in regulated sectors (healthcare, finance, legal, public sector).

## Architecture (user view)

The app computes embeddings locally and hands them to the client library, which encrypts them before anything leaves the user's environment. The server stores and searches only ciphertexts and returns encrypted scores, which only the client can decrypt.

```
Client (user's environment)                 Server (Docker, GPU)
  app (LangChain / custom)                    encrypted storage
  embedding model (any, run locally)            vector ciphertexts
  Python client library                         sealed records and documents
    holds the secret key      --encrypted-->    public evaluation keys
    encrypts vectors/queries  <--enc scores--  encrypted search engine (GPU)
    decrypts scores, top-k                      never sees plaintext
```

## Modes

Every vector is stored encrypted. There is one storage mode. The client computes the final ranking itself, with the exact vectors.

## Key features

1. **Server as a Docker image.** A GPU image for performance and a CPU image with the client and local mode for testing and evaluation. A `docker compose` file starts the server on one machine.
2. **Python client library.** Installed with pip. It generates and stores keys, encrypts vectors and queries, decrypts results and picks the top-k. The secret key never leaves the client.
3. **User-computed embeddings.** Users bring their own vectors, 384, 768 or 1536 dimensions, of unit length. The client refuses vectors that are not normalized.
4. **LangChain integration.** `langchain-ombra` is a vector store for LangChain.
5. **Local mode.** A CPU server in a child process of the client, so someone can try Ombra without a GPU or a server.
6. **gRPC API with streaming.** Ciphertexts are large, so the data API is gRPC. A small HTTP port serves `/healthz` and `/metrics`.
7. **Encrypted documents and metadata.** The text stored with each vector is encrypted on the client with AES-256-GCM.
8. **Core in C++/CUDA.** The client uses openfhe-python for all cryptography.
9. **A recall tool.** `ombra-recall` compares encrypted and plaintext recall on the user's own data.

## Design points

### Metadata filtering

Filters work on integer fields declared when the collection is created. The field values are encrypted, and the server evaluates filters under FHE: each filter becomes an additive penalty on the similarity scores. The server does not learn the field values or what a field means. It does learn how many filter fields there are and how many values each has. In Fast mode the number of rounds of a query also shows whether it was filtered. The cardinalities can reveal something when a field is, say, a customer id with a distinctive number of values. Bucketing the cardinalities is the caller's job: declare a larger count than needed. Only equality filters exist.

### Key management

The secret key lives in a passphrase-protected key file on the client and never leaves it. The file can be copied to share the key between apps and team members, and it works offline and in air-gapped setups. Losing the key file or the passphrase means losing the data, so users must back it up.

The key file also holds a document master key, which protects the AES keys that encrypt documents. Evaluation keys are public. The client generates them from the secret key and uploads them once when a collection is created.

### Threat model

We protect against an honest-but-curious server: it runs the software correctly but tries to learn what it can. This covers the hosting provider, insiders and attackers after a breach. A server that deliberately returns wrong or incomplete results is out of scope, and the client cannot detect it.

- **The server cannot see:** vectors, queries, documents, filter values or scores.
- **The server can still learn:** collection size, number of queries, query timing, the cluster of every record (each insert names it), the ids of all records, which clusters a query searched (except in Strict), and the ids of the records the client fetches after the top-k.
- **Ids are plaintext.** Do not put sensitive data in an id. Use random ids, such as `uuid4`.
- **The fetch after a search.** After every search, at every level, the client fetches the record blobs of its best candidates by id (3 times k by default). Because the server knows each id's cluster, this reveals which records were candidates and how large k is. The levels Decoy and Strict hide only which clusters a query requested. They do not hide this fetch.
- **Fast and Decoy modes:** the server sees which clusters the client asks for, and these choices come from decrypted results. We make no IND-CPA-D claim: decrypted values never leave the client, and the ordinal leak from fixed-noise decryption is stated, not analysed.
- **Selective filters:** a filter that matches few vectors can leave fewer than k matches in the clusters chosen, so the client searches more clusters. A filter below 0.5% expected selectivity makes the client scan every cluster, which the server can tell apart from a normal query.
- **Filter-aware cluster choice:** the client keeps encrypted per-cluster counts of each filter value and skips clusters with no expected match. Every filtered query does exactly two rounds. In Fast mode the chosen clusters can hint at the filter value, and the round count shows whether a query was filtered. Decoy and Strict always do two rounds with filter-independent request sizes.
- **Single tenant.** The server has no tenancy. Any read key reads every collection and blob on it, and any write key can overwrite or delete data in any collection.
- **Out of scope:** a malicious server, side channels, a compromised client.

### Search and top-k

Search uses a two-stage clustering index.

1. **Clustering by the client.** When a collection is created, the client clusters the vectors with k-means, since the server can't see them. Cluster centers are encrypted and stored like the vectors.
2. **Stage 1, centers.** The server scores the encrypted query against the encrypted cluster centers and returns encrypted scores. The client decrypts them and picks the best clusters.
3. **Stage 2, chosen clusters.** The server scores only the vectors in those clusters, with filter masks applied, and returns encrypted scores.
4. **Top-k on the client.** The client decrypts the scores, picks the best candidates and re-scores them exactly with their vectors from the blob store.

A small collection is a single cluster, which is the same as a full scan. The number of clusters grows with the collection, so query cost depends on the clusters searched. The number searched trades speed for recall. Clusters drift as data is inserted, so the client has a `recluster` operation, which the user runs.

Users choose the security level per collection and can override it per query. The levels cover only which clusters a query requests:

| Level | What is requested | What the server learns about the query |
| --- | --- | --- |
| Fast | Only the chosen clusters | Which clusters were requested, in rank order (a coarse hint about the topic) |
| Decoy | Chosen clusters plus random decoy clusters, shuffled | A set of clusters that contains the real ones. Repeated similar queries can be intersected |
| Strict | All clusters (full scan) | Nothing from the cluster choice |

In all three levels the fetch of record blobs by id after the search still reveals which records were candidates, and k.

### Accuracy

CKKS is approximate, so encrypted scores carry a little noise, and clustering can miss a match in a cluster that wasn't searched.

- **Presets** are tuned for a balance of speed and accuracy. Recall compared with plaintext search is published on the documentation site.
- **Client re-ranking** is always on: the client takes 3 times k candidates, fetches their vectors, and re-scores them exactly.
- **The recall tool** measures recall on the user's own data.

### Updates and deletes

A delete marks the vector as deleted: its score is masked out during search, reusing the filter mechanism. The server drops blocks whose vectors are all deleted, and the client's `repack` rewrites partly deleted blocks. There is no update call: an update is a delete plus an insert.

### FHE parameters

Users pick a preset by name: "fast" or "balanced" (the default for a server). The client derives the exact FHE parameters from the preset and the vector dimension, so users never touch low-level settings. Every preset guarantees at least 128-bit security, and the code refuses any setting below it. There is no expert mode.

### Format versioning

Every ciphertext, key file and collection format carries a version, and a reader refuses a version it does not know.

## Planned

Not built. None of this exists in the code today. The public version of this list is `docs/site/roadmap.md`.

- **LlamaIndex integration.**
- **Plaintext-database mode,** for corpora that any client may read, which would be faster. Only the query and the results would be private.
- **Range filters** (dates, numbers), which need encrypted comparisons.
- **A "precise" preset** with higher precision. The name is reserved and refused.
- **Automatic normalization** of vectors by the client.
- **Router and sharded nodes** for collections larger than one machine, and a Helm chart for Kubernetes.
- **REST API with collection management.**
- **Mutual TLS.**
- **License key check** at server startup.
- **Embedding model migration tool** with an alias switch and no downtime.
- **Key rotation** by server-side key switching for vectors, and re-wrapping of the document keys.
- **A migration tool** for older format versions.
- **Hiding the fetch after a search,** with a fixed-size padded and shuffled id set, so that Decoy and Strict cover it too.
- **Cardinality bucketing** for filter fields in the client.
- **Automatic re-clustering** and a signal for when clusters drift.
