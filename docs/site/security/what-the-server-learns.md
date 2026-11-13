# What the server learns

This page lists what an honest-but-curious server can see, by search level. It follows the client code, not the design goals: where the code does less than we would like, the page says so.

The three levels are Fast, Decoy and Strict. Each collection has a default level, and a query can override it. There is no fourth, plaintext-database mode. Every vector is encrypted in every collection, so there is nothing to list for it.

## In every mode

The server never sees vectors, queries, scores, filter values, document text, metadata values or the secret key. It does see the following, whatever the level.

- **Vector ids.** You choose the ids and they are stored in plaintext. They arrive with each insert and delete, and they come back with every score the server returns. Use random ids.
- **Cluster membership.** Each insert says which cluster a block belongs to, so the server knows which ids sit in the same cluster and how many vectors each cluster holds. It does not know what a cluster is about.
- **Collection shape.** The preset, the vector dimension, the number of clusters, the number of blocks, the number of vectors (tombstoned ones included until they are compacted), and the filter layout: how many filter fields there are and how many values each has. It does not see what a field means, or its values.
- **Deletes.** The ids you delete, and when. A deleted vector stays as a tombstone in its block, still scored and returned, until the block is dropped or repacked. The count of deleted vectors per block is visible. See [Deletes](../how-it-works/deletes.md).
- **Blob sizes and ids.** Every record and document blob is padded to one of four sizes ({{ const("ombra.blobstore.BUCKET_SIZES")[0] }}, {{ const("ombra.blobstore.BUCKET_SIZES")[1] }}, {{ const("ombra.blobstore.BUCKET_SIZES")[2] }} or {{ const("ombra.blobstore.BUCKET_SIZES")[3] }} bytes). The server sees the size class, not the true length. A vector's record and document use the same id as the vector. The client also stores its per-cluster statistics as one sealed blob under a reserved id, which the server sees as a blob like any other.
- **Query count and timing.** When each query arrives, how long each round takes, and how many bytes move.
- **Candidate ids after the search.** To re-rank, the client fetches the record blobs of its best `rerank_factor * k` candidates (3 times k by default, never fewer than k). The server sees those ids. It cannot tell which of them end up in the final k, and it cannot read the vectors inside, but the ids are near neighbours of the query. A document fetched afterwards is one more blob request. These fetches happen in Strict too.
- **Keys that are public by design.** The key tag, the crypto context and the evaluation keys (relinearization and automorphism keys), uploaded once per collection. The public encryption key stays on the client.
- **Network metadata.** The client address, the API key and request sizes, as for any service.

What stays hidden in every mode: the query vector, the scores, the filter values, and which of the returned candidates are the real top k. An unfiltered query and a filtered one have the same encrypted query format, because unused filter entries carry a fixed value, so the query ciphertext does not say whether a filter was applied.

## Fast

Fast scores only the clusters whose centers are closest to the query. The server learns this on top of the common list.

- **The clusters asked for.** The cluster ids in each request are the clusters nearest to the query, which is a coarse hint at its topic. The ids arrive in rank order, best first, so the order shows which cluster scored highest.
- **The first-round size.** By default the square root of the cluster count, rounded up, or whatever `n_probe` is set to.
- **Whether the query was filtered.** An unfiltered query normally takes one round and a filtered one takes exactly two, so the round count gives it away. An unfiltered query takes more rounds when the first returns fewer than k candidates.
- **Hints about the filter value.** A filtered query skips clusters that the client's statistics say have no matches, so the clusters it picks can point at the filter value.
- **A rare filter.** When the expected selectivity is under 0.5%, the client scans every cluster in two rounds. The server sees a full scan, split at the middle cluster id, and knows the filter matched very little.

## Decoy

Decoy does exactly two rounds for every query. Each round asks for a fixed number of clusters that depends only on the first-round size and the cluster count: twice the first-round size in round 1, then four times the first-round size (or whatever is left) in round 2. The real clusters are mixed with decoys picked at random from clusters not yet asked. A cluster is never asked twice in one query.

What the server learns on top of the common list:

- **The number of rounds and clusters per round.** Always two rounds with fixed counts, so it cannot tell a filtered query from an unfiltered one, or a narrow query from a wide one.
- **A set of clusters that contains the real ones.** By set membership alone, real and decoy clusters look the same.
- **A full scan for a rare filter.** As in Fast. This scan has different request sizes from a normal Decoy query, so it stands out.

!!! warning "Decoy does not hide the real clusters from a server that reads the order"
    In the current client, each request lists the real clusters first, in score order, and appends the decoys after them. The server keeps that order. A server that looks at the order of the ids in a request can read off the real clusters, and Decoy gives no protection against it. Until the client shuffles the ids, treat Decoy as protection against a server that only looks at the set of clusters, and use Strict when the cluster choice must stay hidden.

Even with a shuffle, decoys protect one query at a time. Over many queries on the same topic the real clusters repeat and the decoys do not, see [Out of scope](out-of-scope.md).

## Strict

Strict asks for every cluster in two rounds: the first half of the cluster ids, then the second half, in a fixed order that does not depend on the query. A filtered query looks the same as an unfiltered one, and so does a rare-filter query, because the full scan is already the normal plan.

On top of the common list, the server learns nothing about the query from the cluster choice, because every query asks for everything.

What Strict does not hide is the candidate ids fetched for re-ranking, the count and timing of queries, and the cost. A Strict query scores every vector, so it is the slowest level. See [Clusters and search levels](../how-it-works/clusters.md).

## Side by side

| What the server sees | Fast | Decoy | Strict |
| --- | --- | --- | --- |
| Which clusters the query is closest to | yes, with rank order | yes, if it reads the id order (see the warning) | no |
| Number of rounds | 1 unfiltered, 2 filtered | always 2 | always 2 |
| Whether a filter was used | yes, from the round count | no, unless the filter is rare | no |
| Rare filter (under 0.5%) | visible as a full scan | visible as a full scan | not visible |
| Record ids fetched for re-ranking | yes | yes | yes |
| Vectors, queries, scores, filter values | no | no | no |

!!! deep-dive "Where each item comes from"
    The level logic is `run_search` in `client/ombra/search.py`. Its `run_round` pads Decoy rounds with `rng.sample` and appends the decoys after the real ids, `scan_all` splits a full scan with `strict_halves`, and `_rerank` fetches the record blobs. `decoy_round_sizes` and the filter-aware `ProbePlanner` are in `client/ombra/probe.py`. On the server, `server/grpc/src/service.cpp` answers the clusters of a request in the order it received them, and logs only counts, sizes and timings, never ids. The messages are in `client/ombra/resources/ombra.proto`: the filter layout travels as `FilterField` cardinalities, and the blob sizes are the `LIMIT_BLOB_BUCKET_*` values.
