# Deletes and compaction

A vector lives in one slot of each column of its block, as one number inside a ciphertext. The server cannot remove one slot from a ciphertext, because it cannot read it. So a delete works in two steps: the server marks the vector as dead right away, and the client later rewrites the blocks without the dead slots. This page covers both, and re-clustering, which rebuilds the clusters from the live vectors.

The code is in [`client/ombra/collection.py`](https://github.com/erailabs/ombra/blob/main/client/ombra/collection.py) (`delete`, `repack`, `recluster`), [`client/ombra/repack.py`](https://github.com/erailabs/ombra/blob/main/client/ombra/repack.py) (the repack plan) and [`server/search/include/ombra/search/layout.hpp`](https://github.com/erailabs/ombra/blob/main/server/search/include/ombra/search/layout.hpp) (the tombstone penalty).

## Delete and tombstones

`delete(ids)` does four things. The client opens the vector's record blob to learn its field values and finds its cluster by the nearest center, and lowers the cluster statistics ([filters](filters.md)). It tells the server to tombstone the ids, which means the server keeps the id in a set of deleted ids. It deletes the record and document blobs of those ids. Unknown ids and ids that are already deleted are ignored, and the call returns how many it deleted.

At search time the server still scores the dead slot, since it is inside the ciphertext, but it adds a penalty to that slot with a plaintext addition before it returns the scores:

$$\text{penalty} = -\lambda\,(F + 1)$$

where $\lambda = {{ const("ombra.presets.FILTER_LAMBDA") }}$ is the filter weight and $F$ is the number of filter fields of the collection (0 when there are none). The slot then holds score + $\lambda c-\lambda (F+1)$. Since $c$ is between 0 and $F$, the band the client decodes is $c-(F + 1) \le -1$ ([filters](filters.md) explains bands). A live vector that passes the filter has band $F$, so a tombstoned vector can never be a candidate.

Example with $F = 2$ and $\lambda = {{ const("ombra.presets.FILTER_LAMBDA") }}$: a vector with score 0.9 that matched both fields has slot $0.9 + {{ 2 * const("ombra.presets.FILTER_LAMBDA") }} = {{ (0.9 + 2 * const("ombra.presets.FILTER_LAMBDA")) | round(1) }}$ before the penalty. The penalty is $-{{ const("ombra.presets.FILTER_LAMBDA") * 3 }}$, so the client sees ${{ (0.9 + 2 * const("ombra.presets.FILTER_LAMBDA")-3 * const("ombra.presets.FILTER_LAMBDA")) | round(1) }}$, band $\lfloor (-3.1 + 2) / {{ const("ombra.presets.FILTER_LAMBDA") }} \rfloor = -1$. That is not 2, so it is dropped.

The penalty is also why $F_{max}$ keeps a spare unit of headroom (see the limits in [filters](filters.md)). The server adds the penalty only to blocks that have at least one tombstoned id.

What a delete does not do: it frees nothing. The ciphertexts keep their size, searching a block costs the same with or without dead slots, and the stored bytes stay as they were. A block whose slots are all dead is dropped by the server on its own. Anything else needs compaction.

## Compaction (repack)

Compaction is done by the client, because only the client can decrypt a block. `repack` asks the server for the list of blocks with their counts of used and deleted slots, and picks the clusters worth rewriting. For each one it downloads the blocks, decrypts every column, drops the dead slots, merges blocks that now fit together, encrypts the result and sends it back in a `ReplaceBlock` call. The server swaps the new block in for the old ones in one step, so searches keep working.

A cluster is rewritten when any of its blocks has at least

$$\frac{\text{deleted slots}}{\text{used slots}} \ge \tau$$

with $\tau = {{ const("ombra.repack.DEFAULT_THRESHOLD") }}$ by default (the `threshold` argument changes it). Inside a flagged cluster, the blocks that still have live vectors are packed first-fit, largest first, into groups of at most $S$ live vectors, where $S$ is {{ preset("balanced", "batch_size") }} slots on Balanced and {{ preset("fast", "batch_size") }} on Fast. Each group becomes one new block. A group is written if it holds a flagged block or merges more than one.

{% set S = preset("balanced", "batch_size") %}Example on Balanced ($S = {{ S }}$). One cluster has three full blocks of {{ S }} slots with {{ S-5000 }}, {{ S-2500 }} and {{ S-900 }} deleted. Their live counts are 5000, 2500 and 900, and the deleted fractions are {{ "%.2f" | format((S-5000) / S) }}, {{ "%.2f" | format((S-2500) / S) }} and {{ "%.2f" | format((S-900) / S) }}, all above {{ const("ombra.repack.DEFAULT_THRESHOLD") }}. Largest first: 5000, then 2500 fits ($7500 \le {{ S }}$), then 900 does not ($8400 > {{ S }}$), so it starts a second group. The result is two blocks in place of three, holding 7500 and 900 vectors. The 900 block is still mostly empty. It stays that way until more vectors are inserted into the cluster and another repack, or until the collection is re-clustered.

Repack is slow and moves data. Every column of every rewritten block, which is $d + \sum_f m_f$ ciphertexts per block, is downloaded, decrypted, encrypted and uploaded. It needs the secret key, so it runs on the client and cannot be handed to the server. Run it when the deleted fraction is high, not after every delete. It returns a report with the block counts and bytes before and after.

Each `insert_vectors` call also adds a new block per touched cluster, so a collection with many small inserts has many half-empty blocks. Repack merges those too, but only in clusters that have a flagged block.

`repack` and `recluster` are methods on the `Collection` class in `ombra.collection`. The `Client` wrapper has `delete` but does not expose them yet.

## Re-clustering

The centers are fixed when the collection is created. Deletes remove vectors from clusters, inserts add them to the nearest old center, and over time the clusters stop matching the data, which costs recall at a given number of probed clusters. Re-clustering rebuilds them.

`recluster` reads every block with live vectors, decrypts it, and runs k-means again on the live vectors on your machine, in the same child process as at creation, as in [clusters](clusters.md). It builds a new collection called `<name>.g<n>`, with the number of clusters you pass or the old count, using the same keys. The new blocks have no dead slots. The record and document blobs are copied byte for byte, because their encryption is tied to the base name and not to the generation, so they need no re-encryption.

Search keeps working on the old collection the whole time. When `recluster` returns, you switch your searches to the new collection and call `retire()` on the old one, which drops it from the server. Until then both exist, so the server stores about twice the data.

This is the slowest operation. It decrypts and re-encrypts the whole collection, runs k-means on all of it, and copies every blob over gRPC. It is meant for a maintenance window, not for regular use. There is no automatic trigger, and nothing yet that tells you when the clusters have drifted.

!!! deep-dive "Why the statistics are rebuilt, not copied"
    The cluster statistics ([filters](filters.md)) count live vectors per cluster and per field value. After re-clustering every vector can sit in a different cluster, so the new collection builds fresh statistics from the new labels and seals them as an encrypted blob. The old collection's blob is left alone.
