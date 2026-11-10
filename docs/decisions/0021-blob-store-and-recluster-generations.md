# 0021. Encrypted blob store and re-clustering by generation

- **Status:** proposed
- **Date:** 2026-11-10

## Context

T3.4 needs the server to hold encrypted documents without learning their sizes, and lets large files stay in the customer's storage. T3.5 needs re-clustering, but the v1 API cannot move a vector between clusters or replace the center block (ReplaceBlock keeps the cluster).

## Options considered

- **Blob sizes.** A. Four fixed buckets, 16 KiB, 256 KiB, 2 MiB, 16 MiB (`LIMIT_BLOB_BUCKET_*`), enforced by client and server; bodies above 1 MiB (client setting) default to an encrypted reference {uri, sha256, size}. B. Power-of-two buckets: finer, but leaks more size bits.
- **Blob keys.** Per blob AES-256-GCM key wrapped with the document master key; associated data binds layout version, kind, base collection name and id. Two kinds: RECORD (vector copy and metadata) and DOCUMENT (body or reference), so re-ranking does not fetch large bodies.
- **Re-clustering.** A. New collection `<base>.g<n>` under the same key pair, blobs copied byte for byte, old one searchable until `retire()`. B. New server RPCs to replace the center block and move vectors in place: cleaner names, more server work.

## Decision

Blob sizes A; the blob layout above; re-clustering A for M3.

## Consequences

The server sees only the bucket and kind of each blob and the number of blobs per id. A re-clustering briefly doubles storage and needs a client-side name switch (the client resolves the current generation). Blob replacement is not protected against rollback by the server (no version counter in the associated data). Option B stays open for after M3.
