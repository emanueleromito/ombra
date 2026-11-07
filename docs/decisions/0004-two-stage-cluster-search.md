# 0004. Two-stage cluster search with user-selectable security levels

- **Status:** accepted
- **Date:** 2026-10-31

## Context

A full encrypted scan is secure but grows linearly with collection size. We need a design that scales easily, while letting users choose how much the server may learn.

## Options considered

- **A. Client-side top-k over a full scan.** Simple, no leakage, but cost and download grow with size.
- **B. Homomorphic top-k on the server.** Small response, very expensive.
- **C. Clustering index.** Much faster, but the server learns which clusters were searched.
- **D. Hybrid.** Client clusters the data; stage 1 scores encrypted centers; stage 2 scores only the chosen clusters; top-k on the client.

## Decision

D, with three levels set per collection and overridable per query:
- **Fast:** only chosen clusters; server learns which clusters.
- **Decoy:** chosen clusters plus a configurable number of random decoys.
- **Strict:** all clusters (full scan); nothing learned about matches.

## Consequences

A small collection is one cluster, so it equals a full scan. The client must run occasional re-clustering. The threat model documents cluster-access leakage.
