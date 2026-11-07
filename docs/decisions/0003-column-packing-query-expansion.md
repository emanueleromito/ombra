# 0003. Column packing with server-side query expansion

- **Status:** accepted
- **Date:** 2026-10-31

## Context

How vectors are laid out in CKKS slots decides query size, the number of rotations (the slowest operation) and how scores come out.

## Options considered

- **A. Row packing.** Small query, but about log2(d) rotations per database ciphertext; scores scattered.
- **B. Column packing.** No rotations, neat scores, but the query is one ciphertext per dimension (hundreds of MB).
- **C. Column packing + query expansion on the server.** Client sends one ciphertext; the server expands it once per query; search needs no rotations.
- **D. Blocked packing.** A few dimensions per ciphertext; a dial between A and B.

## Decision

C. Rotation cost is fixed per query and does not grow with database size, and scores come out one per slot, aligned with filter masks and clusters.

## Consequences

Query expansion cost must be measured in the spike. D is the fallback if expansion is too slow for small collections.
