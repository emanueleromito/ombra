# 0007. Router plus sharded nodes, clusters as the unit of distribution

- **Status:** accepted
- **Date:** 2026-11-06

## Context

The server must scale beyond one machine. In our search the GPU is mostly limited by memory reads, so batching queries is the main speed lever.

## Options considered

- **A. One process per machine, all its GPUs.** Simple, limited to one machine.
- **B. A plus sharding across machines.** A router forwards each query to the nodes holding its clusters and merges encrypted results.
- **C. Separate compute and storage.** Elastic, most complex, slow on-demand loading.

## Decision

B. Each node runs one process with a query queue and a batching scheduler over all its GPUs. A single machine is the same design with one node.

## Consequences

The router never decrypts anything. We need cluster placement, routing and failure handling. C stays an option for a managed cloud later.
