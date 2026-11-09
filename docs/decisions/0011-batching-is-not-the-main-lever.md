# 0011. Batching is not the main throughput lever

- **Status:** accepted (2026-11-09); amends the premise of 0007
- **Date:** 2026-11-09

## Context

Decision 0007 says the GPU is mostly limited by memory reads, so batching queries is the main speed lever, and builds a batching scheduler on that. T0.7 measured this on an A100 at `m0_depth2`. Against one query at a time, batched search gains 1.2x to 1.54x, and is 0.93x to 1.03x against the single-query T0.4 kernel. Batched expansion gains 1.01x to 1.05x. The search kernel holds about 1.5e11 modular multiplications per second at every batch size, using 18% to 21% of the A100's published memory bandwidth when batching. Both workloads are compute-bound. Batch size is also limited by memory: at a 34 GiB budget, 25 to 97 queries fit per search pass and 11 to 46 per expansion pass.

## Options considered

- **A. Keep 0007's premise.** Simple, but the scheduler would be designed around a gain that was not found.
- **B. Treat batching as a memory and scheduling tool, not a speed lever.** Keep a queue and group work where it reduces memory or host overhead, and look for throughput in the arithmetic instead.
- **C. Drop batching altogether.** Simpler, but T0.5b shows that batching within a query (per tree level) is worth 8.8x, so "no batching" would be wrong at that level.

## Decision

B. Batch within one query where it fills the GPU, as T0.5b does. Do not count on batching across queries for throughput on one A100. Router and sharding from 0007 are unchanged; only the premise changes.

## Consequences

The search kernel's modular multiplication becomes the first thing to speed up. The scheduler in M1 can be simpler and is sized by memory rather than by a hoped-for speedup. The result covers one A100 at this preset; it should be re-measured if the kernels or hardware change.

## Acceptance note (2026-11-09)

Accepted. If the search kernel's modular multiplication gets much cheaper (M1 tasks T1.4 and T1.5), memory bandwidth may become the limit again and batching across queries may start to pay. T1.8 re-measures and revisits this decision if so.
