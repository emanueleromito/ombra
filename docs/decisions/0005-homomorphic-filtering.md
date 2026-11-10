# 0005. Homomorphic metadata filtering

- **Status:** accepted; amended by 0019 (the filter is an additive penalty, not a multiplied mask)
- **Date:** 2026-10-31

## Context

Real RAG apps filter results (per customer, per date), but the server can't read encrypted fields.

## Options considered

- **A. Plaintext filter fields.** Fast, any filter, but the server sees the values.
- **B. Deterministic encryption.** Equality only; reveals which records share a value.
- **C. Client-side post-filtering.** Nothing revealed; poor results for selective filters.
- **D. Homomorphic filtering.** Filters evaluated under FHE as encrypted masks multiplied into the scores.
- **E. One collection per filter value.** Only works for a few fixed values like tenants.

## Decision

D. The server learns neither the field values nor which filter was used.

## Consequences

Equality filters are cheap. Range filters need encrypted comparisons and cost much more; they may come after equality filters. The mask uses one multiplication level, which the parameter presets must budget for.
