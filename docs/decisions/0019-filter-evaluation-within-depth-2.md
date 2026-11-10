# 0019. Filter evaluation within depth 2

- **Status:** proposed
- **Date:** 2026-11-10

## Context

Decision 0005 chose encrypted filters, and said the mask uses one multiplication level. Search already spends two levels of the `m0_depth2` budget (query x stored, then the mask multiply of T0.6/T1.2). A filter result is itself computed under FHE, so it must be ready before the score is masked, and it must not add a third level. This record costs the options; nothing is measured yet. Figures come from T1.4/T1.5 and T2.2: a column ciphertext holds 8192 vectors (4096 at `m0_compact_n8192`) in 786,432 B (about 394,000 B), which is **96 B per vector per column** in both presets; expansion costs about 0.047 ms per leaf on the A100 (18.33 ms at d = 384, 70.83 ms at d = 1536); the search kernel is memory-bound, so its time follows the number of stored columns read.

A field has V possible values, fixed when the collection is created. Every vector holds exactly one value per field, so a one-hot column for value v has a 1 in the slots of vectors with that value.

## Options considered

- **A. Multiplicative mask (the plan's starting point).** The client sends a one-hot query for the filter value; the server expands it and takes a dot product with the field's V one-hot columns (level 0 to 1), then multiplies the result with the score (level 1 to 2).
  - Levels: expansion 0, filter dot 1, score 1 (in parallel), mask multiply 2. Exactly depth 2, no spare level.
  - One field and OR within a field (several ones in the query) work. AND across fields does not: multiplying two field masks needs level 2, then the score needs level 3. Only declared joint fields work (columns for every pair of values, V1 x V2 columns).
  - Filtered-out vectors come out as 0, which is a legal cosine score. Without an extra shift of the score, a non-matching vector outranks a matching one with a negative score.
  - Cost per block: V extra columns read, plus one extra relinearize, rescale and ciphertext multiply (GPU path exists from T1.2, but not with a mask coming from the pipeline: needs a new oracle test).
- **B. Additive penalty (recommended).** The client puts the filter one-hots into the same query, scaled by a constant lambda (4), so the query has d + sum(V) entries. One dot product over the vector columns and all filter columns gives `score + lambda * c`, where c is the number of filter fields the vector matches (0 to F). No mask multiply.
  - Levels: expansion 0, one dot product 1. **Total 1 of 2; the second level is unused**, so the level budget and the preset do not change.
  - The client knows F, so it reads c from the band of the decrypted value (lambda = 4 and |score| <= 1 keep the bands 2 apart, against a measured score error of at most 9.0e-6), drops the vectors that do not match and subtracts lambda * F from the rest. AND across any fields, OR within a field, NOT (the other values) and "at least m of F" are all one decode rule. No new kernel: it is the existing search sum with more columns.
  - Cost: the key set and expansion grow with the leaves (about +0.047 ms per added value, about +4.6 KB of automorphism key per leaf); the search reads sum(V) more columns. Example, d = 768 and three fields of 16 values: 48 extra columns, +6.3% search read, +2.3 ms expansion. One relinearize and one rescale per block instead of two.
  - Deleted vectors (T2.7): the server adds a plaintext vector `-lambda` at the tombstoned slots (EvalAdd of a plaintext, no level), so deletes also cost no multiplication.
- **C. Binary-coded fields with an equality polynomial.** log2(V) columns per field, but equality needs a product of the bits: at least one more level for a 2-bit field, more for larger. Rejected: over the budget.
- **D. Client-built mask per block.** The client keeps the metadata, builds the 0/1 mask for each block it searches and uploads it. Nothing stored on the server, but one 786 KB ciphertext per block per query, and the client must hold every field value of every vector. Rejected for upload cost and client state.
- **E. Plaintext fields on the server.** Rejected by 0005 (the server would see values).

## Common to A and B

- **Storage:** 96 B per value per vector: a field of 8 values costs 768 B, 32 values 3,072 B, 64 values 6,144 B, against 73,728 B for a vector of d = 768 (1% to 8%). Three fields of 16 values cost 4,608 B. Columns sit in the same segments as the vector columns (inside the 0016 envelope).
- **High cardinality:** a field with thousands of values costs more than the vector itself. Hash it into at most 64 buckets and let the client check the exact value on the returned ids (extra false positives, so ask for a larger k). Tenant-style fields stay as one collection per tenant (0005 option E).
- **What the server learns:** the number of filter fields, the number of values (columns) of each, and the layout of the query. Not the values, not which value a query asks for. If a query has no filter, the client sends all-ones for the fields so the server cannot tell filtered from unfiltered queries; field names stay on the client (the server sees numbered fields). Bucketing V to 8, 16, 32 or 64 hides the exact cardinality.
- **Not an access boundary:** the client holds the secret key of the whole collection and decrypts every score in the blocks it searches. Filters shape relevance; they do not protect other tenants' data from a client with the key.
- **Left out:** ranges (dates, numbers) as such, text search, and substring match. A range can be approximated by one-hot bins (month, price band), where a range is then an OR of bins and costs the bins' columns.

## Decision

Proposed: **B**, the additive penalty, folded into the vector query. It stays within depth 2 (it uses 1), removes the AND limit of A and the zero-score ambiguity, and needs no new kernel. A is the fallback if the decode rule in B turns out to hurt recall.

Not yet checked, and required before this is accepted (T2.7, with an oracle test as for any new path): that the compact preset's score error stays far below the band gap when scores reach lambda * F, and that the decision-0012 recall rule still passes with the filter columns in the sum. Nothing in this record is measured.

## Consequences

If accepted: the collection schema lists the filter fields and their (bucketed) value counts, and the query ciphertext carries d + sum(V) entries (T2.4 message shape: `filter_fields[{id, cardinality}]` in the collection, no separate filter message). The client decodes bands (M3). T2.7 builds the extended dot product and the tombstone addition, and adds the oracle test. 0005 is amended: filters are additive penalties, not a multiplied mask. If rejected for A: T2.4 gets a filter ciphertext per query and AND across fields is limited to declared joint fields.
