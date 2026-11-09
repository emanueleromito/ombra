# 0018. Plaintext-DB mode: the corpus is readable by clients

- **Status:** accepted
- **Date:** 2026-11-09

## Context

`docs/product-requirements.md` offers "encrypted query + plaintext DB" for a corpus the server owner already holds, "public or licensed data". The result holds one score per vector in each searched block. A client that encrypts the unit vector e_j, which passes the norm check, receives coordinate j of every vector, so d queries recover each searched block (finding F10, `docs/plan/results/m1-crypto-review.md`). The norm check runs on the client, and the server cannot verify anything about an encrypted query. CKKS noise in the result is not flooded in this mode either. The threat model covers only a curious server, so this loss for the data owner is undocumented.

## Options considered

- **A. Keep the mode for public or licensed data and document the limit.** The query is protected and the corpus is not. It leaves "licensed" data on offer, though a licence may well forbid handing the vectors to every client.
- **B. Restrict the mode to corpora any client is allowed to read in full.** The query is protected; the corpus is treated as public to its clients. Honest about what the mode does, at the cost of a narrower market.
- **C. Protect the corpus too, or remove the mode.** Protection would need a server-side check on encrypted queries, or noise added to results, neither designed nor measured. Removing the mode gives up its speed benefit for public corpora.

Rate limits do not solve it: at d = 384 to 1536, recovery takes one query per dimension.

## Decision

B. This mode protects the client's query and results from the server, and does nothing to protect the corpus from clients. Offer it only for corpora whose owner accepts that any client with query access can read the searched vectors. Remove "licensed" unless the licence allows that.

## Consequences

Accepted 2026-11-09 (option B). `docs/product-requirements.md` now states the limit in the mode's row and in the threat model. Option C stays open if a customer needs a protected corpus; it would need its own design and measurements.
