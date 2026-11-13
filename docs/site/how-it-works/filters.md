# Filters

The server cannot read your metadata, so it cannot filter on it. Ombra moves the filter into the score. Each vector carries its filter values as extra encrypted columns, the query asks for them with a weight, and a vector that fails the filter ends up in a different range of score values. The client decrypts the scores, reads the range and drops the vectors that failed. The server never learns which fields were filtered.

The code is in [`client/ombra/search.py`](https://github.com/erailabs/ombra/blob/main/client/ombra/search.py) (`filter_query`, `decode_band`), [`client/ombra/presets.py`](https://github.com/erailabs/ombra/blob/main/client/ombra/presets.py) (`filter_field_limit`) and [`client/ombra/probe.py`](https://github.com/erailabs/ombra/blob/main/client/ombra/probe.py) (cluster statistics and the planner).

## The additive penalty

A collection can have $F$ filter fields. Field $f$ takes one of $m_f$ values, numbered $0$ to $m_f-1$ (a language field with 5 languages has $m_f = 5$). Each field becomes $m_f$ extra columns next to the $d$ vector columns, a one-hot encoding: the column for value $v$ holds 1 for the vectors whose field equals $v$ and 0 for the rest (see [encoding](encoding.md)). The query vector grows the same way, to $d + \sum_f m_f$ entries.

For the $d$ vector entries the query holds the normal query values $q_j$. For the filter entries it holds the weight $\lambda = {{ const("ombra.presets.FILTER_LAMBDA") }}$ at each allowed value of a filtered field and 0 at the others. A field you do not filter has $\lambda$ at every value, so every vector matches it. The server computes the same sum over all columns for every query ([encrypted search](encrypted-search.md)), so it cannot tell a filtered query from an unfiltered one by the shape of the work.

What comes out for one vector is

$$\text{slot} = \underbrace{\sum_{j} q_j v_j}_{\text{score}} + \lambda c$$

where $c$ is the number of fields on which the vector matches the filter. A field that is not filtered always counts as matching. A vector passes the filter when $c = F$.

Example with $\lambda = {{ const("ombra.presets.FILTER_LAMBDA") }}$ and two fields, language (3 values) and year (2 values). The filter is language in {0, 2} and year = 1. The query extension is $[4, 0, 4,\ 0, 4]$. A vector with language 2, year 1 gets $4 + 4 = 8$ on top of its score, so $c = 2$. A vector with language 1, year 1 gets $0 + 4$, so $c = 1$. A vector with language 0, year 0 gets $4 + 0$, also $c = 1$.

Matching inside a field is OR, across fields it is AND. Fields and values are numbered integers; there are no ranges and no text matching.

## Bands

The client has the decrypted slot value, which is the score (a cosine in $[-1, 1]$) plus a multiple of $\lambda$. Because $\lambda$ is larger than the score range, each value of $c$ owns its own band of width $\lambda$ and the two parts can be separated:

$$c = \left\lfloor \frac{\text{slot} + 2}{\lambda} \right\rfloor \qquad \text{score} = \text{slot}-\lambda c$$

The shift by 2 centers each band on $\lambda c$, so it covers $[\lambda c-2,\ \lambda c + 2)$. A score of at most 1 in size leaves a margin of 1 on each side for the small noise CKKS adds (see [CKKS](../concepts/ckks.md)).

Example with $\lambda = {{ const("ombra.presets.FILTER_LAMBDA") }}$. Slot value 8.3 gives $c = \lfloor 10.3 / 4 \rfloor = 2$ and score $8.3-8 = 0.3$. Slot value 3.7 gives $c = \lfloor 5.7 / 4 \rfloor = 1$ and score $3.7-4 = -0.3$. With two fields only the first one passes, and the client keeps vectors with $c = 2$. The score it keeps is the approximate one. The final ranking uses the exact vectors from the record blobs.

## Where tombstones fit

A deleted vector stays in its block until the block is rewritten. The server adds $-\lambda(F + 1)$ to its slot. With $c$ between 0 and $F$, the band index drops to at most $-1$, below every live vector, and never equals $F$. So a deleted vector can never pass the filter, with or without a filter in the query. [Deletes and compaction](deletes.md) has the rest.

## Limits per preset

The decrypted slot must stay inside the range CKKS can represent at the lowest level. For a first modulus of $q_0$ bits and a scale of $\Delta$ bits, that range is values below $2^{q_0-\Delta-1}$. The client takes 3 bits less as a margin:

$$A = 2^{\,q_0-\Delta-4} \qquad F_{max} = \min\!\left(F_{cap},\ \left\lfloor \frac{A-1}{\lambda} \right\rfloor-1\right)$$

where $F_{cap}$ is the server's cap on filter fields, {{ const("ombra.presets.MAX_FILTER_FIELDS") }}. A field adds at most $\lambda$ to the slot, so $F$ fields add at most $\lambda F$, and the extra $-1$ leaves room for the tombstone penalty.

{% set fq = preset("fast", "first_mod_size") %}{% set fs = preset("fast", "scaling_mod_size") %}{% set bq = preset("balanced", "first_mod_size") %}{% set bs = preset("balanced", "scaling_mod_size") %}{% set lam = const("ombra.presets.FILTER_LAMBDA") %}Fast has $q_0 = {{ fq }}$ and $\Delta = {{ fs }}$, so $A = 2^{ {{ fq-fs-4 }} } = {{ 2 ** (fq-fs-4) }}$ and the formula gives $\lfloor {{ 2 ** (fq-fs-4)-1 }} / {{ lam }} \rfloor-1 = {{ (2 ** (fq-fs-4)-1) // lam-1 }}$, which the cap brings down to $F_{max} = {{ fmax("fast") }}$. Balanced has $q_0 = {{ bq }}$ and $\Delta = {{ bs }}$, so $A = 2^{ {{ bq-bs-4 }} } = {{ 2 ** (bq-bs-4) }}$ and $\lfloor {{ 2 ** (bq-bs-4)-1 }} / {{ lam }} \rfloor-1 = {{ (2 ** (bq-bs-4)-1) // lam-1 }}$, so $F_{max} = {{ fmax("balanced") }}$. Balanced has the larger scale, so less room is left for the filter weight.

Creating a collection with more than $F_{max}$ fields fails. Two more limits apply: a field has between 1 and {{ const("ombra.presets.MAX_FILTER_CARDINALITY") }} values, and $d + \sum_f m_f$ must be at most half the ring dimension ({{ preset("fast", "ring_dim") // 2 }} for Fast, {{ preset("balanced", "ring_dim") // 2 }} for Balanced). A field with many distinct values, such as a user id, does not fit. Bucket it into at most {{ const("ombra.presets.MAX_FILTER_CARDINALITY") }} values or keep it out of the filter.

## Choosing clusters for a filter

With a filter, the best clusters by center score may hold no matching vectors at all. The client keeps statistics for this: per cluster the number of live vectors, and per field and value how many of them have that value. The statistics are stored as an encrypted blob. The server does not see them. The client updates them on insert and delete and rebuilds them when it re-clusters.

From these counts it estimates the matches in cluster $i$ as

$$E_i = n_i \prod_{f \in \text{filtered}} \frac{n_{i,f,A_f}}{n_i}$$

where $n_i$ is the number of live vectors in the cluster, and $n_{i,f,A_f}$ the number of them whose field $f$ is in the allowed set $A_f$. The product treats the fields as independent. When they are correlated the estimate is off, and the result is still correct, since every candidate is checked exactly by its band; only the cluster choice suffers.

Example: a cluster of $n_i = 1000$ vectors, 100 with language in {0, 2} and 500 with year 1. $E_i = 1000 \times 0.1 \times 0.5 = 50$.

The first round goes down the clusters in center-score order. It skips clusters with $E_i = 0$ and adds clusters until the running sum of $E_i$ reaches $3k$ ($k$ is the number of results asked for), but never fewer clusters than an unfiltered query would take. With $k = 10$ it aims for 30 expected matches. In Decoy the real part of round 1 is capped at $2p$ clusters (see [clusters](clusters.md)).

## The extra round

A filtered query always makes a second round: the next clusters with $E_i > 0$, twice as many as round 1 used. It runs even if round 1 found enough matches. That keeps the round count the same for every filtered query at a level. The cost is a second trip to the server, and more scoring. There is no third round. If fewer than $k$ vectors passed the filter after two rounds, the result says `incomplete` and is not padded.

Without statistics (a collection opened without its stats blob), a filtered query still makes two rounds, but picks clusters by center score only, as an unfiltered Decoy query does: the first $p$ clusters, then the next $2p$.

If the estimated selectivity, $\sum_i E_i / \sum_i n_i$, is under 0.5%, Fast and Decoy give up planning and scan all clusters in two rounds with no decoys. That is slow, and the server sees it. [Clusters](clusters.md) covers it, and [what the server learns](../security/what-the-server-learns.md) says what it reveals.

In Fast, an unfiltered query normally makes one round and a filtered query two, so the server can tell them apart there. Decoy and Strict use two rounds for every query.

## What the server sees of the filter

The server does not see field values, names or which values a query allows. It does see the number of filter fields and the number of values of each field, exactly. A field with a distinctive number of values can say something about what it holds. Reducing that leak, for example by declaring a larger value count than you need so that fields look alike, is up to you: the client does not bucket the counts. In Fast, the round count also shows whether a query was filtered, as above.

!!! deep-dive "Why not stop at k matches in round 1"
    The first version stopped as soon as the first round had $k$ matches. At around 1% selectivity this lost recall, because the matches found first were not the nearest ones. The planner now works from the statistics and always takes a second round. Its weak spot is the independence assumption above.
