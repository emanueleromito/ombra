# Choosing a preset and dimension

Start with Balanced, which is the default. Move to Fast if latency or key size matters more than the last bit of score precision. Pick the dimension from your embedding model, because it sets storage and latency more than any other choice you make.

## Balanced or Fast

Both presets meet the 128-bit security level, and both gave the same recall as plaintext search in our tests. They differ in cost.

| | Balanced | Fast |
| --- | --- | --- |
| Ring dimension | {{ preset("balanced", "ring_dim") }} | {{ preset("fast", "ring_dim") }} |
| Vectors per ciphertext | {{ preset("balanced", "batch_size") }} | {{ preset("fast", "batch_size") }} |
| Query latency, A100, relative to Balanced | 1.00 | 0.49 to 0.65 |
| Query upload | 787,423 bytes | 394,207 bytes |
| Evaluation keys | 28.8 to 34.1 MB | about 40% smaller |
| Storage per vector | same | same |
| Largest score error measured | 1.7e-8 | 9.0e-6 |
| Recall gap to plaintext, worst case | 0.0000 | 0.0004 |

The latency ratio comes from one A100 run of 20 queries per cell, 384 to 1536 dimensions, 100,000 and 1,000,000 vectors, 5% and 20% searched.

The recall gaps come from 1,000 queries on each of 3 datasets. Both presets have to stay within a recall gap of 0.005 to ship, and 0.0004 is well inside it. Fast's largest score error is about 500 times Balanced's.

Pick Balanced when you want the margin and your latency budget has room. Pick Fast for interactive search, small keys, or a smaller upload per query. There is no higher precision preset: Balanced already shows no measurable recall loss, so there is nothing for one to win.

Filter fields are limited per preset: Balanced carries up to {{ fmax("balanced") }} filter fields and Fast up to {{ fmax("fast") }}.

## Dimension

Ombra supports 384, 768 and 1536. Use the dimension your embedding model produces. If you can choose, a smaller one is cheaper on every axis:

- Storage grows in step with it: 18.6 GB, 37.2 GB and 74.3 GB per million vectors with seeded ciphertexts, the default.
- Latency grows slower than storage. On an A100 with 100,000 vectors and 5% searched, a Balanced query takes 63.2 ms at 384, 80.4 ms at 768 and 118.1 ms at 1536 in total, of which the server spends 32.6, 49.6 and 87.2 ms.
- The number of vectors that fit in the memory of one A100 (40 GB) falls from 1,105,920 at 384 to 548,864 at 768 and 270,336 at 1536. Beyond that the server pages from disk, which is slow at large sizes.

## Collection size and hardware

Up to about a million vectors at 384 dimensions, one A100 can hold everything encrypted in GPU memory. Above the GPU's memory the collection has to be paged in, and a query then takes seconds. We have not measured multi-GPU setups. At 1,000,000 vectors with small cache budgets, a query took 9.2 s at the median on Balanced and 2.0 s on Fast, see [latency and throughput](latency-and-throughput.md).

## Security level and latency

Fast, Decoy and Strict are the security levels of a collection, a separate choice from the presets above (and Fast there is not the Fast preset). Decoy adds random clusters to each query and Strict reads every cluster, so both cost more than the default. We have timed neither on its own yet, so there is no number to give. Roughly, the time follows the share of the collection a query reads, and the latency page shows how that share moves the total.

Measure with your own data before you commit. The [reproduction page](reproducing.md) says how.
