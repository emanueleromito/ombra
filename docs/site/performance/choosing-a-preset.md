# Choosing a preset and dimension

Start with Balanced, which is the default. Move to Fast if latency or key size matters more than the last bit of score precision. Pick the dimension from your embedding model, because it sets storage and latency more than any other choice you make.

## Balanced or Fast

Both presets meet the 128-bit security level, and both gave the same recall as plaintext search in our tests. They differ in cost.

| | Balanced | Fast |
| --- | --- | --- |
| Ring dimension | {{ preset("balanced", "ring_dim") }} | {{ preset("fast", "ring_dim") }} |
| Vectors per ciphertext | {{ preset("balanced", "batch_size") }} | {{ preset("fast", "batch_size") }} |
| Query latency, A100, relative to Balanced | 1.00 | 0.52 to 0.65 |
| Query upload | 787,423 bytes | 394,207 bytes |
| Evaluation keys | 28.8 to 34.1 MB | about 40% smaller |
| Storage per vector | same | same |
| Largest score error measured | 1.7e-8 | 9.0e-6 |
| Recall gap to plaintext, worst case | 0.0000 | 0.0004 |

The latency ratios come from twelve cases on one A100 (three dimensions, two collection sizes, two search shares). The recall gaps come from 1,000 queries on each of 3 datasets. Both presets have to stay within a recall gap of 0.005 to ship, and 0.0004 is well inside it. Fast's largest score error is about 500 times Balanced's.

Pick Balanced when you want the margin and your latency budget has room. Pick Fast for interactive search, small keys, or a smaller upload per query. There is no higher precision preset: Balanced already shows no measurable recall loss, so there is nothing for one to win.

Filter fields are limited per preset: Balanced carries up to {{ fmax("balanced") }} filter fields and Fast up to {{ fmax("fast") }}.

## Dimension

Ombra supports 384, 768 and 1536. Use the dimension your embedding model produces. If you can choose, a smaller one is cheaper on every axis:

- Storage grows in step with it: 37.2 GB, 74.4 GB and 148.8 GB per million vectors.
- Latency grows slower. On an A100 with 100,000 vectors and 5% searched, Balanced takes 65.5 ms at 384, 87.1 ms at 768 and 130.7 ms at 1536.
- The number of vectors that fit in the memory of one A100 (40 GB) falls from 1,105,920 at 384 to 548,864 at 768 and 270,336 at 1536. Beyond that the server pages from disk, which is slow at large sizes.

## Collection size and hardware

Up to about a million vectors at 384 dimensions, one A100 can hold everything encrypted in GPU memory, and a query took 36 to 334 ms in our runs. Above the GPU's memory the collection has to be paged in. The one test of a 1,000,000 vector collection that did not fit in the cache took seconds per query. We have not measured multi-GPU setups.

On an A10, the Fast preset on 10,003 vectors at d = 384 answered in 70 to 148 ms.

## Security level and latency

Fast, Decoy and Strict are the security levels of a collection, a separate choice from the presets above (and Fast there is not the Fast preset). Decoy adds random clusters to each query and Strict reads every cluster, so both cost more than the default. We have timed neither on its own yet, so there is no number to give. Roughly, the time follows the share of the collection a query reads, and the latency page shows how that share moves the total: from 134.8 ms at 5% to 333.6 ms at 20% of a million vectors at 384 dimensions on Balanced.

Measure with your own data before you commit. The [reproduction page](reproducing.md) says how.
