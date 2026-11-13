# Storage

Encrypted vectors are large. Each dimension of each vector takes 48 bytes on disk, against 4 bytes for a plain float32. That is 12 times the size, and it is the price of keeping the vectors encrypted during search.

The 48 bytes come from the ciphertext layout. A ciphertext holds one dimension of many vectors, one vector per slot. Balanced has {{ preset("balanced", "batch_size") }} slots per ciphertext and a ring dimension of {{ preset("balanced", "ring_dim") }}. Fast has {{ preset("fast", "batch_size") }} slots and a ring dimension of {{ preset("fast", "ring_dim") }}. Both end up at the same size per vector, because halving the ring halves the slots.

A ciphertext is a pair of polynomials, and one of them is a uniformly random polynomial. The client stores a 32-byte seed in its place, and the server regenerates that polynomial when it loads the block. This is called a seeded ciphertext. A stored Balanced ciphertext is 393,336 bytes for 8,192 slots, which is 48.01 bytes per dimension per vector. Without the seed it would be 787,423 bytes, or 96.12 bytes. The client writes seeded ciphertexts when its compiled module is installed, which the wheels include. The GPU holds the same data as before, so memory on the card and search speed do not change.

## Per million vectors

For one million vectors, whole blocks included, an encrypted collection takes:

| Dimension | Encrypted (seeded) | Encrypted (not seeded) | Plain float32 |
| --- | --- | --- | --- |
| 384 | 18.6 GB | 37.2 GB | 1.5 GB |
| 768 | 37.2 GB | 74.4 GB | 3.1 GB |
| 1536 | 74.3 GB | 148.8 GB | 6.1 GB |

![Storage per million vectors by dimension](img/storage.svg)

The seeded size is 12 times the float32 size at every dimension.

These figures count the stored column ciphertexts of Balanced. They are the serialized object sizes (393,336 and 787,423 bytes per ciphertext) multiplied by the number of columns and blocks. The encrypted figure rounds up to whole blocks of 8,192 vectors (123 blocks for a million), so it overstates slightly. There is no plaintext mode: every vector is stored encrypted.

## What a real collection takes

A collection takes more than the column ciphertexts alone. It also holds the keys, the sealed record blobs and the cluster centers, and clusters fill their blocks only partly. Filter fields add columns: two fields of 8 values add 16 columns, so 400 columns at d = 384.

Measured on an A100 run of the gRPC benchmark at d = 384. That benchmark's client uploads stored columns without seeding, so these sizes are the unseeded case. A client with seeding, the default in the wheels, stores the columns in about half the space. The 1,000,000 row waits for the next run.

| Vectors | Filter fields | Balanced | Fast |
| --- | --- | --- | --- |
| 100,000 | 2 fields of 8 values | 5.70 GB (57.0 KB per vector) | 4.91 GB (49.1 KB per vector) |
| 1,000,000 | 2 fields of 8 values | TBD | TBD |

The 400 columns alone would be 19.2 KB per vector at 48 bytes each. Measure your own collection before you plan disk.

## Keys

The server holds public evaluation keys, never the secret key. For Balanced they take:

| Dimension | Rotation keys | Relinearization key | Total |
| --- | --- | --- | --- |
| 384 | 26.2 MB | 2.6 MB | 28.8 MB |
| 768 | 28.8 MB | 2.6 MB | 31.5 MB |
| 1536 | 31.5 MB | 2.6 MB | 34.1 MB |

Fast keys are about 40% smaller. These keys are uploaded once per collection.

## Memory on the GPU

How many vectors fit in the memory of one A100 (40 GB), for Balanced, with 1 GiB left for scratch and one expanded query? This is calculated from the object sizes, and nothing was allocated to check it. The card holds full ciphertexts, so seeding does not change it:

| Dimension | Vectors |
| --- | --- |
| 384 | 1,105,920 |
| 768 | 548,864 |
| 1536 | 270,336 |

A bigger collection still works. The server keeps the rest on disk and in host memory and pages clusters in as queries ask for them, which is slower. See [latency and throughput](latency-and-throughput.md) for what that costs.
