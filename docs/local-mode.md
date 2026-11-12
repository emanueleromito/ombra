# Local mode

`pip install "ombra[local]"` then `Client.local("./ombra-data")` runs a CPU server inside your
process. The same client code works against a GPU server with `Client("host:port")`.
Everything the server does on ciphertexts is OpenFHE on the CPU (`client/ombra/_local.py`); it
gets only a context and public evaluation keys, never the secret key, and refuses what the real
server refuses (key tag, preset, sizes, F_max, block layouts, blob buckets).

## Requirements

- CPython 3.12 on x86-64 Linux (the `openfhe` wheel on PyPI is built for it), libstdc++ from
  GCC 13 or newer (Ubuntu 24.04) and `libgomp1`.
- No GPU.

## Speed limits (measured on Modal, 4 vCPU, "fast" preset, d = 384)

| What | Time |
| --- | --- |
| create a collection of 12 vectors (keys, encryption, upload) | 5.4 s |
| one search, 1 cluster | 19.7 s |
| the end-to-end search test (about 40 searches, 150 vectors, 6 clusters) | 351 s |

A search costs about 10 key switches per query entry (d + filter values = 387 entries), plus
one ciphertext product per entry and block. The GPU server needs about two key switches per
entry. Local mode is for trying Ombra, tests and small collections (some thousands of vectors),
not for production. The "balanced" preset (ring 16384) is about 4 times slower.

## Why the expansion differs from the GPU server

openfhe-python cannot multiply a ciphertext exactly by X^-k, which the server's expansion tree
needs. Local mode gets the same entries with a real plaintext product and a subgroup trace over
the same automorphism keys (see the module docstring); results agree to about 1e-6 and the
top-k after exact re-ranking is identical (the end-to-end test runs on both backends).
