# Local mode

`pip install "ombra[local]"` then `Client.local("./ombra-data")` runs a CPU server inside your
process. The same client code works against a GPU server with `Client("host:port")`.
Everything the server does on ciphertexts is OpenFHE on the CPU (`client/ombra/_local.py`); it
gets only a context and public evaluation keys, never the secret key, and refuses what the real
server refuses (key tag, preset, sizes, F_max, block layouts, blob buckets).

## Requirements

- CPython 3.11, 3.12 or 3.13 on Linux (x86-64 or aarch64) or macOS (arm64 or x86-64). The
  `ombra` wheel carries the compiled expansion and the OpenFHE libraries, so it needs no compiler.
- openfhe-python 1.6.1 for keys and encryption. The `openfhe` wheel on PyPI is a CPython 3.12
  x86-64 Linux build only, so `pip install "ombra[local]"` pulls it in there and nowhere else;
  on other platforms install openfhe-python 1.6.1 yourself (CMake, against OpenFHE 1.6.1).
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

## The expansion

Local mode runs the server's exact expansion tree (`core/include/ombra/core/expansion_tree.hpp`
with the CPU operations of `core/src/openfhe_expand_ops.hpp`) in `ombra._native`, a C++ OpenFHE
extension in the wheel. It uses the same levels as the GPU server, and the score blocks come back
at the same ciphertext level (`tests/wheel_check.py` checks it on every wheel, and
`tests/test_gpu_local_parity.py` compares the top-k with the server's).
