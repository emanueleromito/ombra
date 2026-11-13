# Local mode

`pip install "ombra[local]"` then `Client.local("./ombra-data")` runs a CPU server inside your
process. The same client code works against a GPU server with `Client("host:port")`.
Everything the server does on ciphertexts is OpenFHE on the CPU (`client/ombra/_local.py`); it
gets only a context and public evaluation keys, never the secret key, and refuses what the real
server refuses (key tag, preset, sizes, F_max, block layouts, blob buckets).

## Requirements

- CPython 3.11, 3.12 or 3.13 on Linux (x86-64 or aarch64) or macOS (arm64 or x86-64). The
  `ombra` wheel carries the compiled expansion and the OpenFHE libraries, so it needs no compiler.
- openfhe-python 1.6.1 for keys and encryption. `pip install "ombra[local]"` installs it as
  `ombra-openfhe`, our build of openfhe-python 1.6.1.0 with OpenFHE inside, for the same
  platforms as the `ombra` wheel. It has the same import name (`openfhe`) as the `openfhe`
  package on PyPI, so the two conflict: install only `ombra-openfhe`, and uninstall `openfhe`
  first if you have it.
- No GPU.

## Speed limits (d = 768, "fast" preset, 10,000 vectors, 4 logical CPUs of a Ryzen 5 5600H)

| What | Time |
| --- | --- |
| create the collection (keys, encryption, upload) | 35 s |
| one search, 2 clusters | 2.8 s |

Details and the command are in `docs/plan/results/2026-11-12-t4.1-local-mode-wheel.md`. Local mode
is for trying Ombra, tests and small collections, not for production. The "balanced" preset
(ring 16384) is about 4 times slower.

## The expansion

Local mode runs the server's exact expansion tree (`core/include/ombra/core/expansion_tree.hpp`
with the CPU operations of `core/src/openfhe_expand_ops.hpp`) in `ombra._native`, a C++ OpenFHE
extension in the wheel. It uses the same levels as the GPU server, and the score blocks come back
at the same ciphertext level (`tests/wheel_check.py` checks it on every wheel, and
`tests/test_gpu_local_parity.py` compares the top-k with the server's).
