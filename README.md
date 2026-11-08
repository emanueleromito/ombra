# Ombra

A vector database where your vectors and your search queries stay encrypted with fully homomorphic encryption (FHE). The server stores and searches only ciphertexts, so it never sees your data or what you search for.

> **Status:** early design and prototyping. Nothing here is usable yet.

## How it works

- A **Python client** holds the secret key. It encrypts vectors and queries, decrypts results and picks the top-k.
- A **server** (Docker image, GPU or CPU) stores ciphertexts and runs encrypted similarity search on GPUs with custom CUDA kernels.
- Cryptography uses the **CKKS** scheme: OpenFHE for everything security-critical, our own CUDA kernels for the fast path.
- Works with **your own embeddings** from any model, with **LangChain** and **LlamaIndex** integrations planned.

See [docs/](docs/) for the product requirements, architecture and decisions.

## Repository layout

| Folder | What lives there |
| --- | --- |
| `core/` | CUDA kernels and OpenFHE ciphertext format handling (C++/CUDA) |
| `server/` | API, storage, clustering and GPU scheduling (C++) |
| `client/` | Python client: keys, encryption, top-k, re-ranking |
| `integrations/` | LangChain and LlamaIndex adapters |
| `tests/` | Cross-component and end-to-end tests |
| `docs/` | Requirements, architecture, decisions and plan |

## Development

GPU work runs on [Modal](https://modal.com). Targets for now: NVIDIA Ampere (A10 for development, A100 for benchmarks).

```bash
make help        # list all commands
make test        # CPU tests
make test-gpu    # GPU tests on Modal
make bench       # benchmarks on Modal (A100)
```

## License

Free for personal, academic and non-commercial use under the [PolyForm Noncommercial License 1.0.0](LICENSE). **Companies need a commercial license**: see [COMMERCIAL.md](COMMERCIAL.md).
