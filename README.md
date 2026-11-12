# Ombra

Ombra is available under the [PolyForm Noncommercial License 1.0.0](LICENSE) or a commercial license. See [License](#license) for which one applies to you.

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

Ombra is source-available and dual-licensed. You may use it under either:

1. the [PolyForm Noncommercial License 1.0.0](LICENSE), free for noncommercial use, or
2. a commercial license, required for any business use, including internal use inside a company. Contact: <CONTACT_LINK_PLACEHOLDER>

You only need one of them. Source files carry the SPDX identifier `PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial`. The full legal text in [LICENSE](LICENSE) is what counts; this section is a summary.

| Who | Free? | What you need |
| --- | --- | --- |
| Individuals, hobbyists, students | Yes | Nothing |
| Universities and research institutes | Yes | Nothing |
| Charities, public bodies, non-commercial organizations | Yes | Nothing |
| Companies evaluating Ombra before buying | Yes, for a limited trial | A free evaluation license |
| Any company using Ombra for business, including internally for its own AI, products or services | No | A paid commercial license |

**What counts as commercial:** any use aimed at commercial advantage or private monetary compensation. For example, running Ombra in a product or service you sell, or inside a company for its own business operations. Personal use, research, education and use by charities and public bodies are noncommercial.

**Contributions:** outside contributors sign a Contributor License Agreement (CLA) before their first pull request is merged, so their code can be included in commercial licenses.
