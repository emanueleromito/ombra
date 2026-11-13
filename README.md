# Ombra

Ombra is available under the [PolyForm Noncommercial License 1.0.0](LICENSE) or a commercial license. See [License](#license) for which one applies to you.

Ombra is a vector database where your vectors, your queries and the scores stay encrypted with fully homomorphic encryption (FHE). The server stores and searches ciphertexts only, so it never sees your data or what you search for.

You keep the secret key in a Python client. The client encrypts vectors and queries, the server computes similarity on the ciphertexts with CUDA kernels on an NVIDIA GPU, and the client decrypts the scores and picks the top results. Results are exact, because the client re-ranks the candidates itself. The cryptography is CKKS, and OpenFHE does key generation, encryption and decryption.

An encrypted search is slower than a plain one and ciphertexts are much larger than vectors. The docs say by how much.

## Try it

```bash
pip install "ombra[local]" sentence-transformers
python examples/local_search.py
```

Local mode runs a CPU server inside your process, so you need no GPU. It takes several seconds per search and is for trying Ombra and small collections. Wheels exist for Python 3.11 to 3.13 on Linux (x86_64, aarch64) and macOS 15 or newer (arm64, x86_64). There is no Windows build.

For GPU speed, run the server with `docker compose up -d` on a host with a [supported NVIDIA GPU](docs/site/operations/gpus.md), and connect with `Client("host:port", ...)`.

## Documentation

- [Getting started](docs/site/getting-started/install.md): install, quickstart, your first search, local mode or a server, examples
- [Concepts](docs/site/concepts/homomorphic-encryption.md): what FHE and CKKS are, and who holds which key
- [How it works](docs/site/how-it-works/encoding.md): encoding, query expansion, search, clusters, filters, deletes
- [Security](docs/site/security/threat-model.md): the threat model and what the server learns
- [Performance](docs/site/performance/latency-and-throughput.md): latency, throughput and storage

The `examples/` folder has runnable scripts, and the LangChain adapter is in `integrations/langchain-ombra/`.

## Repository layout

| Folder | What lives there |
| --- | --- |
| `core/` | CUDA kernels and OpenFHE ciphertext format handling (C++/CUDA) |
| `server/` | API, storage, clustering and GPU scheduling (C++) |
| `client/` | Python client: keys, encryption, top-k, re-ranking |
| `integrations/` | LangChain adapter |
| `examples/` | Runnable scripts |
| `tests/` | Cross-component and end-to-end tests |
| `docs/` | The documentation site (`docs/site/`) and internal notes |

## Development

GPU work runs on [Modal](https://modal.com). Reference cards: NVIDIA Ampere (A10 for development, A100 for benchmarks). Turing, Ada and Hopper cards are built and tested too. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
make help        # list all commands
make test        # CPU tests
make test-gpu    # GPU tests on Modal
make bench       # benchmarks on Modal (A100)
```

## License

Ombra is source-available and dual-licensed. You may use it under either:

1. the [PolyForm Noncommercial License 1.0.0](LICENSE), free for noncommercial use, or
2. a commercial license, required for any business use, including internal use inside a company. Contact: emanuele.romito@erailabs.com

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
