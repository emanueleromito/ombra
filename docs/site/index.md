# Ombra

Ombra is a vector database where the vectors, the queries and the scores stay encrypted. The server stores and searches ciphertexts only, so it never sees your data or what you ask for. It uses fully homomorphic encryption (FHE), and the search runs on GPUs.

You hold the secret key in a Python client. The client encrypts your vectors and your query, the server computes similarity scores on the ciphertexts, and the client decrypts the scores and picks the top results. Results are exact: encryption costs no recall, and the client re-ranks the candidates itself.

## Who it is for

Teams that want similarity search over embeddings (documents, chat history, customer records) without trusting the host with the plain data. The cost is speed and size: an encrypted search is slower than a plain one and ciphertexts are much larger than vectors. [Performance](performance/latency-and-throughput.md) has the numbers.

Ombra works with embeddings from any model that gives 384, 768 or 1536 dimensions. It has filters on integer fields, inserts, deletes and a LangChain adapter. It needs an NVIDIA GPU for the server, and there is a slow CPU local mode for trying it.

## Where to start

- To run something now, go to [Install](getting-started/install.md) and then the [quickstart](getting-started/quickstart.md).
- To understand the ideas first, read [what homomorphic encryption is](concepts/homomorphic-encryption.md).
- To see what the server can and cannot learn, read the [threat model](security/threat-model.md).
- To see how the search works, start at [encoding](how-it-works/encoding.md).

## License

Ombra is source-available. It is free for noncommercial use under the PolyForm Noncommercial License 1.0.0, and any business use needs a commercial license. The [README](https://github.com/erailabs/ombra#license) has the details.
