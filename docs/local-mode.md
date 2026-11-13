# Local mode

Local mode runs a CPU server in a child process of your program, so you can try Ombra without a GPU or a server. The documentation site covers it:

- [Install](site/getting-started/install.md): `pip install "ombra[local]"`, the supported Python versions and platforms
- [Quickstart](site/getting-started/quickstart.md): a first search with `Client.local(...)`
- [Local mode or a server](site/getting-started/local-or-server.md): what changes when you move to a GPU server
- [Latency and throughput](site/performance/latency-and-throughput.md): how slow local mode is

Local mode is for trying Ombra, tests and small collections, not for production. The child process gets only a context and public evaluation keys, never the secret key. It runs the same expansion tree as the GPU server, in the compiled module of the `ombra` wheel (`client/native/`).
