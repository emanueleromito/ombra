# Install

The client is a Python package. The server is a Docker image. You need the client to try Ombra at all, and the server only when you want GPU speed.

## The client

```bash
pip install "ombra[local]"
```

The `local` extra adds local mode, which runs a CPU server inside your process so you can try Ombra without a GPU or a server. It installs `ombra-openfhe`, our build of openfhe-python on OpenFHE 1.6.1. Plain `pip install ombra` is enough when you only talk to a server, and it skips the OpenFHE wheel.

Python 3.11, 3.12 and 3.13 are supported. Wheels exist for these platforms, so no compiler is needed:

| Platform | Architectures |
| --- | --- |
| Linux (manylinux_2_28) | x86_64, aarch64 |
| macOS 15 or newer | arm64, x86_64 |

There is no Windows build.

Local mode needs about 1 GB of free disk space and a few GB of memory. A collection of six short texts writes a few hundred MB of keys and ciphertexts.

The quickstart also uses an embedding model. Any model works as long as its vectors have 384, 768 or 1536 dimensions. The examples use `sentence-transformers`:

```bash
pip install sentence-transformers
```

Check that it worked:

```bash
python -c "import ombra; print(ombra.__version__)"
```

## The server

The server needs a host with a supported NVIDIA GPU (an A10 or A100, for example; see [GPUs](../operations/gpus.md) for the list) and Docker with the NVIDIA container toolkit. Start it from the compose file in the repository:

```bash
docker compose up -d
```

The data lives in a volume, so it survives restarts. [Docker](../operations/docker.md) covers the image, the volume, TLS and API keys, and [GPUs](../operations/gpus.md) covers the card requirements. If you have no GPU, stay with local mode.

Next: the [quickstart](quickstart.md).
