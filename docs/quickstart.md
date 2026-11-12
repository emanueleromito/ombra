# Quickstart

Run your first encrypted search. Your vectors and your query are encrypted on your machine with
fully homomorphic encryption (FHE); the server only computes on ciphertexts. Ombra is
source-available (see `LICENSING.md`).

You start in local mode, which runs a CPU server inside your process and needs no GPU and no
network. Then you run the same code against a server. The steps take about 10 minutes once Ombra
is installed.

## What you need

- CPython 3.12 on x86-64 Linux, with libstdc++ from GCC 13 or newer (Ubuntu 24.04 has it) and
  `libgomp1`. Other Python versions, macOS and aarch64 are not supported yet.
- About 1 GB of free disk space and a few GB of memory. A collection of six short texts writes
  about 300 MB of keys and ciphertexts.
- No GPU for local mode.
- **Ombra has no published wheels yet** (they are planned). Until then you
  build the client from source, and that needs a compiler, CMake and **OpenFHE 1.6.1** (C++,
  built with the default 64-bit native size). The OpenFHE build takes a while, so budget extra
  time for the first install. `modal_app.py` (`local_install_image`) shows the exact steps that
  are tested.

## 1. Install

Install OpenFHE 1.6.1 first (from `https://github.com/openfheorg/openfhe-development`, tag
`v1.6.1`, with `cmake --install`). Then, from a clone of this repository:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "./client[local]" sentence-transformers
```

The `local` extra adds `openfhe-python` 1.6.1. If Python cannot load `libOPENFHEcore`, add the
OpenFHE `lib` folder to `LD_LIBRARY_PATH`.

## 2. First search, in local mode

This embeds six sentences with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions;
Ombra supports 384, 768 and 1536), encrypts them, and searches. The model downloads about 90 MB
on the first run. Save it as `local_search.py`:

```python
"""First encrypted search in local mode: python local_search.py"""

import tempfile
from pathlib import Path

from ombra import Client
from sentence_transformers import SentenceTransformer


def main() -> None:
    texts = [
        "The cat sat on the warm windowsill.",
        "Kittens love to chase a ball of yarn.",
        "Interest rates rose again this quarter.",
        "The central bank raised its policy rate.",
        "A GPU runs thousands of threads in parallel.",
        "Homomorphic encryption lets a server compute on data it cannot read.",
    ]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384 dimensions
    vectors = model.encode(texts, normalize_embeddings=True).tolist()  # Ombra needs unit length

    workdir = Path(tempfile.mkdtemp())  # use a folder you keep for real data, e.g. "./ombra-data"
    client = Client.local(workdir / "server")  # a CPU server inside this process
    client.create_collection(
        "notes",
        vectors,
        ids=[str(i) for i in range(len(texts))],
        key_file=workdir / "notes.key",  # holds the secret key: back it up, never share it
        passphrase="change this passphrase",
    )

    query = model.encode(["What did the central bank do?"], normalize_embeddings=True)[0].tolist()
    for hit in client.search("notes", query, k=3):
        print(f"{hit.score:.3f}  {texts[int(hit.id)]}")  # the score is the exact cosine similarity
    client.close()
    print(f"Keys and encrypted data are in {workdir}; delete it when you are done.")


if __name__ == "__main__":  # needed: local mode starts a child process
    main()
```

```bash
python local_search.py
```

Expected output (scores can differ in the last digit):

```text
0.593  The central bank raised its policy rate.
0.333  Interest rates rose again this quarter.
0.133  A GPU runs thousands of threads in parallel.
```

On a laptop-class machine the whole run, model load included, took about 22 seconds. Creating
the collection (keys, encryption, upload) takes most of it; a search takes a few seconds.

Things to know:

- Vectors must have unit length. `normalize_embeddings=True` does that.
- The scores are exact cosine similarities (higher is better), computed on your machine after
  decryption. The ranking is re-ranked exactly on the client.
- The `if __name__ == "__main__":` guard is required, because local mode starts a child process.
- Local mode is for trying Ombra, tests and small collections (some thousands of vectors). It
  is not for production. There is no plaintext mode: local mode encrypts exactly like a server.
- Delete the printed folder when you are done; it holds the secret key and large files.

## 3. The same search against a server

The code is the same; only the client object changes. Replace `Client.local(...)` with
`Client(address, api_key=..., root_certificates=...)`. TLS is on unless you pass
`insecure=True`, which is for test servers on a loopback address. The API key is an access token
for the server; it is not your secret key and is not related to it.

The server runs from a Docker image. It needs an NVIDIA GPU (Ampere: A100, A10, RTX 30xx) and the
NVIDIA container toolkit. From a checkout of this repository:

```bash
docker compose up -d
docker compose logs ombra                                   # shows the admin API key, once
docker compose cp ombra:/data/tls/server.crt ./server.crt   # the certificate, for the client
```

The first start makes a self-signed certificate (valid for `localhost` and `127.0.0.1`) and one
admin API key. Both stay in the `ombra-data` volume with your collections. A machine without a
GPU cannot run the server; use local mode there. You can also build `ombra_server` yourself with
`make build`. Its options are listed at the top of `server/grpc/src/ombra_server.cpp`, and
`docs/architecture.md` describes the design.

The server refuses to start without TLS and without an API key unless you pass `--insecure` or
`--no-auth`, and those two are limited to loopback addresses. Save this as `server_search.py`,
set `OMBRA_API_KEY` to the token from the log and `OMBRA_PASSPHRASE` to a passphrase for your key
file, and run it with the server address and the certificate (or its CA), for example
`python server_search.py localhost:50051 server.crt`:

```python
"""The same search against an Ombra server: python server_search.py HOST:PORT CA_CERT"""

import os
import sys
from pathlib import Path

from ombra import Client
from sentence_transformers import SentenceTransformer


def main() -> None:
    address, ca_cert = sys.argv[1], Path(sys.argv[2])
    texts = [
        "The cat sat on the warm windowsill.",
        "Interest rates rose again this quarter.",
        "The central bank raised its policy rate.",
    ]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = model.encode(texts, normalize_embeddings=True).tolist()

    client = Client(
        address,  # TLS is on unless you pass insecure=True (loopback test servers only)
        api_key=os.environ["OMBRA_API_KEY"],  # an access token, not your secret key
        root_certificates=ca_cert.read_bytes(),
    )
    client.create_collection(
        "notes",
        vectors,
        ids=[str(i) for i in range(len(texts))],
        key_file="notes.key",  # holds the secret key: back it up, never share it
        passphrase=os.environ["OMBRA_PASSPHRASE"],
    )

    query = model.encode(["What did the central bank do?"], normalize_embeddings=True)[0].tolist()
    for hit in client.search("notes", query, k=2):
        print(f"{hit.score:.3f}  {texts[int(hit.id)]}")
    client.close()


if __name__ == "__main__":
    main()
```

```bash
pip install "./client[grpc]"
export OMBRA_API_KEY=...        # the token printed above
export OMBRA_PASSPHRASE=...     # protects notes.key
python server_search.py localhost:50051 server.crt
```

Against a server the default preset is "balanced" (local mode uses "fast"), so keys and
ciphertexts are larger and everything is slower. This script needs a running server, so CI
checks that it compiles but does not run it.

## 4. LangChain

`langchain-ombra` is a LangChain `VectorStore` that encrypts texts, vectors and metadata on your
machine. Install it with `pip install ./integrations/langchain-ombra`. This example wraps the
same model; save it as `langchain_search.py`:

```python
"""The same search through LangChain: python langchain_search.py"""

import tempfile
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_ombra import OmbraVectorStore
from ombra import Client
from sentence_transformers import SentenceTransformer


class MiniLM(Embeddings):
    """all-MiniLM-L6-v2 (384 dimensions) as a LangChain embedding; vectors are unit length."""

    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def main() -> None:
    workdir = Path(tempfile.mkdtemp())
    store = OmbraVectorStore.from_texts(
        [
            "The cat sat on the warm windowsill.",
            "Interest rates rose again this quarter.",
            "The central bank raised its policy rate.",
        ],
        MiniLM(),
        client=Client.local(workdir / "server"),
        collection_name="notes",
        key_file=workdir / "notes.key",  # holds the secret key: back it up, never share it
        passphrase="change this passphrase",
    )
    for doc, score in store.similarity_search_with_score("What did the central bank do?", k=2):
        print(f"{score:.3f}  {doc.page_content}")
    store.close()
    print(f"Keys and encrypted data are in {workdir}; delete it when you are done.")


if __name__ == "__main__":
    main()
```

Pass a server `Client(...)` instead of `Client.local(...)` to use a server. Filters, scores and
what is not supported are in `integrations/langchain-ombra/README.md`.

## Your key file

The key file (`notes.key` above) holds your secret key, protected by the passphrase. It also
holds the key that protects stored documents.

- **Back it up** (and remember the passphrase). If you lose either, the data cannot be recovered,
  by you or by anyone else.
- **Never send it to the server** or put it in a repository. The server only receives ciphertexts
  and public evaluation keys, and Ombra's server code never accepts a secret key.
- Anyone who has the file and the passphrase can read your data.

## What the server can still learn

Ombra protects against an honest-but-curious server: one that runs the software correctly but
tries to learn what it can (the hosting provider, insiders, an attacker after a breach). It is
not a defence against a server that returns wrong or incomplete results on purpose. In local mode
the "server" is a child process on your own machine, so these limits matter only for a real
server.

- The server cannot see vectors, queries, documents, filter values or scores.
- The server can see the collection size, the number of queries, query timing, which document
  IDs you fetch after the top-k, and (except at the Strict level) which clusters a query searched.
- At the default Fast level the clusters you search come from decrypted scores, so they hint at
  the topic of a query. Decoy hides which clusters are real by adding random ones; Strict scans
  everything and hides the most, at the highest cost. Choose with `security_level=` when you
  create the collection, or `level=` per search.
- In Fast and Decoy modes the server sees which clusters and IDs the client picks, and these
  choices come from decrypted results. Ombra makes no IND-CPA-D claim: decrypted
  values never leave your machine, and the leak through these choices is stated, not analysed.
- A filter that matches few vectors makes the client run extra rounds, so the server can tell
  that a query was filtered and that the filter was selective. In Fast mode the clusters chosen can
  hint at the filter value. Decoy and Strict hide whether a filter was used.

See `docs/product-requirements.md` (threat model) for the full list.
