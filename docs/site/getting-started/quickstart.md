# Quickstart

This page runs your first encrypted search in about ten minutes. Your vectors and your query are encrypted on your machine, and the search runs on ciphertexts.

You start in local mode: a CPU server runs inside your process, so you need no GPU and no network. Install first, as described in [Install](install.md):

```bash
pip install "ombra[local]" sentence-transformers
```

## Run it

The script embeds six sentences with `sentence-transformers/all-MiniLM-L6-v2`, which makes 384-dimension vectors. It encrypts them, then asks which sentence answers "What did the central bank do?". The model downloads about 90 MB on the first run. Save this as `local_search.py`, or take it from `examples/local_search.py`:

```python
"""First encrypted search in local mode: python local_search.py"""

import tempfile
import uuid
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
    ids = [str(uuid.uuid4()) for _ in texts]  # random ids: the server sees ids in the clear
    text_of = dict(zip(ids, texts))

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384 dimensions
    vectors = model.encode(texts, normalize_embeddings=True).tolist()  # Ombra needs unit length

    workdir = Path(tempfile.mkdtemp())  # use a folder you keep for real data, e.g. "./ombra-data"
    client = Client.local(workdir / "server")  # a CPU server inside this process
    client.create_collection(
        "notes",
        vectors,
        ids=ids,
        key_file=workdir / "notes.key",  # holds the secret key: back it up, never share it
        passphrase="change this passphrase",
    )

    query = model.encode(["What did the central bank do?"], normalize_embeddings=True)[0].tolist()
    for hit in client.search("notes", query, k=3):
        print(f"{hit.score:.3f}  {text_of[hit.id]}")  # the score is the exact cosine similarity
    client.close()
    print(f"Keys and encrypted data are in {workdir}; delete it when you are done.")


if __name__ == "__main__":  # needed: local mode starts a child process
    main()
```

```bash
python local_search.py
```

You should see output like this. The scores can differ in the last digit:

```text
0.593  The central bank raised its policy rate.
0.333  Interest rates rose again this quarter.
0.133  A GPU runs thousands of threads in parallel.
```

Creating the collection (keys, encryption, upload) takes most of the run time, and a single search takes a second or more in local mode. The exact times are on the [latency page](../performance/latency-and-throughput.md).

## Things to know

Ids are stored on the server in plain text, so the script uses random ones (`uuid.uuid4()`). Do not put names, e-mail addresses or file names in ids.

The search level (`security_level` when you create a collection) decides which clusters a query asks the server for. The default is `fast`. `decoy` and `strict` hide more about the cluster choice, and they hide only that: after every search, at every level, the client fetches the best candidate records by id, and the server sees those ids. See [what the server learns](../security/what-the-server-learns.md).

Vectors must have unit length, which `normalize_embeddings=True` does. The scores are cosine similarities, higher is better, and they are exact: the client decrypts the server's answer and re-ranks the candidates itself.

The `if __name__ == "__main__":` guard is required, because local mode starts a child process.

Local mode is for trying Ombra, for tests and for small collections of a few thousand vectors. It is slow and not meant for production. There is no plaintext mode: local mode encrypts exactly like a server does.

The script prints a folder at the end. It holds the secret key and the encrypted collection, which is a few hundred MB, so delete it when you are done.

Next: [your first search, step by step](first-search.md).
