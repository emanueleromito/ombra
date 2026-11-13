# Local mode or a server

Ombra runs search in one of two places. Your client code is the same in both. Only the line that creates the client changes.

| | Local mode | Server |
| --- | --- | --- |
| Where search runs | a child process on your CPU | a GPU server |
| Needs | `pip install "ombra[local]"` | Docker and an NVIDIA Ampere GPU |
| Speed | several seconds per search | much faster, and many searches at once |
| Size | a few thousand vectors | larger collections |
| Default preset | fast | balanced |
| Use it for | trying Ombra, tests, small collections | production |

Local mode runs the same encrypted search as the server, on the CPU. It is slow: a search takes several seconds, and the time grows with the dimension and the collection size. The server is built for the GPU and is the way to run real workloads. [Latency and throughput](../performance/latency-and-throughput.md) has the measured numbers.

Local mode is also not a plaintext shortcut. It encrypts and decrypts exactly as a server setup does, so code that works locally works against a server.

## Switch to a server

Start the server as in [Install](install.md) and create an API key as described in [Docker](../operations/docker.md). The API key is an access token for the server. It is not your secret key and has nothing to do with it.

Then replace `Client.local(...)` with `Client(address, api_key=..., root_certificates=...)`. TLS is on unless you pass `insecure=True`, which is only for a test server on a loopback address. The server refuses to start without TLS and an API key, except in the same loopback test setup.

This script is the quickstart against a server. Run it with the server address and the certificate (or its CA):

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
export OMBRA_API_KEY=...        # the token you created
export OMBRA_PASSPHRASE=...     # protects notes.key
python server_search.py localhost:50051 server.crt
```

Against a server the default preset is "balanced", so keys and ciphertexts are larger than in local mode. Creating the collection takes longer, and every search is faster than a local one.

## What a server can learn

Ombra protects against a server that runs the software correctly but tries to learn what it can: the hosting provider, an insider, an attacker after a breach. It does not defend against a server that returns wrong results on purpose. In local mode the "server" is a child process on your own machine, so these limits matter only for a real server.

The server cannot see vectors, queries, filter values or scores. It can see the collection size, the number of queries, their timing, and which document ids you fetch after the top-k. At the default Fast level it can also see which clusters a query searched, and those hint at the topic. The Decoy level adds random clusters to hide the real ones, and Strict scans everything. You choose with `security_level=` when you create a collection, or `level=` per search. [What the server learns](../security/what-the-server-learns.md) has the full list, and [the threat model](../security/threat-model.md) says what Ombra assumes.
