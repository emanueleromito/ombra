# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""The same search against an Ombra server: python server_search.py HOST:PORT CA_CERT"""

import os
import sys
import uuid
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
    ids = [str(uuid.uuid4()) for _ in texts]  # random ids: the server sees ids in the clear
    text_of = dict(zip(ids, texts, strict=True))

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
        ids=ids,
        key_file="notes.key",  # holds the secret key: back it up, never share it
        passphrase=os.environ["OMBRA_PASSPHRASE"],
    )

    query = model.encode(["What did the central bank do?"], normalize_embeddings=True)[0].tolist()
    for hit in client.search("notes", query, k=2):
        print(f"{hit.score:.3f}  {text_of[hit.id]}")
    client.close()


if __name__ == "__main__":
    main()
