# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Filters, inserts, deletes and reopening a collection: python filters_and_updates.py"""

import tempfile
from pathlib import Path

from ombra import Client
from sentence_transformers import SentenceTransformer

TOPICS = {0: "pets", 1: "finance", 2: "tech"}


def main() -> None:
    texts = [
        "The cat sat on the warm windowsill.",
        "Kittens love to chase a ball of yarn.",
        "Interest rates rose again this quarter.",
        "The central bank raised its policy rate.",
        "A GPU runs thousands of threads in parallel.",
        "Homomorphic encryption lets a server compute on data it cannot read.",
    ]
    topics = [0, 0, 1, 1, 2, 2]  # one integer per text; the filter compares against it

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(items: list[str]) -> list[list[float]]:
        return model.encode(items, normalize_embeddings=True).tolist()

    def show(title: str, hits) -> None:
        print(title)
        for hit in hits:
            print(f"  {hit.score:.3f}  [{TOPICS[topics[int(hit.id)]]}] {texts[int(hit.id)]}")

    workdir = Path(tempfile.mkdtemp())
    key_file = workdir / "notes.key"
    client = Client.local(workdir / "server")
    client.create_collection(
        "notes",
        embed(texts),
        ids=[str(i) for i in range(len(texts))],
        key_file=key_file,
        passphrase="change this passphrase",
        metadata=[{"topic": t} for t in topics],
        filter_fields={"topic": 3},  # the field "topic" takes the values 0, 1 and 2
    )

    query = embed(["What did the central bank do?"])[0]
    show("no filter", client.search("notes", query, k=3))
    show("only topic 2 (tech)", client.search("notes", query, k=2, filter={"topic": 2}))

    texts.append("Bond yields fell after the announcement.")
    topics.append(1)
    client.insert("notes", embed(texts[-1:]), ids=["6"], metadata=[{"topic": 1}])
    show("after an insert, topic 1", client.search("notes", query, k=3, filter={"topic": 1}))

    client.delete("notes", ["3"])
    show("after deleting id 3, topic 1", client.search("notes", query, k=3, filter={"topic": 1}))
    client.close()

    # A new process only needs the server folder, the key file and the passphrase.
    client = Client.local(workdir / "server")
    client.open_collection(
        "notes", key_file=key_file, passphrase="change this passphrase", filter_fields=["topic"]
    )
    show("after reopening", client.search("notes", query, k=1))
    client.close()
    print(f"Keys and encrypted data are in {workdir}; delete it when you are done.")


if __name__ == "__main__":  # needed: local mode starts a child process
    main()
