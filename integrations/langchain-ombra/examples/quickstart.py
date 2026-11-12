"""End to end in local mode: encrypt, store and search a few documents with LangChain.

    python quickstart.py [DIRECTORY]

Needs `pip install "ombra[local]" langchain-ombra`. Everything goes under DIRECTORY (a temporary
directory by default). The fake embedding is deterministic and needs no model download; it is not
meaningful, so only an exact repeat of a text ranks it first. Use a real model in your own code
(its dimension must be 384, 768 or 1536). Local mode takes a few seconds per search.
"""

import sys
import tempfile
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_ombra import OmbraVectorStore
from ombra import Client


def main(directory: Path) -> list[tuple[str, float]]:
    embedding = DeterministicFakeEmbedding(size=384)
    client = Client.local(directory / "server")
    store = OmbraVectorStore.from_texts(
        ["encrypted search", "gpu kernels", "vector databases"],
        embedding,
        metadatas=[{"topic": 0}, {"topic": 1}, {"topic": 0}],
        client=client,
        collection_name="quickstart",
        key_file=directory / "quickstart.key",  # back this file up: it holds the secret key
        passphrase="change me",
        filter_fields={"topic": 2},  # integer metadata you can filter on
    )
    try:
        # Score = cosine similarity, higher is better.
        found = store.similarity_search_with_score("gpu kernels", k=2, filter={"topic": 1})
        for doc, score in found:
            print(f"{score:.3f}  {doc.page_content}")
        return [(doc.page_content, score) for doc, score in found]
    finally:
        store.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            main(Path(tmp))
