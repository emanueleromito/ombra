# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
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
