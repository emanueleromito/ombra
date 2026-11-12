"""`OmbraVectorStore`: a LangChain `VectorStore` over an Ombra collection.

Vectors, metadata and the text of each document are encrypted on this machine before they leave
it; the server (or the in-process local mode) only computes on ciphertexts. The secret key lives
in the key file you name, protected by your passphrase. Back that file up: without it the data
cannot be read.

Scores. `similarity_search_with_score` returns the exact cosine similarity between the query and
the stored vector (computed on the client from the decrypted stored copy), in [-1, 1]. HIGHER IS
BETTER. This differs from stores that return a distance. Results come best first.

Filters. Ombra filters on integer fields whose number of values is fixed when the collection is
created. Pass `filter_fields={"topic": 5}` to the constructor, put an integer in [0, 5) under
"topic" in every document's metadata, and search with `filter={"topic": 2}` (equality only). Any
other filter, on a field that was not declared or with a value other than an integer, raises
`ValueError`: this class never filters results on the client behind your back.

Not supported: `get_by_ids` (the client API cannot fetch a vector by id), `max_marginal_relevance`
searches (they need the stored vectors), filters other than equality on declared fields, and
deleting everything with `delete(ids=None)`.

Async methods use LangChain's default: the synchronous method runs in a worker thread. The Ombra
async client does its work in the calling thread too, so mapping onto it would gain nothing.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

_TEXT_KEY = "_ombra_page_content"  # the document text lives inside the encrypted record


class OmbraVectorStore(VectorStore):
    """A LangChain vector store backed by one Ombra collection.

    The collection is created on the first `add_texts` (Ombra needs data to cluster); until then
    searches return an empty list. If it already exists on the server it is reopened with
    `key_file` and `passphrase`. A key file that exists while the collection does not is refused,
    so that a key is never overwritten.
    """

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embedding: Embeddings,
        *,
        key_file: str | Path,
        passphrase: str,
        filter_fields: Mapping[str, int] | None = None,
        preset: str | None = None,
        security_level: str = "fast",
        scrypt_params: tuple[int, int, int] | None = None,
    ) -> None:
        self._client = client
        self._name = collection_name
        self._embedding = embedding
        self._key_file = Path(key_file)
        self.__passphrase = passphrase  # never part of repr or of any error message
        self._filter_fields = dict(filter_fields or {})
        self._preset = preset
        self._security_level = security_level
        self._scrypt_params = scrypt_params
        self._exists = False
        if collection_name in client.list_collections():
            client.open_collection(
                collection_name,
                key_file=self._key_file,
                passphrase=self.__passphrase,
                filter_fields=list(self._filter_fields) or None,
            )
            self._exists = True

    def __repr__(self) -> str:
        return f"OmbraVectorStore(collection={self._name!r})"

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def close(self) -> None:
        """Close the Ombra client (and the local-mode server process, if any)."""
        self._client.close()

    # ---- writing ------------------------------------------------------------------------
    @staticmethod
    def _unit(vectors: Sequence[Sequence[float]]) -> np.ndarray:
        x = np.asarray(vectors, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("embeddings must be a list of equal-length vectors")
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if not np.all(norms > 0):
            raise ValueError("an embedding has zero length and cannot be normalised")
        return x / norms

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str | None] | list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed, encrypt and store `texts`; return their ids.

        An id that already exists is replaced (delete, then insert), so adding the same ids
        twice does not duplicate. Limitation: the server keeps a deleted id reserved until the
        segment holding it is gone, so replacing one id next to live vectors in the same segment
        can fail with "duplicate id" after the old version was deleted (see the error text).
        Ids not given are generated.
        """
        texts = list(texts)
        if not texts:
            return []
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("one metadata dict per text")
        if ids is not None and len(ids) != len(texts):
            raise ValueError("one id per text")
        given = list(ids) if ids is not None else [None] * len(texts)
        final_ids = [i if i is not None else uuid.uuid4().hex for i in given]
        if len(set(final_ids)) != len(final_ids):
            raise ValueError("ids must be unique")
        metas: list[dict[str, Any]] = []
        for m, text in zip(metadatas or [{}] * len(texts), texts, strict=True):
            if _TEXT_KEY in m:
                raise ValueError(f"metadata key {_TEXT_KEY!r} is reserved")
            metas.append({**m, _TEXT_KEY: text})
        vectors = self._unit(self._embedding.embed_documents(texts))

        if not self._exists:
            if self._key_file.exists():
                raise FileExistsError(
                    "the key file exists but the collection does not; refusing to overwrite a "
                    "key. Use another key_file, or restore the collection."
                )
            self._client.create_collection(
                self._name,
                vectors,
                final_ids,
                key_file=self._key_file,
                passphrase=self.__passphrase,
                preset=self._preset,
                metadata=metas,
                filter_fields=self._filter_fields or None,
                security_level=self._security_level,
                scrypt_params=self._scrypt_params,
            )
            self._exists = True
            return final_ids
        explicit = [i for i in given if i is not None]
        if explicit:
            self._client.delete(self._name, explicit)
        try:
            self._client.insert(self._name, vectors, final_ids, metas)
        except Exception as e:
            if explicit:
                raise RuntimeError(
                    "insert failed after the previous versions of the given ids were deleted; "
                    "they are not restored. Ombra keeps a deleted id reserved until compaction, "
                    "so replacing an id can be refused with 'duplicate id'."
                ) from e
            raise
        return final_ids

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete by id. Unknown ids are ignored."""
        if ids is None:
            raise ValueError("delete needs ids; deleting everything is not supported")
        if self._exists and ids:
            self._client.delete(self._name, list(ids))
        return True

    # ---- reading ------------------------------------------------------------------------
    def _check_filter(self, filter: dict[str, Any] | None) -> dict[str, int] | None:
        if not filter:
            return None
        for name, want in filter.items():
            if name not in self._filter_fields:
                raise ValueError(
                    f"cannot filter on {name!r}: Ombra filters only on fields declared with "
                    f"filter_fields (declared: {sorted(self._filter_fields)})"
                )
            if isinstance(want, bool) or not isinstance(want, int | np.integer):
                raise ValueError(f"filter on {name!r} must be an integer (equality only)")
        return {k: int(v) for k, v in filter.items()}

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """The k nearest documents and their cosine similarity (higher is better)."""
        flt = self._check_filter(filter)
        if not self._exists or k <= 0:
            return []
        q = self._unit([embedding])[0]
        hits = self._client.search(self._name, q, k, flt, **kwargs)
        out = []
        for h in hits:
            meta = dict(h.metadata)
            text = meta.pop(_TEXT_KEY, "")
            out.append((Document(id=h.id, page_content=text, metadata=meta), float(h.score)))
        return out

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        self._check_filter(filter)
        if not self._exists:
            return []
        return self.similarity_search_with_score_by_vector(
            self._embedding.embed_query(query), k, filter, **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        hits = self.similarity_search_with_score_by_vector(embedding, k, filter, **kwargs)
        return [d for d, _ in hits]

    def similarity_search(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Document]:
        return [d for d, _ in self.similarity_search_with_score(query, k, filter, **kwargs)]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return lambda cosine: (cosine + 1.0) / 2.0  # [-1, 1] -> [0, 1], higher is better

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        client: Any = None,
        local_path: str | Path | None = None,
        collection_name: str,
        key_file: str | Path,
        passphrase: str,
        **kwargs: Any,
    ) -> OmbraVectorStore:
        """Build a store and add `texts`. Give `client`, or `local_path` for local mode."""
        if client is None:
            if local_path is None:
                raise ValueError("give client=... or local_path=... (local mode)")
            from ombra import Client

            client = Client.local(local_path)
        store = cls(
            client, collection_name, embedding, key_file=key_file, passphrase=passphrase, **kwargs
        )
        store.add_texts(texts, metadatas, ids=ids)
        return store
