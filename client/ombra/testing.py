# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""`FakeClient`: an in-memory test double with the API of `ombra.Client`, for unit tests of code
that uses Ombra.

It is NOT a mode of Ombra and not a way to run it: nothing is encrypted, nothing is persisted,
there is no network and no key file is written. Ranking is an exact cosine search over the vectors
you give it, so a test of your own code can run in milliseconds without OpenFHE. Use `Client.local`
to try Ombra itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ombra._core import CollectionInfo, SearchHit
from ombra.retry import RetryPolicy


@dataclass
class _Coll:
    preset: str
    security_level: str
    field_names: tuple[str, ...]
    cards: tuple[int, ...]
    ids: list[str] = field(default_factory=list)
    vectors: list[np.ndarray] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    documents: list[str | bytes | None] = field(default_factory=list)
    deleted: int = 0


class FakeClient:
    """Same public methods and signatures as `ombra.Client`; keeps everything in memory."""

    def __init__(
        self,
        address: str,
        *,
        api_key: str | None = None,
        root_certificates: bytes | None = None,
        insecure: bool = False,
        timeout: float | None = None,
        stream_timeout: float | None = None,
        deadlines: Mapping[str, float] | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        self._colls: dict[str, _Coll] = {}

    @classmethod
    def local(cls, path: str | Path) -> FakeClient:
        return cls("memory")

    def create_collection(
        self,
        name: str,
        vectors: Sequence[Sequence[float]],
        ids: Sequence[str] | None = None,
        *,
        key_file: str | Path,
        passphrase: str,
        preset: str | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
        documents: Sequence[str | bytes | None] | None = None,
        filter_fields: Mapping[str, int] | None = None,
        security_level: str = "fast",
        n_clusters: int | None = None,
        scrypt_params: tuple[int, int, int] | None = None,
    ) -> str:
        if name in self._colls:
            raise ValueError(f"collection {name!r} exists")
        self._colls[name] = _Coll(
            preset or "fast",
            security_level,
            tuple(filter_fields or {}),
            tuple((filter_fields or {}).values()),
        )
        self.insert(name, vectors, ids, metadata, documents)
        return name

    def open_collection(
        self,
        name: str,
        *,
        key_file: str | Path,
        passphrase: str,
        filter_fields: Sequence[str] | None = None,
    ) -> str:
        self._get(name)
        return name

    def get_collection(self, name: str) -> CollectionInfo:
        c = self._get(name)
        dim = len(c.vectors[0]) if c.vectors else 0
        return CollectionInfo(
            name=name,
            preset=c.preset,
            dimension=dim,
            security_level=c.security_level,
            filter_cardinalities=c.cards,
            cluster_count=1,
            vector_count=len(c.ids),
            deleted_count=c.deleted,
        )

    def list_collections(self) -> list[str]:
        return sorted(self._colls)

    def drop_collection(self, name: str) -> None:
        self._get(name)
        del self._colls[name]

    def insert(
        self,
        collection: str,
        vectors: Sequence[Sequence[float]],
        ids: Sequence[str] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
        documents: Sequence[str | bytes | None] | None = None,
    ) -> list[str]:
        c = self._get(collection)
        x = np.asarray(vectors, dtype=np.float64)
        n = len(x)
        new = list(ids) if ids is not None else [f"fake-{len(c.ids) + i}" for i in range(n)]
        if len(new) != n or len(set(new)) != n or set(new) & set(c.ids):
            raise ValueError("one unique new id per vector")
        for seq in (metadata, documents):
            if seq is not None and len(seq) != n:
                raise ValueError("one metadata and one document entry per vector")
        for i, ident in enumerate(new):
            c.ids.append(ident)
            c.vectors.append(x[i] / np.linalg.norm(x[i]))
            c.metadata.append(dict(metadata[i]) if metadata is not None else {})
            c.documents.append(documents[i] if documents is not None else None)
        return new

    def delete(self, collection: str, ids: Sequence[str]) -> None:
        c = self._get(collection)
        for ident in ids:
            if ident in c.ids:
                i = c.ids.index(ident)
                for lst in (c.ids, c.vectors, c.metadata, c.documents):
                    del lst[i]
                c.deleted += 1

    def search(
        self,
        collection: str,
        query: Sequence[float],
        k: int = 10,
        filter: dict[str, Any] | None = None,
        *,
        level: str | None = None,
        n_probe: int | None = None,
    ) -> list[SearchHit]:
        c = self._get(collection)
        q = np.asarray(query, dtype=np.float64)
        q = q / np.linalg.norm(q)
        rows = []
        for i, vec in enumerate(c.vectors):
            if all(self._match(c.metadata[i].get(f), want) for f, want in (filter or {}).items()):
                rows.append((float(vec @ q), i))
        rows.sort(key=lambda r: (-r[0], r[1]))
        return [SearchHit(c.ids[i], s, c.metadata[i]) for s, i in rows[:k]]

    def close(self) -> None:
        self._colls.clear()

    def _get(self, name: str) -> _Coll:
        try:
            return self._colls[name]
        except KeyError:
            raise KeyError(f"collection {name!r} does not exist") from None

    @staticmethod
    def _match(value: Any, want: Any) -> bool:
        if isinstance(want, list | tuple | set):
            return value in want
        return bool(value == want)
