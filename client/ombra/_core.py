# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Core layer shared by the sync and async clients. No network or disk I/O of its own: it drives a
transport (a gRPC channel, or the in-process local server of `ombra._local`) through the
`Collection` code of the client library, so both modes run the same client code.

A transport is an object with `pb` (the ombra.v1 messages module), `stub` (an object with the
method names of the gRPC stub of service ombra.v1.Ombra), `openfhe_version` and `close()`.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ombra import presets
from ombra.collection import Collection, CollectionError, KeyKit, check_server_spec
from ombra.search import Hit

MAX_ID_BYTES = 256  # LIMIT_MAX_ID_BYTES


@dataclass(frozen=True)
class CollectionInfo:
    """What the server stores about a collection; filter field names live on the client."""

    name: str
    preset: str
    dimension: int
    security_level: str
    filter_cardinalities: tuple[int, ...]
    cluster_count: int
    vector_count: int
    deleted_count: int


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float = field(repr=False)  # exact cosine score (from the re-ranking record)
    metadata: dict[str, Any] = field(repr=False, default_factory=dict)


class ClientCore:
    """Holds the transport and the open collections (with their keys); builds the requests."""

    def __init__(self, transport: Any) -> None:
        self._t = transport
        self._open: dict[str, tuple[Collection, tuple[str, ...]]] = {}
        self.closed = False

    def __repr__(self) -> str:
        return f"ClientCore(transport={type(self._t).__name__})"

    # ---- helpers ----------------------------------------------------------------------------
    def _get(self, name: str) -> tuple[Collection, tuple[str, ...]]:
        try:
            return self._open[name]
        except KeyError:
            raise KeyError(
                f"collection {name!r} is not open in this client: create_collection or "
                "open_collection first"
            ) from None

    @staticmethod
    def _id(x: str | bytes) -> bytes:
        b = x.encode() if isinstance(x, str) else bytes(x)
        if not 1 <= len(b) <= MAX_ID_BYTES:
            raise ValueError(f"ids must be 1 to {MAX_ID_BYTES} bytes")
        return b

    @staticmethod
    def _field_values(
        names: Sequence[str],
        cards: Sequence[int],
        metadata: Sequence[dict[str, Any]] | None,
        n: int,
    ) -> np.ndarray | None:
        if not names:
            return None
        if metadata is None or len(metadata) != n:
            raise ValueError("a collection with filter fields needs one metadata dict per vector")
        out = np.zeros((n, len(names)), dtype=np.int64)
        for i, m in enumerate(metadata):
            for fi, (fname, card) in enumerate(zip(names, cards, strict=True)):
                v = m.get(fname)
                if not isinstance(v, int | np.integer) or not 0 <= int(v) < card:
                    raise ValueError(
                        f"metadata field {fname!r} must be an integer in [0, {card}) "
                        "for every vector"
                    )
                out[i, fi] = int(v)
        return out

    # ---- collections ------------------------------------------------------------------------
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
        preset = preset or getattr(self._t, "default_preset", presets.DEFAULT_PRESET)
        x = np.asarray(vectors, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("vectors must be a non-empty 2-D array")
        names = tuple(filter_fields or {})
        cards = tuple((filter_fields or {}).values())
        id_bytes = self._make_ids(ids, len(x))
        derived = presets.derive(preset, x.shape[1])
        presets.check_filter_schema(derived, cards)  # refused here, before any key is made
        fields = self._field_values(names, cards, metadata, len(x))
        kit = KeyKit(derived, self._t.openfhe_version, derived.dimension + sum(cards))
        save = {} if scrypt_params is None else {"scrypt_params": scrypt_params}
        kit.save(key_file, passphrase, **save)  # the key is on disk before the server has a byte
        col = Collection.create(
            self._t.pb,
            self._t.stub,
            name,
            x,
            id_bytes,
            preset=preset,
            fields=fields,
            cards=cards,
            n_clusters=n_clusters,
            openfhe_version=self._t.openfhe_version,
            kit=kit,
            metadata=metadata,
            documents=documents,
            security_level=security_level,
        )
        self._open[name] = (col, names)
        return name

    def open_collection(
        self,
        name: str,
        *,
        key_file: str | Path,
        passphrase: str,
        filter_fields: Sequence[str] | None = None,
    ) -> str:
        """Reopen an existing collection with its key file. `filter_fields` are the field names
        in the order they were created in (the server stores only the value counts)."""
        pb = self._t.pb
        r = self._t.stub.GetCollection(pb.GetCollectionRequest(name=name))
        if r.WhichOneof("result") == "error":
            raise CollectionError(r.error.code, r.error.message)
        spec = r.info.spec
        cards = tuple(f.cardinality for f in sorted(spec.filter_fields, key=lambda f: f.field_id))
        names = tuple(filter_fields or ())
        if names and len(names) != len(cards):
            raise ValueError(f"the collection has {len(cards)} filter fields, not {len(names)}")
        derived = presets.derive(spec.preset, spec.dimension)
        check_server_spec(derived, cards, spec.cluster_count)  # before any key is generated
        kit = KeyKit.load(
            key_file, passphrase, derived, self._t.openfhe_version, derived.dimension + sum(cards)
        )
        col = Collection.open(pb, self._t.stub, name, kit)
        self._open[name] = (col, names)
        return name

    def get_collection(self, name: str) -> CollectionInfo:
        pb = self._t.pb
        r = self._t.stub.GetCollection(pb.GetCollectionRequest(name=name))
        if r.WhichOneof("result") == "error":
            raise CollectionError(r.error.code, r.error.message)
        s = r.info.spec
        return CollectionInfo(
            name=s.name,
            preset=s.preset,
            dimension=s.dimension,
            security_level=pb.SecurityLevel.Name(s.security_level)
            .removeprefix("SECURITY_LEVEL_")
            .lower(),
            filter_cardinalities=tuple(
                f.cardinality for f in sorted(s.filter_fields, key=lambda f: f.field_id)
            ),
            cluster_count=s.cluster_count,
            vector_count=r.info.vector_count,
            deleted_count=r.info.deleted_count,
        )

    def list_collections(self) -> list[str]:
        r = self._t.stub.ListCollections(self._t.pb.ListCollectionsRequest())
        return sorted(c.spec.name for c in r.collections)

    def drop_collection(self, name: str) -> None:
        r = self._t.stub.DropCollection(self._t.pb.DropCollectionRequest(name=name))
        if r.error.code != 0:
            raise CollectionError(r.error.code, r.error.message)
        opened = self._open.pop(name, None)
        if opened is not None:
            opened[0].close()

    # ---- data -------------------------------------------------------------------------------
    @staticmethod
    def _make_ids(ids: Sequence[str] | None, n: int) -> list[bytes]:
        if ids is None:
            return [uuid.uuid4().hex.encode() for _ in range(n)]
        if len(ids) != n:
            raise ValueError("one id per vector")
        out = [ClientCore._id(i) for i in ids]
        if len(set(out)) != n:
            raise ValueError("ids must be unique")
        return out

    def insert(
        self,
        collection: str,
        vectors: Sequence[Sequence[float]],
        ids: Sequence[str] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
        documents: Sequence[str | bytes | None] | None = None,
    ) -> list[str]:
        col, names = self._get(collection)
        x = np.asarray(vectors, dtype=np.float64)
        id_bytes = self._make_ids(ids, len(x))
        fields = self._field_values(names, col.cards, metadata, len(x))
        col.insert_vectors(x, id_bytes, fields, metadata, documents)
        return [i.decode() for i in id_bytes]

    def delete(self, collection: str, ids: Sequence[str]) -> None:
        col, _ = self._get(collection)
        col.delete([self._id(i) for i in ids])

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
        col, names = self._get(collection)
        fidx: dict[int, Any] = {}
        for fname, want in (filter or {}).items():
            if fname not in names:
                raise ValueError(f"unknown filter field {fname!r}; known: {list(names)}")
            fidx[names.index(fname)] = want
        res = col.search(
            np.asarray(query, dtype=np.float64),
            k,
            filter=fidx or None,
            level=level,
            n_probe=n_probe,
        )
        hits: list[Hit] = res.hits
        return [SearchHit(h.id.decode(), h.score, h.metadata) for h in hits]

    def close(self) -> None:
        for col, _ in self._open.values():
            col.close()
        self._open.clear()
        self._t.close()
        self.closed = True
