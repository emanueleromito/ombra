"""Core layer shared by the sync and async clients. No I/O happens here."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class ClientCore:
    """Holds connection settings and builds requests; transport is added later."""

    def __init__(self, address: str, *, api_key: str | None = None) -> None:
        self.address = address
        self._api_key = api_key
        self.closed = False

    def __repr__(self) -> str:
        return f"ClientCore(address={self.address!r})"

    @staticmethod
    def _todo(name: str) -> NotImplementedError:
        return NotImplementedError(f"{name} is not implemented yet")

    def create_collection(self, name: str, dimension: int, preset: str = "balanced") -> Any:
        raise self._todo("create_collection")

    def get_collection(self, name: str) -> Any:
        raise self._todo("get_collection")

    def list_collections(self) -> list[str]:
        raise self._todo("list_collections")

    def drop_collection(self, name: str) -> None:
        raise self._todo("drop_collection")

    def insert(
        self,
        collection: str,
        vectors: Sequence[Sequence[float]],
        ids: Sequence[str] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> list[str]:
        raise self._todo("insert")

    def delete(self, collection: str, ids: Sequence[str]) -> None:
        raise self._todo("delete")

    def search(
        self,
        collection: str,
        query: Sequence[float],
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        raise self._todo("search")

    def close(self) -> None:
        self.closed = True
