"""Synchronous and asynchronous Ombra clients with identical public methods."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ombra._core import ClientCore


class Client:
    """Synchronous client."""

    def __init__(self, address: str, *, api_key: str | None = None) -> None:
        self._core = ClientCore(address, api_key=api_key)

    def create_collection(self, name: str, dimension: int, preset: str = "balanced") -> Any:
        return self._core.create_collection(name, dimension, preset)

    def get_collection(self, name: str) -> Any:
        return self._core.get_collection(name)

    def list_collections(self) -> list[str]:
        return self._core.list_collections()

    def drop_collection(self, name: str) -> None:
        return self._core.drop_collection(name)

    def insert(
        self,
        collection: str,
        vectors: Sequence[Sequence[float]],
        ids: Sequence[str] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> list[str]:
        return self._core.insert(collection, vectors, ids, metadata)

    def delete(self, collection: str, ids: Sequence[str]) -> None:
        return self._core.delete(collection, ids)

    def search(
        self,
        collection: str,
        query: Sequence[float],
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        return self._core.search(collection, query, k, filter)

    def close(self) -> None:
        self._core.close()


class AsyncClient:
    """Asynchronous client."""

    def __init__(self, address: str, *, api_key: str | None = None) -> None:
        self._core = ClientCore(address, api_key=api_key)

    async def create_collection(self, name: str, dimension: int, preset: str = "balanced") -> Any:
        return self._core.create_collection(name, dimension, preset)

    async def get_collection(self, name: str) -> Any:
        return self._core.get_collection(name)

    async def list_collections(self) -> list[str]:
        return self._core.list_collections()

    async def drop_collection(self, name: str) -> None:
        return self._core.drop_collection(name)

    async def insert(
        self,
        collection: str,
        vectors: Sequence[Sequence[float]],
        ids: Sequence[str] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> list[str]:
        return self._core.insert(collection, vectors, ids, metadata)

    async def delete(self, collection: str, ids: Sequence[str]) -> None:
        return self._core.delete(collection, ids)

    async def search(
        self,
        collection: str,
        query: Sequence[float],
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        return self._core.search(collection, query, k, filter)

    async def close(self) -> None:
        self._core.close()
