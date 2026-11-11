"""Synchronous and asynchronous Ombra clients with identical public methods.

    client = Client("host:443", api_key="...")   # a GPU server (pip install ombra[grpc])
    client = Client.local("./ombra-data")         # in-process CPU server (pip install ombra[local])

Both modes run the same client code (keys, encryption, clustering, top-k and re-ranking happen in
this process); only the transport differs. The secret key stays in the key file you name in
`create_collection` and is never sent anywhere. The async client has the same methods as
coroutines; the work itself is synchronous (it runs in the calling thread).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ombra._core import ClientCore, CollectionInfo, SearchHit


def _grpc_transport(
    address: str, api_key: str | None, root_certificates: bytes | None, insecure: bool
) -> Any:
    from ombra._transport import GrpcTransport

    return GrpcTransport(address, api_key, root_certificates, insecure)


def _local_transport(path: str | Path) -> Any:
    from ombra._local import LocalTransport

    return LocalTransport(path)


class Client:
    """Synchronous client."""

    def __init__(
        self,
        address: str,
        *,
        api_key: str | None = None,
        root_certificates: bytes | None = None,
        insecure: bool = False,
    ) -> None:
        """Connect to a server. TLS is used unless `insecure=True` (for a local test server)."""
        self._core = ClientCore(_grpc_transport(address, api_key, root_certificates, insecure))

    @classmethod
    def local(cls, path: str | Path) -> Client:
        """An in-process CPU (OpenFHE) server that keeps its files under `path`."""
        self = cls.__new__(cls)
        self._core = ClientCore(_local_transport(path))
        return self

    def create_collection(
        self,
        name: str,
        vectors: Sequence[Sequence[float]],
        ids: Sequence[str] | None = None,
        *,
        key_file: str | Path,
        passphrase: str,
        preset: str = "balanced",
        metadata: Sequence[dict[str, Any]] | None = None,
        filter_fields: Mapping[str, int] | None = None,
        security_level: str = "fast",
        n_clusters: int | None = None,
        scrypt_params: tuple[int, int, int] | None = None,
    ) -> str:
        """Cluster and encrypt `vectors` (unit length), write the key file, create the collection.

        `filter_fields` maps a field name to its number of values; every metadata dict then needs
        an integer in range under that name. Back up the key file: without it the data is lost.
        """
        return self._core.create_collection(
            name,
            vectors,
            ids,
            key_file=key_file,
            passphrase=passphrase,
            preset=preset,
            metadata=metadata,
            filter_fields=filter_fields,
            security_level=security_level,
            n_clusters=n_clusters,
            scrypt_params=scrypt_params,
        )

    def open_collection(
        self,
        name: str,
        *,
        key_file: str | Path,
        passphrase: str,
        filter_fields: Sequence[str] | None = None,
    ) -> str:
        """Reopen a collection with its key file; name the filter fields in their creation order."""
        return self._core.open_collection(
            name, key_file=key_file, passphrase=passphrase, filter_fields=filter_fields
        )

    def get_collection(self, name: str) -> CollectionInfo:
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
        *,
        level: str | None = None,
        n_probe: int | None = None,
    ) -> list[SearchHit]:
        """The k nearest vectors (unit-length query) that pass `filter`, best first."""
        return self._core.search(collection, query, k, filter, level=level, n_probe=n_probe)

    def close(self) -> None:
        self._core.close()


class AsyncClient:
    """Asynchronous client."""

    def __init__(
        self,
        address: str,
        *,
        api_key: str | None = None,
        root_certificates: bytes | None = None,
        insecure: bool = False,
    ) -> None:
        """Connect to a server. TLS is used unless `insecure=True` (for a local test server)."""
        self._core = ClientCore(_grpc_transport(address, api_key, root_certificates, insecure))

    @classmethod
    def local(cls, path: str | Path) -> AsyncClient:
        """An in-process CPU (OpenFHE) server that keeps its files under `path`."""
        self = cls.__new__(cls)
        self._core = ClientCore(_local_transport(path))
        return self

    async def create_collection(
        self,
        name: str,
        vectors: Sequence[Sequence[float]],
        ids: Sequence[str] | None = None,
        *,
        key_file: str | Path,
        passphrase: str,
        preset: str = "balanced",
        metadata: Sequence[dict[str, Any]] | None = None,
        filter_fields: Mapping[str, int] | None = None,
        security_level: str = "fast",
        n_clusters: int | None = None,
        scrypt_params: tuple[int, int, int] | None = None,
    ) -> str:
        """Cluster and encrypt `vectors` (unit length), write the key file, create the collection.

        `filter_fields` maps a field name to its number of values; every metadata dict then needs
        an integer in range under that name. Back up the key file: without it the data is lost.
        """
        return self._core.create_collection(
            name,
            vectors,
            ids,
            key_file=key_file,
            passphrase=passphrase,
            preset=preset,
            metadata=metadata,
            filter_fields=filter_fields,
            security_level=security_level,
            n_clusters=n_clusters,
            scrypt_params=scrypt_params,
        )

    async def open_collection(
        self,
        name: str,
        *,
        key_file: str | Path,
        passphrase: str,
        filter_fields: Sequence[str] | None = None,
    ) -> str:
        """Reopen a collection with its key file; name the filter fields in their creation order."""
        return self._core.open_collection(
            name, key_file=key_file, passphrase=passphrase, filter_fields=filter_fields
        )

    async def get_collection(self, name: str) -> CollectionInfo:
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
        *,
        level: str | None = None,
        n_probe: int | None = None,
    ) -> list[SearchHit]:
        """The k nearest vectors (unit-length query) that pass `filter`, best first."""
        return self._core.search(collection, query, k, filter, level=level, n_probe=n_probe)

    async def close(self) -> None:
        self._core.close()
