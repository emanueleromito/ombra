# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Blob store access for encrypted records and documents.

The server stores opaque blobs keyed by (collection, kind, id); this module is the client side
of that API. `BlobStore` is the protocol, `InMemoryBlobStore` is for tests, and `GrpcBlobStore`
uses the generated stubs (passed in, so this module needs neither grpc nor the stubs).
Blobs, ids and collection names never appear in reprs or error messages.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Protocol

from ombra.errors import BlobStoreError, from_proto_error
from ombra.retry import Replayable

KIND_RECORD = 1  # BlobKind.BLOB_KIND_RECORD
KIND_DOCUMENT = 2  # BlobKind.BLOB_KIND_DOCUMENT
BUCKET_SIZES = (16384, 262144, 2097152, 16777216)  # LIMIT_BLOB_BUCKET_0..3
API_VERSION = 1
CHUNK_BYTES = 1 << 20
MAX_CHUNK_BYTES = 4194304  # LIMIT_MAX_CHUNK_BYTES


class BlobStore(Protocol):
    def put(self, collection: str, kind: int, blob_id: bytes, data: bytes) -> None: ...
    def get(self, collection: str, kind: int, blob_id: bytes) -> bytes: ...
    def delete(self, collection: str, kind: int, ids: Iterable[bytes]) -> int: ...


class InMemoryBlobStore:
    """Dict-backed store for tests. Enforces the bucket-size rule like the server."""

    def __init__(self) -> None:
        self.blobs: dict[tuple[str, int, bytes], bytes] = {}

    def __repr__(self) -> str:
        return f"InMemoryBlobStore(<{len(self.blobs)} blobs>)"

    def put(self, collection: str, kind: int, blob_id: bytes, data: bytes) -> None:
        if len(data) not in BUCKET_SIZES:
            raise BlobStoreError("blob size is not a bucket size")
        self.blobs[(collection, kind, bytes(blob_id))] = bytes(data)

    def get(self, collection: str, kind: int, blob_id: bytes) -> bytes:
        try:
            return self.blobs[(collection, kind, bytes(blob_id))]
        except KeyError:
            raise BlobStoreError("blob not found") from None

    def delete(self, collection: str, kind: int, ids: Iterable[bytes]) -> int:
        return sum(self.blobs.pop((collection, kind, bytes(i)), None) is not None for i in ids)


class GrpcBlobStore:
    """Thin wrapper over the Ombra gRPC stubs. `pb2` is the ombra_pb2 module, `stub` an OmbraStub.

    Not exercised in `make test` (needs the generated stubs and a server).
    """

    def __init__(self, stub: Any, pb2: Any) -> None:
        self._stub = stub
        self._pb = pb2

    def __repr__(self) -> str:
        return "GrpcBlobStore()"

    @staticmethod
    def _check(error: Any) -> None:
        if getattr(error, "code", 0):
            raise from_proto_error(
                error.code, error.message, BlobStoreError, f"server error code {error.code}"
            )

    def put(self, collection: str, kind: int, blob_id: bytes, data: bytes) -> None:
        pb = self._pb

        def requests() -> Iterator[Any]:
            header = pb.PutBlobHeader(
                collection=collection, kind=kind, id=bytes(blob_id), size=len(data)
            )
            yield pb.PutBlobRequest(api_version=API_VERSION, header=header)
            for i in range(0, len(data), CHUNK_BYTES):
                yield pb.PutBlobRequest(api_version=API_VERSION, data=data[i : i + CHUNK_BYTES])

        resp = self._stub.PutBlob(Replayable(requests))
        if resp.HasField("error"):
            self._check(resp.error)

    def get(self, collection: str, kind: int, blob_id: bytes) -> bytes:
        pb = self._pb
        req = pb.GetBlobRequest(
            api_version=API_VERSION, collection=collection, kind=kind, id=bytes(blob_id)
        )
        size: int | None = None
        parts: list[bytes] = []
        total = 0
        for msg in self._stub.GetBlob(req):
            which = msg.WhichOneof("part")
            if which == "error":
                self._check(msg.error)
            elif which == "size":
                # a blob is one of the bucket sizes; a server cannot make us allocate more
                if size is not None or msg.size not in BUCKET_SIZES:
                    raise BlobStoreError("blob stream announces an invalid size")
                size = msg.size
            elif which == "data":
                total += len(msg.data)
                if size is None or total > size or len(msg.data) > MAX_CHUNK_BYTES:
                    raise BlobStoreError("blob stream is larger than announced")
                parts.append(msg.data)
        data = b"".join(parts)
        if size is None or len(data) != size:
            raise BlobStoreError("incomplete blob stream")
        return data

    def delete(self, collection: str, kind: int, ids: Iterable[bytes]) -> int:
        from ombra._transport import id_batches

        pb = self._pb
        deleted = 0
        for batch in id_batches(ids):  # each request stays under the server's receive limit
            req = pb.DeleteBlobsRequest(
                api_version=API_VERSION, collection=collection, kind=kind, ids=batch
            )
            resp = self._stub.DeleteBlobs(req)
            if resp.HasField("error"):
                self._check(resp.error)
            deleted += int(resp.deleted)
        return deleted
