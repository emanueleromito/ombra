"""The encrypted blob store through the gRPC server `ombra_server`.

Blobs are opaque bytes (synthetic here, not ciphertexts: the server never parses them). Cases: put,
get, replace, delete round trip across all four bucket sizes; refusals (a size that is not a
bucket, an unknown kind, a stream shorter or longer than the header's size, a missing collection);
missing blob; blobs survive a server restart; blobs go away with the collection. No FHE keys are
needed, so this test does not use openfhe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")
grpc = pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")

import e2e_client as ec  # noqa: E402

pytestmark = pytest.mark.gpu

VERSION = "test-openfhe"
PRESET = "m0_compact_n8192"
BUCKETS = [16384, 262144, 2097152, 16777216]
CHUNK = 4194304
ERROR_SIZE_LIMIT = 5
ERROR_INVALID_ARGUMENT = 4
ERROR_BAD_SEQUENCE = 10
ERROR_COLLECTION_NOT_FOUND = 2
ERROR_BLOB_NOT_FOUND = 14


def _data(n: int, seed: int) -> bytes:
    return bytes((seed + i * 7) % 251 for i in range(1024)) * (n // 1024) + bytes(n % 1024)


class Rig:
    def __init__(self, tmp: Path) -> None:
        binary = ec.find_server_binary()
        if binary is None:
            pytest.fail("ombra_server not found; build with `make build` or `make test-gpu`")
        self.binary, self.tmp = binary, tmp
        self.pb, self.pbg = ec.load_stubs(tmp / "stubs")
        self.data_dir = tmp / "data"
        self.n_starts = 0
        self.start()

    def start(self) -> None:
        self.n_starts += 1
        self.server = ec.ServerProcess(
            self.binary, self.data_dir, VERSION, self.tmp / f"server{self.n_starts}.log"
        )
        opts = [
            ("grpc.max_send_message_length", 1 << 28),
            ("grpc.max_receive_message_length", 1 << 28),
        ]
        self.channel = grpc.insecure_channel(self.server.address, options=opts)
        self.stub = self.pbg.OmbraStub(self.channel)

    def stop(self) -> None:
        self.channel.close()
        self.server.stop()

    def create(self, name: str) -> None:
        pb = self.pb
        spec = pb.CollectionSpec(
            api_version=1,
            name=name,
            preset=PRESET,
            dimension=16,
            security_level=pb.SECURITY_LEVEL_FAST,
            key_tag="synthetic-tag",
            cluster_count=2,
        )
        spec.filter_fields.add(field_id=0, cardinality=3)
        r = self.stub.CreateCollection(pb.CreateCollectionRequest(spec=spec))
        assert r.WhichOneof("result") == "spec", r

    def put(self, coll: str, kind: int, ident: bytes, data: bytes, size: int | None = None):
        pb = self.pb

        def msgs():
            yield pb.PutBlobRequest(
                api_version=1,
                header=pb.PutBlobHeader(
                    collection=coll,
                    kind=kind,
                    id=ident,
                    size=len(data) if size is None else size,
                ),
            )
            for pos in range(0, len(data), CHUNK):
                yield pb.PutBlobRequest(api_version=1, data=data[pos : pos + CHUNK])

        return self.stub.PutBlob(msgs()).error.code

    def get(self, coll: str, kind: int, ident: bytes) -> bytes | int:
        """The blob bytes, or the error code."""
        pb = self.pb
        size, chunks = None, []
        for m in self.stub.GetBlob(
            pb.GetBlobRequest(api_version=1, collection=coll, kind=kind, id=ident)
        ):
            part = m.WhichOneof("part")
            if part == "error":
                return m.error.code
            if part == "size":
                size = m.size
            else:
                assert len(m.data) <= CHUNK
                chunks.append(m.data)
        out = b"".join(chunks)
        assert size == len(out)
        return out

    def delete(self, coll: str, kind: int, ids: list[bytes]) -> tuple[int, int]:
        r = self.stub.DeleteBlobs(
            self.pb.DeleteBlobsRequest(api_version=1, collection=coll, kind=kind, ids=ids)
        )
        return r.deleted, r.error.code


@pytest.fixture
def rig(tmp_path: Path):
    r = Rig(tmp_path)
    yield r
    r.stop()


def test_blob_round_trip_refusals_restart_drop(rig: Rig) -> None:
    pb = rig.pb
    rec, doc = pb.BLOB_KIND_RECORD, pb.BLOB_KIND_DOCUMENT
    rig.create("blobs")

    # Put and get at every bucket size (the largest needs several chunks).
    for i, n in enumerate(BUCKETS):
        assert rig.put("blobs", rec, b"id%d" % i, _data(n, i)) == 0
        assert rig.get("blobs", rec, b"id%d" % i) == _data(n, i)
    # Same id, other kind: separate namespace.
    assert rig.get("blobs", doc, b"id0") == ERROR_BLOB_NOT_FOUND
    assert rig.put("blobs", doc, b"id0", _data(BUCKETS[0], 50)) == 0
    assert rig.get("blobs", doc, b"id0") == _data(BUCKETS[0], 50)
    assert rig.get("blobs", rec, b"id0") == _data(BUCKETS[0], 0)

    # Replace, also into another bucket.
    assert rig.put("blobs", rec, b"id0", _data(BUCKETS[1], 99)) == 0
    assert rig.get("blobs", rec, b"id0") == _data(BUCKETS[1], 99)

    # Refusals leave the stored blob untouched.
    assert rig.put("blobs", rec, b"id0", b"x" * 100) == ERROR_SIZE_LIMIT
    assert rig.put("blobs", rec, b"id0", b"x" * (BUCKETS[0] + 1)) == ERROR_SIZE_LIMIT
    assert rig.put("blobs", rec, b"id0", b"") == ERROR_SIZE_LIMIT
    assert rig.put("blobs", pb.BLOB_KIND_UNSPECIFIED, b"id0", _data(BUCKETS[0], 1)) == (
        ERROR_INVALID_ARGUMENT
    )
    assert rig.put("blobs", 77, b"id0", _data(BUCKETS[0], 1)) == ERROR_INVALID_ARGUMENT
    assert rig.put("blobs", rec, b"", _data(BUCKETS[0], 1)) == ERROR_INVALID_ARGUMENT
    assert rig.put("blobs", rec, b"i" * 257, _data(BUCKETS[0], 1)) == ERROR_INVALID_ARGUMENT
    # Stream shorter / longer than the header says.
    assert rig.put("blobs", rec, b"id0", _data(BUCKETS[0], 1)[:-1], size=BUCKETS[0]) == (
        ERROR_INVALID_ARGUMENT
    )
    assert rig.put("blobs", rec, b"id0", _data(BUCKETS[0], 1) + b"x", size=BUCKETS[0]) == (
        ERROR_INVALID_ARGUMENT
    )
    assert rig.put("nope", rec, b"id0", _data(BUCKETS[0], 1)) == ERROR_COLLECTION_NOT_FOUND
    assert rig.get("blobs", rec, b"id0") == _data(BUCKETS[1], 99)
    assert rig.get("blobs", rec, b"missing") == ERROR_BLOB_NOT_FOUND
    assert rig.get("blobs", pb.BLOB_KIND_UNSPECIFIED, b"id0") == ERROR_INVALID_ARGUMENT
    assert rig.get("nope", rec, b"id0") == ERROR_COLLECTION_NOT_FOUND

    # Restart: everything is still there.
    rig.stop()
    rig.start()
    assert rig.get("blobs", rec, b"id0") == _data(BUCKETS[1], 99)
    assert rig.get("blobs", rec, b"id3") == _data(BUCKETS[3], 3)
    assert rig.get("blobs", doc, b"id0") == _data(BUCKETS[0], 50)

    # Delete: missing ids are not an error; counts are exact.
    assert rig.delete("blobs", rec, [b"id0", b"missing", b"id1"]) == (2, 0)
    assert rig.get("blobs", rec, b"id0") == ERROR_BLOB_NOT_FOUND
    assert rig.get("blobs", doc, b"id0") == _data(BUCKETS[0], 50)
    assert rig.delete("blobs", rec, [b"id0"]) == (0, 0)
    assert rig.delete("blobs", 77, [b"id2"])[1] == ERROR_INVALID_ARGUMENT
    assert rig.delete("nope", rec, [b"id2"])[1] == ERROR_COLLECTION_NOT_FOUND

    # Dropped with the collection; a new collection of the same name starts empty.
    assert rig.stub.DropCollection(pb.DropCollectionRequest(name="blobs")).error.code == 0
    blob_dir = rig.data_dir / "blobs"
    assert not any(blob_dir.iterdir())
    rig.create("blobs")
    assert rig.get("blobs", rec, b"id2") == ERROR_BLOB_NOT_FOUND
