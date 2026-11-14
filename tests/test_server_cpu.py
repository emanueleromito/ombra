# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Smoke test of the CPU-only `ombra_server` with the real Python client.

The CPU build (`make build-cpu`) has no GPU: every call works except Search, which must fail with
a clear error. This test covers create, key upload, insert, delete, ListBlocks, GetBlock, repack
(ReplaceBlock), blobs, a restart on the same data dir, and the Search refusal. Synthetic data only.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")
pytest.importorskip("openfhe")

from server_harness import ServerProcess, find_server_binary  # noqa: E402

pytestmark = pytest.mark.openfhe

if find_server_binary() is None:
    pytest.skip(
        "ombra_server (CPU build) not found: run `make build-cpu` or set OMBRA_SERVER_BIN",
        allow_module_level=True,
    )

from ombra import Client  # noqa: E402
from ombra.blobstore import BUCKET_SIZES, KIND_RECORD  # noqa: E402

D = 384  # the smallest supported dimension
N = 40
NAME = "cpu_smoke"


def _vectors() -> np.ndarray:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(N, D))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _collection(client: Client):
    """The open Collection object of the client (blocks and blobs are below the public API)."""
    return client._core._get(NAME)[0]


def test_cpu_server_everything_but_search(tmp_path) -> None:
    x = _vectors()
    ids = [f"v{i}" for i in range(N)]
    key_file = tmp_path / "smoke.key"
    server = ServerProcess(tmp_path, rest=True)
    with server:
        client = server.client()
        client.create_collection(
            NAME,
            x,
            ids,
            key_file=key_file,
            passphrase="pw",  # noqa: S106
            preset="fast",
            n_clusters=1,
            scrypt_params=(2**4, 8, 1),
        )
        info = client.get_collection(NAME)
        assert info.preset == "m0_compact_n8192"
        assert (info.vector_count, info.deleted_count) == (N, 0)
        assert client.list_collections() == [NAME]

        # insert more vectors (one more block per touched cluster)
        more = _vectors()[:5]
        client.insert(NAME, more, [f"w{i}" for i in range(5)])
        assert client.get_collection(NAME).vector_count == N + 5

        # delete most of one cluster's vectors, so the repack threshold is passed
        col = _collection(client)
        stats = col.block_stats()
        assert stats and sum(b.n_vectors for b in stats) == N + 5
        victims = ids[:30]
        client.delete(NAME, victims)
        stats = col.block_stats()
        assert sum(b.n_deleted for b in stats) == len(victims)
        live = {i for b in stats for i in col.read_live(b.segment)[0]}  # GetBlock + decrypt
        assert live == {i.encode() for i in ids[30:]} | {f"w{i}".encode() for i in range(5)}

        # ReplaceBlock (repack): deleted slots are gone afterwards, the live ids are unchanged
        report = col.repack(threshold=0.1)
        assert report.groups >= 1
        stats = col.block_stats()
        assert sum(b.n_deleted for b in stats) == 0
        assert {i for b in stats for i in col.read_live(b.segment)[0]} == live

        # blobs
        blob = bytes(range(256)) * (BUCKET_SIZES[0] // 256)
        col.blobs.put(col.name, KIND_RECORD, b"rec1", blob)
        assert col.blobs.get(col.name, KIND_RECORD, b"rec1") == blob
        assert col.blobs.delete(col.name, KIND_RECORD, [b"rec1"]) == 1
        col.blobs.put(col.name, KIND_RECORD, b"rec2", blob)

        # Search: a clear error, not a crash and not an internal-error message
        with pytest.raises(Exception, match="without GPU support"):
            client.search(NAME, x[35], k=3)
        assert server.running

        # the REST port reports that there is no GPU
        import json
        import urllib.request

        with urllib.request.urlopen(f"http://{server.rest_address}/healthz", timeout=10) as r:
            assert json.load(r)["status"] == "ok"
        with urllib.request.urlopen(f"http://{server.rest_address}/metrics", timeout=10) as r:
            assert json.load(r)["gpu_available"] is False

        # restart on the same data dir: everything is still there
        server.restart(kill=True)
        client = server.client()
        client.open_collection(NAME, key_file=key_file, passphrase="pw")  # noqa: S106
        assert client.list_collections() == [NAME]
        info = client.get_collection(NAME)
        assert (info.vector_count, info.deleted_count) == (15, 0)
        col = _collection(client)
        stats = col.block_stats()
        assert {i for b in stats for i in col.read_live(b.segment)[0]} == live
        assert col.blobs.get(col.name, KIND_RECORD, b"rec2") == blob
        with pytest.raises(Exception, match="without GPU support"):
            client.search(NAME, x[35], k=3)
        # keys survived the restart: a new insert (key tag check, key load) still works
        client.insert(NAME, x[:1], ["z0"])
        assert client.get_collection(NAME).vector_count == 16
        client.drop_collection(NAME)
        assert client.list_collections() == []
