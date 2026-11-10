"""T3.5: collection create, re-clustering and repack against the real gRPC server.

Synthetic data only. d = 384 (the smallest supported dimension) on the "fast" preset, one filter
field with 3 values. Checks: (a) centers, clusters and keys exist after create; (b) search equals
numpy brute force; (c) a cluster with blocks 30 % or more deleted is repacked: fewer blocks, fewer
stored bytes on disk, same search results; (d) recluster: the old collection keeps working while
the new one is built, the new one is correct, deleted vectors are gone.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")
pytest.importorskip("openfhe")
pytest.importorskip("faiss")

import e2e_client as ec  # noqa: E402
from ombra.collection import Collection  # noqa: E402

pytestmark = pytest.mark.gpu

VERSION = "test-openfhe"
D = 384
CARDS = (3,)
K = 5
TOL = 1e-2  # compact preset noise


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _dir_bytes(path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


class Env:
    def __init__(self, tmp) -> None:
        binary = ec.find_server_binary()
        if binary is None:
            pytest.fail("ombra_server not found")
        self.pb, pbg = ec.load_stubs(tmp / "stubs")
        self.data_dir = tmp / "data"
        self.server = ec.ServerProcess(binary, self.data_dir, VERSION, tmp / "server.log")
        self.api = ec.OmbraClient(self.pb, pbg, self.server.address, "m0_compact_n8192", VERSION)
        self.stub = self.api.stub

    def search(self, col: Collection, q, wants=(None,)):
        """Use the test client's search with this collection's keys (same process, same key)."""
        a = self.api
        a.cc, a.keys, a.key_tag = col.kit.cc, col.kit._keys, col.kit.key_tag
        return a.search(col.name, q, list(col.cards), list(wants), K)

    def close(self) -> None:
        self.api.close()
        self.server.stop()


def _brute(x, fields, live, q, wants=(None,)):
    rows = [
        i
        for i in range(len(x))
        if live[i] and all(w is None or fields[i, f] == w for f, w in enumerate(wants))
    ]
    rows.sort(key=lambda i: (-float(x[i] @ q), i))
    return [b"v%04d" % i for i in rows[:K]], {b"v%04d" % i: float(x[i] @ q) for i in rows}


def _same(got, want, true):
    kth = true[want[-1]]
    assert len(got) == len(want)
    for i in set(got) ^ set(want):
        assert abs(true[i] - kth) <= 2 * TOL


def test_create_recluster_repack(tmp_path) -> None:
    env = Env(tmp_path)
    try:
        rng = np.random.default_rng(7)
        n, n_extra = 90, 90
        x = _unit(rng.standard_normal((n + n_extra, D)))
        fields = rng.integers(0, CARDS[0], size=(n + n_extra, 1))
        ids = [b"v%04d" % i for i in range(n + n_extra)]
        live = np.ones(n + n_extra, dtype=bool)
        queries = [_unit(rng.standard_normal((1, D)))[0] for _ in range(2)]

        col = Collection.create(
            env.pb,
            env.stub,
            "docs",
            x[:n],
            ids[:n],
            preset="fast",
            fields=fields[:n],
            cards=CARDS,
            n_clusters=3,
            openfhe_version=VERSION,
        )
        # (a) clusters, centers and keys are present
        info = env.api.get_collection(col.name)
        assert info.keys_uploaded and info.vector_count == n
        assert info.spec.cluster_count == col.cluster_count == 3
        stats = col.block_stats()
        assert sum(b.n_vectors for b in stats) == n
        assert {b.cluster for b in stats} == set(range(3))
        centers = [b for b in env.api.list_blocks(col.name) if b.cluster_id == 3]
        assert len(centers) == 1 and centers[0].n_vectors == 3
        assert "secret" not in repr(col.kit).lower()

        # (b) search equals brute force
        for q in queries:
            res = env.search(col, q)
            want, true = _brute(x[:n], fields[:n], live[:n], q)
            _same(res.ids, want, true)

        # (c) repack: two more small blocks per cluster, then delete 40 % of the vectors
        col.insert_vectors(x[n : n + 45], ids[n : n + 45], fields[n : n + 45])
        col.insert_vectors(x[n + 45 :], ids[n + 45 :], fields[n + 45 :])
        victims = [i for i in range(n + n_extra) if i % 5 in (0, 1)]
        assert col.delete([ids[i] for i in victims]) == len(victims)
        live[victims] = False
        before_blocks = len(col.block_stats())
        before_disk = _dir_bytes(env.data_dir)
        rep = col.repack(0.3)
        after_disk = _dir_bytes(env.data_dir)
        assert rep.groups > 0 and rep.blocks_after < rep.blocks_before
        assert rep.bytes_after < rep.bytes_before and after_disk < before_disk
        assert len(col.block_stats()) < before_blocks
        assert all(b.n_deleted == 0 for b in col.block_stats())
        for q in queries:
            res = env.search(col, q)
            want, true = _brute(x, fields, live, q)
            _same(res.ids, want, true)
        assert col.repack(0.3).groups == 0  # nothing left over the threshold

        # (d) recluster: the old collection works until retired; the new one is correct
        new = col.recluster(n_clusters=2)
        assert new.name != col.name and new.cluster_count == 2
        for q in queries:
            want, true = _brute(x, fields, live, q)
            _same(env.search(col, q).ids, want, true)  # during: old still answers
        col.retire()
        info = env.api.get_collection(new.name)
        assert info.keys_uploaded and info.vector_count == int(live.sum())
        for q in queries:
            want, true = _brute(x, fields, live, q)
            _same(env.search(new, q).ids, want, true)
            for wants in ((1,), (2,)):
                w, t = _brute(x, fields, live, q, wants)
                _same(env.search(new, q, wants).ids, w, t)
    finally:
        env.close()
