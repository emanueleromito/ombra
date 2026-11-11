"""T3.6: insert, delete and search through the client library against the real gRPC server.

Synthetic data only; d = 384 on the "fast" preset, one filter field with 3 values, 6 clusters of
well separated vectors. Checks that top-k with re-ranking equals a plaintext search at each
security level (Fast, Decoy, Strict) and with a filter; that a schema above F_max is refused; that
a filtered query probes at least the unfiltered clusters and widens only below k matches; that
deleted vectors (and their blobs) are gone; and that a collection can be reopened from the key file.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")
pytest.importorskip("openfhe")
pytest.importorskip("faiss")

import e2e_client as ec  # noqa: E402
from ombra import presets  # noqa: E402
from ombra.blobstore import KIND_DOCUMENT, KIND_RECORD, BlobStoreError  # noqa: E402
from ombra.collection import Collection, KeyKit  # noqa: E402
from ombra.params import ParamsError  # noqa: E402

BACKENDS = [
    pytest.param("server", marks=pytest.mark.gpu),
    pytest.param("local", marks=pytest.mark.openfhe),  # the in-process CPU server (T3.7)
]

VERSION = "test-openfhe"
D = 384
CARDS = (3,)
K = 5
CLUSTERS = 6
PER = 25
SCORE_TOL = 1e-5  # the re-ranked score is exact (float32 record), not the noisy encrypted one


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _plain(x, fields, live, q, want=None, rows=None):
    """Plaintext top-K ids and scores among live vectors (matching the filter, in `rows`)."""
    cand = [
        i
        for i in (range(len(x)) if rows is None else rows)
        if live[i] and (want is None or fields[i, 0] == want)
    ]
    cand.sort(key=lambda i: (-float(x[i] @ q), b"v%04d" % i))
    return [b"v%04d" % i for i in cand[:K]], [float(x[i] @ q) for i in cand[:K]]


def _check(res, want):
    ids, scores = want
    assert [h.id for h in res.hits] == ids
    assert np.allclose([h.score for h in res.hits], scores, atol=SCORE_TOL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_insert_search_delete(tmp_path, backend) -> None:
    version = VERSION
    if backend == "server":
        binary = ec.find_server_binary()
        if binary is None:
            pytest.fail("ombra_server not found")
        pb, pbg = ec.load_stubs(tmp_path / "stubs")
        server = ec.ServerProcess(binary, tmp_path / "data", version, tmp_path / "server.log")
        api = ec.OmbraClient(pb, pbg, server.address, "m0_compact_n8192", version)
        stub, stop = api.stub, lambda: (api.close(), server.stop())
    else:
        from ombra._local import LocalTransport

        transport = LocalTransport(tmp_path / "local")
        pb, stub, version = transport.pb, transport.stub, transport.openfhe_version
        stop = transport.close
    col = None
    try:
        rng = np.random.default_rng(11)
        base = _unit(rng.standard_normal((CLUSTERS, D)))
        n = CLUSTERS * PER
        x = _unit(np.repeat(base, PER, axis=0) + 0.3 * _unit(rng.standard_normal((n, D))))
        fields = rng.integers(0, 2, size=(n, 1))  # values 0 and 1 are common ...
        rare = rng.choice(n, 4, replace=False)
        fields[rare, 0] = 2  # ... value 2 has only 4 vectors, fewer than K
        ids = [b"v%04d" % i for i in range(n)]
        live = np.ones(n, dtype=bool)
        queries = [_unit(base[c] + 0.05 * rng.standard_normal(D)) for c in (0, 3)]

        # schema above F_max is refused, before anything is sent
        with pytest.raises(ParamsError):
            Collection.create(
                pb,
                stub,
                "x",
                x[:1],
                ids[:1],
                preset="fast",
                fields=np.zeros((1, 17)),
                cards=[2] * 17,
                openfhe_version=version,
            )
        with pytest.raises(ParamsError):
            Collection.create(
                pb,
                stub,
                "x",
                x[:1],
                ids[:1],
                preset="balanced",
                fields=np.zeros((1, 15)),
                cards=[2] * 15,
                openfhe_version=version,
            )

        col = Collection.create(
            pb,
            stub,
            "docs",
            x[: n - 30],
            ids[: n - 30],
            preset="fast",
            fields=fields[: n - 30],
            cards=CARDS,
            n_clusters=CLUSTERS,
            openfhe_version=version,
            metadata=[{"i": i} for i in range(n - 30)],
            documents=[f"doc {i}" for i in range(n - 30)],
        )
        col.insert_vectors(
            x[n - 30 :], ids[n - 30 :], fields[n - 30 :], [{"i": i} for i in range(n - 30, n)]
        )
        assert col.blobs.get(col.name, KIND_DOCUMENT, ids[0])

        # top-k with re-ranking equals plaintext search at each security level
        for q in queries:
            want = _plain(x, fields, live, q)
            _check(col.search(q, K, n_probe=2), want)  # fast (the collection's level)
            _check(col.search(q, K, level="decoy", decoys=2, n_probe=2, seed=1), want)
            res = col.search(q, K, level="strict")
            _check(res, want)
            assert sorted(res.probed) == list(range(col.cluster_count))
            assert res.hits[0].metadata == {"i": int(res.hits[0].id[1:])}
        _check(col.search(queries[0], K, n_probe=2, workers=2), _plain(x, fields, live, queries[0]))

        # with a filter; a filtered query probes at least what the unfiltered one does
        for q in queries:
            unf = col.search(q, K, n_probe=2)
            for v in (0, 1):
                fil = col.search(q, K, filter={0: v}, n_probe=2, probe="basic")
                _check(fil, _plain(x, fields, live, q, v))
                assert set(unf.probed) <= set(fil.probed)
                assert fil.rounds == 1 and fil.probed == unf.probed  # K matches: no widening
            both = col.search(q, K, filter={0: [0, 1]}, n_probe=2)
            _check(
                both, _plain(x, fields, live, q, None, [i for i in range(n) if fields[i, 0] < 2])
            )

        # a selective filter (4 matches, K = 5) widens 2x per round up to max_probe, then stops
        q = queries[0]
        rare_want = _plain(x, fields, live, q, 2)
        res = col.search(
            q, K, filter={0: 2}, n_probe=1, max_probe_factor=2, probe="basic", strict_fallback=False
        )
        assert res.incomplete and len(res.probed) == 2 and res.rounds == 2
        assert not res.strict_fallback and all(fields[int(h.id[1:]), 0] == 2 for h in res.hits)
        res = col.search(
            q, K, filter={0: 2}, n_probe=1, max_probe_factor=2, probe="basic"
        )  # Strict fallback
        assert res.strict_fallback and res.incomplete and sorted(res.probed) == list(range(6))
        assert [h.id for h in res.hits] == rare_want[0] and len(res.hits) == 4

        # a new vector comes back first, with its metadata and document
        new = _unit(queries[1] + 0.01 * rng.standard_normal(D))
        col.insert_vectors(new[None], [b"new"], np.array([[1]]), [{"tag": "new"}], ["hello"])
        top = col.search(new, K, n_probe=2).hits[0]
        assert top.id == b"new" and top.metadata == {"tag": "new"}
        assert col.delete([b"new"]) == 1

        # deleted vectors are not returned, and their blobs are gone
        q = queries[1]
        victims = [h.id for h in col.search(q, K, n_probe=2).hits[:2]]
        assert col.delete(victims) == 2
        for v in victims:
            live[int(v[1:])] = False
            with pytest.raises(BlobStoreError):
                col.blobs.get(col.name, KIND_RECORD, v)
        res = col.search(q, K, n_probe=2)
        assert not {h.id for h in res.hits} & set(victims)
        _check(res, _plain(x, fields, live, q))
        _check(col.search(q, K, filter={0: 1}, n_probe=2), _plain(x, fields, live, q, 1))

        # the keys come back from the key file: a collection can be reopened
        key_path = tmp_path / "docs.key"
        col.kit.save(key_path, "pw", scrypt_params=(2**4, 8, 1))
        kit2 = KeyKit.load(key_path, "pw", presets.derive("fast", D), version, col.kit.n_leaves)
        again = Collection.open(pb, stub, col.name, kit2)
        assert again.cards == CARDS and again.cluster_count == CLUSTERS
        assert np.allclose(again.centers, col.centers, atol=1e-2)
        _check(again.search(q, K, n_probe=2), _plain(x, fields, live, q))
        again.insert_vectors(x[:1] * 1.0, [b"again"], fields[:1])
        assert again.codec.open_record(b"again", again.blobs.get(again.name, KIND_RECORD, b"again"))
    finally:
        if col is not None:
            col.close()
        stop()
