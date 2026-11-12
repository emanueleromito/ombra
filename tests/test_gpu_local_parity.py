"""Local mode parity: the same data and queries through local mode (CPU, child process, compiled
expansion) and through the real GPU server give the same top-k, and the score ciphertexts come back
at the same level. Synthetic data only; d = 384 on the "fast" preset, one filter field.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")
pytest.importorskip("openfhe")
pytest.importorskip("faiss")
pytest.importorskip("ombra._native", reason="the compiled extension is not built")

import e2e_client as ec  # noqa: E402
from ombra._local import LocalTransport  # noqa: E402
from ombra.collection import Collection  # noqa: E402
from ombra.file_format import Kind, read_envelope  # noqa: E402

pytestmark = pytest.mark.gpu

D = 384
K = 5
CLUSTERS = 4
PER = 25


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _level(col, pb, stub, query, version) -> int:
    import tempfile
    from pathlib import Path

    import openfhe as ofhe

    start = pb.SearchStart(
        collection=col.name,
        key_tag=col.kit.key_tag,
        query=col.kit.encrypt_query(query),
        filter_leaves=sum(col.cards),
    )
    stream = stub.Search(iter([pb.SearchRequest(start=start)]))
    centers = next(stream).centers
    if hasattr(stream, "cancel"):
        stream.cancel()
    else:
        stream.close()
    env = read_envelope(
        bytes(centers.score_blocks[0]), Kind.CIPHERTEXT, "m0_compact_n8192", version
    )
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "c.bin"
        p.write_bytes(env.payload)
        ct, ok = ofhe.DeserializeCiphertext(str(p), ofhe.BINARY)
    assert ok
    return ct.GetLevel()


def test_same_top_k_through_local_mode_and_the_server(tmp_path) -> None:
    binary = ec.find_server_binary()
    if binary is None:
        pytest.fail("ombra_server not found")
    rng = np.random.default_rng(5)
    base = _unit(rng.standard_normal((CLUSTERS, D)))
    n = CLUSTERS * PER
    x = _unit(np.repeat(base, PER, axis=0) + 0.3 * _unit(rng.standard_normal((n, D))))
    fields = rng.integers(0, 2, size=(n, 1))
    ids = [b"v%04d" % i for i in range(n)]
    queries = [_unit(base[c] + 0.05 * rng.standard_normal(D)) for c in (0, 2)]

    local = LocalTransport(tmp_path / "local")
    version = "test-openfhe"
    pb, pbg = ec.load_stubs(tmp_path / "stubs")
    server = ec.ServerProcess(binary, tmp_path / "data", version, tmp_path / "server.log")
    api = ec.OmbraClient(pb, pbg, server.address, "m0_compact_n8192", version)
    cols = []
    try:
        results, levels = {}, {}
        for name, (tpb, stub, ver) in {
            "local": (local.pb, local.stub, local.openfhe_version),
            "server": (pb, api.stub, version),
        }.items():
            col = Collection.create(
                tpb,
                stub,
                "parity",
                x,
                ids,
                preset="fast",
                fields=fields,
                cards=(2,),
                n_clusters=CLUSTERS,
                openfhe_version=ver,
            )
            cols.append(col)
            out = []
            for q in queries:
                out.append([h.id for h in col.search(q, K, n_probe=2).hits])
                out.append([h.id for h in col.search(q, K, filter={0: 1}, n_probe=2).hits])
                out.append([h.id for h in col.search(q, K, level="strict").hits])
            results[name] = out
            levels[name] = _level(col, tpb, stub, queries[0], ver)
        assert results["local"] == results["server"]
        assert levels["local"] == levels["server"]
    finally:
        for col in cols:
            col.close()
        api.close()
        server.stop()
        local.close()
