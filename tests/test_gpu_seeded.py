# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""SEED-2: seeded stored columns (docs/decisions/0023) through the real gRPC server.

Synthetic data, d = 384 on the "fast" preset. The same small collection is created twice on one
server, once with OMBRA_SEEDED=0 (kind 2 columns) and once seeded (kind 4). The server loads the
seeded blocks by expanding the seed on the host, so the answers must be identical and the stored
column bytes (read back with GetBlock) about half.
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
from ombra.collection import Collection  # noqa: E402
from ombra.file_format import Kind, read_column_envelope  # noqa: E402

pytestmark = pytest.mark.gpu

VERSION = "test-openfhe"
D = 384
CARDS = (3,)
K = 5


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _stored(col: Collection) -> tuple[int, set[int]]:
    """Bytes of the stored data columns of every block (via GetBlock) and the kinds found."""
    total, kinds = 0, set()
    for b in col.block_stats():
        _, cols = col._get_block(b.segment)
        total += sum(len(c) for c in cols)
        kinds |= {
            int(read_column_envelope(c, col.derived.preset, col.kit.openfhe_version).kind)
            for c in cols
        }
    return total, kinds


def test_seeded_collection_equals_plain_on_the_server(tmp_path, monkeypatch) -> None:
    binary = ec.find_server_binary()
    if binary is None:
        pytest.fail("ombra_server not found")
    pb, pbg = ec.load_stubs(tmp_path / "stubs")
    server = ec.ServerProcess(binary, tmp_path / "data", VERSION, tmp_path / "server.log")
    api = ec.OmbraClient(pb, pbg, server.address, "m0_compact_n8192", VERSION)
    cols = []
    try:
        rng = np.random.default_rng(11)
        n = 60
        x = _unit(rng.standard_normal((n, D)))
        fields = rng.integers(0, CARDS[0], size=(n, 1))
        ids = [b"v%04d" % i for i in range(n)]
        queries = [_unit(rng.standard_normal((1, D)))[0] for _ in range(3)]

        out: dict = {}
        for name, seeded in (("plain", False), ("seeded", True)):
            monkeypatch.setenv("OMBRA_SEEDED", "1" if seeded else "0")
            col = Collection.create(
                pb,
                api.stub,
                name,
                x,
                ids,
                preset="fast",
                fields=fields,
                cards=CARDS,
                n_clusters=2,
                openfhe_version=VERSION,
            )
            cols.append(col)
            total, kinds = _stored(col)
            top = []
            for q in queries:
                top.append([h.id for h in col.search(q, K).hits])
            top.append([h.id for h in col.search(queries[0], K, filter={0: 1}).hits])
            top.append([h.id for h in col.search(queries[1], K, filter={0: 2}).hits])
            out[name] = {"bytes": total, "kinds": kinds, "top": top}

        assert out["plain"]["kinds"] == {int(Kind.CIPHERTEXT)}
        assert out["seeded"]["kinds"] == {int(Kind.SEEDED_CIPHERTEXT)}
        assert all(len(t) > 0 for t in out["plain"]["top"])
        assert out["seeded"]["top"] == out["plain"]["top"]
        ratio = out["seeded"]["bytes"] / out["plain"]["bytes"]
        assert 0.45 <= ratio <= 0.60, ratio
    finally:
        for c in cols:
            c.close()
        api.close()
        server.stop()
