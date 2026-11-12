"""End-to-end through the gRPC server, with a RESTART of the server on the same data dir.

tests/test_gpu_e2e_grpc.py already covers create, key upload, insert, search with and
without a filter, delete and the numpy / CPU OpenFHE oracles, but never restarts the server. This
file adds what is missing: the same flow, then the server process is killed (SIGKILL, no clean
shutdown) and started again on the same data directory, and searching again must give

  * the same top-10 ids as before the restart, for no filter and for filters (one and both
    fields), with scores equal within the preset's noise tolerance;
  * no deleted id, and the delete is still recorded (deleting the same ids again removes none);
  * the same answers as the numpy brute force over the live vectors;
  * the same collection info (vector and deleted counts, keys uploaded).

Both presets, d = 16, 300 vectors in 6 clusters (the Deployment of the gRPC server test). The secret
key
stays in the test process; the restarted server gets no key again (it must have kept the public
keys on disk). Synthetic data only.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")
pytest.importorskip("openfhe")

import e2e_client as ec  # noqa: E402
from test_gpu_e2e_grpc import CARDS, PRESETS, VERSION, Deployment, K, _ident  # noqa: E402

pytestmark = pytest.mark.gpu

WANTS = ([None, None], [1, None], [None, 2], [1, 2])


@pytest.fixture(scope="module", params=PRESETS)
def dep(request, tmp_path_factory):
    d = Deployment(tmp_path_factory.mktemp("restart_" + request.param), request.param)
    yield d
    d.close()


def _restart(dep: Deployment, tmp) -> None:
    """Kill the server (SIGKILL), start a new one on the same data dir, reconnect the client."""
    old = dep.client
    old.close()
    dep.server.proc.kill()
    dep.server.proc.wait(timeout=30)
    dep.server.stop()
    binary = ec.find_server_binary()
    assert binary is not None
    dep.server = ec.ServerProcess(binary, dep.data_dir, VERSION, tmp / "server2.log")
    _, pbg = ec.load_stubs(tmp / "stubs2")
    new = ec.OmbraClient(dep.pb, pbg, dep.server.address, dep.preset, VERSION)
    new.cc, new.keys, new.other, new.key_tag = old.cc, old.keys, old.other, old.key_tag
    dep.client = new


def test_restart_keeps_data_deletes_and_results(dep: Deployment, tmp_path) -> None:
    cl = dep.client
    queries = dep.queries[:2]

    def search_all() -> dict:
        out = {}
        for qi, q in enumerate(queries):
            for wants in WANTS:
                res = dep.client.search("docs", q, CARDS, wants, K)
                want, true = dep.brute_force(q, wants)
                dep.assert_same_topk(res.ids, want, true, f"q{qi} {wants}")
                out[(qi, tuple(wants))] = res
        return out

    # Before any delete: both kinds of search match numpy.
    search_all()

    # Delete: the best hits of query 0 (so the results change) and some others.
    top, _ = dep.brute_force(queries[0], [None, None])
    victims = list(dict.fromkeys([int(i[1:]) for i in top[:3]] + [7, 100, 250]))
    assert cl.delete("docs", [_ident(i) for i in victims]) == len(victims)
    dep.deleted.update(victims)
    gone = {_ident(i) for i in victims}

    before = search_all()
    for res in before.values():
        assert not gone & set(res.ids)
    info_before = cl.get_collection("docs")
    assert info_before.keys_uploaded
    assert info_before.vector_count == dep.x.shape[0]
    assert info_before.deleted_count == len(victims)

    _restart(dep, tmp_path)
    cl = dep.client

    info = cl.get_collection("docs")
    assert info.keys_uploaded
    assert info.vector_count == info_before.vector_count
    assert info.deleted_count == info_before.deleted_count
    assert info.spec.cluster_count == info_before.spec.cluster_count

    after = search_all()  # equal to numpy again (brute_force skips the deleted ids)
    assert after.keys() == before.keys()
    for key, res in after.items():
        assert res.ids == before[key].ids, key
        assert np.allclose(res.scores, before[key].scores, atol=2 * dep.tol), key
        assert not gone & set(res.ids), key
    # The delete is durable: nothing is left to delete, and the centers are still all there.
    assert cl.delete("docs", [_ident(i) for i in victims]) == 0
    assert cl.get_collection("docs").deleted_count == len(victims)
    assert len(CARDS) == 2 and K == 10
