# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Round counts and cluster requests of run_search per security level (no OpenFHE, fake server)."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
from ombra import search
from ombra.presets import FILTER_LAMBDA
from ombra.probe import ClusterStats

C, N = 36, 20  # clusters, vectors per cluster; first round = ceil(sqrt(36)) = 6
GOOD = FILTER_LAMBDA * 1 + 0.1  # slot of a vector matching the one filter field


def _ns(**kw):
    return SimpleNamespace(**kw)


class _Msg(SimpleNamespace):
    def WhichOneof(self, _):  # noqa: N802
        return self.kind


def _fake(level, matches_per_cluster, requests):
    rng = np.random.default_rng(0)
    cs = rng.uniform(-1, 1, C)
    stats = ClusterStats.empty(C, (2,))
    stats.sizes[:] = N
    stats.counts[0][:, 1] = matches_per_cluster
    stats.counts[0][:, 0] = N - matches_per_cluster

    def search_stream(it):
        next(it)
        yield _Msg(kind="centers", centers=_ns(score_blocks=[cs.tobytes()], center_count=C))
        for req in it:
            requests.append(list(req.chosen.cluster_ids))
            for c in req.chosen.cluster_ids:
                ids = [bytes([c, j]) for j in range(N)]
                blk = np.full(N, GOOD).tobytes()
                yield _Msg(kind="scores", scores=_ns(cluster_id=c, score_blocks=[blk], ids=ids))

    pb = _ns(
        ListBlocksRequest=lambda **kw: None,
        SearchStart=lambda **kw: None,
        SearchRequest=lambda start=None, chosen=None: _ns(start=start, chosen=chosen),
        SearchChosen=lambda cluster_ids: _ns(cluster_ids=cluster_ids),
    )
    stub = _ns(
        ListBlocks=lambda r: _ns(blocks=[_ns(cluster_id=c, n_vectors=N) for c in range(C)]),
        Search=search_stream,
    )
    kit = _ns(
        key_tag="t",
        encrypt_query=lambda ext: b"",
        decrypt_column=lambda env: np.frombuffer(env, dtype=np.float64),
    )
    return _ns(
        pb=pb,
        stub=stub,
        kit=kit,
        name="c",
        security_level=level,
        derived=_ns(dimension=4),
        cards=(2,),
        cluster_count=C,
        stats=stats,
    )


@pytest.fixture(autouse=True)
def _no_rerank(monkeypatch):
    monkeypatch.setattr(search, "_rerank", lambda col, q, top, k: [])


def _run(level, flt, per_cluster=5, seed=1):
    requests: list[list[int]] = []
    col = _fake(level, per_cluster, requests)
    q = np.array([1.0, 0, 0, 0])
    res = search.run_search(col, q, 3, filter=flt, level=level, _rng=random.Random(seed))  # noqa: S311
    return res, requests


@pytest.mark.parametrize("level", ["fast", "decoy", "strict"])
def test_filtered_queries_do_exactly_two_rounds(level):
    for per_cluster in (5, 1, 0):  # 25%, 5%, 0% expected matches (0 -> scan)
        res, req = _run(level, {0: 1}, per_cluster)
        assert res.rounds == 2
        flat = [c for r in req for c in r]
        assert len(flat) == len(set(flat))  # no cluster requested twice
        if level == "fast":
            assert len(req) <= 2


@pytest.mark.parametrize("level", ["decoy", "strict"])
def test_unfiltered_queries_do_exactly_two_rounds(level):
    res, req = _run(level, None)
    assert res.rounds == 2 and len(req) == 2


def test_decoy_requested_counts_do_not_depend_on_filter():
    shapes = set()
    for flt, per in ((None, 5), ({0: 1}, 5), ({0: 1}, 1), ({0: [0, 1]}, 10), ({0: 1}, 0)):
        _, req = _run("decoy", flt, per)
        if per == 0 and flt:
            continue  # the low-selectivity full scan is a different, documented shape
        shapes.add(tuple(len(r) for r in req))
    assert shapes == {(12, 24)}


def test_strict_covers_every_cluster_exactly_once():
    for flt in (None, {0: 1}):
        res, req = _run("strict", flt)
        assert req == [list(range(18)), list(range(18, 36))]
        assert sorted(res.probed) == list(range(C))


def test_fast_unfiltered_keeps_single_round():
    res, req = _run("fast", None)
    assert res.rounds == 1 and len(req) == 1


def test_decoy_requests_are_shuffled_with_real_clusters_not_always_first():
    firsts_real = []
    for seed in range(30):
        res, req = _run("decoy", None, seed=seed)
        firsts_real.append(req[0][0] in set(res.probed))
    assert not all(firsts_real)  # chance of 30 real-first requests: 2**-30
    assert any(firsts_real)


def test_public_search_refuses_a_seed():
    from ombra.collection import Collection

    col = Collection.__new__(Collection)
    for opt in ({"seed": 1}, {"_rng": random.Random(1)}):  # noqa: S311
        with pytest.raises(TypeError):
            col.search(np.zeros(4), 3, **opt)


def test_server_center_count_and_cluster_ids_are_checked():
    requests: list[list[int]] = []
    col = _fake("fast", 5, requests)
    good = col.stub.Search

    def lying_count(it):
        for m in good(it):
            if m.kind == "centers":
                m.centers.center_count = C * 1000
            yield m

    col.stub.Search = lying_count
    with pytest.raises(RuntimeError, match="center scores"):
        search.run_search(col, np.array([1.0, 0, 0, 0]), 3, level="fast")

    col = _fake("fast", 5, requests)
    good = col.stub.Search

    def lying_cluster(it):
        for m in good(it):
            if m.kind == "scores":
                m.scores.cluster_id = 10_000
            yield m

    col.stub.Search = lying_cluster
    with pytest.raises(RuntimeError, match="not asked"):
        search.run_search(col, np.array([1.0, 0, 0, 0]), 3, level="fast")
