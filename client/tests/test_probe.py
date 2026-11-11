"""Probe policy and cluster statistics (T3.11): pure logic, no OpenFHE, no server."""

import math

import numpy as np
import pytest
from ombra.blobstore import KIND_RECORD, InMemoryBlobStore
from ombra.clustering import assign
from ombra.documents import DocumentCodec
from ombra.probe import STATS_ID, ClusterStats, ProbePlanner, check_user_id


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _data(seed: int = 0, n: int = 600, d: int = 8, nc: int = 6):
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(size=(nc, d)))
    x = _unit(centers[rng.integers(0, nc, n)] + 0.6 * rng.normal(size=(n, d)))
    labels = assign(x, centers)
    fields = np.stack([rng.integers(0, 5, n), rng.integers(0, 3, n)], axis=1)
    stats = ClusterStats.build(x, labels, centers, fields, (5, 3))
    return x, centers, labels, fields, stats


def test_counts_and_expected_matches():
    x, centers, labels, fields, stats = _data()
    assert stats.sizes.tolist() == np.bincount(labels, minlength=6).tolist()
    exp = stats.expected({0: [1, 2]})
    for c in range(6):
        assert exp[c] == np.sum((labels == c) & np.isin(fields[:, 0], [1, 2]))
    # AND across fields: size * product of fractions
    both = stats.expected({0: 1, 1: [0, 2]})
    c = 0
    m = labels == c
    want = m.sum() * np.mean(fields[m, 0] == 1) * np.mean(np.isin(fields[m, 1], [0, 2]))
    assert both[c] == pytest.approx(want)
    assert np.array_equal(stats.expected(None), stats.sizes)


def test_upper_bound_holds_for_every_member():
    x, centers, labels, _, stats = _data(1)
    q = _unit(np.random.default_rng(5).normal(size=(50, 8)))
    for qv in q:
        cs = centers @ qv
        ub = stats.upper_bounds(cs)
        for c in range(6):
            members = x[labels == c]
            if len(members):
                assert (members @ qv).max() <= ub[c] + 1e-12


def test_insert_delete_update():
    x, centers, labels, fields, stats = _data(2)
    extra = _unit(np.random.default_rng(9).normal(size=(10, 8)))
    ef = np.zeros((10, 2), dtype=np.int64)
    el = assign(extra, centers)
    stats.add(extra, el, centers, ef)
    assert stats.sizes.sum() == 610
    for c in range(6):
        if (el == c).any():
            assert stats.cos_min[c] <= (extra[el == c] @ centers[c]).min() + 1e-12
    before = stats.cos_min.copy()
    stats.remove(el, ef)
    assert stats.sizes.sum() == 600 and np.array_equal(stats.cos_min, before)
    assert (stats.counts[0] >= 0).all()


def test_stats_blob_round_trip_is_sealed():
    *_, stats = _data(3)
    codec = DocumentCodec(b"k" * 32, "col")
    store = InMemoryBlobStore()
    store.put("col", KIND_RECORD, STATS_ID, stats.seal(codec))
    blob = store.get("col", KIND_RECORD, STATS_ID)
    assert stats.to_bytes()[:20] not in blob  # nothing readable in the stored bytes
    back = ClusterStats.unseal(codec, blob)
    assert back.cards == stats.cards
    assert np.array_equal(back.sizes, stats.sizes)
    assert np.array_equal(back.cos_min, stats.cos_min)
    assert all(np.array_equal(a, b) for a, b in zip(back.counts, stats.counts, strict=True))
    with pytest.raises(ValueError):
        ClusterStats.from_bytes(b"XXXX" + stats.to_bytes()[4:])


def test_reserved_ids_refused():
    check_user_id(b"doc-1")
    with pytest.raises(ValueError):
        check_user_id(b"\x00ombra/anything")


def _planner(policy, stats, cs, flt, k=3, first=2, **kw):
    order = [int(i) for i in np.argsort(-cs, kind="stable")]
    return order, ProbePlanner(
        order, policy, k, first, stats=stats, filter=flt, center_scores=cs, **kw
    )


def _toy_stats(sizes, match_counts, cos_min):
    n = len(sizes)
    s = ClusterStats.empty(n, (2,))
    s.sizes = np.array(sizes)
    s.counts[0][:, 1] = match_counts
    s.counts[0][:, 0] = np.array(sizes) - np.array(match_counts)
    s.cos_min = np.array(cos_min, dtype=float)
    return s


def test_first_round_skips_zero_expected_and_reaches_3k():
    stats = _toy_stats([100] * 5, [0, 4, 0, 4, 4], [0.5] * 5)
    cs = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    order, p = _planner("e", stats, cs, {0: 1}, k=2, first=1)
    # expected per cluster 0,4,0,4,4: clusters 1 and 3 reach 8 >= 3k = 6; cluster 0 and 2 skipped
    assert p.first_round() == [1, 3]


def test_first_round_never_below_unfiltered_first_round():
    stats = _toy_stats([100] * 5, [50] * 5, [0.5] * 5)
    cs = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    _, p = _planner("e", stats, cs, {0: 1}, k=2, first=3)
    assert p.first_round() == [0, 1, 2]  # 3k expected after one cluster, floor of 3 wins


def test_strict_scan_below_half_percent_selectivity():
    stats = _toy_stats([1000] * 4, [1, 1, 1, 0], [0.5] * 4)  # 3 of 4000 = 0.075%
    _, p = _planner("e_a", stats, np.array([0.9, 0.8, 0.7, 0.6]), {0: 1})
    assert p.first_round() is None


def test_stopping_check_widens_only_while_a_cluster_could_win():
    # cluster 1 has a wide radius (cos_min 0.0), cluster 2 is tight (cos_min 0.99)
    stats = _toy_stats([100] * 3, [50, 50, 50], [0.99, 0.0, 0.99])
    cs = np.array([0.95, 0.6, 0.5])
    _, p = _planner("e_a", stats, cs, {0: 1}, k=1, first=1)
    assert p.first_round() == [0]
    # k-th match score 0.94: cluster 1 could reach cos(max(0, acos(.6) - pi/2)) = 1, cluster 2
    # at most cos(acos(.5) - acos(.99)) = cos(1.047 - 0.1415) = 0.618 < 0.94
    assert p.next_round(0.94, 5) == [1]
    assert p.next_round(0.94, 5) == []  # cluster 2 cannot win: stop
    # below k matches: kth = -inf widens to the next cluster
    _, p2 = _planner("e_a", stats, cs, {0: 1}, k=1, first=1)
    p2.first_round()
    assert p2.next_round(-math.inf, 0) == [1, 2]


def test_max_rounds_caps_widening_and_e_stops_at_k():
    stats = _toy_stats([100] * 8, [50] * 8, [0.0] * 8)
    cs = np.linspace(0.9, 0.2, 8)
    _, p = _planner("e_a", stats, cs, {0: 1}, k=1, first=1, max_rounds=3)
    assert p.first_round() == [0]
    assert p.next_round(0.99, 1) == [1, 2]  # 2x growth
    assert p.next_round(0.99, 1) == [3, 4, 5, 6]
    assert p.next_round(0.99, 1) == []  # three rounds done
    _, pe = _planner("e", stats, cs, {0: 1}, k=1, first=1)
    pe.first_round()
    assert pe.next_round(0.99, 1) == []  # k matches reached: E does not widen


def test_basic_policy_is_the_0019_rule():
    stats = _toy_stats([10] * 8, [5] * 8, [0.5] * 8)
    cs = np.linspace(0.9, 0.2, 8)
    order = list(range(8))
    p = ProbePlanner(order, "basic", 3, 2)
    assert p.first_round() == [0, 1]
    assert p.next_round(-math.inf, 1) == [2, 3, 4, 5]  # 2x
    assert p.next_round(-math.inf, 1) == [6, 7]  # capped at 8x first (all 8 here)
    assert p.next_round(-math.inf, 1) == []
    q = ProbePlanner(order, "basic", 3, 2)
    q.first_round()
    assert q.next_round(0.1, 3) == []  # k matches: stop
    del stats, cs
