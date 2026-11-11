"""Probe policy and cluster statistics (T3.11): pure logic, no OpenFHE, no server."""

import numpy as np
import pytest
from ombra.blobstore import KIND_RECORD, InMemoryBlobStore
from ombra.clustering import assign
from ombra.documents import DocumentCodec
from ombra.probe import (
    STATS_ID,
    ClusterStats,
    ProbePlanner,
    check_user_id,
    decoy_round_sizes,
    strict_halves,
)


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _data(seed: int = 0, n: int = 600, d: int = 8, nc: int = 6):
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(size=(nc, d)))
    x = _unit(centers[rng.integers(0, nc, n)] + 0.6 * rng.normal(size=(n, d)))
    labels = assign(x, centers)
    fields = np.stack([rng.integers(0, 5, n), rng.integers(0, 3, n)], axis=1)
    stats = ClusterStats.build(labels, nc, fields, (5, 3))
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


def test_insert_delete_update():
    x, centers, labels, fields, stats = _data(2)
    extra = _unit(np.random.default_rng(9).normal(size=(10, 8)))
    ef = np.zeros((10, 2), dtype=np.int64)
    el = assign(extra, centers)
    stats.add(el, ef)
    assert stats.sizes.sum() == 610
    stats.remove(el, ef)
    assert stats.sizes.sum() == 600
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
    assert all(np.array_equal(a, b) for a, b in zip(back.counts, stats.counts, strict=True))
    with pytest.raises(ValueError):
        ClusterStats.from_bytes(b"XXXX" + stats.to_bytes()[4:])
    old = bytearray(stats.to_bytes())
    old[4:6] = (2).to_bytes(2, "little")  # version 2 held radii: refused, not misread
    with pytest.raises(ValueError):
        ClusterStats.from_bytes(bytes(old))


def test_reserved_ids_refused():
    check_user_id(b"doc-1")
    with pytest.raises(ValueError):
        check_user_id(b"\x00ombra/anything")


def _planner(policy, stats, cs, flt, k=3, first=2, **kw):
    order = [int(i) for i in np.argsort(-cs, kind="stable")]
    return order, ProbePlanner(order, policy, k, first, stats=stats, filter=flt, **kw)


def _toy_stats(sizes, match_counts):
    n = len(sizes)
    s = ClusterStats.empty(n, (2,))
    s.sizes = np.array(sizes)
    s.counts[0][:, 1] = match_counts
    s.counts[0][:, 0] = np.array(sizes) - np.array(match_counts)
    return s


def test_first_round_skips_zero_expected_and_reaches_3k():
    stats = _toy_stats([100] * 5, [0, 4, 0, 4, 4])
    cs = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    order, p = _planner("e2", stats, cs, {0: 1}, k=2, first=1)
    # expected per cluster 0,4,0,4,4: clusters 1 and 3 reach 8 >= 3k = 6; cluster 0 and 2 skipped
    assert p.first_round() == [1, 3]


def test_first_round_never_below_unfiltered_first_round():
    stats = _toy_stats([100] * 5, [50] * 5)
    cs = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    _, p = _planner("e2", stats, cs, {0: 1}, k=2, first=3)
    assert p.first_round() == [0, 1, 2]  # 3k expected after one cluster, floor of 3 wins


def test_strict_scan_below_half_percent_selectivity():
    stats = _toy_stats([1000] * 4, [1, 1, 1, 0])  # 3 of 4000 = 0.075%
    _, p = _planner("e2", stats, np.array([0.9, 0.8, 0.7, 0.6]), {0: 1})
    assert p.first_round() is None


def test_e2_second_round_is_one_capped_widening_round():
    stats = _toy_stats([100] * 10, [50, 0, 50, 50, 50, 0, 50, 50, 50, 50])
    cs = np.linspace(0.9, 0.2, 10)
    _, p = _planner("e2", stats, cs, {0: 1}, k=1, first=1)
    assert p.first_round() == [0]
    # round 2: 2x round 1's real clusters, skipping the zero-match cluster 1
    assert p.next_round(5) == [2, 3]
    assert p.next_round(0) == []  # never a third round, even with no matches
    _, p2 = _planner("e2", stats, cs, {0: 1}, k=1, first=1)
    p2.first_round()
    assert p2.next_round(0, {2}) == [3, 4]  # clusters asked as decoys are not repeated


def test_e2_second_round_always_happens_even_when_empty():
    stats = _toy_stats([100] * 3, [50, 50, 50])
    _, p = _planner("e2", stats, np.array([0.9, 0.8, 0.7]), {0: 1}, k=1, first=3)
    assert p.first_round() == [0, 1, 2]
    assert p.next_round(10) == []
    assert p.rounds == 2


def test_cap1_limits_round_one():
    stats = _toy_stats([1] * 20, [1] * 20)  # 3k = 30 expected needs all 20 clusters
    _, p = _planner("e2", stats, np.linspace(0.9, 0.1, 20), {0: 1}, k=10, first=2, cap1=4)
    assert p.first_round() == [0, 1, 2, 3]


def test_plain2_is_first_then_next_twice_as_many():
    order = list(range(12))
    p = ProbePlanner(order, "plain2", 3, 2)
    assert p.first_round() == [0, 1]
    assert p.next_round(0) == [2, 3, 4, 5]
    assert p.next_round(0) == []


def test_strict_halves_cover_every_cluster_once_by_id():
    for n in (1, 2, 7, 36):
        a, b = strict_halves(n)
        assert a + b == list(range(n))


def test_decoy_round_sizes_do_not_depend_on_filter():
    assert decoy_round_sizes(6, 36) == (12, 24)
    assert decoy_round_sizes(6, 16) == (12, 4)  # fewer clusters left than the nominal size
    assert decoy_round_sizes(10, 12) == (12, 0)


def test_basic_policy_is_the_0019_rule():
    order = list(range(8))
    p = ProbePlanner(order, "basic", 3, 2)
    assert p.first_round() == [0, 1]
    assert p.next_round(1) == [2, 3, 4, 5]  # 2x
    assert p.next_round(1) == [6, 7]  # capped at 8x first (all 8 here)
    assert p.next_round(1) == []
    q = ProbePlanner(order, "basic", 3, 2)
    q.first_round()
    assert q.next_round(3) == []  # k matches: stop
