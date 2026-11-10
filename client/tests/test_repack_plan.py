"""T3.5: block repack planning and clustering (CPU only, no OpenFHE)."""

from __future__ import annotations

import numpy as np
import pytest
from ombra.repack import BlockStat, plan_repack

SLOTS = 100


def _b(seg: str, cluster: int, n: int, deleted: int) -> BlockStat:
    return BlockStat(seg, cluster, n, deleted)


def test_below_threshold_untouched() -> None:
    assert plan_repack([_b("a", 0, 100, 29), _b("b", 0, 100, 0)], SLOTS) == []


def test_threshold_is_inclusive_and_merges() -> None:
    blocks = [_b("a", 0, 40, 12), _b("b", 0, 30, 0), _b("c", 0, 30, 0)]  # 30 % in a
    plan = plan_repack(blocks, SLOTS)
    assert len(plan) == 1 and {b.segment for b in plan[0]} == {"a", "b", "c"}  # 28+30+30 <= 100


def test_does_not_overfill_and_skips_dead_blocks() -> None:
    blocks = [_b("a", 0, 100, 40), _b("b", 0, 100, 0), _b("dead", 0, 10, 10), _b("z", 1, 5, 0)]
    plan = plan_repack(blocks, SLOTS)
    assert [[b.segment for b in g] for g in plan] == [["a"]]  # b (100 live) does not fit with a


def test_threshold_configurable() -> None:
    assert plan_repack([_b("a", 0, 100, 10)], SLOTS, 0.1) != []
    with pytest.raises(ValueError):
        plan_repack([], SLOTS, 0.0)


def test_clustering_covers_all_vectors() -> None:
    pytest.importorskip("faiss")
    from ombra.clustering import cluster_vectors

    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 16))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    c = cluster_vectors(x, 5, seed=3)
    assert c.centers.shape[1] == 16 and len(c.labels) == 200
    assert set(c.labels) == set(range(len(c.centers)))
    assert np.allclose(np.linalg.norm(c.centers, axis=1), 1.0)
