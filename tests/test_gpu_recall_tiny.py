"""T0.8: the recall measurement code path end to end on tiny synthetic data.

Seeded synthetic unit vectors (d = 16, one database block, a few queries), a faiss k-means with
4 clusters, the real pipeline of core/bench/recall.py (encrypt blocks and centers, encode queries,
GPU expansion and search through `ombra_recall_search`, Relinearize, Rescale, encrypted all-ones
mask, Rescale, decrypt). Checks that encrypted scores are within the 1e-6 bound of numpy, that the
encrypted top-10 equals the plaintext top-10, and that probing all clusters loses nothing. Prints
aggregates only; the data is synthetic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260111
TOLERANCE = 1e-6


def _load_recall():
    path = REPO_ROOT / "core" / "bench" / "recall.py"
    spec = importlib.util.spec_from_file_location("ombra_recall", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recall_pipeline_tiny(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    recall = _load_recall()
    rng = np.random.default_rng(SEED)
    d, n, nq, nlist = 16, 2000, 4, 4
    db = rng.standard_normal((n, d))
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    qs = rng.standard_normal((nq, d))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)

    work = tmp_path / "work"
    work.mkdir()
    result = recall.measure(db, qs, (nlist,), (1, 2, 4), work)

    assert result["max_abs_score_error_scan"] < TOLERANCE
    assert result["max_abs_score_error_center_scores"] < TOLERANCE
    full = result["full_scan"]
    assert full["encrypted_top10_equals_plaintext_top10_fraction"] == 1.0
    assert full["encrypted_no_rerank"] == 1.0
    assert full["encrypted_rerank"] == 1.0
    assert full["plaintext"] == 1.0
    rows = result["clustered"][str(nlist)]
    assert set(rows) == {"1", "2", "4"}
    assert rows["4"]["encrypted_no_rerank"] == 1.0
    assert rows["4"]["plaintext_exact_scoring"] == 1.0
    for row in rows.values():
        assert 0.0 <= row["encrypted_no_rerank"] <= 1.0
        assert 0.0 <= row["encrypted_rerank"] <= 1.0
    print(
        f"tiny recall: err_scan={result['max_abs_score_error_scan']:.3e} "
        f"err_centers={result['max_abs_score_error_center_scores']:.3e}"
    )
