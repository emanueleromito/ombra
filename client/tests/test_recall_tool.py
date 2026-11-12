"""Argument parsing and verdict logic of the recall tool (no OpenFHE needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from ombra import recall  # noqa: E402


def test_parser_defaults_and_options() -> None:
    a = recall.build_parser().parse_args(["v.npy"])
    assert (a.preset, a.security_level, a.k) == ("fast", "fast", 10)
    assert (a.n_probe, a.n_queries) == (None, 100)
    a = recall.build_parser().parse_args(
        ["v.npy", "--filter-field", "f.npy", "--queries", "q.npy", "--n-probe", "3", "--k", "5"]
    )
    assert (a.filter_field, a.queries, a.n_probe, a.k) == ("f.npy", "q.npy", 3, 5)
    with pytest.raises(SystemExit):
        recall.build_parser().parse_args(["v.npy", "--queries", "q.npy", "--n-queries", "5"])
    with pytest.raises(SystemExit):
        recall.build_parser().parse_args(["v.npy", "--security-level", "nope"])


def test_verdict_logic() -> None:
    assert recall.verdict(0.0, 1000) == recall.PASS
    assert recall.verdict(-0.005, 5000) == recall.PASS
    assert recall.verdict(0.0, 999) == recall.NOT_ELIGIBLE
    assert recall.verdict(-0.006, 5000) == recall.FAIL
    assert recall.verdict(0.02, 10) == recall.FAIL


def test_recall_at_k() -> None:
    assert recall.recall_at_k([1, 2, 3], [1, 2, 4, 5], 4) == 0.5
    assert recall.recall_at_k([], [], 10) == 1.0


def test_thresholds_match_the_0012_check() -> None:
    path = Path(__file__).resolve().parents[2] / "core" / "bench" / "recall_check.py"
    if not path.exists():
        pytest.skip("repository checkout only")
    spec = importlib.util.spec_from_file_location("recall_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (mod.TOLERANCE, mod.MIN_QUERIES) == (recall.TOLERANCE, recall.MIN_QUERIES)
    assert (mod.PASS, mod.FAIL, mod.NOT_ELIGIBLE) == (recall.PASS, recall.FAIL, recall.NOT_ELIGIBLE)


def test_bad_input_is_refused(tmp_path) -> None:
    np.save(tmp_path / "v.npy", np.ones(5))
    assert recall.main([str(tmp_path / "v.npy")]) == 2
