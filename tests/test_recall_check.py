"""T1.6: pass/fail logic of the recall acceptance check (core/bench/recall_check.py), CPU only."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "recall_check", REPO_ROOT / "core" / "bench" / "recall_check.py"
)
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


def _result(nq: int = 1000, enc: float = 0.90, plain: float = 0.90) -> dict:
    return {
        "n_queries": nq,
        "full_scan": {"encrypted_no_rerank": 1.0, "plaintext": 1.0},
        "clustered": {
            "100": {
                "1": {"encrypted_no_rerank": enc, "plaintext_exact_scoring": plain},
                "5": {"encrypted_no_rerank": 0.99, "plaintext_exact_scoring": 0.99},
            }
        },
    }


def test_equal_recall_passes() -> None:
    c = rc.check_dataset(_result())
    assert c["verdict"] == rc.PASS
    assert c["settings_checked"] == 3


def test_boundary_is_inclusive() -> None:
    assert rc.check_dataset(_result(enc=0.895, plain=0.900))["verdict"] == rc.PASS
    assert rc.check_dataset(_result(enc=0.8949, plain=0.900))["verdict"] == rc.FAIL


def test_two_sided() -> None:
    # "within" is two-sided: a large positive gap means the comparison is broken, not a pass
    assert rc.check_dataset(_result(enc=0.95, plain=0.90))["verdict"] == rc.FAIL


def test_one_bad_setting_fails_even_if_the_rest_is_exact() -> None:
    r = _result()
    r["full_scan"]["encrypted_no_rerank"] = 0.98
    c = rc.check_dataset(r)
    assert c["verdict"] == rc.FAIL
    assert c["worst_setting"] == "full scan"


def test_too_few_queries_is_not_eligible() -> None:
    assert rc.check_dataset(_result(nq=5))["verdict"] == rc.NOT_ELIGIBLE
    assert rc.check_dataset(_result(nq=5), min_queries=5)["verdict"] == rc.PASS
    # a failure is still a failure on a small run
    assert rc.check_dataset(_result(nq=5, enc=0.5))["verdict"] == rc.FAIL


def test_overall_needs_two_passing_datasets() -> None:
    good = rc.check_dataset(_result())
    bad = rc.check_dataset(_result(enc=0.5))
    small = rc.check_dataset(_result(nq=5))
    assert rc.check_overall({"a": good, "b": copy.deepcopy(good)})["verdict"] == rc.PASS
    assert rc.check_overall({"a": good})["verdict"] == rc.NOT_ELIGIBLE
    assert rc.check_overall({"a": good, "b": small})["verdict"] == rc.NOT_ELIGIBLE
    assert rc.check_overall({"a": good, "b": bad})["verdict"] == rc.FAIL
    assert rc.check_overall({"a": good}, min_datasets=1)["verdict"] == rc.PASS


def test_report_is_aggregates_only() -> None:
    c = rc.check_dataset(_result())
    text = rc.format_report({"a": c}, rc.check_overall({"a": c}), {"a": 1e-7})
    assert "OVERALL" in text and "PASS" in text
