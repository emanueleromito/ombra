"""Pass/fail rule of the recall-based preset acceptance check. Standard library only.

A preset passes when, on at least MIN_DATASETS public embedding datasets with at least MIN_QUERIES
queries each, recall@10 under encryption is within TOLERANCE of the plaintext recall at the same
cluster settings, for every setting checked. The score error is reported as a safety margin and is
not part of the rule.

Verdicts: "PASS", "FAIL" (some setting is out of tolerance) and "NOT ELIGIBLE" (every setting is
within tolerance, but the run is too small to count: fewer queries or fewer datasets than required,
for example a dry run).
"""

from __future__ import annotations

from typing import Any

TOLERANCE = 0.005
MIN_DATASETS = 2
MIN_QUERIES = 1000

PASS = "PASS"  # noqa: S105 - a verdict label, not a password
FAIL = "FAIL"
NOT_ELIGIBLE = "NOT ELIGIBLE"


def _setting(label: str, encrypted: float, plaintext: float, tolerance: float) -> dict[str, Any]:
    gap = encrypted - plaintext
    return {
        "setting": label,
        "encrypted_recall": encrypted,
        "plaintext_recall": plaintext,
        "gap": gap,
        "within_tolerance": abs(gap) <= tolerance + 1e-12,
    }


def check_dataset(
    result: dict[str, Any], tolerance: float = TOLERANCE, min_queries: int = MIN_QUERIES
) -> dict[str, Any]:
    """Apply the rule to one dataset's result dict (as made by recall.measure)."""
    rows = [
        _setting(
            "full scan",
            result["full_scan"]["encrypted_no_rerank"],
            result["full_scan"]["plaintext"],
            tolerance,
        )
    ]
    for nlist, per_probe in result["clustered"].items():
        for nprobe, row in per_probe.items():
            rows.append(
                _setting(
                    f"nlist={nlist} nprobe={nprobe}",
                    row["encrypted_no_rerank"],
                    row["plaintext_exact_scoring"],
                    tolerance,
                )
            )
    settings_ok = all(r["within_tolerance"] for r in rows)
    enough_queries = result["n_queries"] >= min_queries
    if not settings_ok:
        verdict = FAIL
    elif not enough_queries:
        verdict = NOT_ELIGIBLE
    else:
        verdict = PASS
    worst = max(rows, key=lambda r: abs(r["gap"]))
    return {
        "tolerance": tolerance,
        "min_queries": min_queries,
        "n_queries": result["n_queries"],
        "settings": rows,
        "settings_checked": len(rows),
        "all_settings_within_tolerance": settings_ok,
        "worst_setting": worst["setting"],
        "worst_gap": worst["gap"],
        "verdict": verdict,
    }


def check_overall(
    checks: dict[str, dict[str, Any]], min_datasets: int = MIN_DATASETS
) -> dict[str, Any]:
    """Combine per-dataset checks (name -> check_dataset output) into the preset verdict."""
    verdicts = {name: c["verdict"] for name, c in checks.items()}
    if any(v == FAIL for v in verdicts.values()):
        verdict = FAIL
    elif len(checks) < min_datasets or any(v != PASS for v in verdicts.values()):
        verdict = NOT_ELIGIBLE
    else:
        verdict = PASS
    return {"min_datasets": min_datasets, "datasets": verdicts, "verdict": verdict}


def format_report(
    checks: dict[str, dict[str, Any]], overall: dict[str, Any], margins: dict[str, float]
) -> str:
    """Aggregates only: verdicts, gaps and the score-error margin. No vectors or scores."""
    lines = []
    for name, c in checks.items():
        lines.append(
            f"{c['verdict']:>12}  {name}: {c['settings_checked']} settings, "
            f"{c['n_queries']} queries, worst gap {c['worst_gap']:+.4f} at {c['worst_setting']} "
            f"(tolerance {c['tolerance']}), max score error {margins.get(name, float('nan')):.3e}"
        )
    why = ""
    if overall["verdict"] == NOT_ELIGIBLE:
        why = (
            f" (settings within tolerance, but the rule needs at least {overall['min_datasets']} "
            "datasets that each reach the query minimum)"
        )
    lines.append(f"{overall['verdict']:>12}  OVERALL{why}")
    return "\n".join(lines)
