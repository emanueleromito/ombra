# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Repo-wide pytest options."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--expansion-large",
        action="store_true",
        default=False,
        help="also run the query expansion tests at d = 384, 768, 1536 (slow, CPU minutes)",
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """With OMBRA_FAIL_ON_SKIP=1 (set by modal_app.py for the gpu and openfhe runs), any skipped
    test fails the run: a broken image would otherwise turn the oracle tests into skips with
    exit code 0. Collection-time skips (importorskip at module level) count too."""
    import os

    if os.environ.get("OMBRA_FAIL_ON_SKIP") != "1" or exitstatus != 0:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter is not None else []
    # Count only what the run is about: tests marked gpu or openfhe, and collection skips of the
    # gpu and openfhe test modules. A docs or licence module skipped at collection is not a run
    # of the oracle tests (it is deselected by -m anyway).
    skipped = [r for r in skipped if _counts_as_oracle(r)]
    if skipped:
        names = ", ".join(sorted({getattr(r, "nodeid", "?") for r in skipped}))[:2000]
        print(f"\nOMBRA_FAIL_ON_SKIP: {len(skipped)} skipped test(s): {names}")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def _counts_as_oracle(report: object) -> bool:
    keywords = getattr(report, "keywords", None) or {}
    if "gpu" in keywords or "openfhe" in keywords:
        return True
    name = str(getattr(report, "nodeid", "") or getattr(report, "fspath", "")).rsplit("/", 1)[-1]
    return name.startswith(("test_gpu_", "test_openfhe_"))
