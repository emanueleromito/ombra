"""T0.4: the column-search benchmark runs end to end at a tiny size (d = 4) on the test GPU.

Runs core/bench/run.py, which builds the context from the preset, writes only the context and the
public key, and runs the C++ benchmark binary. Checks the output record, not the speed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bench_column_search_tiny(tmp_path: Path) -> None:
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(REPO_ROOT / "core" / "bench" / "run.py"),
            "--task",
            "t0.4",
            "--out",
            str(tmp_path),
            "--dims",
            "4",
            "--blocks",
            "5",
            "--warmup",
            "1",
            "--iters",
            "3",
            "--name",
            "t0.4-test.json",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"bench failed (exit {proc.returncode}): {proc.stderr}"
    record = json.loads((tmp_path / "t0.4-test.json").read_text())
    assert record["preset"] == "m0_depth2"
    assert record["gpu_name"] and record["gpu_name"] != "unknown"
    assert "excluded" in record["note"]
    (entry,) = record["results"]
    assert entry["dims"] == 4
    assert entry["blocks"] == 5
    assert entry["total_vectors_scored"] == 5 * entry["vectors_per_block"]
    assert entry["kernel_seconds_median"] > 0
    assert entry["vectors_per_second"] > 0
    assert entry["copy_h2d_seconds_per_block"] > 0
    assert "copy_d2h_seconds_result" in entry
