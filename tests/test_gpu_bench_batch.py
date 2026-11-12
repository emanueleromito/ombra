# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""The batching benchmark runs end to end at a tiny size (d = 4) on the test GPU.

Runs core/bench/run.py, which builds the context from the preset, writes only public material and
runs the C++ benchmark binary. Checks the record, the memory plan and the multi-pass path (a cap of
2 queries per pass), not the speed.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bench_batch_tiny(tmp_path: Path) -> None:
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(REPO_ROOT / "core" / "bench" / "run.py"),
            "--task",
            "batching",
            "--out",
            str(tmp_path),
            "--dims",
            "4",
            "--stored-cts",
            "20",
            "--queries",
            "1,3,5",
            "--max-per-pass",
            "2",
            "--mem-budget-gib",
            "6",
            "--warmup",
            "1",
            "--iters",
            "3",
            "--name-batching",
            "batching-test.json",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"bench failed (exit {proc.returncode}): {proc.stderr}"
    record = json.loads((tmp_path / "batching-test.json").read_text())
    assert record["preset"] == "m0_depth2"
    assert record["published_peak_memory_bandwidth_gb_per_s"] == 1555.0
    assert "command_output" in record["gpu_memory_query"]
    (entry,) = record["results"]
    assert entry["stored_blocks"] == 5
    for kind in ("search", "expansion"):
        rows = entry[kind]
        assert [r["queries"] for r in rows] == [1, 3, 5]
        for r in rows:
            assert r["queries_per_pass"] <= 2
            assert r["passes"] == math.ceil(r["queries"] / r["queries_per_pass"])
            assert r["stored_blocks"] == 5
            assert r["stored_vectors"] == 5 * 8192
            assert r["within_budget"] is True
            assert r["peak_device_bytes_used"] > 0
            assert r["total_seconds_median"] > 0
            assert r["queries_per_second"] > 0
            assert r["speedup_vs_q1"] > 0
    for r in entry["expansion"]:
        # Slab pool: a ciphertext costs its nominal size plus rounding well under 1 percent, the
        # pool holds at least one slab at the peak, and release_pool() frees all of it.
        assert r["ciphertext_footprint_bytes"] >= r["ciphertext_nominal_bytes"]
        assert r["ciphertext_footprint_bytes"] < 1.01 * r["ciphertext_nominal_bytes"]
        assert r["pool_bytes_at_peak"] >= r["slab_bytes"] > 0
        assert r["pool_bytes_after_release"] == 0
    assert "phase search d=4 Q=1" in proc.stdout
    assert "phase expansion d=4 Q=5" in proc.stdout
    for r in entry["search"]:
        assert r["pass_bytes_loaded"] > 0
        assert r["pass_bytes_written"] > 0
        assert r["achieved_gb_per_s_access_pattern"] > 0
        assert r["achieved_gb_per_s_compulsory"] > 0
