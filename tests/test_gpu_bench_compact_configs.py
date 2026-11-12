"""The compact-configurations benchmark runs end to end at a tiny size (d = 16).

Runs core/bench/run.py --task compact-configs, which creates the context, writes only public
material and
runs the C++ benchmark binaries and the GPU pipeline. Checks the output record, not the speed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bench_compact_configs_tiny(tmp_path: Path) -> None:
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(REPO_ROOT / "core" / "bench" / "run.py"),
            "--task",
            "compact-configs",
            "--out",
            str(tmp_path),
            "--dims",
            "16",
            "--presets-compact-configs",
            "m0_compact_n8192",
            "--warmup",
            "1",
            "--iters",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert proc.returncode == 0, f"bench failed (exit {proc.returncode}): {proc.stderr}"
    record = json.loads((tmp_path / "compact-configs-a100.json").read_text())
    assert record["gpu_name"] != "unknown"
    by_kind: dict[str, list[dict]] = {}
    for r in record["results"]:
        by_kind.setdefault(r["kind"], []).append(r)
    assert len(by_kind["expansion"]) == 1 and by_kind["expansion"][0]["key_bytes"] > 0
    modes = {r["mode"]: r for r in by_kind["search"]}
    assert set(modes) == {"ct_x_ct", "pt_preencoded", "fp16", "int8"}
    assert modes["fp16"]["storage_bytes_per_vector"] == 2 * 16
    assert modes["int8"]["storage_bytes_per_vector"] == 16
    assert all(r["vectors_per_second"] > 0 for r in modes.values())
    assert {r["mode"] for r in by_kind["quantization"]} == {
        "fp16",
        "int8_global_scale",
        "int8_fixed_127",
    }
    (err,) = by_kind["end_to_end_error"]
    assert err["within_bound"] is True
