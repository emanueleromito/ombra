"""The end-to-end latency measurement path on tiny synthetic data.

Runs core/bench/e2e.py (client in this process, `ombra_bench_e2e` as the server) at d = 16 with
2 chosen blocks, 3 queries: client encode and encrypt, GPU expansion, the device-to-device leaf
handoff into the stage 1 and stage 2 searches, relinearize, rescale and mask on the GPU, client
decrypt, cluster selection and top-k. Checks that the decrypted scores are within 1e-6 of numpy and
that the record has every step, the byte counts and the memory plan. Prints aggregates only; the
data is synthetic.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_e2e():
    path = REPO_ROOT / "core" / "bench" / "e2e.py"
    spec = importlib.util.spec_from_file_location("ombra_e2e", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_binary(name: str) -> Path:
    for c in (
        os.environ.get("OMBRA_BENCH_E2E_BIN", ""),
        f"/root/ombra-build/core/{name}",
        str(REPO_ROOT / "build" / "core" / name),
    ):
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail(f"{name} not found; build with `make build` or run `make test-gpu`")


def test_e2e_tiny(tmp_path: Path) -> None:
    e2e = _load_e2e()
    # N = 20,000 vectors, 60% searched = 12,000 vectors = 2 blocks of 8192 (the last one masked).
    out = e2e.run_e2e_latency(
        tmp_path,
        (16,),
        queries=3,
        warmup=1,
        find_binary=_find_binary,
        openfhe_version="test",
        cases_spec=[(20_000, 0.6, 8)],
        name="e2e-latency-tiny.json",
    )
    rec = json.loads(out.read_text())
    assert len(rec["cases"]) == 1
    c = rec["cases"][0]
    assert c["chosen_blocks"] == 2 and c["memory"]["resident_blocks"] == 2
    assert set(c["steps_seconds"]) == set(e2e.STEPS)
    assert abs(sum(c["step_share"].values()) - 1.0) < 1e-9
    err = c["max_abs_score_error"]
    assert err["stage1_center_scores"] < e2e.ERROR_BOUND
    assert err["stage2_block0"] < e2e.ERROR_BOUND
    assert err["stage2_last_masked_block"] < e2e.ERROR_BOUND
    assert c["bytes"]["upload_query"] > 0 and c["bytes"]["download_total"] > 0
    print(
        f"tiny e2e: total={c['total_seconds_sum_of_median_steps'] * 1000:.1f} ms "
        f"errs={err['stage1_center_scores']:.2e}/{err['stage2_block0']:.2e}/"
        f"{err['stage2_last_masked_block']:.2e}"
    )
