"""T1.15: every core driver refuses a context that fails the security floor.

Hard rule 1 forbids building an insecure context, even in tests. Instead the drivers read
OMBRA_TEST_LOG2QP_LIMIT, which can only LOWER the log2(QP) limit. With it set to 1 a valid preset
context must be refused by each driver (non-zero exit, clear error, before any other work). A
control run without the variable shows the same context passes the check (the driver then fails
later, on the missing input files, with a different message). The pure check is covered on plain
numbers by the C++ ctest `security_floor`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from ombra import params as ombra_params

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS = ["m0_depth2", "m0_compact_n8192"]

# driver name -> arguments after the context file (all other inputs are dummies: the check runs
# right after the context is read, before anything else is opened).
DRIVERS: dict[str, list[str]] = {
    "ombra_roundtrip": ["a", "b", "p", "out"],
    "ombra_column_search": ["in", "out", "1", "1"],
    "ombra_column_search_batch": ["in", "out", "1", "1", "1"],
    "ombra_expand": ["q", "k", "out", "1"],
    "ombra_expand_gpu": ["q", "k", "1"],
    "ombra_expand_gpu_batch": ["k", "1", "1", "dir"],
    "ombra_encoding_check": ["s", "q", "1"],
    "ombra_plain_search": ["dir", "1", "f64", "1", "1"],
    "ombra_search_gpu": ["k", "in", "out", "1", "m"],
    "ombra_recall_search": ["k", "data", "out", "1", "1", "1", "1"],
    "ombra_bench_batch": ["pk", "k", "1", "1", "1", "1", "1", "1", "1"],
    "ombra_bench_column_search": ["pk", "1", "1", "1", "1", "1"],
    "ombra_bench_expand": ["pk", "k", "1", "1", "1", "1"],
    "ombra_bench_plain_search": ["pk", "1", "1", "1", "1", "1"],
    "ombra_bench_e2e": ["k", "m", "dir", "1"],
}


def _driver(name: str) -> Path:
    for c in ("/root/ombra-build/core", str(REPO_ROOT / "build" / "core")):
        p = Path(c) / name
        if p.is_file():
            return p
    pytest.fail(f"{name} not found; build with `make build` or run `make test-gpu`")


@pytest.fixture(scope="module", params=PRESETS)
def context_file(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    import openfhe as ofhe

    _, cc = ombra_params.make_context(request.param)
    path = tmp_path_factory.mktemp("floor") / "context.bin"
    assert ofhe.SerializeToFile(str(path), cc, ofhe.BINARY)
    return path


def _run(name: str, ctx: Path, tmp: Path, limit: str | None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "OMBRA_TEST_LOG2QP_LIMIT"}
    if limit is not None:
        env["OMBRA_TEST_LOG2QP_LIMIT"] = limit
    args = [str(ctx)] + [str(tmp / a) if not a.isdigit() and a != "f64" else a for a in DRIVERS[name]]
    return subprocess.run(
        [str(_driver(name)), *args], capture_output=True, text=True, env=env, timeout=120
    )


@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_driver_refuses_below_limit(name: str, context_file: Path, tmp_path: Path) -> None:
    refused = _run(name, context_file, tmp_path, "1")
    assert refused.returncode != 0
    assert "security floor" in refused.stdout + refused.stderr

    # Control: without the variable the check passes, so any failure is about the dummy inputs.
    control = _run(name, context_file, tmp_path, None)
    assert "security floor" not in control.stdout + control.stderr
