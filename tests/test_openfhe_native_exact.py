"""PERF-B: the fast CPU operations of local mode are bit-identical to the reference code.

Runs the C++ test `ombra_native_exact` (client/native/exact_test.cpp) once per preset of
core/params/ckks_presets.json, with the parameters read through ombra.params (never hard-coded).
It compares every RNS word and the metadata (level, noise scale degree, scaling factor, key tag)
of (a) the leaves of the evaluation-form expansion split against cpu_ref::OpenFheOps for d = 13, 16
(and 384 with --expansion-large) and (b) the fused 3-component score sum against the
EvalMultNoRelin / EvalAdd loop for 1, 5, 17 and 400 columns on 1 and 4 OpenMP threads, and checks
that the final score decrypts to the expected dot products. Synthetic data only.

The binary is found through OMBRA_NATIVE_EXACT_BIN, /root/ombra-build/client/ombra_native_exact or
build-native/ombra_native_exact; the test fails (not skips) when it is missing.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from ombra import params as ombra_params

pytestmark = pytest.mark.openfhe

REPO_ROOT = Path(__file__).resolve().parents[1]


def _binary() -> Path:
    candidates = [
        os.environ.get("OMBRA_NATIVE_EXACT_BIN", ""),
        "/root/ombra-build/client/ombra_native_exact",
        str(REPO_ROOT / "build-native" / "ombra_native_exact"),
        str(REPO_ROOT / "build-openfhe" / "client" / "ombra_native_exact"),
    ]
    for c in candidates:
        if c and os.access(c, os.X_OK) and Path(c).is_file():
            return Path(c)
    pytest.fail("ombra_native_exact not found; build client/ (see client/CMakeLists.txt)")


@pytest.mark.parametrize("name", sorted(ombra_params.load_presets()))
def test_fast_ops_are_bit_identical(name: str, request: pytest.FixtureRequest) -> None:
    p = ombra_params.get_preset(name)
    ombra_params.validate_preset(p)
    assert p["scaling_technique"] == "FIXEDMANUAL" and p["key_switch_technique"] == "HYBRID"
    args = [
        str(_binary()),
        str(p["ring_dim"]),
        str(p["multiplicative_depth"]),
        str(p["scaling_mod_size"]),
        str(p["first_mod_size"]),
        str(p["batch_size"]),
        str(p.get("num_large_digits", 0)),
        "1" if request.config.getoption("--expansion-large") else "0",
    ]
    done = subprocess.run(  # noqa: S603
        args, capture_output=True, text=True, timeout=3000, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "FAIL" not in done.stdout
    assert done.stdout.count("ok  ") >= 16
