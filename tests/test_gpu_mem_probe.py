"""Nominal versus actual device memory of ciphertext-sized allocations.

Runs `ombra_mem_probe`, which allocates many buffers of one ciphertext's size with separate
cudaMalloc calls and again as one slab, and reads cudaMemGetInfo around each. The numbers are
printed (run with -s) so the footprint of the pool can be documented. The assertion is the claim
the slab allocator relies on: a slab costs about its nominal size.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from ombra import params as ombra_params

pytestmark = pytest.mark.gpu

PRESET = "m0_depth2"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _probe() -> Path:
    candidates = [
        os.environ.get("OMBRA_MEM_PROBE_BIN", ""),
        "/root/ombra-build/core/ombra_mem_probe",
        str(REPO_ROOT / "build" / "core" / "ombra_mem_probe"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail("ombra_mem_probe not found; build with `make build` or run `make test-gpu`")


def test_slab_footprint_close_to_nominal() -> None:
    preset = ombra_params.get_preset(PRESET)
    towers = preset["multiplicative_depth"] + 1  # one Q tower per level, plus the first
    nominal = 2 * towers * preset["ring_dim"] * 8  # two components of 64-bit words
    proc = subprocess.run(  # noqa: S603
        [str(_probe()), str(nominal), "1000"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    print(proc.stdout)
    values = dict(line.split("=") for line in proc.stdout.split())
    assert int(values["nominal_bytes_per_buffer"]) == nominal
    assert float(values["slab_over_nominal"]) < 1.01
    assert float(values["separate_over_nominal"]) >= 1.0
