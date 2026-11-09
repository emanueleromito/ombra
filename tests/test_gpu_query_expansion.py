"""T0.5: GPU query expansion against OpenFHE (decision 0010).

Python makes keys (preset m0_depth2), encodes a normalized query as the symmetric polynomial,
generates automorphism keys for exactly N/2^k + 1, k < ceil(log2(2d)), and writes only public
material (context, query ciphertext, automorphism keys) to disk. The C++ driver `ombra_expand_gpu`
runs the staged oracle checks (NTT, key-switch digits, inner product, ModDown, one automorphism)
and then the whole expansion tree on the CPU (OpenFHE) and on the GPU, comparing every RNS word,
modulus and metadata field of every leaf EXACTLY (non-zero exit on any mismatch). This test then
decrypts the GPU and CPU leaves and asserts the decrypted errors. The secret key stays in this
process and is used only to decrypt. Nothing prints values.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
from ombra import params as ombra_params
from ombra import query as oq

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.gpu

PRESETS = ["m0_depth2", "m0_compact_n8192", "m1_composite_n16384"]  # T0.11, T1.5
SEED = 20260109

# Max |decrypted slot - q_j| for GPU leaves, and between GPU and CPU leaves. Same agreed bound as
# the CPU expansion test (tests/test_query_expansion.py); do not loosen.
TOLERANCE = 1e-6

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIMS = [13, 16]
LARGE_DIMS = [384, 768, 1536]  # `--expansion-large`: every tree depth the benchmark uses


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "preset" in metafunc.fixturenames:
        metafunc.parametrize("preset", PRESETS)
    if "d" in metafunc.fixturenames:
        large = metafunc.config.getoption("--expansion-large")
        metafunc.parametrize("d", DEFAULT_DIMS + (LARGE_DIMS if large else []))


def _driver() -> Path:
    candidates = [
        os.environ.get("OMBRA_EXPAND_GPU_BIN", ""),
        "/root/ombra-build/core/ombra_expand_gpu",
        str(REPO_ROOT / "build" / "core" / "ombra_expand_gpu"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail("ombra_expand_gpu not found; build with `make build` or run `make test-gpu`")


def test_gpu_expansion_matches_openfhe(tmp_path: Path, d: int, preset: str) -> None:
    import openfhe as ofhe

    _, cc = ombra_params.make_context(preset)
    keys = cc.KeyGen()
    ring = cc.GetRingDimension()
    depth = oq.expansion_depth(d)
    indices = oq.automorphism_indices(ring, d)
    assert len(indices) == depth

    cc.ClearEvalAutomorphismKeys()  # the key store is static: hold exactly this d's indices
    cc.EvalAutomorphismKeyGen(keys.secretKey, indices)
    assert ofhe.SerializeToFile(str(tmp_path / "context.bin"), cc, ofhe.BINARY)
    assert cc.SerializeEvalAutomorphismKey(str(tmp_path / "autokeys.bin"), ofhe.BINARY)

    rng = np.random.default_rng(SEED + d)
    q = rng.standard_normal(d)
    q /= np.linalg.norm(q)
    ct = oq.encode_query(cc, keys.publicKey, q)
    assert ofhe.SerializeToFile(str(tmp_path / "query.bin"), ct, ofhe.BINARY)

    proc = subprocess.run(  # noqa: S603
        [
            str(_driver()),
            str(tmp_path / "context.bin"),
            str(tmp_path / "query.bin"),
            str(tmp_path / "autokeys.bin"),
            str(d),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    print(proc.stdout)
    # Exit code 0 means every stage and every leaf matched OpenFHE exactly.
    msg = f"driver failed (exit {proc.returncode}): {proc.stdout}{proc.stderr}"
    assert proc.returncode == 0, msg
    assert "GPU expansion exact RNS match with OpenFHE: PASS" in proc.stdout
    assert "FAIL" not in proc.stdout
    match = re.search(r"cpu_seconds=([0-9.]+) gpu_seconds=([0-9.]+)", proc.stdout)
    assert match is not None, "driver summary line missing"

    slots = cc.GetRingDimension() // 2

    def decrypt(name: str) -> np.ndarray:
        leaf, ok = ofhe.DeserializeCiphertext(str(tmp_path / name), ofhe.BINARY)
        assert ok
        dec = cc.Decrypt(leaf, keys.secretKey)
        dec.SetLength(slots)
        return np.array(dec.GetRealPackedValue())

    worst_gpu = worst_between = 0.0
    for j in range(d):
        gpu = decrypt(f"gpu_{j}.bin")
        cpu = decrypt(f"cpu_{j}.bin")
        worst_gpu = max(worst_gpu, float(np.max(np.abs(gpu - q[j]))))
        worst_between = max(worst_between, float(np.max(np.abs(gpu - cpu))))
    print(
        f"d={d} L={depth} keys={len(indices)} cpu_seconds={match.group(1)} "
        f"gpu_seconds={match.group(2)} max_err_gpu={worst_gpu:.3e} gpu_vs_cpu={worst_between:.3e}"
    )
    assert worst_gpu < TOLERANCE
    assert worst_between < TOLERANCE
