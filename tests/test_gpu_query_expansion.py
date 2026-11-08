"""T0.5: query expansion by automorphisms, OpenFHE CPU reference (decision 0010).

Python makes keys (preset m0_depth2), encodes a normalized query as polynomial coefficients through
OpenFHE, generates automorphism keys for exactly N/2^k + 1, k < ceil(log2 d), and writes only public
material (context, query ciphertext, automorphism keys) to disk. The C++ driver `ombra_expand`
runs the tree and writes d ciphertexts. The secret key stays in this process and is used only to
decrypt the outputs. Nothing prints values, only max errors and tree sizes.

The coefficient-level encoding check uses `ombra_encoding_check` because openfhe-python cannot
reach the encoded polynomial (hence this file is marked gpu, like the driver builds).
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

PRESET = "m0_depth2"
SEED = 20260108

# Max |decrypted slot - q_j| after expansion. Agreed bound for T0.5; do not loosen.
EXPANSION_TOLERANCE = 1e-6
# Max |coefficient / scale - q_j| of OpenFHE's encoded polynomial (double-precision FFT plus
# rounding to integers at scale 2^50, so about 2^-49 relative; this leaves a wide margin).
ENCODING_TOLERANCE = 1e-12

REPO_ROOT = Path(__file__).resolve().parents[1]


def _binary(name: str) -> Path:
    candidates = [
        os.environ.get(f"OMBRA_{name.upper()}_BIN", ""),
        f"/root/ombra-build/core/{name}",
        str(REPO_ROOT / "build" / "core" / name),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail(f"{name} not found; build with `make build` or run `make test-gpu`")


def _normalized(rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):
    import openfhe as ofhe

    work = tmp_path_factory.mktemp("expansion")
    _, cc = ombra_params.make_context(PRESET)
    cc.ClearEvalAutomorphismKeys()  # static store: keep only this test's keys in the key file
    keys = cc.KeyGen()
    ring = cc.GetRingDimension()
    # d = 16 and d = 13 both need L = 4, so one key set serves both.
    indices = oq.automorphism_indices(ring, 16)
    assert indices == oq.automorphism_indices(ring, 13)
    cc.EvalAutomorphismKeyGen(keys.secretKey, indices)
    assert ofhe.SerializeToFile(str(work / "context.bin"), cc, ofhe.BINARY)
    assert cc.SerializeEvalAutomorphismKey(str(work / "autokeys.bin"), ofhe.BINARY)
    key_bytes = (work / "autokeys.bin").stat().st_size
    print(f"automorphism keys: {len(indices)} keys, serialized size {key_bytes} bytes")
    return cc, keys, work


@pytest.mark.parametrize("d", [16, 13])
def test_expansion_matches_query(setup, d: int) -> None:
    import openfhe as ofhe

    cc, keys, work = setup
    slots = cc.GetRingDimension() // 2
    q = _normalized(np.random.default_rng(SEED + d), d)
    out_dir = work / f"out_{d}"
    out_dir.mkdir()

    ct = oq.encode_query(cc, keys.publicKey, q)
    assert ofhe.SerializeToFile(str(work / f"query_{d}.bin"), ct, ofhe.BINARY)

    proc = subprocess.run(  # noqa: S603
        [
            str(_binary("ombra_expand")),
            str(work / "context.bin"),
            str(work / f"query_{d}.bin"),
            str(work / "autokeys.bin"),
            str(out_dir),
            str(d),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"driver failed (exit {proc.returncode}): {proc.stderr}"
    print(proc.stdout.strip())
    match = re.search(r"depth=(\d+) outputs=(\d+)", proc.stdout)
    assert match is not None
    assert int(match.group(1)) == oq.expansion_depth(d)
    assert int(match.group(2)) == d

    files = sorted(out_dir.glob("expanded_*.bin"))
    assert len(files) == d

    worst = 0.0
    for j in range(d):
        out, ok = ofhe.DeserializeCiphertext(str(out_dir / f"expanded_{j}.bin"), ofhe.BINARY)
        assert ok
        dec = cc.Decrypt(out, keys.secretKey)
        dec.SetLength(slots)
        got = np.array(dec.GetRealPackedValue())
        worst = max(worst, float(np.max(np.abs(got - q[j]))))
    print(f"d={d}: tree_depth={oq.expansion_depth(d)} outputs={len(files)} max_err={worst:.3e}")
    assert worst < EXPANSION_TOLERANCE


@pytest.mark.parametrize("d", [16, 13])
def test_encoded_coefficients_equal_query(setup, d: int) -> None:
    cc, _, work = setup
    q = _normalized(np.random.default_rng(SEED + d), d)
    slot_values = oq.canonical_embedding(q, cc.GetRingDimension())
    inter = np.empty(2 * slot_values.size)
    inter[0::2] = slot_values.real
    inter[1::2] = slot_values.imag
    (work / f"enc_slots_{d}.bin").write_bytes(inter.astype("<f8").tobytes())
    (work / f"enc_q_{d}.bin").write_bytes(q.astype("<f8").tobytes())

    proc = subprocess.run(  # noqa: S603
        [
            str(_binary("ombra_encoding_check")),
            str(work / "context.bin"),
            str(work / f"enc_slots_{d}.bin"),
            str(work / f"enc_q_{d}.bin"),
            str(d),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"driver failed (exit {proc.returncode}): {proc.stderr}"
    print(proc.stdout.strip())
    match = re.search(r"max_coeff_error_in_scale_units=([0-9.eE+-]+)", proc.stdout)
    assert match is not None
    assert float(match.group(1)) < ENCODING_TOLERANCE

