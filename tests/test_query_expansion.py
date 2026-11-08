"""T0.5: query expansion by automorphisms, OpenFHE CPU reference (decision 0010).

Python makes keys (preset m0_depth2), encodes a normalized query as the symmetric polynomial
through OpenFHE, generates automorphism keys for exactly N/2^k + 1, k < ceil(log2(2d)), and writes
only public material (context, query ciphertext, automorphism keys) to disk. The C++ driver
`ombra_expand` runs the tree with OpenFHE EvalAutomorphism and writes d ciphertexts. The secret
key stays in this process and is used only to decrypt the outputs. Prints one summary line per d,
never values or key material.

Default d = 13, 16; `--expansion-large` adds d = 384, 768, 1536 (CPU minutes).
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

pytestmark = pytest.mark.openfhe

PRESET = "m0_depth2"
SEED = 20260108

# Max |decrypted slot - q_j| after expansion. Agreed bound for T0.5; do not loosen.
EXPANSION_TOLERANCE = 1e-6

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIMS = [13, 16]
LARGE_DIMS = [384, 768, 1536]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "d" in metafunc.fixturenames:
        dims = DEFAULT_DIMS + (LARGE_DIMS if metafunc.config.getoption("--expansion-large") else [])
        metafunc.parametrize("d", dims)


def _binary(name: str) -> Path:
    candidates = [
        os.environ.get(f"OMBRA_{name.upper()}_BIN", ""),
        f"/root/ombra-build/core/{name}",
        str(REPO_ROOT / "build" / "core" / name),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail(f"{name} not found; build it (`make test-openfhe` builds the CPU drivers)")


@pytest.fixture(scope="module")
def context():
    _, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()
    return cc, keys


def test_expansion_matches_query(context, tmp_path: Path, d: int) -> None:
    import openfhe as ofhe

    cc, keys = context
    ring = cc.GetRingDimension()
    slots = ring // 2
    depth = oq.expansion_depth(d)
    indices = oq.automorphism_indices(ring, d)
    assert len(indices) == depth

    # The key store is static: clear it so the key file holds exactly this d's indices.
    cc.ClearEvalAutomorphismKeys()
    cc.EvalAutomorphismKeyGen(keys.secretKey, indices)
    assert ofhe.SerializeToFile(str(tmp_path / "context.bin"), cc, ofhe.BINARY)
    assert cc.SerializeEvalAutomorphismKey(str(tmp_path / "autokeys.bin"), ofhe.BINARY)
    key_bytes = (tmp_path / "autokeys.bin").stat().st_size

    rng = np.random.default_rng(SEED + d)
    q = rng.standard_normal(d)
    q /= np.linalg.norm(q)
    ct = oq.encode_query(cc, keys.publicKey, q)
    assert ofhe.SerializeToFile(str(tmp_path / "query.bin"), ct, ofhe.BINARY)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    proc = subprocess.run(  # noqa: S603
        [
            str(_binary("ombra_expand")),
            str(tmp_path / "context.bin"),
            str(tmp_path / "query.bin"),
            str(tmp_path / "autokeys.bin"),
            str(out_dir),
            str(d),
        ],
        capture_output=True,
        text=True,
        timeout=3000,
    )
    assert proc.returncode == 0, f"driver failed (exit {proc.returncode}): {proc.stderr}"
    match = re.search(
        r"d=(\d+) depth=(\d+) outputs=(\d+) nodes=(\d+) automorphisms=(\d+) seconds=([0-9.]+)",
        proc.stdout,
    )
    assert match is not None, "driver summary line missing"
    drv_d, drv_depth, outputs, nodes, autos, seconds = match.groups()
    assert int(drv_d) == d
    assert int(drv_depth) == depth
    assert int(outputs) == d
    assert len(sorted(out_dir.glob("expanded_*.bin"))) == d

    worst = 0.0
    for j in range(d):
        out, ok = ofhe.DeserializeCiphertext(str(out_dir / f"expanded_{j}.bin"), ofhe.BINARY)
        assert ok
        dec = cc.Decrypt(out, keys.secretKey)
        dec.SetLength(slots)
        got = np.array(dec.GetRealPackedValue())
        assert got.size == slots
        worst = max(worst, float(np.max(np.abs(got - q[j]))))
    print(
        f"d={d} L={depth} nodes={nodes} automorphisms={autos} keys={len(indices)} "
        f"key_bytes={key_bytes} cpu_seconds={seconds} max_err={worst:.3e}"
    )
    assert worst < EXPANSION_TOLERANCE
