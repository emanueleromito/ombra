# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Query expansion by automorphisms, OpenFHE CPU reference.

Python makes keys (preset m0_depth2), encodes a normalized query as the symmetric polynomial
through OpenFHE, generates automorphism keys for exactly N/2^k + 1, k < ceil(log2(2d)), and writes
only public material (context, query ciphertext, automorphism keys) to disk. The C++ driver
`ombra_expand` runs the tree with OpenFHE EvalAutomorphism and writes d ciphertexts. The secret
key stays in this process and is used only to decrypt the outputs. Prints one summary line per d,
never values or key material.

The query is pre-scaled by 2^-L on the client, so the server does no multiplication on expansion.
Two measurements per d, both against 1e-6: (a) max error of every expanded leaf against the
unscaled q_j, and (b) max error of a full pipeline (column-packed score, then a ciphertext 0/1
mask) against numpy, which must use exactly 2 levels.

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

# Max |decrypted slot - q_j| after expansion. Agreed bound for expansion; do not loosen.
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
    cc.EvalMultKeyGen(keys.secretKey)
    return cc, keys


@pytest.fixture(scope="module")
def expansion_cache(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """d -> result of the single expansion run for that d, shared by both tests."""
    return {}


def _expanded(context, cache: dict, factory: pytest.TempPathFactory, d: int) -> dict:
    """Run the C++ expansion once per d (public material on disk) and return its description."""
    if d in cache:
        return cache[d]
    import openfhe as ofhe

    cc, keys = context
    tmp_path = factory.mktemp(f"expand_d{d}")
    ring = cc.GetRingDimension()
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
    cache[d] = {
        "out_dir": out_dir,
        "q": q,
        "depth": depth,
        "nodes": nodes,
        "autos": autos,
        "seconds": seconds,
        "key_bytes": key_bytes,
        "n_keys": len(indices),
    }
    return cache[d]


def test_expansion_matches_query(context, expansion_cache, tmp_path_factory, d: int) -> None:
    import openfhe as ofhe

    cc, keys = context
    r = _expanded(context, expansion_cache, tmp_path_factory, d)
    out_dir, q = r["out_dir"], r["q"]
    slots = cc.GetRingDimension() // 2

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
        f"d={d} L={r['depth']} nodes={r['nodes']} automorphisms={r['autos']} keys={r['n_keys']} "
        f"key_bytes={r['key_bytes']} cpu_seconds={r['seconds']} max_err={worst:.3e}"
    )
    assert worst < EXPANSION_TOLERANCE


def test_expansion_end_to_end_score_with_mask(
    context, expansion_cache, tmp_path_factory, d: int
) -> None:
    """Expanded query -> column-packed score (1 level) -> ciphertext 0/1 mask (1 level)."""
    import openfhe as ofhe

    cc, keys = context
    r = _expanded(context, expansion_cache, tmp_path_factory, d)
    out_dir, q = r["out_dir"], r["q"]
    slots = ombra_params.get_preset(PRESET)["batch_size"]

    rng = np.random.default_rng(SEED + 7 * d + 1)
    stored = rng.standard_normal((slots, d))  # one stored vector per row, one slot per vector
    stored /= np.linalg.norm(stored, axis=1, keepdims=True)
    mask = rng.integers(0, 2, size=slots).astype(np.float64)
    mask[0], mask[1] = 0.0, 1.0  # both values present whatever the draw
    mask_ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(mask.tolist(), 1, 1))

    # Column j of the stored vectors is encrypted at level 0, multiplied with leaf j and summed.
    # Encrypting column by column (instead of holding all d ciphertexts) keeps memory bounded;
    # the ciphertexts are the same as the stored_j of the column search test.
    score = None
    for j in range(d):
        col = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(stored[:, j].tolist()))
        leaf, ok = ofhe.DeserializeCiphertext(str(out_dir / f"expanded_{j}.bin"), ofhe.BINARY)
        assert ok
        term = cc.EvalMultNoRelin(col, leaf)
        score = term if score is None else cc.EvalAdd(score, term)
    score = cc.Rescale(cc.Relinearize(score))  # query x stored: 1 level
    masked = cc.Rescale(cc.EvalMult(score, mask_ct))  # mask: 1 level

    assert masked.GetLevel() == 2
    if hasattr(masked, "GetNoiseScaleDeg"):
        assert masked.GetNoiseScaleDeg() == 1

    dec = cc.Decrypt(masked, keys.secretKey)
    dec.SetLength(slots)
    got = np.array(dec.GetRealPackedValue())
    assert got.size == slots
    worst = float(np.max(np.abs(got - (stored @ q) * mask)))
    print(f"d={d} L={r['depth']} levels_used=2 max_err_final={worst:.3e}")
    assert worst < EXPANSION_TOLERANCE
