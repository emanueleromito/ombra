# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Batched column-packed search (Q queries in one pass) against OpenFHE.

Same setup as tests/test_gpu_column_search.py, with Q distinct queries. The C++ driver
`ombra_column_search_batch` scores all Q queries against the stored blocks in one kernel launch and,
as the oracle, computes each (query, block) with OpenFHE EvalMultNoRelin + EvalAdd; it compares all
RNS words, moduli and metadata exactly (non-zero exit on mismatch). This test then decrypts both
and asserts the errors against numpy. Nothing prints values. The secret key stays in this process.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from ombra import params as ombra_params

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.gpu

PRESET = "m0_depth2"
SEED = 20260110

# Same agreed bound as tests/test_gpu_column_search.py; do not loosen.
DECRYPT_TOLERANCE = 1e-6

REPO_ROOT = Path(__file__).resolve().parents[1]


def _driver() -> Path:
    candidates = [
        os.environ.get("OMBRA_COLUMN_SEARCH_BATCH_BIN", ""),
        "/root/ombra-build/core/ombra_column_search_batch",
        str(REPO_ROOT / "build" / "core" / "ombra_column_search_batch"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail("ombra_column_search_batch not found; build with `make build` or `make test-gpu`")


def _normalized_rows(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    m = rng.standard_normal((rows, cols))
    return m / np.linalg.norm(m, axis=1, keepdims=True)


# (dims, blocks): d = 16 with one fully packed block, and d = 4 with 5 blocks so the partially
# filled block tile (blocks not a multiple of 4) is exercised. Q = 1, 3, 8: Q = 3 is a ragged query
# tile (tile is 4), Q = 8 is two full tiles.
@pytest.mark.parametrize("queries", [1, 3, 8])
@pytest.mark.parametrize(("dims", "blocks"), [(16, 1), (4, 5)])
def test_gpu_batched_search_matches_openfhe(
    tmp_path: Path, dims: int, blocks: int, queries: int
) -> None:
    import openfhe as ofhe

    _, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()
    slots = ombra_params.get_preset(PRESET)["batch_size"]

    rng = np.random.default_rng(SEED + dims)
    stored = [_normalized_rows(rng, slots, dims) for _ in range(blocks)]
    qs = _normalized_rows(rng, queries, dims)

    assert ofhe.SerializeToFile(str(tmp_path / "context.bin"), cc, ofhe.BINARY)
    for q in range(queries):
        for j in range(dims):
            ct = cc.Encrypt(
                keys.publicKey, cc.MakeCKKSPackedPlaintext(np.full(slots, qs[q, j]).tolist())
            )
            assert ofhe.SerializeToFile(str(tmp_path / f"query_{q}_{j}.bin"), ct, ofhe.BINARY)
    for b in range(blocks):
        for j in range(dims):
            ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(stored[b][:, j].tolist()))
            assert ofhe.SerializeToFile(str(tmp_path / f"stored_{b}_{j}.bin"), ct, ofhe.BINARY)

    proc = subprocess.run(  # noqa: S603
        [
            str(_driver()),
            str(tmp_path / "context.bin"),
            str(tmp_path),
            str(tmp_path),
            str(dims),
            str(blocks),
            str(queries),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"driver failed (exit {proc.returncode}): {proc.stderr}"
    assert "batched column search exact RNS match: PASS" in proc.stdout
    assert "FAIL" not in proc.stdout
    print(f"d={dims} blocks={blocks} Q={queries}: exact RNS match with OpenFHE: PASS")

    def decrypt(name: str) -> np.ndarray:
        ct, ok = ofhe.DeserializeCiphertext(str(tmp_path / name), ofhe.BINARY)
        assert ok
        dec = cc.Decrypt(ct, keys.secretKey)
        dec.SetLength(slots)
        return np.array(dec.GetRealPackedValue())

    worst_gpu = worst_between = 0.0
    for q in range(queries):
        for b in range(blocks):
            want = stored[b] @ qs[q]
            gpu = decrypt(f"gpu_{q}_{b}.bin")
            cpu = decrypt(f"cpu_{q}_{b}.bin")
            worst_gpu = max(worst_gpu, float(np.max(np.abs(gpu - want))))
            worst_between = max(worst_between, float(np.max(np.abs(gpu - cpu))))
    print(
        f"d={dims} blocks={blocks} Q={queries}: max_err_gpu={worst_gpu:.3e} "
        f"gpu_vs_cpu={worst_between:.3e}"
    )
    assert worst_gpu < DECRYPT_TOLERANCE
    assert worst_between < DECRYPT_TOLERANCE
