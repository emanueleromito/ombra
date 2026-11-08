"""T0.4: column-packed search kernel against OpenFHE.

Python generates keys and seeded synthetic data (preset m0_depth2), packs the stored vectors
column-wise (one ciphertext per dimension, one vector per slot) and the query pre-expanded (the
j-th ciphertext holds q_j in every slot), encrypts both, and writes only public material (context,
ciphertexts) to disk. The C++ driver `ombra_column_search` computes sum_j stored_j * query_j on the
GPU and with OpenFHE (EvalMultNoRelin + EvalAdd), compares all RNS words, moduli and metadata
exactly (non-zero exit on mismatch) and writes both results. This test decrypts them (3-component
ciphertexts decrypt directly: rns-pke.cpp:208 multiplies by powers of s) and asserts max errors
only; nothing prints values. The secret key never leaves this process.
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
SEED = 20260104

# Max error of the decrypted scores against numpy dot products, and between GPU and OpenFHE CPU.
# Normalized vectors, scale 2^50 each (2^100 product, no rescale): the real error is far below.
# Same agreed bound as T0.2/T0.3; do not loosen.
DECRYPT_TOLERANCE = 1e-6

REPO_ROOT = Path(__file__).resolve().parents[1]


def _driver() -> Path:
    candidates = [
        os.environ.get("OMBRA_COLUMN_SEARCH_BIN", ""),
        "/root/ombra-build/core/ombra_column_search",
        str(REPO_ROOT / "build" / "core" / "ombra_column_search"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail("ombra_column_search not found; build with `make build` or run `make test-gpu`")


def _normalized_rows(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    m = rng.standard_normal((rows, cols))
    return m / np.linalg.norm(m, axis=1, keepdims=True)


# (dims, blocks): the required d = 16 single fully packed block, plus 5 blocks of d = 4 so the
# kernel's partially filled block tile (blocks not a multiple of 4) is exercised too.
@pytest.mark.parametrize(("dims", "blocks"), [(16, 1), (4, 5)])
def test_gpu_column_search_matches_openfhe(tmp_path: Path, dims: int, blocks: int) -> None:
    import openfhe as ofhe

    _, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()
    slots = ombra_params.get_preset(PRESET)["batch_size"]  # fully packed block

    rng = np.random.default_rng(SEED)
    stored = [_normalized_rows(rng, slots, dims) for _ in range(blocks)]  # [block][vector, dim]
    query = _normalized_rows(rng, 1, dims)[0]

    assert ofhe.SerializeToFile(str(tmp_path / "context.bin"), cc, ofhe.BINARY)
    for j in range(dims):
        ct = cc.Encrypt(
            keys.publicKey, cc.MakeCKKSPackedPlaintext(np.full(slots, query[j]).tolist())
        )
        assert ofhe.SerializeToFile(str(tmp_path / f"query_{j}.bin"), ct, ofhe.BINARY)
        for b in range(blocks):
            ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(stored[b][:, j].tolist()))
            assert ofhe.SerializeToFile(str(tmp_path / f"stored_{b}_{j}.bin"), ct, ofhe.BINARY)

    proc = subprocess.run(  # noqa: S603
        [str(_driver()), str(tmp_path / "context.bin"), str(tmp_path), str(tmp_path), str(dims),
         str(blocks)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    # Exit code 0 means every RNS word, modulus and metadata field matched OpenFHE exactly.
    assert proc.returncode == 0, f"driver failed (exit {proc.returncode}): {proc.stderr}"
    assert "column search exact RNS match: PASS" in proc.stdout
    print(f"d={dims} blocks={blocks}: exact RNS match with OpenFHE: PASS")

    def decrypt(name: str) -> np.ndarray:
        ct, ok = ofhe.DeserializeCiphertext(str(tmp_path / name), ofhe.BINARY)
        assert ok
        dec = cc.Decrypt(ct, keys.secretKey)
        dec.SetLength(slots)
        return np.array(dec.GetRealPackedValue())

    worst_gpu = worst_cpu = worst_between = 0.0
    for b in range(blocks):
        want = stored[b] @ query
        gpu = decrypt(f"gpu_{b}.bin")
        cpu = decrypt(f"cpu_{b}.bin")
        worst_gpu = max(worst_gpu, float(np.max(np.abs(gpu - want))))
        worst_cpu = max(worst_cpu, float(np.max(np.abs(cpu - want))))
        worst_between = max(worst_between, float(np.max(np.abs(gpu - cpu))))
    print(
        f"d={dims} blocks={blocks}: max_err_gpu={worst_gpu:.3e} max_err_cpu={worst_cpu:.3e} "
        f"gpu_vs_cpu={worst_between:.3e}"
    )
    assert worst_gpu < DECRYPT_TOLERANCE
    assert worst_cpu < DECRYPT_TOLERANCE
    assert worst_between < DECRYPT_TOLERANCE
