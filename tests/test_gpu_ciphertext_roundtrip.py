"""Ciphertext round trip through the CUDA kernels.

Python generates keys and synthetic seeded data, encrypts, and writes only public material (the
CryptoContext and ciphertexts, OpenFHE BINARY serialization) to disk. The C++ driver
`ombra_roundtrip` computes ct + ct and ct * pt on the GPU and with OpenFHE, compares the raw RNS
polynomials exactly (non-zero exit on mismatch), and writes both results back. This test decrypts
them in Python; the secret key never leaves this process. Assertions use max error only; nothing
prints plaintext values.
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import pytest
from ombra import params as ombra_params

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.gpu

PRESET = "m0_depth2"
SEED = 20260102
N_VALUES = 256

# Maximum error between a decrypted result and the same computation in numpy (and between the GPU
# and OpenFHE CPU results). Values are normalized, scale is 2^50, depth is 1 here: the real error
# is orders of magnitude lower. Same agreed bound as the CPU baseline; do not loosen.
DECRYPT_TOLERANCE = 1e-6

REPO_ROOT = Path(__file__).resolve().parents[1]


def _driver() -> Path:
    candidates = [
        os.environ.get("OMBRA_ROUNDTRIP_BIN", ""),
        "/root/ombra-build/core/ombra_roundtrip",
        str(REPO_ROOT / "build" / "core" / "ombra_roundtrip"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail("ombra_roundtrip not found; build with `make build` or run `make test-gpu`")


def _normalized(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


def _write_values(path: Path, values: np.ndarray) -> None:
    """Test-only plaintext input: uint32 version=1, uint32 count, float64 values."""
    path.write_bytes(struct.pack("<II", 1, len(values)) + values.astype("<f8").tobytes())


def test_gpu_add_and_mult_match_openfhe(tmp_path: Path) -> None:
    import openfhe as ofhe

    _, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()

    rng = np.random.default_rng(SEED)
    a, b, p = (_normalized(rng, N_VALUES) for _ in range(3))
    ct_a = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(a.tolist()))
    ct_b = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(b.tolist()))

    # Public material only: no key is serialized.
    assert ofhe.SerializeToFile(str(tmp_path / "context.bin"), cc, ofhe.BINARY)
    assert ofhe.SerializeToFile(str(tmp_path / "ct_a.bin"), ct_a, ofhe.BINARY)
    assert ofhe.SerializeToFile(str(tmp_path / "ct_b.bin"), ct_b, ofhe.BINARY)
    _write_values(tmp_path / "pt.bin", p)

    proc = subprocess.run(  # noqa: S603
        [
            str(_driver()),
            str(tmp_path / "context.bin"),
            str(tmp_path / "ct_a.bin"),
            str(tmp_path / "ct_b.bin"),
            str(tmp_path / "pt.bin"),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    # Exit code 0 means every RNS word and metadata field matched OpenFHE exactly.
    assert proc.returncode == 0, f"driver failed (exit {proc.returncode}): {proc.stderr}"
    assert "add exact RNS match: PASS" in proc.stdout
    assert "mult-by-plaintext exact RNS match: PASS" in proc.stdout

    def decrypt(name: str) -> np.ndarray:
        ct, ok = ofhe.DeserializeCiphertext(str(tmp_path / name), ofhe.BINARY)
        assert ok
        dec = cc.Decrypt(ct, keys.secretKey)
        dec.SetLength(N_VALUES)
        return np.array(dec.GetRealPackedValue())

    expected = {"add": a + b, "mul": a * p}
    for op, want in expected.items():
        gpu = decrypt(f"{op}_gpu.bin")
        cpu = decrypt(f"{op}_cpu.bin")
        err_gpu = float(np.max(np.abs(gpu - want)))
        err_cpu = float(np.max(np.abs(cpu - want)))
        err_between = float(np.max(np.abs(gpu - cpu)))
        print(
            f"{op}: max_err_gpu={err_gpu:.3e} max_err_cpu={err_cpu:.3e} "
            f"gpu_vs_cpu={err_between:.3e}"
        )
        assert err_gpu < DECRYPT_TOLERANCE
        assert err_cpu < DECRYPT_TOLERANCE
        assert err_between < DECRYPT_TOLERANCE
