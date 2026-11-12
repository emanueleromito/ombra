# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""GPU CKKS encoding of compact plaintext columns and ciphertext x plaintext search.

Python generates keys and seeded synthetic normalized vectors, stores them as fp16 or int8 (or as
exact doubles), encrypts the pre-expanded query (query value j in every slot) and writes only
public material and synthetic data to disk. The C++ driver `ombra_plain_search` (see its header)
checks, against OpenFHE on the same doubles:
  * the GPU encoder against MakeCKKSPackedPlaintext: FFT and rounding within
    COEF_TOLERANCE units (1 unit = 1/scale in slot space); tower reduction and NTT of OpenFHE's own
    integer coefficients exactly; the GPU evaluation form exactly against OpenFHE's NTT of the GPU's
    own coefficients (never skipped); the GPU's coefficient words are one integer in every tower.
    test_gpu_encoder_oracle_detects_faults injects one wrong
    reduction or NTT word beside a one-unit difference and needs the driver to fail;
  * the ct x pt search with OpenFHE-encoded plaintexts against EvalMult + EvalAdd, EXACTLY;
  * the fused encode-and-search path against encode-all then search.
This test then (a) decodes the GPU polynomial with numpy's canonical embedding and compares the
slots with the input doubles (DECODE_TOLERANCE), and (b) decrypts the search results and compares
with numpy (DECRYPT_TOLERANCE for the exact decoded values; the error from quantization against
the unquantized vectors is printed, with no bound). The secret key stays in this process. Nothing
prints data values.

int8 scaling: codes c = clip(rint(x * s), -127, 127), collection-wide scale s = 127 / max|x|,
decoded as c / s. fp16: numpy float16 codes, decoded exactly.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from ombra import params as ombra_params
from ombra import query as oq

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.gpu

PRESETS = ["m0_depth2", "m0_compact_n8192"]
SEED = 20260111
D = 16

# GPU vs OpenFHE coefficient of the encoded polynomial, in integer units. The GPU repeats
# OpenFHE's floating-point steps in the same order without fused multiply-adds, so identical
# results are expected; 2 units leaves room for a different rounding of one operation.
COEF_TOLERANCE = 2
# Max |decoded slot - input double| of the GPU polynomial. The rounding error of a coefficient is at
# most 0.5 unit; a slot sums N of them, about 0.3 * sqrt(N) / scale (2e-11 at scale 2^40).
DECODE_TOLERANCE = 1e-9
# Max |decrypted score - numpy score on the decoded values|: the agreed bound, do not loosen.
DECRYPT_TOLERANCE = 1e-6

REPO_ROOT = Path(__file__).resolve().parents[1]


def _driver() -> Path:
    candidates = [
        os.environ.get("OMBRA_PLAIN_SEARCH_BIN", ""),
        "/root/ombra-build/core/ombra_plain_search",
        str(REPO_ROOT / "build" / "core" / "ombra_plain_search"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail("ombra_plain_search not found; build with `make build` or run `make test-gpu`")


def _normalized_rows(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    m = rng.standard_normal((rows, cols))
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def quantize(stored: np.ndarray, fmt: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (codes, decoded float64 values, code_scale) for `stored` (any shape)."""
    if fmt == "f64":
        return stored.astype(np.float64), stored.astype(np.float64), 1.0
    if fmt == "fp16":
        codes = stored.astype(np.float16)
        return codes, codes.astype(np.float64), 1.0
    scale = 127.0 / float(np.max(np.abs(stored)))
    codes = np.clip(np.rint(stored * scale), -127, 127).astype(np.int8)
    return codes, codes.astype(np.float64) / scale, scale


def _run_driver(tmp_path: Path, preset: str, fmt: str, inject: str | None = None):
    """Write the inputs and run the driver; return (proc, context, keys, state for the checks)."""
    import openfhe as ofhe

    params, cc = ombra_params.make_context(preset)
    keys = cc.KeyGen()
    ring = cc.GetRingDimension()
    slots = ombra_params.get_preset(preset)["batch_size"]
    scale = float(2 ** ombra_params.get_preset(preset)["scaling_mod_size"])

    rng = np.random.default_rng(SEED)
    stored = _normalized_rows(rng, slots, D)  # one vector per row
    q = _normalized_rows(rng, 1, D)[0]
    codes, decoded, code_scale = quantize(stored, fmt)

    assert ofhe.SerializeToFile(str(tmp_path / "context.bin"), cc, ofhe.BINARY)
    for j in range(D):
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(np.full(slots, q[j]).tolist()))
        assert ofhe.SerializeToFile(str(tmp_path / f"query_{j}.bin"), ct, ofhe.BINARY)
    columns = np.ascontiguousarray(decoded.T)  # [dim][slot]
    (tmp_path / "values.bin").write_bytes(columns.tobytes())
    (tmp_path / "codes.bin").write_bytes(np.ascontiguousarray(codes.T).tobytes())

    cmd = [
        str(_driver()),
        str(tmp_path / "context.bin"),
        str(tmp_path),
        str(D),
        fmt,
        repr(code_scale),
        str(COEF_TOLERANCE),
    ]
    if inject:
        cmd.append(inject)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)  # noqa: S603
    return proc, cc, keys, (ring, slots, scale, stored, q, decoded, columns)


@pytest.mark.parametrize("inject", ["reduction", "ntt", "tower"])
@pytest.mark.parametrize("fmt", ["f64", "fp16", "int8"])
@pytest.mark.parametrize("preset", PRESETS)
def test_gpu_encoder_oracle_detects_faults(
    tmp_path: Path, preset: str, fmt: str, inject: str
) -> None:
    """One wrong reduction or NTT word is caught although every coefficient is within one unit."""
    proc, *_ = _run_driver(tmp_path, preset, fmt, inject)
    print(proc.stdout)
    assert proc.returncode == 1, f"driver did not fail (exit {proc.returncode}): {proc.stdout}"
    assert "encoding FFT and rounding" in proc.stdout
    fft_line = next(x for x in proc.stdout.splitlines() if x.startswith("encoding FFT"))
    assert fft_line.endswith("PASS"), fft_line  # the one-unit difference alone is tolerated
    wanted = {
        "reduction": "tower reduction of OpenFHE's integer coefficients, exact words: FAIL",
        "ntt": "NTT of OpenFHE's integer coefficients, exact words: FAIL",
        "tower": "GPU coefficient words are one integer in every tower, exact: FAIL",
    }[inject]
    assert wanted in proc.stdout
    if inject == "ntt":
        assert "OpenFHE NTT of the GPU coefficients, exact: FAIL" in proc.stdout
    assert "plain search oracle: FAIL" in proc.stdout


@pytest.mark.parametrize("fmt", ["f64", "fp16", "int8"])
@pytest.mark.parametrize("preset", PRESETS)
def test_gpu_plain_search_matches_openfhe(tmp_path: Path, preset: str, fmt: str) -> None:
    import openfhe as ofhe

    proc, cc, keys, state = _run_driver(tmp_path, preset, fmt)
    ring, slots, scale, stored, q, decoded, columns = state
    print(proc.stdout)
    msg = f"driver failed (exit {proc.returncode}): {proc.stdout}{proc.stderr}"
    assert proc.returncode == 0, msg
    assert "plain search oracle: PASS" in proc.stdout
    assert "FAIL" not in proc.stdout

    # (a) Decode the GPU polynomial (centered integer coefficients) back to slots.
    gpu_coef = np.frombuffer((tmp_path / "gpu_coef.bin").read_bytes(), dtype=np.int64)
    gpu_coef = gpu_coef.reshape(D, ring)
    worst_decode = 0.0
    for c in range(D):
        z = oq.canonical_embedding(gpu_coef[c].astype(np.float64), ring)[:slots] / scale
        worst_decode = max(worst_decode, float(np.max(np.abs(z - columns[c]))))

    # (b) Decrypt the search results.
    def decrypt(name: str) -> np.ndarray:
        ct, ok = ofhe.DeserializeCiphertext(str(tmp_path / name), ofhe.BINARY)
        assert ok
        dec = cc.Decrypt(ct, keys.secretKey)
        dec.SetLength(slots)
        return np.array(dec.GetRealPackedValue())

    want = decoded @ q
    err_openfhe_pt = float(np.max(np.abs(decrypt("result_openfhe_pt.bin") - want)))
    err_gpu_pt = float(np.max(np.abs(decrypt("result_gpu_pt.bin") - want)))
    err_quant = float(np.max(np.abs(decrypt("result_gpu_pt.bin") - stored @ q)))
    print(
        f"preset={preset} fmt={fmt} d={D}: decode_err={worst_decode:.3e} "
        f"decrypt_err_openfhe_pt={err_openfhe_pt:.3e} decrypt_err_gpu_pt={err_gpu_pt:.3e} "
        f"err_vs_unquantized={err_quant:.3e}"
    )
    assert worst_decode < DECODE_TOLERANCE
    assert err_openfhe_pt < DECRYPT_TOLERANCE
    assert err_gpu_pt < DECRYPT_TOLERANCE
    if fmt == "f64":
        assert err_quant < DECRYPT_TOLERANCE


@pytest.mark.parametrize("inject", ["meta_level", "meta_degree", "meta_scale"])
@pytest.mark.parametrize("preset", PRESETS)
def test_gpu_plain_search_oracle_detects_wrong_metadata(
    tmp_path: Path, preset: str, inject: str
) -> None:
    """A wrong level, degree or scale in gpu_eval_plain_search is caught by the exact check."""
    proc, *_ = _run_driver(tmp_path, preset, "f64", inject)
    print(proc.stdout)
    assert proc.returncode == 1, f"driver did not fail (exit {proc.returncode}): {proc.stdout}"
    assert "exact RNS match with EvalMult: FAIL" in proc.stdout
    assert "plain search oracle: FAIL" in proc.stdout
