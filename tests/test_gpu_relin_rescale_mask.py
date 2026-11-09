"""T1.2: GPU relinearize, rescale and ciphertext mask multiply, and the GPU-only search.

Oracle (hard rule 3): the driver `ombra_search_gpu` feeds the same OpenFHE ciphertexts to each GPU
kernel (relinearize at both levels, rescale at both levels, ciphertext x ciphertext multiply at the
mask level) and to OpenFHE on the CPU, and compares every RNS word, the moduli, level, noise-scale
degree and scaling factor exactly. It then runs the GPU-only pipeline (multiply-and-add kernel,
relinearize, rescale, mask, rescale; no OpenFHE call in between) and compares every stage with the
OpenFHE chain. This test then decrypts the GPU-only output for three masks and reports the max
error against numpy, with the same 1e-6 bound as T0.6 (test_gpu_filter_mask.py), same inputs.

Query leaves come from the GPU expansion, as in T0.6. Only public material goes to disk; the secret
key stays in this process. Data is synthetic; only aggregates are printed.
Default d = 16; `--expansion-large` adds d = 384, 768, 1536.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from ombra import params as ombra_params
from ombra import query as oq

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.gpu

PRESETS = ["m0_depth2", "m0_compact_n8192", "m1_composite_n16384"]
SEED = 20260110  # same seed and data as test_gpu_filter_mask.py

# Max |decrypted slot - (stored @ q) * mask|. Same agreed bound as T0.2 to T0.6; do not loosen.
TOLERANCE = 1e-6
# m0_compact_n8192 is known to miss 1e-6 at d >= 384 (T0.11: 3.5e-6 to 6.3e-6; here up to 7.6e-6 on
# the CPU result too). The GPU output is bit-identical to OpenFHE's, so this is the preset's own
# noise, not the kernels'. Its acceptance rule is T1.6/T1.7; here it only gets a sanity bound.
COMPACT_SANITY_BOUND = 1e-5

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIMS = [16]
LARGE_DIMS = [384, 768, 1536]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "preset" in metafunc.fixturenames:
        metafunc.parametrize("preset", PRESETS)
    if "d" in metafunc.fixturenames:
        large = metafunc.config.getoption("--expansion-large")
        metafunc.parametrize("d", DEFAULT_DIMS + (LARGE_DIMS if large else []))


def _driver(name: str) -> Path:
    candidates = [
        os.environ.get(f"OMBRA_{name.removeprefix('ombra_').upper()}_BIN", ""),
        f"/root/ombra-build/core/{name}",
        str(REPO_ROOT / "build" / "core" / name),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail(f"{name} not found; build with `make build` or run `make test-gpu`")


def _run(args: list[str], timeout: int) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)  # noqa: S603
    msg = f"{Path(args[0]).name} failed (exit {proc.returncode}): {proc.stdout}{proc.stderr}"
    assert proc.returncode == 0, msg
    assert "FAIL" not in proc.stdout
    return proc.stdout


def test_gpu_relin_rescale_mask_and_search(tmp_path: Path, d: int, preset: str) -> None:
    import openfhe as ofhe

    _, cc = ombra_params.make_context(preset)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    slots = ombra_params.get_preset(preset)["batch_size"]
    cd = cc.GetCompositeDegree()  # towers per rescale: 1 under FIXEDMANUAL, 2 for the T1.5 preset

    indices = oq.automorphism_indices(cc.GetRingDimension(), d)
    cc.ClearEvalAutomorphismKeys()  # the key store is static: hold exactly this d's indices
    cc.EvalAutomorphismKeyGen(keys.secretKey, indices)
    assert ofhe.SerializeToFile(str(tmp_path / "context.bin"), cc, ofhe.BINARY)
    assert cc.SerializeEvalAutomorphismKey(str(tmp_path / "autokeys.bin"), ofhe.BINARY)
    assert ofhe.CryptoContext.SerializeEvalMultKey(
        str(tmp_path / "evalmult.bin"), ofhe.BINARY, keys.publicKey.GetKeyTag()
    )

    rng = np.random.default_rng(SEED + d)
    stored = rng.standard_normal((slots, d))  # one vector per slot
    stored /= np.linalg.norm(stored, axis=1, keepdims=True)
    q = rng.standard_normal(d)
    q /= np.linalg.norm(q)

    in_dir = tmp_path / "search_in"
    in_dir.mkdir()
    for j in range(d):
        col = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(stored[:, j].tolist()))
        assert ofhe.SerializeToFile(str(in_dir / f"stored_0_{j}.bin"), col, ofhe.BINARY)
    qct = oq.encode_query(cc, keys.publicKey, q)
    assert ofhe.SerializeToFile(str(tmp_path / "query.bin"), qct, ofhe.BINARY)

    exp_dir = tmp_path / "expand_out"
    exp_dir.mkdir()
    out = _run(
        [
            str(_driver("ombra_expand_gpu")),
            str(tmp_path / "context.bin"),
            str(tmp_path / "query.bin"),
            str(tmp_path / "autokeys.bin"),
            str(d),
            str(exp_dir),
        ],
        timeout=1800,
    )
    assert "GPU expansion exact RNS match with OpenFHE: PASS" in out
    for j in range(d):
        shutil.copy(exp_dir / f"gpu_{j}.bin", in_dir / f"query_{j}.bin")

    expected = stored @ q
    random_mask = rng.integers(0, 2, size=slots).astype(np.float64)
    random_mask[0], random_mask[1] = 0.0, 1.0  # both values present whatever the draw
    masks = {"random": random_mask, "ones": np.ones(slots), "zeros": np.zeros(slots)}
    for kind, mask in masks.items():
        mask_ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(mask.tolist(), 1, cd))
        assert ofhe.SerializeToFile(str(in_dir / f"mask_{kind}.bin"), mask_ct, ofhe.BINARY)

    out_dir = tmp_path / "search_out"
    out_dir.mkdir()
    out = _run(
        [
            str(_driver("ombra_search_gpu")),
            str(tmp_path / "context.bin"),
            str(tmp_path / "evalmult.bin"),
            str(in_dir),
            str(out_dir),
            str(d),
            *masks,
        ],
        timeout=3600,
    )
    print(f"preset={preset} d={d}")
    for line in out.splitlines():
        print(line)  # PASS verdicts, shapes and timings only
    for stage in (
        "relinearize, level 0: PASS",
        "rescale, level 0 -> 1: PASS",
        "relinearize, level 1: PASS",
        "ciphertext multiply with relinearization, level 1: PASS",
        "rescale, level 1 -> 2: PASS",
        "pipeline final: rescale",
    ):
        assert stage in out, stage
    assert "GPU relinearize/rescale/mask exact match with OpenFHE: PASS" in out

    def decrypt(name: str) -> np.ndarray:
        ct, ok = ofhe.DeserializeCiphertext(str(out_dir / name), ofhe.BINARY)
        assert ok
        assert ct.GetLevel() == 2 * cd
        dec = cc.Decrypt(ct, keys.secretKey)
        dec.SetLength(slots)
        got = np.array(dec.GetRealPackedValue())
        assert got.size == slots
        return got

    errs: dict[str, tuple[float, float]] = {}
    for kind, mask in masks.items():
        want = expected * mask
        err_gpu = float(np.max(np.abs(decrypt(f"gpu_final_{kind}.bin") - want)))
        err_cpu = float(np.max(np.abs(decrypt(f"cpu_final_{kind}.bin") - want)))
        print(
            f"preset={preset} d={d} mask={kind} levels_used=2 gpu_only_pipeline "
            f"max_err_gpu={err_gpu:.3e} max_err_cpu={err_cpu:.3e}"
        )
        errs[kind] = (err_gpu, err_cpu)

    bound = COMPACT_SANITY_BOUND if preset == "m0_compact_n8192" else TOLERANCE
    for kind, (err_gpu, err_cpu) in errs.items():
        assert err_gpu < bound, kind
        assert err_cpu < bound, kind
