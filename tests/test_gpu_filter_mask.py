"""T0.6: encrypted 0/1 filter mask on GPU-produced scores (no new kernel, no parameter change).

Real pipeline with preset m0_depth2: GPU query expansion (`ombra_expand_gpu`), GPU column-packed
search (`ombra_column_search`, 3-component degree-2 result at level 0), then in OpenFHE:
Relinearize, Rescale (query x stored, level 1), EvalMult with an encrypted 0/1 mask at level 1,
Rescale (mask, level 2). The same steps run on the CPU search result. Both are decrypted and
compared with (stored @ q) * mask. Only public material goes to disk; the secret key stays in this
process. Masks and data are synthetic. Prints aggregates only, never vectors or scores.

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

PRESETS = ["m0_depth2", "m0_compact_n8192"]  # T0.11: every test runs on both
SEED = 20260110

# Max |decrypted slot - (stored @ q) * mask|. Same agreed bound as T0.2 to T0.5; do not loosen.
TOLERANCE = 1e-6

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


def test_gpu_scores_with_encrypted_filter_mask(tmp_path: Path, d: int, preset: str) -> None:
    import openfhe as ofhe

    _, cc = ombra_params.make_context(preset)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    slots = ombra_params.get_preset(preset)["batch_size"]

    indices = oq.automorphism_indices(cc.GetRingDimension(), d)
    cc.ClearEvalAutomorphismKeys()  # the key store is static: hold exactly this d's indices
    cc.EvalAutomorphismKeyGen(keys.secretKey, indices)
    assert ofhe.SerializeToFile(str(tmp_path / "context.bin"), cc, ofhe.BINARY)
    assert cc.SerializeEvalAutomorphismKey(str(tmp_path / "autokeys.bin"), ofhe.BINARY)

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

    # 1. GPU expansion: the d leaves are the query ciphertexts of the search.
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

    # 2. GPU column-packed search over the GPU leaves.
    search_dir = tmp_path / "search_out"
    search_dir.mkdir()
    out = _run(
        [
            str(_driver("ombra_column_search")),
            str(tmp_path / "context.bin"),
            str(in_dir),
            str(search_dir),
            str(d),
            "1",
        ],
        timeout=1800,
    )
    assert "column search exact RNS match: PASS" in out

    expected = stored @ q
    random_mask = rng.integers(0, 2, size=slots).astype(np.float64)
    random_mask[0], random_mask[1] = 0.0, 1.0  # both values present whatever the draw
    masks = {"random": random_mask, "ones": np.ones(slots), "zeros": np.zeros(slots)}

    def mask_and_decrypt(name: str, mask_ct) -> np.ndarray:
        ct, ok = ofhe.DeserializeCiphertext(str(search_dir / name), ofhe.BINARY)
        assert ok
        score = cc.Rescale(cc.Relinearize(ct))  # query x stored: 1 level
        masked = cc.Rescale(cc.EvalMult(score, mask_ct))  # mask: 1 level
        assert masked.GetLevel() == 2
        dec = cc.Decrypt(masked, keys.secretKey)
        dec.SetLength(slots)
        got = np.array(dec.GetRealPackedValue())
        assert got.size == slots
        return got

    errs: dict[str, tuple[float, float]] = {}
    for kind, mask in masks.items():
        mask_ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(mask.tolist(), 1, 1))
        want = expected * mask
        err_gpu = float(np.max(np.abs(mask_and_decrypt("gpu_0.bin", mask_ct) - want)))
        err_cpu = float(np.max(np.abs(mask_and_decrypt("cpu_0.bin", mask_ct) - want)))
        print(
            f"preset={preset} d={d} mask={kind} levels_used=2 "
            f"max_err_gpu={err_gpu:.3e} max_err_cpu={err_cpu:.3e}"
        )
        errs[kind] = (err_gpu, err_cpu)

    for kind, (err_gpu, err_cpu) in errs.items():
        assert err_gpu < TOLERANCE, kind
        assert err_cpu < TOLERANCE, kind
