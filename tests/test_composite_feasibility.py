"""T1.5 stage 1: preset m1_composite_n16384 (OpenFHE composite scaling, 32-bit limbs) on the CPU.

Checks, with OpenFHE only (no GPU):
  * the baseline preset m0_depth2 is unchanged;
  * the real tower layout and log2(QP) against the HomomorphicEncryption.org 128-bit classic limit
    for N = 16384 (438 bits), and that every prime (Q and P) fits a 32-bit limb;
  * level, noise-scale degree and scaling-factor metadata under COMPOSITESCALINGMANUAL: a rescale
    drops compositeDegree towers, the scaling factor is not exactly 2^s;
  * the level budget (expansion 0, query x stored 1, mask 1) and the client pre-scaling by 2^-L;
  * CPU error of query expansion and of the final masked scores (same measures as T0.5, T0.6).

Finding (OpenFHE v1.6.1, rns-leveledshe.cpp:352): CryptoContext::Rescale / ModReduce act only under
FIXEDMANUAL, so under COMPOSITESCALINGMANUAL the public Rescale is a no-op. The rescale of the
score happens inside EvalMult(score, mask) (AdjustLevelsAndDepthInPlace calls ModReduceInternal with
compositeDegree levels), so this test never calls Rescale for the pipeline. Synthetic data only;
prints scalars only.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest
from ombra import params as ombra_params
from ombra import query as oq

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.openfhe

PRESET = "m1_composite_n16384"
SEED = 20260110
LIMIT_N16384 = 438  # HE standard, 128-bit classic, ternary secret
TOLERANCE = 1e-6  # same bound as T0.5/T0.6, reported as a safety margin (decision 0012)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIMS = [13, 16]
LARGE_DIMS = [384, 768, 1536]

BASELINE = {
    "security_level": "HEStd_128_classic",
    "ring_dim": 16384,
    "multiplicative_depth": 2,
    "scaling_mod_size": 50,
    "first_mod_size": 60,
    "batch_size": 8192,
    "scaling_technique": "FIXEDMANUAL",
    "key_switch_technique": "HYBRID",
}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "d" in metafunc.fixturenames:
        dims = DEFAULT_DIMS + (LARGE_DIMS if metafunc.config.getoption("--expansion-large") else [])
        metafunc.parametrize("d", dims)


def _binary(name: str) -> Path:
    for c in (
        os.environ.get(f"OMBRA_{name.upper()}_BIN", ""),
        f"/root/ombra-build/core/{name}",
        str(REPO_ROOT / "build" / "core" / name),
    ):
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail(f"{name} not found; build it (`make test-openfhe` builds the CPU drivers)")


@pytest.fixture(scope="module")
def context():
    _, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    return cc, keys


def test_baseline_preset_unchanged() -> None:
    assert ombra_params.get_preset("m0_depth2") == BASELINE


def test_candidate_fields() -> None:
    p = ombra_params.get_preset(PRESET)
    assert p["security_level"] == "HEStd_128_classic"
    assert p["ring_dim"] == 16384
    assert p["multiplicative_depth"] == 2
    assert p["scaling_technique"] == "COMPOSITESCALINGMANUAL"
    assert p["composite_degree"] == 2
    assert p["register_word_size"] == 32


def test_tower_layout_and_log_qp(context, tmp_path) -> None:
    import openfhe as ofhe

    cc, keys = context
    assert cc.GetRingDimension() == 16384
    assert cc.GetCompositeDegree() == 2
    key_file = tmp_path / "evalmult.json"
    assert ofhe.CryptoContext.SerializeEvalMultKey(
        str(key_file), ofhe.JSON, keys.publicKey.GetKeyTag()
    )
    entry = json.loads(key_file.read_text())["value0"][0]["value"][0]["ptr_wrapper"]["data"]
    qp = [int(t["v"]["ptr_wrapper"]["data"]["m"]["v"]) for t in entry["ak"][0]["v"]]
    digits = len(entry["ak"])
    preset = ombra_params.get_preset(PRESET)
    num_q = (preset["multiplicative_depth"] + 1) * preset["composite_degree"]
    q_bits = math.log2(cc.GetElementParams().GetModulus())
    log_qp = sum(math.log2(m) for m in qp)
    print(
        f"{PRESET}: towers={len(qp)} (Q={num_q}, P={len(qp) - num_q}) "
        f"bits={[round(math.log2(m), 3) for m in qp]} digits={digits} "
        f"log2(Q)={q_bits:.2f} log2(QP)={log_qp:.2f} limit={LIMIT_N16384}"
    )
    assert abs(sum(math.log2(m) for m in qp[:num_q]) - q_bits) < 1e-6
    assert len(qp) > num_q, "evaluation key does not contain the special primes P"
    assert log_qp <= LIMIT_N16384
    # 32-bit limbs: every prime below 2^31, so that a sum of two residues fits 32 bits.
    assert all(m < 2**31 for m in qp), "a prime does not fit a 31-bit limb"


def test_metadata_and_level_budget(context) -> None:
    cc, keys = context
    cd = cc.GetCompositeDegree()
    slots = ombra_params.get_preset(PRESET)["batch_size"]
    rng = np.random.default_rng(SEED)
    a = rng.uniform(-1, 1, slots)
    b = rng.uniform(-1, 1, slots)
    m = rng.integers(0, 2, slots).astype(np.float64)
    ca = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(a.tolist()))
    cb = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(b.tolist()))
    cm = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(m.tolist(), 1, cd))

    sf0 = cc.GetScalingFactorReal(0)
    sf_mask = cc.GetScalingFactorReal(cd)
    print(f"scaling factors: level0={sf0!r} level{cd}={sf_mask!r} 2^58={2.0**58!r}")
    assert sf0 != 2.0**58, "the composite scaling factor is the product of two primes, not 2^58"
    assert abs(sf0 / 2.0**58 - 1) < 1e-3
    assert (ca.GetLevel(), ca.GetNoiseScaleDeg()) == (0, 1)
    assert (cm.GetLevel(), cm.GetNoiseScaleDeg()) == (cd, 1)

    prod = cc.EvalMultNoRelin(ca, cb)  # query x stored
    assert (prod.GetLevel(), prod.GetNoiseScaleDeg()) == (0, 2)
    rel = cc.Relinearize(prod)
    assert (rel.GetLevel(), rel.GetNoiseScaleDeg()) == (0, 2)
    # Public Rescale is a no-op under COMPOSITESCALINGMANUAL (OpenFHE acts only for FIXEDMANUAL).
    again = cc.Rescale(rel)
    assert (again.GetLevel(), again.GetNoiseScaleDeg()) == (0, 2)

    masked = cc.EvalMult(rel, cm)  # rescales the score by cd towers, then multiplies by the mask
    assert (masked.GetLevel(), masked.GetNoiseScaleDeg()) == (cd, 2)
    dec = cc.Decrypt(masked, keys.secretKey)
    dec.SetLength(slots)
    got = np.array(dec.GetRealPackedValue())
    err = float(np.max(np.abs(got - a * b * m)))
    print(
        f"levels used = {masked.GetLevel() // cd} (towers dropped {masked.GetLevel()}) "
        f"err={err:.3e}"
    )
    assert err < TOLERANCE


def _expanded(context, cache: dict, factory: pytest.TempPathFactory, d: int) -> dict:
    if d in cache:
        return cache[d]
    import openfhe as ofhe

    cc, keys = context
    tmp_path = factory.mktemp(f"expand_d{d}")
    indices = oq.automorphism_indices(cc.GetRingDimension(), d)
    cc.ClearEvalAutomorphismKeys()
    cc.EvalAutomorphismKeyGen(keys.secretKey, indices)
    assert ofhe.SerializeToFile(str(tmp_path / "context.bin"), cc, ofhe.BINARY)
    assert cc.SerializeEvalAutomorphismKey(str(tmp_path / "autokeys.bin"), ofhe.BINARY)
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
    match = re.search(r"seconds=([0-9.]+)", proc.stdout)
    assert match is not None
    cache[d] = {"out_dir": out_dir, "q": q, "seconds": match.group(1)}
    return cache[d]


@pytest.fixture(scope="module")
def expansion_cache() -> dict:
    return {}


def test_expansion_error(context, expansion_cache, tmp_path_factory, d: int) -> None:
    import openfhe as ofhe

    cc, keys = context
    r = _expanded(context, expansion_cache, tmp_path_factory, d)
    slots = cc.GetRingDimension() // 2
    worst = 0.0
    for j in range(d):
        out, ok = ofhe.DeserializeCiphertext(str(r["out_dir"] / f"expanded_{j}.bin"), ofhe.BINARY)
        assert ok
        dec = cc.Decrypt(out, keys.secretKey)
        dec.SetLength(slots)
        got = np.array(dec.GetRealPackedValue())
        worst = max(worst, float(np.max(np.abs(got - r["q"][j]))))
    print(f"d={d} cpu_seconds={r['seconds']} max_err_expansion={worst:.3e}")
    assert worst < TOLERANCE


def test_final_score_error(context, expansion_cache, tmp_path_factory, d: int) -> None:
    import openfhe as ofhe

    cc, keys = context
    cd = cc.GetCompositeDegree()
    r = _expanded(context, expansion_cache, tmp_path_factory, d)
    slots = ombra_params.get_preset(PRESET)["batch_size"]
    rng = np.random.default_rng(SEED + 7 * d + 1)
    stored = rng.standard_normal((slots, d))
    stored /= np.linalg.norm(stored, axis=1, keepdims=True)
    mask = rng.integers(0, 2, size=slots).astype(np.float64)
    mask[0], mask[1] = 0.0, 1.0
    mask_ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(mask.tolist(), 1, cd))
    score = None
    for j in range(d):
        col = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(stored[:, j].tolist()))
        leaf, ok = ofhe.DeserializeCiphertext(str(r["out_dir"] / f"expanded_{j}.bin"), ofhe.BINARY)
        assert ok
        term = cc.EvalMultNoRelin(col, leaf)
        score = term if score is None else cc.EvalAdd(score, term)
    score = cc.Relinearize(score)  # level 0, degree 2
    masked = cc.EvalMult(score, mask_ct)  # score rescaled by cd towers inside, mask: level cd
    assert masked.GetLevel() == cd
    dec = cc.Decrypt(masked, keys.secretKey)
    dec.SetLength(slots)
    got = np.array(dec.GetRealPackedValue())
    worst = float(np.max(np.abs(got - (stored @ r["q"]) * mask)))
    print(f"d={d} max_err_final={worst:.3e}")
    assert worst < TOLERANCE
