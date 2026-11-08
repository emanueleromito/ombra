"""T0.2: CKKS parameter security, CPU round trip and CPU timings.

All data is synthetic, seeded and normalized. Nothing prints plaintext values or keys; assertions
compare scalar errors only.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path

import pytest
from ombra import params as ombra_params

np = pytest.importorskip("numpy")  # not needed by `make test`, which deselects these tests

pytestmark = pytest.mark.openfhe

PRESET = "m0_depth2"
SEED = 20260101
N_VALUES = 256  # packed slots actually used; the context has batch_size slots

# Maximum error between the CPU FHE result and the same computation in numpy. The values are
# normalized (|x| <= 1), the scale is 2^50 and the circuit has depth 2, so the expected error is
# many orders of magnitude below this; 1e-6 is the agreed bound and must not be loosened.
ROUND_TRIP_TOLERANCE = 1e-6

# HomomorphicEncryption.org standard, 128-bit classic security, uniform ternary secret:
# ring dimension -> maximum log2 of the full modulus (QP, including key-switching primes).
HE_STD_128_CLASSIC_MAX_LOG_QP = {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881}

TIMING_ITERATIONS = 5
RESULTS_FILE = Path("/data/results/t0.2-cpu.json")


def _normalized(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


def _evalkey_moduli(tree) -> list[int]:
    """RNS moduli of the towers of the first polynomial of a serialized EvalMult key (JSON).

    Path (cereal layout of OpenFHE v1.6.1): key map entry -> polymorphic EvalKeyRelin -> 'ak'[0]
    (a DCRTPoly over QP) -> 'v' (towers) -> tower['v'] (unique_ptr to a NativeVector) -> 'm'.
    """
    key_entry = tree["value0"][0]["value"][0]
    dcrt = key_entry["ptr_wrapper"]["data"]["ak"][0]
    return [int(t["v"]["ptr_wrapper"]["data"]["m"]["v"]) for t in dcrt["v"]]


@pytest.fixture(scope="module")
def ofhe():
    import openfhe

    return openfhe


@pytest.fixture(scope="module")
def setup(ofhe):
    params, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    return params, cc, keys


def test_security_level_and_modulus_bound(ofhe, setup, tmp_path):
    params, cc, keys = setup
    # 1. The parameter object carries HEStd_128_classic. (CryptoContext has no
    #    GetSecurityLevel binding, so the CCParams object is what can be asserted.)
    assert params.GetSecurityLevel() == ofhe.SecurityLevel.HEStd_128_classic
    ring_dim = cc.GetRingDimension()
    assert ring_dim == params.GetRingDim()
    assert ring_dim in HE_STD_128_CLASSIC_MAX_LOG_QP

    # 2. Independent check of log2(QP). P is not exposed by the bindings, so read the moduli of a
    #    (public) evaluation key, which lives in QP, from its JSON serialization.
    q_bits = math.log2(cc.GetElementParams().GetModulus())
    key_file = tmp_path / "evalmult.json"
    ok = ofhe.CryptoContext.SerializeEvalMultKey(
        str(key_file), ofhe.JSON, keys.publicKey.GetKeyTag()
    )
    assert ok, "could not serialize the evaluation multiplication key"
    qp = _evalkey_moduli(json.loads(key_file.read_text()))
    num_q_towers = params.GetMultiplicativeDepth() + 1
    assert len(qp) > num_q_towers, "evaluation key does not contain the special primes P"
    log_qp = sum(math.log2(m) for m in qp)
    # Q (from the context) must be the first towers of QP and add up to the same number of bits.
    assert abs(sum(math.log2(m) for m in qp[:num_q_towers]) - q_bits) < 1e-6
    print(f"log2(Q)={q_bits:.2f} towers_QP={len(qp)} log2(QP)={log_qp:.2f}")
    assert log_qp <= HE_STD_128_CLASSIC_MAX_LOG_QP[ring_dim]


def test_loader_refuses_below_128_bits():
    preset = dict(ombra_params.get_preset(PRESET))
    preset["security_level"] = "HEStd_NotSet"
    with pytest.raises(ombra_params.ParamsError):
        ombra_params.validate_preset(preset)
    # make_ckks_params must refuse before touching OpenFHE, so no context can be created.
    with pytest.raises(ombra_params.ParamsError):
        ombra_params.make_ckks_params(preset)


def test_round_trip_matches_numpy(ofhe, setup):
    _, cc, keys = setup
    rng = np.random.default_rng(SEED)
    x, y, w, v, u = (_normalized(rng, N_VALUES) for _ in range(5))

    def enc(vec, level=0):
        return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec, 1, level))

    p1 = cc.Rescale(cc.EvalMult(enc(x), cc.MakeCKKSPackedPlaintext(y)))  # ct * pt
    p2 = cc.Rescale(cc.EvalMult(enc(w), cc.MakeCKKSPackedPlaintext(v)))
    s = cc.EvalAdd(p1, p2)  # ct + ct
    r = cc.Rescale(cc.EvalMult(s, enc(u, level=1)))  # ct * ct, second level

    dec = cc.Decrypt(r, keys.secretKey)
    dec.SetLength(N_VALUES)
    got = np.array(dec.GetRealPackedValue())
    expected = (x * y + w * v) * u
    max_err = float(np.max(np.abs(got - expected)))
    print(f"round trip max abs error: {max_err:.3e}")
    assert max_err <= ROUND_TRIP_TOLERANCE, f"max error {max_err:.3e} > {ROUND_TRIP_TOLERANCE}"


def _median_ms(fn, iterations: int = TIMING_ITERATIONS) -> float:
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def test_cpu_timings(ofhe):
    preset = ombra_params.get_preset(PRESET)
    _, cc = ombra_params.make_context(PRESET)
    rng = np.random.default_rng(SEED + 1)
    a, b = _normalized(rng, N_VALUES), _normalized(rng, N_VALUES)
    timings: dict[str, float] = {}

    holder = {}

    def keygen():
        holder["keys"] = cc.KeyGen()
        cc.EvalMultKeyGen(holder["keys"].secretKey)

    timings["keygen_with_evalmultkey"] = _median_ms(keygen, 3)
    keys = holder["keys"]
    pt_a = cc.MakeCKKSPackedPlaintext(a)
    pt_b = cc.MakeCKKSPackedPlaintext(b)
    ct_a = cc.Encrypt(keys.publicKey, pt_a)
    ct_b = cc.Encrypt(keys.publicKey, pt_b)
    prod = cc.EvalMult(ct_a, ct_b)

    timings["encrypt"] = _median_ms(lambda: cc.Encrypt(keys.publicKey, pt_a))
    timings["mult_ct_pt"] = _median_ms(lambda: cc.EvalMult(ct_a, pt_b))
    timings["add_ct_ct"] = _median_ms(lambda: cc.EvalAdd(ct_a, ct_b))
    timings["mult_ct_ct"] = _median_ms(lambda: cc.EvalMult(ct_a, ct_b))
    timings["rescale"] = _median_ms(lambda: cc.Rescale(prod))
    res = cc.Rescale(prod)
    timings["decrypt"] = _median_ms(lambda: cc.Decrypt(res, keys.secretKey))

    assert all(t > 0 for t in timings.values())
    report = {
        "task": "T0.2",
        "preset": PRESET,
        "ring_dim": preset["ring_dim"],
        "cpu_count": os.cpu_count(),
        "openfhe_python_version": ofhe.__version__,
        "iterations": TIMING_ITERATIONS,
        "median_ms": timings,
    }
    print(json.dumps(report))
    if RESULTS_FILE.parent.parent.exists():
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_FILE.write_text(json.dumps(report, indent=2))
