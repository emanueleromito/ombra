"""T0.11 step 1: preset m0_compact_n8192 (ring dimension 8192) against the 128-bit limit.

The real log2(QP) is read from the crypto context's public evaluation key (the only place the
bindings expose P) and compared with the HomomorphicEncryption.org 128-bit classic limit for
N = 8192, 218 bits. OpenFHE itself also refuses a ring dimension that fails its own check at
context creation, so creating the context at all is a second, independent confirmation. Also pins
the baseline preset m0_depth2, which must stay unchanged. Scalars only; no data.
"""

from __future__ import annotations

import json
import math

import pytest
from ombra import params as ombra_params

pytestmark = [pytest.mark.openfhe, pytest.mark.gpu]  # also run by `make test-gpu`

COMPACT = "m0_compact_n8192"
LIMIT_N8192 = 218  # HE standard, 128-bit classic, ternary secret

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


def test_baseline_preset_unchanged() -> None:
    assert ombra_params.get_preset("m0_depth2") == BASELINE


def test_compact_preset_fields() -> None:
    p = ombra_params.get_preset(COMPACT)
    assert p["security_level"] == "HEStd_128_classic"
    assert p["ring_dim"] == 8192
    assert p["multiplicative_depth"] == 2
    assert p["scaling_mod_size"] == 40
    assert p["batch_size"] == 4096
    assert p["scaling_technique"] == "FIXEDMANUAL"
    assert p["key_switch_technique"] == "HYBRID"


def test_compact_log_qp_within_128_bit_limit(tmp_path) -> None:
    import openfhe as ofhe

    params, cc = ombra_params.make_context(COMPACT)  # OpenFHE raises if N is non-compliant
    assert params.GetSecurityLevel() == ofhe.SecurityLevel.HEStd_128_classic
    assert cc.GetRingDimension() == 8192
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    key_file = tmp_path / "evalmult.json"
    assert ofhe.CryptoContext.SerializeEvalMultKey(
        str(key_file), ofhe.JSON, keys.publicKey.GetKeyTag()
    )
    tree = json.loads(key_file.read_text())
    entry = tree["value0"][0]["value"][0]["ptr_wrapper"]["data"]
    qp = [int(t["v"]["ptr_wrapper"]["data"]["m"]["v"]) for t in entry["ak"][0]["v"]]
    digits = len(entry["ak"])
    num_q = ombra_params.get_preset(COMPACT)["multiplicative_depth"] + 1
    q_bits = math.log2(cc.GetElementParams().GetModulus())
    assert abs(sum(math.log2(m) for m in qp[:num_q]) - q_bits) < 1e-6
    log_qp = sum(math.log2(m) for m in qp)
    sizes = [round(math.log2(m)) for m in qp]
    print(
        f"{COMPACT}: log2(Q)={q_bits:.2f} log2(QP)={log_qp:.2f} limit={LIMIT_N8192} "
        f"towers={len(qp)} bits={sizes} digits={digits}"
    )
    assert len(qp) > num_q, "evaluation key does not contain the special primes P"
    assert log_qp <= LIMIT_N8192
