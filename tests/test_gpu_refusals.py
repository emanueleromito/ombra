"""T1.12, T1.13, T1.14: GPU entry points refuse inputs that do not belong together.

The C++ driver `ombra_refusal_check` sends mixed keys, wrong metadata and mismatched shapes to
every GPU entry point and prints one line per case ("REFUSED <message>" or "ACCEPTED"). Each
negative test asserts the specific message, so another failure cannot pass as a refusal; each
valid control must be ACCEPTED. Two key pairs of one context are made here with the normal OpenFHE
client API; the secret keys stay in this process and nothing secret is written to disk (only the
context, ciphertexts and public evaluation keys). Synthetic data only.
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

PRESET = "m0_depth2"
SMALL_PRESET = "m0_compact_n8192"  # 128-bit, ring 8192 (PRESET has ring 16384)
D = 4
SEED = 20260111

DIFFERENT_KEY = "encrypted under different keys"
OTHER_KEY_EVAL = "evaluation key belongs to a different key"


def _driver() -> Path:
    candidates = [
        os.environ.get("OMBRA_REFUSAL_CHECK_BIN", ""),
        "/root/ombra-build/core/ombra_refusal_check",
        str(Path(__file__).resolve().parents[1] / "build" / "core" / "ombra_refusal_check"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail("ombra_refusal_check not found; build with `make build` or `make test-gpu`")


def _run(args: list[str]) -> dict[str, str]:
    proc = subprocess.run(  # noqa: S603
        [str(_driver()), *args], capture_output=True, text=True, timeout=900
    )
    assert proc.returncode == 0, f"driver failed (exit {proc.returncode}): {proc.stderr}"
    cases: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("CASE "):
            name, _, result = line[len("CASE ") :].partition(": ")
            cases[name] = result
    return cases


def _refused(cases: dict[str, str], name: str, message: str) -> None:
    assert name in cases, f"case {name} missing: {sorted(cases)}"
    assert cases[name].startswith("REFUSED "), f"{name}: {cases[name]}"
    assert message in cases[name], f"{name}: wrong refusal: {cases[name]}"


def _accepted(cases: dict[str, str], name: str) -> None:
    assert cases.get(name) == "ACCEPTED", f"{name}: {cases.get(name)}"


@pytest.fixture(scope="module")
def keys_cases(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    import openfhe as ofhe

    tmp_path = tmp_path_factory.mktemp("refusals")
    _, cc = ombra_params.make_context(PRESET)
    slots = ombra_params.get_preset(PRESET)["batch_size"]
    pair = {"a": cc.KeyGen(), "b": cc.KeyGen()}
    assert pair["a"].publicKey.GetKeyTag() != pair["b"].publicKey.GetKeyTag()

    indices = oq.automorphism_indices(cc.GetRingDimension(), D)
    cc.ClearEvalAutomorphismKeys()  # the key store is static: hold exactly this d's indices
    rng = np.random.default_rng(SEED)
    q = rng.standard_normal(D)
    q /= np.linalg.norm(q)
    for name, kp in pair.items():
        cc.EvalMultKeyGen(kp.secretKey)
        cc.EvalAutomorphismKeyGen(kp.secretKey, indices)
        assert ofhe.CryptoContext.SerializeEvalMultKey(
            str(tmp_path / f"evalmult_{name}.bin"), ofhe.BINARY, kp.publicKey.GetKeyTag()
        )
        for j in range(2):
            ct = cc.Encrypt(
                kp.publicKey, cc.MakeCKKSPackedPlaintext(rng.standard_normal(slots).tolist())
            )
            assert ofhe.SerializeToFile(str(tmp_path / f"{name}_{j}.bin"), ct, ofhe.BINARY)
        mask = cc.Encrypt(kp.publicKey, cc.MakeCKKSPackedPlaintext([1.0] * slots, 1, 1))
        assert ofhe.SerializeToFile(str(tmp_path / f"{name}_mask.bin"), mask, ofhe.BINARY)
        root = oq.encode_query(cc, kp.publicKey, q)
        assert ofhe.SerializeToFile(str(tmp_path / f"{name}_root.bin"), root, ofhe.BINARY)
    assert ofhe.SerializeToFile(str(tmp_path / "context.bin"), cc, ofhe.BINARY)
    assert cc.SerializeEvalAutomorphismKey(str(tmp_path / "autokeys.bin"), ofhe.BINARY)
    return _run(["keys", str(tmp_path / "context.bin"), str(tmp_path), str(D)])


# ---- T1.12 ----------------------------------------------------------------------------------

META = "level, noise degree or scale differ from the sample"


def test_expansion_upload_refuses_other_degree_scale_level(keys_cases: dict[str, str]) -> None:
    _accepted(keys_cases, "expansion_upload_valid")
    _refused(keys_cases, "expansion_upload_degree", META)
    _refused(keys_cases, "expansion_upload_scale", META)
    _refused(keys_cases, "expansion_upload_level", META)


# ---- T1.13 ----------------------------------------------------------------------------------


def test_valid_inputs_still_pass(keys_cases: dict[str, str]) -> None:
    for name in (
        "expansion_keys_valid",
        "add_valid",
        "mult_plain_valid",
        "plain_search_valid",
        "column_search_valid",
        "column_search_batched_valid",
        "ops_ctor_valid",
        "ops_multiply_valid",
        "ops_rescale_valid",
        "search_ctor_valid",
        "search_upload_block_valid",
        "search_upload_query_valid",
        "search_upload_mask_valid",
    ):
        _accepted(keys_cases, name)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("expansion_upload_other_key", "expansion upload: ciphertexts are " + DIFFERENT_KEY),
        ("expansion_upload_other_encoding", "expansion upload: ciphertexts differ in encoding"),
        ("expansion_keys_of_other_key", OTHER_KEY_EVAL),
        ("add_other_key", "add: ciphertexts are " + DIFFERENT_KEY),
        ("add_other_encoding", "add: ciphertexts differ in encoding type"),
        ("mult_plain_other_encoding", "mult: ciphertext and plaintext differ in encoding type"),
        ("plain_search_other_key", "plain search: ciphertexts are " + DIFFERENT_KEY),
        ("column_search_other_key", "stored and query are " + DIFFERENT_KEY),
        ("column_search_mixed_query", "flatten: ciphertexts are " + DIFFERENT_KEY),
        ("column_search_batched_other_key", "stored and query are " + DIFFERENT_KEY),
        ("column_search_batched_mixed_queries", "queries are " + DIFFERENT_KEY),
        ("ops_ctor_relin_key_of_other_key", OTHER_KEY_EVAL),
        ("ops_multiply_other_key", "ciphertext is encrypted under a different key"),
        ("ops_rescale_other_key", "ciphertext is encrypted under a different key"),
        ("ops_relinearize_other_key", "ciphertext is encrypted under a different key"),
        ("search_ctor_mask_of_other_key", "search samples: ciphertexts are " + DIFFERENT_KEY),
        ("search_ctor_relin_key_of_other_key", OTHER_KEY_EVAL),
        ("search_upload_block_mixed", "flatten: ciphertexts are " + DIFFERENT_KEY),
        ("search_upload_block_other_key", "ciphertext is encrypted under a different key"),
        ("search_upload_block_flat_other_key", "block is encrypted under a different key"),
        ("search_upload_query_other_key", "ciphertext is encrypted under a different key"),
        ("search_upload_mask_other_key", "ciphertext is encrypted under a different key"),
    ],
)
def test_mixed_keys_are_refused(keys_cases: dict[str, str], case: str, message: str) -> None:
    _refused(keys_cases, case, message)


# ---- T1.14 ----------------------------------------------------------------------------------


def test_batched_search_refuses_query_with_other_ring_dimension(tmp_path: Path) -> None:
    import openfhe as ofhe

    ring = {}
    cts = {}
    rng = np.random.default_rng(SEED + 1)
    for tag, preset in (("1", PRESET), ("2", SMALL_PRESET)):
        _, cc = ombra_params.make_context(preset)
        ring[tag] = cc.GetRingDimension()
        kp = cc.KeyGen()
        slots = ombra_params.get_preset(preset)["batch_size"]
        assert ofhe.SerializeToFile(str(tmp_path / f"context{tag}.bin"), cc, ofhe.BINARY)
        names = ["q1_0", "q1_1", "stored_0", "stored_1"] if tag == "1" else ["q2_0", "q2_1"]
        for name in names:
            ct = cc.Encrypt(
                kp.publicKey, cc.MakeCKKSPackedPlaintext(rng.standard_normal(slots).tolist())
            )
            assert ofhe.SerializeToFile(str(tmp_path / f"{name}.bin"), ct, ofhe.BINARY)
            cts[name] = True
    assert ring["1"] != ring["2"]
    cases = _run(
        ["shape", str(tmp_path / "context1.bin"), str(tmp_path / "context2.bin"), str(tmp_path)]
    )
    _accepted(cases, "batched_search_valid")
    msg = "queries differ in ring dimension or tower count"
    _refused(cases, "batched_search_smaller_ring_query", msg)
    _refused(cases, "batched_search_smaller_ring_first_query", msg)
