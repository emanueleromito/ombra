# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""PERF-A: decrypting after dropping to the first RNS tower equals the full decrypt."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
ofhe = pytest.importorskip("openfhe")
from ombra import presets  # noqa: E402
from ombra._decrypt import decrypt_values, tower_count  # noqa: E402
from ombra.params import make_context  # noqa: E402
from ombra.topk import serial_topk  # noqa: E402

pytestmark = pytest.mark.openfhe

# preset -> absolute tolerance between the dropped and the full decrypt
TOL = {"m0_depth2": 1e-6, "m0_compact_n8192": 1e-5}
SLOTS = 64
_CACHE: dict[str, tuple] = {}


def _ctx(name: str):
    if name not in _CACHE:
        _, cc = make_context(name)
        kp = cc.KeyGen()
        cc.EvalMultKeyGen(kp.secretKey)
        _CACHE[name] = (cc, kp)
    return _CACHE[name]


def _full(cc, sk, ct, slots=SLOTS) -> np.ndarray:
    dec = cc.Decrypt(ct, sk)
    dec.SetLength(slots)
    return np.asarray(dec.GetRealPackedValue(), dtype=np.float64)


def _at_level(cc, kp, values, level: int):
    pt = cc.MakeCKKSPackedPlaintext([float(v) for v in values], 1, level)
    return cc.Encrypt(kp.publicKey, pt)


def _mult_rescaled(cc, kp, values):
    """The server-equivalent path: values * 1 (EvalMult of two ciphertexts), then Rescale."""
    a = cc.Encrypt(kp.publicKey, cc.MakeCKKSPackedPlaintext([float(v) for v in values]))
    one = cc.Encrypt(kp.publicKey, cc.MakeCKKSPackedPlaintext([1.0] * len(values)))
    return cc.Rescale(cc.EvalMult(a, one))


@pytest.mark.parametrize("name", sorted(TOL))
def test_tower_count(name) -> None:
    cc, kp = _ctx(name)
    depth = cc.GetMultiplicativeDepth()
    for level in range(depth + 1):
        ct = _at_level(cc, kp, [0.5, 0.25], level)
        assert tower_count(cc, ct) == depth + 1 - level
    assert tower_count(cc, _mult_rescaled(cc, kp, [0.5])) == depth  # served: 2 towers


@pytest.mark.parametrize("name", sorted(TOL))
def test_matches_full_decrypt_random(name) -> None:
    cc, kp = _ctx(name)
    vals = np.random.default_rng(1).uniform(-1, 1, SLOTS)
    for level in (0, 1, 2):
        ct = _at_level(cc, kp, vals, level)
        got = decrypt_values(cc, kp.secretKey, ct, SLOTS)
        assert got.dtype == np.float64 and got.shape == (SLOTS,)
        assert np.max(np.abs(got - _full(cc, kp.secretKey, ct))) < TOL[name]
        assert np.max(np.abs(got - vals)) < TOL[name]


@pytest.mark.parametrize("name", sorted(TOL))
def test_max_magnitude_survives_drop(name) -> None:
    """Largest value the client produces: score 1 plus the filter band and tombstone penalty.

    The drop is valid only while |value| * scale < q0 / 2 (see ombra/_decrypt.py).
    """
    cc, kp = _ctx(name)
    derived = presets.derive(name, 384)
    fmax = presets.filter_field_limit(derived)
    top = 1.0 + presets.FILTER_LAMBDA * (fmax + 1)
    vals = np.zeros(SLOTS)
    vals[:4] = [top, -top, top - 1.0, 0.5]
    vals[4:] = np.random.default_rng(2).uniform(-top, top, SLOTS - 4)
    # Decode adds fresh noise each call, so two decrypts of one ciphertext differ slightly.
    for ct, exact in ((_at_level(cc, kp, vals, 1), True), (_mult_rescaled(cc, kp, vals), False)):
        assert tower_count(cc, ct) == 2
        got = decrypt_values(cc, kp.secretKey, ct, SLOTS)
        assert np.max(np.abs(got - _full(cc, kp.secretKey, ct))) < TOL[name]
        if exact:  # the EvalMult path carries its own approximation error, same in both decrypts
            assert np.max(np.abs(got - vals)) < TOL[name]


@pytest.mark.parametrize("name", sorted(TOL))
def test_one_tower_passthrough(name) -> None:
    cc, kp = _ctx(name)
    vals = np.linspace(-1, 1, SLOTS)
    ct = _at_level(cc, kp, vals, cc.GetMultiplicativeDepth())
    assert tower_count(cc, ct) == 1
    got = decrypt_values(cc, kp.secretKey, ct, SLOTS)
    assert ct.GetLevel() == cc.GetMultiplicativeDepth()  # input not modified
    assert np.max(np.abs(got - _full(cc, kp.secretKey, ct))) < TOL[name]


@pytest.mark.parametrize("name", sorted(TOL))
def test_serial_topk_same_indices(name, tmp_path) -> None:
    cc, kp = _ctx(name)
    rng = np.random.default_rng(3)
    perm = rng.permutation(3 * SLOTS) / (3 * SLOTS)  # distinct, so decode noise cannot reorder
    blocks = [perm[i * SLOTS : (i + 1) * SLOTS] for i in range(3)]
    files = []
    for i, b in enumerate(blocks):
        p = tmp_path / f"b{i}.bin"
        assert ofhe.SerializeToFile(str(p), _at_level(cc, kp, b, 1), ofhe.BINARY)
        files.append(p)
    valid_last, k = 50, 7
    scores, idx = serial_topk(files, cc, kp.secretKey, SLOTS, valid_last, k)
    all_s, all_i = [], []
    for i, f in enumerate(files):
        ct, ok = ofhe.DeserializeCiphertext(str(f), ofhe.BINARY)
        assert ok
        v = _full(cc, kp.secretKey, ct)[: valid_last if i == 2 else SLOTS]
        all_s.append(v)
        all_i.append(np.arange(len(v)) + i * SLOTS)
    s, ix = np.concatenate(all_s), np.concatenate(all_i)
    o = np.lexsort((ix, -s))[:k]
    assert np.array_equal(idx, ix[o])
    assert np.allclose(scores, s[o], atol=TOL[name])


def test_depth_after_deserialization() -> None:
    """Parallel workers hold a deserialized context; it must report the same depth."""
    cc, _ = _ctx("m0_depth2")
    cc2 = ofhe.DeserializeCryptoContextString(ofhe.Serialize(cc, ofhe.BINARY), ofhe.BINARY)
    assert cc2.GetMultiplicativeDepth() == cc.GetMultiplicativeDepth()
