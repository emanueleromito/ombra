# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Audit fixes A-01/A-73 (no key file on disk) and A-11/A-70 (refuse foreign ciphertexts)."""

from __future__ import annotations

import tempfile

import pytest

np = pytest.importorskip("numpy")
ofhe = pytest.importorskip("openfhe")
from ombra import presets  # noqa: E402
from ombra._decrypt import CiphertextRefusedError  # noqa: E402
from ombra.collection import KeyKit  # noqa: E402
from ombra.file_format import Kind, write_envelope  # noqa: E402

pytestmark = pytest.mark.openfhe

SCRYPT = (2**4, 8, 1)
VERSION = "test-openfhe"


@pytest.fixture(scope="module")
def kit() -> KeyKit:
    return KeyKit(presets.derive("fast", 384), VERSION, 387)


def test_no_file_for_key_save_and_load(tmp_path, monkeypatch, kit) -> None:
    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(watch))
    monkeypatch.chdir(watch)

    def no_file(*a, **k):
        raise AssertionError("OpenFHE file serialization must not be used")

    for name in ("SerializeToFile", "DeserializePrivateKey", "DeserializePublicKey"):
        monkeypatch.setattr(ofhe, name, no_file)
    monkeypatch.setattr(ofhe.CryptoContext, "SerializeEvalAutomorphismKey", no_file)
    monkeypatch.setattr(ofhe.CryptoContext, "SerializeEvalMultKey", no_file)

    path = out / "k.key"
    kit.save(path, "pw", scrypt_params=SCRYPT)
    again = KeyKit.load(path, "pw", kit.derived, VERSION, 387)
    vals = np.array([0.25, -0.5])
    assert np.allclose(again.decrypt_column(kit.encrypt_column(vals))[:2], vals, atol=1e-3)
    kit.encrypt_query(np.ones(387) / 20)

    class Pb:
        KEY_KIND_CONTEXT, KEY_KIND_EVAL_MULT, KEY_KIND_AUTOMORPHISM = 1, 2, 3

    assert all(kit.key_files(Pb).values())
    assert list(watch.iterdir()) == []  # nothing was written to the temp dir or the cwd
    assert [p.name for p in out.iterdir()] == ["k.key"]  # and no sidecar next to the key file


def _env(kit: KeyKit, ct) -> bytes:
    return write_envelope(
        Kind.CIPHERTEXT, kit.derived.preset, VERSION, ofhe.Serialize(ct, ofhe.BINARY)
    )


def _fresh(kit: KeyKit, values=(0.5, 0.25)):
    return kit.cc.Encrypt(kit.public_key, kit.cc.MakeCKKSPackedPlaintext(list(values)))


def test_genuine_ciphertexts_are_accepted(kit) -> None:
    fresh = _fresh(kit)
    assert np.allclose(kit.decrypt_column(_env(kit, fresh))[:2], [0.5, 0.25], atol=1e-3)
    prod = kit.cc.Rescale(kit.cc.EvalMult(fresh, fresh))  # level 1, as a served score block
    assert np.allclose(kit.decrypt_column(_env(kit, prod))[:2], [0.25, 0.0625], atol=1e-3)


def test_tampered_ciphertexts_are_refused(kit) -> None:
    cc = kit.cc
    other = cc.KeyGen()  # another key pair in the same context
    foreign = cc.Encrypt(other.publicKey, cc.MakeCKKSPackedPlaintext([0.5]))
    fresh = _fresh(kit)
    prod = cc.Rescale(cc.EvalMult(fresh, fresh))

    wrong_level = _fresh(kit)
    wrong_level.SetLevel(1)  # claims level 1 with the towers of level 0
    wrong_level_up = cc.Rescale(cc.EvalMult(fresh, fresh))
    wrong_level_up.SetLevel(0)  # claims level 0 with the towers of level 1

    bad_scale = _fresh(kit)
    bad_scale.SetScalingFactor(bad_scale.GetScalingFactor() * 2.0)

    bad_deg = _fresh(kit)
    bad_deg.SetNoiseScaleDeg(2)

    three = cc.EvalMultNoRelin(fresh, fresh)  # three components
    assert len(three.GetElements()) == 3

    too_deep = cc.LevelReduce(prod, None, 1)  # level 2: one tower only

    for ct in (foreign, wrong_level, wrong_level_up, bad_scale, bad_deg, three, too_deep):
        with pytest.raises(CiphertextRefusedError):
            kit.decrypt_column(_env(kit, ct))
    # the original, untampered one still works
    assert kit.decrypt_column(_env(kit, prod)).shape == (kit.slots,)


def test_garbage_payload_is_refused(kit) -> None:
    env = write_envelope(Kind.CIPHERTEXT, kit.derived.preset, VERSION, b"not a ciphertext")
    with pytest.raises(ValueError):
        kit.decrypt_column(env)


def test_insecure_warns_for_non_loopback_hosts() -> None:
    import warnings

    pytest.importorskip("grpc")
    pytest.importorskip("grpc_tools")
    from ombra._transport import GrpcTransport

    with pytest.warns(UserWarning, match="not encrypted"):
        GrpcTransport("10.255.255.1:1", None, None, True).close()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GrpcTransport("localhost:1", None, None, True).close()
        GrpcTransport("[::1]:1", None, None, True).close()
