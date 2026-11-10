import logging
import os
import stat

import pytest
from ombra import keyfile
from ombra.keyfile import KeyFileError, KeyMaterial

FAST = (2**4, 8, 1)
PASS = "correct horse battery staple"  # noqa: S105


def material() -> KeyMaterial:
    return KeyMaterial.generate(os.urandom(300))


def test_round_trip_file_and_mode(tmp_path):
    m, path = material(), tmp_path / "k.key"
    keyfile.export_key_file(path, m, PASS, scrypt_params=FAST)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(tmp_path.iterdir()) == [path]
    assert keyfile.import_key_file(path, PASS) == m


def test_overwrite_is_atomic_replace(tmp_path):
    path = tmp_path / "k.key"
    keyfile.export_key_file(path, material(), PASS, scrypt_params=FAST)
    m2 = material()
    keyfile.export_key_file(path, m2, PASS, scrypt_params=FAST)
    assert keyfile.import_key_file(path, PASS) == m2


def test_wrong_passphrase_refused():
    blob = keyfile.export_key_bytes(material(), PASS, scrypt_params=FAST)
    with pytest.raises(KeyFileError):
        keyfile.import_key_bytes(blob, "wrong passphrase")


def test_every_changed_byte_refused():
    blob = keyfile.export_key_bytes(material(), PASS, scrypt_params=FAST)
    # Header bytes (incl. KDF parameters and salt) and ciphertext bytes alike.
    for i in list(range(0, keyfile._HEADER_LEN + 4)) + [len(blob) // 2, len(blob) - 1]:
        bad = bytearray(blob)
        bad[i] ^= 1
        with pytest.raises(KeyFileError):
            keyfile.import_key_bytes(bytes(bad), PASS)


def test_unknown_version_refused():
    blob = bytearray(keyfile.export_key_bytes(material(), PASS, scrypt_params=FAST))
    blob[4] = 2
    with pytest.raises(KeyFileError, match="version"):
        keyfile.import_key_bytes(bytes(blob), PASS)


def test_truncated_and_bad_magic_refused():
    blob = keyfile.export_key_bytes(material(), PASS, scrypt_params=FAST)
    with pytest.raises(KeyFileError):
        keyfile.import_key_bytes(blob[:-1], PASS)
    with pytest.raises(KeyFileError):
        keyfile.import_key_bytes(b"XXXX" + blob[4:], PASS)


def test_hostile_kdf_parameters_refused_before_deriving():
    blob = bytearray(keyfile.export_key_bytes(material(), PASS, scrypt_params=FAST))
    blob[7:11] = (2**30).to_bytes(4, "little")
    with pytest.raises(KeyFileError, match="out of range"):
        keyfile.import_key_bytes(bytes(blob), PASS)


def test_salts_differ_between_exports():
    m = material()
    a = keyfile.export_key_bytes(m, PASS, scrypt_params=FAST)
    b = keyfile.export_key_bytes(m, PASS, scrypt_params=FAST)
    assert a != b


def test_no_secrets_in_logs_errors_or_repr(caplog):
    m, secret = material(), "s3cret-passphrase-XYZ"
    blob = keyfile.export_key_bytes(m, secret, scrypt_params=FAST)
    texts = []
    with caplog.at_level(logging.DEBUG):
        for call in (
            lambda: keyfile.import_key_bytes(blob, "another-wrong-pass"),
            lambda: keyfile.import_key_bytes(blob[:-1], secret),
            lambda: keyfile.import_key_bytes(b"x" * 200, secret),
            lambda: KeyMaterial(m.secret_key, b"short"),
            lambda: keyfile.export_key_bytes(m, "", scrypt_params=FAST),
        ):
            with pytest.raises(KeyFileError) as exc:
                call()
            texts += [str(exc.value), repr(exc.value)]
    texts += [repr(m), str(m), caplog.text]
    needles = [
        secret,
        "another-wrong-pass",
        m.secret_key.hex(),
        m.master_key.hex(),
        repr(m.secret_key),
        repr(m.master_key),
    ]
    for t in texts:
        for n in needles:
            assert n not in t
        assert str(list(m.master_key)) not in t
