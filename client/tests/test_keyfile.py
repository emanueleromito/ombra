# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
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
    keyfile.export_key_file(path, m2, PASS, scrypt_params=FAST, overwrite=True)
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


def test_export_refuses_to_overwrite(tmp_path):
    path = tmp_path / "k.ombrakey"
    m = KeyMaterial.generate(b"sk-blob")
    keyfile.export_key_file(path, m, "pw", scrypt_params=(2**4, 1, 1))
    with pytest.raises(KeyFileError, match="already exists"):
        keyfile.export_key_file(path, m, "pw", scrypt_params=(2**4, 1, 1))
    keyfile.export_key_file(path, m, "pw2", scrypt_params=(2**4, 1, 1), overwrite=True)
    assert keyfile.import_key_file(path, "pw2") == m


def test_scrypt_memory_is_capped_on_read_and_write():
    # n = 2**20, r = 32 needs 4 GiB: refused before any derivation
    with pytest.raises(KeyFileError, match="1 GiB"):
        keyfile.export_key_bytes(material(), PASS, scrypt_params=(2**20, 32, 1))
    blob = bytearray(keyfile.export_key_bytes(material(), PASS, scrypt_params=FAST))
    blob[7:11] = (2**20).to_bytes(4, "little")
    blob[11:15] = (32).to_bytes(4, "little")
    with pytest.raises(KeyFileError, match="1 GiB"):
        keyfile.import_key_bytes(bytes(blob), PASS)


def test_weak_scrypt_is_refused_unless_tests_allow_it(monkeypatch):
    monkeypatch.delenv(keyfile.ALLOW_WEAK_ENV, raising=False)
    with pytest.raises(KeyFileError, match="too weak"):
        keyfile.export_key_bytes(material(), PASS, scrypt_params=FAST)
    m = material()
    blob = keyfile.export_key_bytes(m, PASS, scrypt_params=(2**14, 8, 1))
    assert keyfile.import_key_bytes(blob, PASS) == m
    monkeypatch.setenv(keyfile.ALLOW_WEAK_ENV, "1")
    weak = keyfile.export_key_bytes(m, PASS, scrypt_params=FAST)
    monkeypatch.delenv(keyfile.ALLOW_WEAK_ENV)
    with pytest.raises(KeyFileError, match="too weak"):
        keyfile.import_key_bytes(weak, PASS)


def test_no_overwrite_uses_a_link_and_loses_a_race(tmp_path, monkeypatch):
    path = tmp_path / "k.key"
    real_exists = type(path).exists

    def racing_exists(self):
        # the check says "free", then another writer creates the file before the link
        if self == path:
            path.write_bytes(b"other writer")
            return False
        return real_exists(self)

    monkeypatch.setattr(type(path), "exists", racing_exists)
    with pytest.raises(KeyFileError, match="already exists"):
        keyfile.export_key_file(path, material(), PASS, scrypt_params=FAST)
    monkeypatch.undo()
    assert path.read_bytes() == b"other writer"
    assert [p.name for p in tmp_path.iterdir()] == ["k.key"]  # temp file cleaned up


def test_export_without_fchmod(tmp_path, monkeypatch):
    monkeypatch.delattr(keyfile.os, "fchmod")  # as on Windows
    path = tmp_path / "k.key"
    m = material()
    keyfile.export_key_file(path, m, PASS, scrypt_params=FAST)
    assert keyfile.import_key_file(path, PASS) == m
