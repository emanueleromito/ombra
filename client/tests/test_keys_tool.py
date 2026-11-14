# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""`ombra-keys backup` and `check` (CPU, OpenFHE). A real key file of the smallest preset, written
with cheap scrypt (OMBRA_ALLOW_WEAK_SCRYPT, tests only). The passphrase never goes on a command
line; the tool's output never holds key material."""

from __future__ import annotations

import os
import stat

import pytest

pytest.importorskip("numpy")
pytest.importorskip("openfhe")
from ombra import keyfile, keys, presets  # noqa: E402
from ombra.collection import KeyKit  # noqa: E402
from ombra.keyfile import KeyFileError  # noqa: E402

pytestmark = pytest.mark.openfhe

SCRYPT = (2**4, 8, 1)
PW = "correct horse battery staple"  # noqa: S105


@pytest.fixture(autouse=True)
def _weak_scrypt(monkeypatch):
    monkeypatch.setenv(keyfile.ALLOW_WEAK_ENV, "1")
    monkeypatch.delenv(keys.PASSPHRASE_ENV, raising=False)


@pytest.fixture(scope="module")
def kit() -> KeyKit:
    return KeyKit(presets.derive("fast", 384), "test-openfhe", 387)


@pytest.fixture
def key_file(tmp_path, kit, monkeypatch):
    monkeypatch.setenv(keyfile.ALLOW_WEAK_ENV, "1")
    path = tmp_path / "k.key"
    kit.save(path, PW, scrypt_params=SCRYPT)
    return path


def test_backup_copies_atomically_and_checks(tmp_path, key_file, capsys) -> None:
    dst = tmp_path / "backup" / "k.bak"
    dst.parent.mkdir()
    assert keys.main(["backup", str(key_file), str(dst), "--passphrase-file", _pwfile(tmp_path)]) == 0
    assert dst.read_bytes() == key_file.read_bytes()
    assert stat.S_IMODE(dst.stat().st_mode) == 0o600
    assert [p.name for p in dst.parent.iterdir()] == ["k.bak"]  # no temp file left
    out = capsys.readouterr().out
    assert "OK" in out and "passphrase" in out
    # refuses to overwrite without --force; the existing file is untouched
    dst.write_bytes(b"older backup")
    assert keys.main(["backup", str(key_file), str(dst)]) == 1
    assert dst.read_bytes() == b"older backup"
    assert [p.name for p in dst.parent.iterdir()] == ["k.bak"]
    assert keys.main(["backup", str(key_file), str(dst), "--force"]) == 0
    assert dst.read_bytes() == key_file.read_bytes()


def test_backup_refuses_something_that_is_not_a_key_file(tmp_path) -> None:
    bad = tmp_path / "x"
    bad.write_bytes(b"not a key file" * 20)
    assert keys.main(["backup", str(bad), str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out").exists()


def test_backup_with_wrong_passphrase_reports_failure(tmp_path, key_file, monkeypatch) -> None:
    monkeypatch.setenv("MY_PW", "wrong")
    dst = tmp_path / "b"
    assert keys.main(["backup", str(key_file), str(dst), "--passphrase-env", "MY_PW"]) == 1


def _pwfile(tmp_path) -> str:
    p = tmp_path / "pw"
    p.write_text(PW + "\n")
    return str(p)


def test_check_ok_offline_with_tag(tmp_path, key_file, kit, capsys, monkeypatch) -> None:
    monkeypatch.setenv(keys.PASSPHRASE_ENV, PW)
    assert keys.main(["check", str(key_file), "--tag", kit.key_tag]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and kit.key_tag in out
    # the secret material and the passphrase are never printed
    assert PW not in out
    assert keys.main(["check", str(key_file)]) == 0  # no tag: only the file is checked


def test_check_wrong_tag_wrong_passphrase_corrupt_file(tmp_path, key_file, kit, monkeypatch, capsys):
    monkeypatch.setenv(keys.PASSPHRASE_ENV, PW)
    assert keys.main(["check", str(key_file), "--tag", "some other tag"]) == 1
    assert "does not match" in capsys.readouterr().err
    monkeypatch.setenv(keys.PASSPHRASE_ENV, "wrong passphrase")
    assert keys.main(["check", str(key_file)]) == 1
    err = capsys.readouterr().err
    assert "FAIL" in err and "wrong passphrase" in err and PW not in err
    monkeypatch.setenv(keys.PASSPHRASE_ENV, PW)
    data = bytearray(key_file.read_bytes())
    data[-5] ^= 1
    bad = tmp_path / "bad.key"
    bad.write_bytes(bytes(data))
    assert keys.main(["check", str(bad)]) == 1
    assert keys.main(["check", str(tmp_path / "missing.key")]) == 1


def test_passphrase_from_a_file_descriptor_and_a_prompt(tmp_path, key_file, monkeypatch) -> None:
    r, w = os.pipe()
    os.write(w, (PW + "\n").encode())
    os.close(w)
    try:
        assert keys.main(["check", str(key_file), "--passphrase-fd", str(r)]) == 0
    finally:
        os.close(r)
    monkeypatch.setattr(keys.getpass, "getpass", lambda prompt="": PW)
    assert keys.main(["check", str(key_file)]) == 0  # prompt (nothing else set)
    assert keys.main(["check", str(key_file), "--passphrase-file", str(tmp_path / "nope")]) == 1


def test_the_passphrase_cannot_be_given_as_an_argument(tmp_path, key_file) -> None:
    for flag in ("--passphrase", "--password"):
        with pytest.raises(SystemExit) as ei:
            keys.main(["check", str(key_file), flag, PW])
        assert ei.value.code == 2


def test_check_against_a_server_tag(tmp_path, key_file, kit, monkeypatch) -> None:
    monkeypatch.setenv(keys.PASSPHRASE_ENV, PW)
    seen = []

    def fake(address, collection, insecure):
        seen.append((address, collection, insecure))
        return kit.key_tag

    monkeypatch.setattr(keys, "_server_tag", fake)
    args = ["check", str(key_file), "--server", "h:1", "--collection", "c", "--insecure"]
    assert keys.main(args) == 0 and seen == [("h:1", "c", True)]
    monkeypatch.setattr(keys, "_server_tag", lambda *a: "another tag")
    assert keys.main(args) == 1
    with pytest.raises(SystemExit):  # --server needs --collection
        keys.main(["check", str(key_file), "--server", "h:1"])


def test_key_file_module_rejects_non_key_files(tmp_path) -> None:
    p = tmp_path / "f"
    p.write_bytes(b"x" * 200)
    with pytest.raises(KeyFileError):
        keyfile.check_key_file(p, PW)
