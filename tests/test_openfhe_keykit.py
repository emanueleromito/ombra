# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""A KeyKit saved to a key file can be reopened, in this and in a new process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("openfhe")
from ombra import presets  # noqa: E402
from ombra.collection import KeyKit  # noqa: E402
from ombra.keyfile import KeyFileError  # noqa: E402

pytestmark = pytest.mark.openfhe

SCRYPT = (2**4, 8, 1)  # cheap for tests
CLIENT = str(Path(__file__).resolve().parents[1] / "client")
CHILD = """
import sys
from ombra import presets
from ombra.collection import KeyKit
d = presets.derive("fast", 384)
kit = KeyKit.load(sys.argv[1], "pw", d, "test-openfhe", 387)
env = open(sys.argv[2], "rb").read()
print(",".join(f"{v:.4f}" for v in kit.decrypt_column(env)[:3]))
"""


def test_save_load_roundtrip(tmp_path) -> None:
    d = presets.derive("fast", 384)
    kit = KeyKit(d, "test-openfhe", 387)
    path = tmp_path / "k.key"
    kit.save(path, "pw", scrypt_params=SCRYPT)
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(KeyFileError):
        KeyKit.load(path, "wrong", d, "test-openfhe", 387)

    again = KeyKit.load(path, "pw", d, "test-openfhe", 387)
    assert again.key_tag == kit.key_tag and again.master_key == kit.master_key
    vals = np.array([0.25, -0.5, 0.125])
    assert np.allclose(again.decrypt_column(kit.encrypt_column(vals))[:3], vals, atol=1e-3)
    assert np.allclose(kit.decrypt_column(again.encrypt_column(vals))[:3], vals, atol=1e-3)
    assert "secret" not in repr(again).lower()

    # a new process reopens the keys and decrypts what this one encrypted
    env = tmp_path / "ct.bin"
    env.write_bytes(kit.encrypt_column(vals))
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", CHILD, str(path), str(env)],
        env={"PYTHONPATH": CLIENT, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert np.allclose([float(v) for v in out.strip().split(",")], vals, atol=1e-3)
