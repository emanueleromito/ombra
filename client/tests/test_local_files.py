# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Local mode: private data directory and temp files (A-47), no pickle between processes (A-48)."""

import re
import stat
from pathlib import Path

from ombra import _local


def test_data_dir_is_private_and_temp_files_are_unique(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir(mode=0o755)
    _local._private_dir(root / "c" / "x")
    _local._private_dir(root)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "c" / "x").stat().st_mode) == 0o700

    seen = []
    real = _local.os.replace

    def spy(src, dst):
        seen.append((src, stat.S_IMODE(_local.os.stat(src).st_mode)))
        real(src, dst)

    monkeypatch.setattr(_local.os, "replace", spy)
    target = root / "c" / "x" / "f"
    _local._atomic(target, b"one")
    _local._atomic(target, b"two")
    assert target.read_bytes() == b"two"
    assert all(mode == 0o600 for _, mode in seen)
    assert seen[0][0] != target.with_name("f.tmp").as_posix()  # not the fixed old name
    assert [p.name for p in target.parent.iterdir()] == ["f"]  # nothing left behind


def test_parent_and_child_do_not_use_pickle():
    src = Path(_local.__file__).read_text()
    assert not re.search(r"\.(send|recv)\(", src)  # send/recv pickle; only *_bytes are used
    assert "pickle" not in src
