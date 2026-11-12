"""The Modal upload filter excludes every secret pattern from .gitignore (F12)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from modal_upload import ALLOWED_TOP_LEVEL, is_excluded  # noqa: E402


def _secret_patterns() -> list[str]:
    """Patterns in the first (secrets) section of .gitignore, negations skipped."""
    lines = (ROOT / ".gitignore").read_text().splitlines()
    out: list[str] = []
    seen_header = False
    for line in lines:
        if line.startswith("# Secrets"):
            seen_header = True
            continue
        if seen_header and line.startswith("# Data"):
            break
        if seen_header and line.strip() and not line.startswith(("#", "!")):
            out.append(line.strip())
    return out


def _sample(pattern: str) -> str:
    """A concrete path that matches a gitignore pattern."""
    p = pattern.removeprefix("**/")
    if p.endswith("/"):
        return p + "file.bin"
    return p.replace("*", "x")


def test_secret_section_parsed() -> None:
    pats = _secret_patterns()
    for expected in (
        ".env",
        ".env.*",
        "*.key",
        "*.pem",
        "*.keyfile",
        "*.ombrakey",
        "keys/",
        "secrets/",
        ".modal.toml",
    ):
        assert expected in pats


@pytest.mark.parametrize("folder", ["", "core/", "client/", "tests/sub/"])
@pytest.mark.parametrize("pattern", _secret_patterns())
def test_secret_patterns_excluded(pattern: str, folder: str) -> None:
    path = folder + _sample(pattern)
    assert is_excluded(path), path


@pytest.mark.parametrize(
    "path",
    [
        ".claude/settings.json",
        "data/x.txt",
        ".git/config",
        "core/build/a.o",
        "client/__pycache__/a.pyc",
        "docs/plan/notes.md",
        "core/data/v.npy",
    ],
)
def test_other_paths_excluded(path: str) -> None:
    assert is_excluded(path)


@pytest.mark.parametrize(
    "path",
    [
        "core/CMakeLists.txt",
        "CMakeLists.txt",
        "pyproject.toml",
        "conftest.py",
        "Makefile",
        "modal_app.py",
        "modal_upload.py",
        "tests/test_modal_upload.py",
        "client/ombra/__init__.py",
        "core/params/ckks_presets.json",
    ],
)
def test_required_files_included(path: str) -> None:
    assert not is_excluded(path)


def test_allowlist_entries_exist() -> None:
    for entry in ALLOWED_TOP_LEVEL:
        assert (ROOT / entry).exists(), entry
