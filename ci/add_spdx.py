#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Add the SPDX license line to every tracked source file we wrote. Idempotent.

Usage: python ci/add_spdx.py [--check]   (--check only lists files that lack the line)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXPRESSION = "PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial"
TAG = "SPDX-License-Identifier:"

SLASH_EXT = {".cpp", ".hpp", ".h", ".cu", ".cuh", ".proto"}
HASH_EXT = {".py", ".sh", ".cmake"}
HASH_NAMES = {"CMakeLists.txt", "Makefile"}
WORKFLOW_DIR = ".github/workflows/"

# Skipped: generated code, redistributed third-party code, fixtures and data, and .claude/
# (agent configuration, not edited by this tooling).
SKIP_PREFIXES = (
    "packaging/ombra-openfhe/",
    "tests/fixtures/",
    ".claude/",
    "third_party/",
    "vendor/",
)
SKIP_SUFFIXES = ("_pb2.py", "_pb2_grpc.py", "_pb2.pyi")

ROOT = Path(__file__).resolve().parent.parent


def comment_prefix(path: str) -> str | None:
    """Comment marker for a source file we own, or None if the file type is not covered."""
    p = Path(path)
    if path.startswith(SKIP_PREFIXES) or path.endswith(SKIP_SUFFIXES):
        return None
    if "fixtures" in p.parts or "data" in p.parts:
        return None
    if p.suffix in SLASH_EXT:
        return "//"
    if p.suffix in HASH_EXT or p.name in HASH_NAMES or p.name.startswith("Dockerfile"):
        return "#"
    if path.startswith(WORKFLOW_DIR) and p.suffix in {".yml", ".yaml"}:
        return "#"
    return None


def tracked_source_files() -> list[tuple[str, str]]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    result = []
    for path in out:
        prefix = comment_prefix(path)
        if prefix and (ROOT / path).is_file():
            result.append((path, prefix))
    return result


def has_header(text: str) -> bool:
    return any(TAG in line for line in text.splitlines()[:5])


def keep_first(line: str, path: str) -> bool:
    """Lines that must stay above the header: shebang, encoding, Dockerfile parser directives."""
    s = line.strip()
    if line.startswith("#!"):
        return True
    if path.endswith(".py") and s.startswith("#") and "coding" in s:
        return True
    return Path(path).name.startswith("Dockerfile") and s.lower().startswith(
        ("# syntax=", "# escape=")
    )


def add_header(path: str, prefix: str) -> bool:
    file = ROOT / path
    text = file.read_bytes().decode("utf-8")
    if has_header(text):
        return False
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    n = 0
    while n < len(lines) and n < 2 and keep_first(lines[n], path):
        n += 1
    header = f"{prefix} {TAG} {EXPRESSION}{nl}"
    file.write_bytes("".join(lines[:n] + [header] + lines[n:]).encode("utf-8"))
    return True


def main() -> int:
    check = "--check" in sys.argv
    changed = 0
    for path, prefix in tracked_source_files():
        text = (ROOT / path).read_bytes().decode("utf-8")
        if has_header(text):
            continue
        changed += 1
        if check:
            print(path)
        else:
            add_header(path, prefix)
    print(f"{changed} file(s) {'missing the header' if check else 'updated'}")
    return 1 if check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
