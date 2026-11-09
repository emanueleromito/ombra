"""T1.15: the C++ security-floor table must agree with the Python loader's table (CPU only)."""

from __future__ import annotations

import re
from pathlib import Path

from ombra import params as ombra_params

HEADER = (
    Path(__file__).resolve().parents[1]
    / "core"
    / "include"
    / "ombra"
    / "core"
    / "security_floor.hpp"
)


def test_cpp_and_python_tables_agree() -> None:
    text = HEADER.read_text(encoding="utf-8")
    block = text.split("SECURITY_TABLE_BEGIN", 1)[1].split("SECURITY_TABLE_END", 1)[0]
    cpp = {int(n): int(q) for n, q in re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*\}", block)}
    assert cpp == ombra_params.MAX_LOG2_QP_128_CLASSIC


def test_table_is_monotonic() -> None:
    items = sorted(ombra_params.MAX_LOG2_QP_128_CLASSIC.items())
    assert [q for _, q in items] == sorted(q for _, q in items)
