# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Style check of the public docs (docs/site/ and README.md): no dashes as punctuation, no filler
words, no emoji, and never "open source". Code blocks and inline code are skipped. Each hit is
reported as file:line.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "docs" / "site").rglob("*.md")) + [ROOT / "README.md"]

BANNED = [
    "seamlessly",
    "robust",
    "leverage",
    "cutting-edge",
    "delve",
    "unlock",
    "empower",
    "it's worth noting",
    "in today's world",
    "open source",
]


def _word(w: str) -> re.Pattern[str]:
    # Whole words only; a space in the phrase also matches a hyphen ("open-source").
    body = re.escape(w).replace(r"\ ", r"[\s-]+")
    return re.compile(r"(?<![\w-])" + body + r"(?![\w-])", re.IGNORECASE)


RULES = [
    ("em dash", re.compile("—")),
    ("en dash", re.compile("–")),
    ("hyphen used as a dash", re.compile(r"(?<=\w) - (?=\w)")),
    ("'not just X, but Y'", re.compile(r"\bnot just\b[^.]*\bbut\b", re.IGNORECASE)),
    ("emoji", re.compile("[\U0001f000-\U0001faff☀-➿⬀-⯿️]")),
] + [(f"'{w}'", _word(w)) for w in BANNED]


def prose_lines(text: str):
    """(line number, text) of each prose line: fenced code dropped, inline code blanked."""
    fence = None
    for n, line in enumerate(text.splitlines(), 1):
        m = re.match(r"\s*(`{3,}|~{3,})", line)
        if m:
            if fence is None:
                fence = m.group(1)
            elif m.group(1).startswith(fence[0]) and len(m.group(1)) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        yield n, re.sub(r"(`+)(.+?)\1", "``", line)


def violations(text: str) -> list[tuple[int, str]]:
    hits = []
    for n, line in prose_lines(text):
        for name, rule in RULES:
            if rule.search(line):
                hits.append((n, name))
    return hits


@pytest.mark.parametrize("path", FILES, ids=[str(p.relative_to(ROOT)) for p in FILES])
def test_docs_style(path):
    hits = violations(path.read_text(encoding="utf-8"))
    rel = path.relative_to(ROOT)
    assert not hits, "\n".join(f"{rel}:{n}: {name}" for n, name in hits)


@pytest.mark.parametrize(
    ("planted", "rule"),
    [
        ("Search is fast — really.", "em dash"),
        ("Pages 3–5.", "en dash"),
        ("It works - mostly.", "hyphen used as a dash"),
        ("We leverage the GPU.", "'leverage'"),
        ("A Robust design.", "'robust'"),
        ("Not just fast, but private.", "'not just X, but Y'"),
        ("Ombra is open source.", "'open source'"),
        ("Done ✅", "emoji"),
    ],
)
def test_planted_violation_is_found(planted, rule):
    assert (1, rule) in violations(planted)


def test_code_is_skipped():
    text = "Run `a - b` here.\n\n```\nx = 1 — 2  # leverage\n```\nPlain text.\n"
    assert violations(text) == []


def test_list_items_and_ranges_pass():
    assert violations("- a list item\n- another\n\nThe ring has 2-3 towers; well-known.\n") == []
