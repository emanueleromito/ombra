"""Repo-wide pytest options."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--expansion-large",
        action="store_true",
        default=False,
        help="also run the query expansion tests at d = 384, 768, 1536 (slow, CPU minutes)",
    )
