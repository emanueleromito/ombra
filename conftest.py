# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Repo-wide pytest options."""

from __future__ import annotations

import os

import pytest

# Tests use cheap scrypt parameters; a real key file refuses them (ombra.keyfile, audit A-41).
os.environ.setdefault("OMBRA_ALLOW_WEAK_SCRYPT", "1")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--expansion-large",
        action="store_true",
        default=False,
        help="also run the query expansion tests at d = 384, 768, 1536 (slow, CPU minutes)",
    )
