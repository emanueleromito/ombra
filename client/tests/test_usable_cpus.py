# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""The worker count works where os.sched_getaffinity does not exist (macOS)."""

import os

from ombra._decrypt import usable_cpus


def test_usable_cpus_is_positive():
    assert usable_cpus() >= 1


def test_usable_cpus_without_sched_getaffinity(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert usable_cpus() == 6
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert usable_cpus() == 1
