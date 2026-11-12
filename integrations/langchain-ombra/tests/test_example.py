# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""The quickstart example runs end to end in local mode (needs OpenFHE)."""

import importlib.util
import shutil
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("openfhe")

pytestmark = pytest.mark.openfhe

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"


def test_quickstart_runs(tmp_path):
    spec = importlib.util.spec_from_file_location("quickstart", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        found = module.main(tmp_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)  # local mode data is large
    assert [text for text, _ in found] == ["gpu kernels"]  # the only document with topic 1
    assert found[0][1] == pytest.approx(1.0, abs=1e-3)
