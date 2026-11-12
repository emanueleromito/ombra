# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""The quickstart page and its sample files stay in sync, and the local sample runs.

The CPU tests run in `make test` (and so in CI). The end-to-end test needs OpenFHE and
sentence-transformers and is marked `openfhe`.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PAGE = HERE.parent / "quickstart.md"
SAMPLES = sorted(HERE.glob("*_search.py"))

# python blocks of the page that are not a sample file (none today; keep this list short)
SNIPPETS: set[str] = set()

BLOCK = re.compile(r"^```python\n(.*?)^```$", re.DOTALL | re.MULTILINE)


def page_blocks() -> list[str]:
    return BLOCK.findall(PAGE.read_text())


def test_samples_exist():
    assert {s.name for s in SAMPLES} == {
        "local_search.py",
        "server_search.py",
        "langchain_search.py",
    }


def test_every_python_block_is_a_sample_or_a_listed_snippet():
    samples = {s.read_text() for s in SAMPLES}
    for block in page_blocks():
        assert block in samples or block in SNIPPETS, (
            "a python block of docs/quickstart.md differs from every file in docs/quickstart/:\n"
            + block[:200]
        )


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.name)
def test_sample_is_on_the_page_and_compiles(sample):
    assert sample.read_text() in page_blocks(), f"{sample.name} is not on the page"
    compile(sample.read_text(), str(sample), "exec")


def test_page_wording():
    text = PAGE.read_text().lower()
    assert "open source" not in text and "open-source" not in text
    assert "source-available" in text


@pytest.mark.openfhe
def test_local_search_runs(tmp_path):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("openfhe")
    # Local mode data is large and the sample leaves it in a temporary folder; point TMPDIR at
    # tmp_path so pytest owns it. Keep tmp_path short: local mode binds a unix socket there, and
    # socket paths are limited to about 100 bytes (use `--basetemp` with a short path).
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    run = subprocess.run(  # noqa: S603 - fixed arguments
        [sys.executable, str(HERE / "local_search.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    lines = [ln for ln in run.stdout.splitlines() if re.match(r"^-?\d\.\d{3}  ", ln)]
    assert len(lines) == 3
    assert lines[0].endswith("The central bank raised its policy rate.")
    scores = [float(ln.split()[0]) for ln in lines]
    assert scores == sorted(scores, reverse=True)
