# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""The scripts in examples/ run, and every Python block in the getting-started pages and the
README is a piece of one of them.

The page and compile checks run in `make test`. The runs need OpenFHE and the compiled local
mode extension; they are marked `openfhe` and run in the CI image (`make test-openfhe-local`).
They use a small stand-in for sentence-transformers (tests/fixtures/fake_sentence_transformers)
unless the real package is installed, so no model is downloaded. `server_search.py` needs a
running GPU server, so it is only compiled here.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
FAKE = ROOT / "tests" / "fixtures" / "fake_sentence_transformers"
SCRIPTS = sorted(EXAMPLES.glob("*.py"))
PAGES = [
    ROOT / "README.md",
    ROOT / "docs" / "site" / "index.md",
    *sorted((ROOT / "docs" / "site" / "getting-started").glob("*.md")),
]
BLOCK = re.compile(r"^```python\n(.*?)^```$", re.DOTALL | re.MULTILINE)
SCORE_LINE = re.compile(r"^\s*-?\d\.\d{3}  ")
CENTRAL_BANK = "The central bank raised its policy rate."


def source(path: Path) -> str:
    """Script source without the SPDX line, which the pages leave out."""
    lines = path.read_text().splitlines(keepends=True)
    return "".join(ln for ln in lines if not ln.startswith("# SPDX-License-Identifier:"))


def flat(text: str) -> str:
    """Lines without their indentation, so a piece of a function compares equal to its page copy."""
    return "\n".join(ln.strip() for ln in text.splitlines()) + "\n"


def test_quickstart_shows_local_search_whole():
    page = (ROOT / "docs" / "site" / "getting-started" / "quickstart.md").read_text()
    assert source(EXAMPLES / "local_search.py") in BLOCK.findall(page)


def test_server_example_is_on_the_server_page():
    page = (ROOT / "docs" / "site" / "getting-started" / "local-or-server.md").read_text()
    assert source(EXAMPLES / "server_search.py") in BLOCK.findall(page)


def test_scripts_exist():
    assert {s.name for s in SCRIPTS} == {
        "filters_and_updates.py",
        "langchain_search.py",
        "local_search.py",
        "server_search.py",
    }


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_compiles(script):
    compile(script.read_text(), str(script), "exec")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_python_blocks_come_from_the_examples(page):
    sources = [flat(source(s)) for s in SCRIPTS]
    for block in BLOCK.findall(page.read_text()):
        assert any(flat(block) in text for text in sources), (
            f"a python block of {page.name} is not a piece of a script in examples/:\n{block[:200]}"
        )


def test_every_script_is_on_the_examples_page():
    page = (ROOT / "docs" / "site" / "getting-started" / "examples.md").read_text()
    for script in SCRIPTS:
        assert f"examples/{script.name}" in page, f"{script.name} is not described on the page"


def run_script(name: str, tmp_path: Path) -> list[str]:
    """Run an example in a subprocess with its data under tmp_path; return its output lines."""
    paths = [str(ROOT / "client"), str(ROOT / "integrations" / "langchain-ombra")]
    if importlib.util.find_spec("sentence_transformers") is None:
        paths.insert(0, str(FAKE))
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    env["PYTHONPATH"] = os.pathsep.join([*paths, env.get("PYTHONPATH", "")])
    run = subprocess.run(  # noqa: S603 - fixed arguments
        [sys.executable, str(EXAMPLES / name)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    return run.stdout.splitlines()


def needs_local_mode():
    pytest.importorskip("numpy")
    pytest.importorskip("openfhe")
    pytest.importorskip("grpc_tools")
    pytest.importorskip("faiss")
    pytest.importorskip("ombra._native", reason="the compiled extension is not built")


# Local mode binds a unix socket under TMPDIR, and socket paths are limited to about 100 bytes,
# so run with a short `--basetemp`.


@pytest.mark.openfhe
def test_local_search_runs(tmp_path):
    needs_local_mode()
    lines = [ln for ln in run_script("local_search.py", tmp_path) if SCORE_LINE.match(ln)]
    assert len(lines) == 3
    assert lines[0].endswith(CENTRAL_BANK)
    scores = [float(ln.split()[0]) for ln in lines]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.openfhe
def test_langchain_search_runs(tmp_path):
    pytest.importorskip("langchain_core")
    needs_local_mode()
    lines = [ln for ln in run_script("langchain_search.py", tmp_path) if SCORE_LINE.match(ln)]
    assert len(lines) == 2
    assert lines[0].endswith(CENTRAL_BANK)


@pytest.mark.openfhe
def test_filters_and_updates_runs(tmp_path):
    needs_local_mode()
    sections: dict[str, list[str]] = {}
    title = ""
    for ln in run_script("filters_and_updates.py", tmp_path):
        if SCORE_LINE.match(ln):
            sections[title].append(ln)
        elif not ln.startswith("Keys and encrypted data"):
            title = ln
            sections[title] = []
    assert list(sections) == [
        "no filter",
        "only topic 2 (tech)",
        "after an insert, topic 1",
        "after deleting id 3, topic 1",
        "after reopening",
    ]
    assert len(sections["no filter"]) == 3
    assert sections["no filter"][0].endswith(CENTRAL_BANK)
    assert len(sections["only topic 2 (tech)"]) == 2
    assert all("[tech]" in ln for ln in sections["only topic 2 (tech)"])
    assert all("[finance]" in ln for ln in sections["after an insert, topic 1"])
    deleted = sections["after deleting id 3, topic 1"]
    assert deleted and not any(CENTRAL_BANK in ln for ln in deleted)
    assert all("[finance]" in ln for ln in deleted)
    assert len(sections["after reopening"]) == 1
