# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Every {{ ... }} in the docs pages still resolves to a value from the code.

Renders each page of docs/site/ with the functions of docs/macros.py under StrictUndefined, as
`mkdocs build` does. A renamed or removed constant, preset or C++ constexpr fails here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2", reason="jinja2 is in docs/requirements.txt")

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"


def _macros():
    spec = importlib.util.spec_from_file_location("ombra_docs_macros", ROOT / "docs" / "macros.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MACROS = _macros()


def _render(text: str) -> str:
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    return env.from_string(text).render(**MACROS.FUNCTIONS)


PAGES = sorted(SITE.rglob("*.md"))


@pytest.mark.parametrize("page", PAGES, ids=[str(p.relative_to(SITE)) for p in PAGES])
def test_page_values_resolve(page):
    _render(page.read_text(encoding="utf-8"))


def test_some_page_uses_the_code_values():
    used = [p for p in PAGES if "{{" in p.read_text(encoding="utf-8")]
    assert used, "no page reads a value from the code"


def test_values_match_the_code():
    from ombra import params, presets  # the real modules, imported by the test run

    assert _render('{{ const("ombra.presets.FILTER_LAMBDA") }}') == str(presets.FILTER_LAMBDA)
    core = presets.PRODUCT_PRESETS["fast"]
    assert _render('{{ preset("fast", "ring_dim") }}') == str(params.get_preset(core)["ring_dim"])


@pytest.mark.parametrize(
    "planted",
    [
        '{{ const("ombra.presets.NO_SUCH_CONSTANT") }}',
        '{{ preset("no-such-preset", "ring_dim") }}',
        '{{ preset("fast", "no_such_key") }}',
        '{{ fmax("no-such-preset") }}',
        '{{ cpp("core/no/such/file.hpp", "kX") }}',
        '{{ cpp("core/include/ombra/core/expansion_tree.hpp", "kNoSuchName") }}',
        "{{ no_such_macro }}",
    ],
)
def test_a_stale_value_fails(planted):
    with pytest.raises((MACROS.DocsValueError, jinja2.UndefinedError)):
        _render(planted)


def test_cpp_reads_a_constexpr(tmp_path, monkeypatch):
    src = tmp_path / "x.hpp"
    src.write_text("// constexpr int kOld = 1;\nconstexpr std::size_t kSlots = 4'096u;\n")
    monkeypatch.setattr(MACROS, "ROOT", tmp_path)
    assert MACROS.cpp("x.hpp", "kSlots") == 4096
    with pytest.raises(MACROS.DocsValueError):
        MACROS.cpp("x.hpp", "kOld")  # only in a comment


def test_ci_image_has_the_docs_packages():
    def pins(path):
        lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
        return {s for s in (x.split("#")[0].strip() for x in lines) if "==" in s}

    docs = pins("docs/requirements.txt")
    assert docs <= pins("ci/requirements.txt"), "ci/requirements.txt must pin the docs packages"
