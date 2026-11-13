# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Values from the code for the docs pages (mkdocs-macros, see mkdocs.yml).

Pages call these functions inside {{ ... }}, so no constant is typed by hand:

    {{ const("ombra.presets.FILTER_LAMBDA") }}           a Python module attribute
    {{ preset("fast", "ring_dim") }}                     a value of a product preset
    {{ fmax("fast") }}                                   the filter field limit of a preset
    {{ cpp("core/include/ombra/core/x.hpp", "kName") }}  a constexpr from a C++ file

Each raises DocsValueError when the name no longer exists, and the build fails
(on_error_fail in mkdocs.yml). tests/test_docs_constants.py renders every page with the same
functions, so `make test` catches a stale page too.
"""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "client"
# F_max depends on the dimension only through the layout check; every supported one gives the same.
FMAX_DIMENSION = 768


class DocsValueError(LookupError):
    """A page asked for a value that the code does not have (any more)."""


def _ombra_module(name: str) -> types.ModuleType:
    # Import ombra.presets and ombra.params without ombra/__init__.py, which imports the whole
    # client (numpy, grpc, ...). The docs build needs only these small modules.
    if "ombra" not in sys.modules:
        stub = types.ModuleType("ombra")
        stub.__path__ = [str(CLIENT / "ombra")]
        sys.modules["ombra"] = stub
    return importlib.import_module(name)


def const(dotted: str) -> Any:
    """The attribute `dotted` ("ombra.presets.FILTER_LAMBDA") of a client module."""
    module, _, attr = dotted.rpartition(".")
    if not module.startswith("ombra."):
        raise DocsValueError(f"const({dotted!r}): only ombra.* modules are read")
    try:
        mod = _ombra_module(module)
    except ImportError as e:
        raise DocsValueError(f"const({dotted!r}): cannot import {module}: {e}") from e
    if not hasattr(mod, attr):
        raise DocsValueError(f"const({dotted!r}): {module} has no {attr}")
    return getattr(mod, attr)


def preset(name: str, key: str) -> Any:
    """The value `key` ("ring_dim", "batch_size", ...) of product preset `name` ("fast")."""
    presets = _ombra_module("ombra.presets")
    params = _ombra_module("ombra.params")
    core = presets.PRODUCT_PRESETS.get(name)
    if core is None:
        raise DocsValueError(f"preset({name!r}): no such product preset")
    values = params.get_preset(core)
    if key not in values:
        raise DocsValueError(f"preset({name!r}, {key!r}): the preset has no {key}")
    return values[key]


def fmax(name: str) -> int:
    """F_max, the most filter fields product preset `name` can carry."""
    presets = _ombra_module("ombra.presets")
    if name not in presets.PRODUCT_PRESETS:
        raise DocsValueError(f"fmax({name!r}): no such product preset")
    return presets.filter_field_limit(presets.derive(name, FMAX_DIMENSION))


_CONSTEXPR = r"\bconstexpr\b[^=;{{]*?\b{name}\s*(?:=\s*([^;]+?)|\{{([^}}]*)\}})\s*;"


def cpp(path: str, name: str) -> int | float | str:
    """The value of `constexpr ... name = value;` (or `name{value}`) in the C++ file `path`."""
    file = ROOT / path
    if not file.is_file():
        raise DocsValueError(f"cpp({path!r}, {name!r}): no such file")
    text = re.sub(r"//[^\n]*", "", file.read_text(encoding="utf-8"))
    m = re.search(_CONSTEXPR.format(name=re.escape(name)), text)
    if m is None:
        raise DocsValueError(f"cpp({path!r}, {name!r}): no constexpr {name} in the file")
    raw = (m.group(1) or m.group(2)).strip()
    literal = re.sub(r"(?<=\d)'(?=\d)", "", raw)  # digit separators: 1'000
    literal = re.sub(r"(?<=[0-9.])(?:[uU]?[lL]{0,2}|[fF])$", "", literal)  # suffixes: 8u, 1.0f
    for parse in (lambda s: int(s, 0), float):
        try:
            return parse(literal)
        except ValueError:
            pass
    return raw


FUNCTIONS = {"const": const, "preset": preset, "fmax": fmax, "cpp": cpp}


def define_env(env: Any) -> None:
    """Entry point of mkdocs-macros."""
    for name, fn in FUNCTIONS.items():
        env.macro(fn, name)
