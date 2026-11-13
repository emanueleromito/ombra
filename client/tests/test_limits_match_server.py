# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""The Python limits and the filter headroom formula equal the proto and the C++ table (A-36).

Reads server/search/ and proto/ as text only; builds and runs nothing.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from ombra import _local, blobstore, presets, search

ROOT = Path(__file__).resolve().parents[2]
PROTO = (ROOT / "proto/ombra/v1/ombra.proto").read_text()
HPP = ROOT / "server/search/include/ombra/search/layout.hpp"
CPP = ROOT / "server/search/src/layout.cpp"
pytestmark = pytest.mark.skipif(not CPP.exists(), reason="server sources are not in this checkout")


def _proto_limit(name: str) -> int:
    return int(re.search(rf"LIMIT_{name} = (\d+);", PROTO).group(1))


def _hpp_const(name: str) -> float:
    return float(re.search(rf"{name} = ([0-9.]+);", HPP.read_text()).group(1))


def test_constants_equal_the_proto_and_the_cpp_header():
    assert presets.MAX_FILTER_FIELDS == _proto_limit("MAX_FILTER_FIELDS")
    assert presets.MAX_FILTER_FIELDS == _hpp_const("kMaxFilterFields")
    assert presets.MAX_FILTER_CARDINALITY == _proto_limit("MAX_FILTER_CARDINALITY")
    assert presets.MAX_FILTER_CARDINALITY == _hpp_const("kMaxFilterCardinality")
    assert presets.FILTER_LAMBDA == _hpp_const("kFilterLambda")
    assert search.MAX_CLUSTERS_PER_ROUND == _proto_limit("MAX_CLUSTERS_PER_ROUND")
    assert _local.MAX_IDS == _proto_limit("MAX_IDS_PER_REQUEST")
    buckets = tuple(_proto_limit(f"BLOB_BUCKET_{i}") for i in range(4))
    assert blobstore.BUCKET_SIZES == _local.BUCKETS == buckets
    assert blobstore.MAX_CHUNK_BYTES == _proto_limit("MAX_CHUNK_BYTES")


def test_the_cpp_formula_has_the_shape_the_python_port_assumes():
    src = CPP.read_text()
    assert "first_mod_bits < scale_bits + 5) return 0" in src
    assert "first_mod_bits - scale_bits - 1 - 3" in src  # presets._HEADROOM_MARGIN_BITS = 3
    assert "allowed < 1 + 2 * static_cast<std::size_t>(kFilterLambda)) return 0" in src
    assert "(allowed - 1) / static_cast<std::size_t>(kFilterLambda) - 1" in src
    assert "std::min(f, kMaxFilterFields)" in src
    assert presets._HEADROOM_MARGIN_BITS == 3


def _cpp_max_filter_fields(first: int, scale: int) -> int:
    """Line by line port of max_filter_fields in layout.cpp."""
    lam = int(_hpp_const("kFilterLambda"))
    if first < scale + 5:
        return 0
    allowed = 1 << (first - scale - 1 - 3)
    if allowed < 1 + 2 * lam:
        return 0
    return min((allowed - 1) // lam - 1, int(_hpp_const("kMaxFilterFields")))


def test_python_headroom_equals_the_cpp_port_everywhere_and_is_never_negative():
    for first in range(20, 64):
        for scale in range(20, 60):
            derived = SimpleNamespace(values={"first_mod_size": first, "scaling_mod_size": scale})
            got = presets.filter_field_limit(derived)
            assert got == _cpp_max_filter_fields(first, scale), (first, scale)
            assert got >= 0


@pytest.mark.parametrize(("preset", "expected"), [("m0_depth2", 14), ("m0_compact_n8192", 16)])
def test_the_presets_give_the_documented_headroom(preset, expected):
    from ombra import params

    derived = SimpleNamespace(values=params.get_preset(preset))
    assert presets.filter_field_limit(derived) == expected
