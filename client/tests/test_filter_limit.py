"""F_max per preset from decision 0019's formula, and the refusals of a filter schema."""

from __future__ import annotations

import pytest
from ombra import presets
from ombra.params import ParamsError


@pytest.mark.parametrize(("name", "fmax"), [("balanced", 14), ("fast", 16)])
def test_filter_field_limit_matches_decision_0019(name: str, fmax: int) -> None:
    d = presets.derive(name, 384)
    assert presets.filter_field_limit(d) == fmax
    presets.check_filter_schema(d, [2] * fmax)
    with pytest.raises(ParamsError):
        presets.check_filter_schema(d, [2] * (fmax + 1))


def test_other_schema_refusals() -> None:
    d = presets.derive("fast", 384)
    with pytest.raises(ParamsError):
        presets.check_filter_schema(d, [65])
    with pytest.raises(ParamsError):
        presets.check_filter_schema(d, [0])
    d2 = presets.derive("fast", 1536)
    presets.check_filter_schema(d2, [64] * 16)  # 1536 + 1024 leaves is within N/2 = 4096
    with pytest.raises(ParamsError):  # the query length d + sum(V) is above N/2
        presets.check_filter_schema(d2, [64] * 16 + [1] * 2600)
