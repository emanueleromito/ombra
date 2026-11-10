"""Product presets and parameter derivation (T3.3).

Users pick a preset name and a vector dimension; they never pick ring sizes or moduli (hard rule
1). Every number comes from core/params/ckks_presets.json through ombra.params, the same file the
C++ server reads (server/grpc/CMakeLists.txt generates its preset table from it), so the two sides
cannot drift. Nothing here is hand-picked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ombra import params as _params
from ombra.params import ParamsError

SUPPORTED_DIMENSIONS = (384, 768, 1536)

# Product names (docs/product-requirements.md) -> core preset. "fast" is decision 0013;
# "balanced" is the default, i.e. the baseline m0_depth2 (decisions 0009, 0013). "precise" has no
# preset yet, so it is refused.
PRODUCT_PRESETS = {
    "fast": "m0_compact_n8192",
    "balanced": "m0_depth2",
}
DEFAULT_PRESET = "balanced"


@dataclass(frozen=True)
class Derived:
    """Parameters and block layout for one (preset, dimension)."""

    preset: str  # core preset name, as sent to the server
    dimension: int
    ring_dim: int
    batch_size: int  # slots per ciphertext = vectors per block
    columns: int  # ciphertexts per block = one per vector dimension
    multiplicative_depth: int
    security_level: str
    values: dict[str, Any]  # the full preset dict, for make_ckks_params


def derive(preset: str, dimension: int, path: Path | None = None) -> Derived:
    """Map (preset, dimension) to parameters and layout; raise ParamsError if not allowed."""
    if dimension not in SUPPORTED_DIMENSIONS:
        raise ParamsError(
            f"unsupported dimension {dimension!r}; supported: {list(SUPPORTED_DIMENSIONS)}"
        )
    core_name = PRODUCT_PRESETS.get(preset, preset)
    if core_name not in _params.load_presets(path):
        raise ParamsError(f"unknown preset {preset!r}; available: {sorted(PRODUCT_PRESETS)}")
    values = _params.get_preset(core_name, path)
    _params.validate_preset(values)  # refuses anything below 128-bit security
    ring_dim = values["ring_dim"]
    if dimension > ring_dim // 2:  # same bound as the server layout: D <= N/2
        raise ParamsError(f"dimension {dimension} does not fit ring dimension {ring_dim}")
    return Derived(
        preset=core_name,
        dimension=dimension,
        ring_dim=ring_dim,
        batch_size=values["batch_size"],
        columns=dimension,
        multiplicative_depth=values["multiplicative_depth"],
        security_level=values["security_level"],
        values=values,
    )
