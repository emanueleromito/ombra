"""Product presets and parameter derivation.

Users pick a preset name and a vector dimension; they never pick ring sizes or moduli (the 128-bit
1). Every number comes from core/params/ckks_presets.json through ombra.params, the same file the
C++ server reads (server/grpc/CMakeLists.txt generates its preset table from it), so the two sides
cannot drift. Nothing here is hand-picked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ombra import params as _params
from ombra.params import ParamsError

SUPPORTED_DIMENSIONS = (384, 768, 1536)

# Product names (docs/product-requirements.md) -> core preset. "fast" is the compact preset;
# "balanced" is the default, i.e. the baseline m0_depth2. "precise" has no
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


# Server limits (proto LIMIT_MAX_FILTER_FIELDS and LIMIT_MAX_FILTER_CARDINALITY).
MAX_FILTER_FIELDS = 16
MAX_FILTER_CARDINALITY = 64
FILTER_LAMBDA = 4  # the penalty weight of one filter entry
_HEADROOM_MARGIN_BITS = 3  # allowed value = textbook bound at the lowest level, less 3 bits


def filter_field_limit(derived: Derived) -> int:
    """F_max of a preset, derived from its first modulus and scale bits.

    A decrypted slot is score + lambda * c. A tombstoned slot reaches 1 + lambda * (F + 1). That
    must stay under the allowed value 2^(first_mod - scale - 1 - 3), the bound at the lowest level
    less a margin. Capped by the server's LIMIT_MAX_FILTER_FIELDS.
    """
    bits = derived.values["first_mod_size"] - derived.values["scaling_mod_size"] - 1
    allowed = 2 ** (bits - _HEADROOM_MARGIN_BITS)
    return min(MAX_FILTER_FIELDS, (allowed - 1) // FILTER_LAMBDA - 1)


def check_filter_schema(derived: Derived, cards: Sequence[int]) -> None:
    """Refuse a filter schema the preset cannot carry (field count, cardinality, query length)."""
    fmax = filter_field_limit(derived)
    if len(cards) > fmax:
        raise ParamsError(f"{len(cards)} filter fields; preset {derived.preset} allows {fmax}")
    if any(not 1 <= c <= MAX_FILTER_CARDINALITY for c in cards):
        raise ParamsError(f"a filter field needs 1 to {MAX_FILTER_CARDINALITY} values")
    if derived.dimension + sum(cards) > derived.ring_dim // 2:
        raise ParamsError("dimension plus filter values exceed half the ring dimension")
