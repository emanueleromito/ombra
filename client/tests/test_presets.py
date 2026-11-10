"""T3.3: preset and dimension mapping, refusals, and agreement with the C++ side."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from ombra import params, presets
from ombra.params import ParamsError

ROOT = Path(__file__).resolve().parents[2]
JSON = ROOT / "core" / "params" / "ckks_presets.json"
CMAKE = ROOT / "server" / "grpc" / "CMakeLists.txt"

EXPECTED = {  # product name -> (core name, ring_dim, batch_size), from decisions 0009 and 0013
    "fast": ("m0_compact_n8192", 8192, 4096),
    "balanced": ("m0_depth2", 16384, 8192),
}


@pytest.mark.parametrize("dim", presets.SUPPORTED_DIMENSIONS)
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_preset_and_dimension(name: str, dim: int) -> None:
    core, ring, batch = EXPECTED[name]
    d = presets.derive(name, dim)
    assert (d.preset, d.ring_dim, d.batch_size) == (core, ring, batch)
    assert d.columns == dim
    assert d.multiplicative_depth == 2
    assert d.batch_size == d.ring_dim // 2  # slots per ciphertext
    assert dim <= ring // 2


def test_core_names_accepted_and_default() -> None:
    assert presets.derive("m0_depth2", 384).preset == "m0_depth2"
    assert presets.derive(presets.DEFAULT_PRESET, 384).preset == "m0_depth2"


@pytest.mark.parametrize("dim", [0, 1, 383, 512, 1024, 3072, -384])
def test_unsupported_dimension_refused(dim: int) -> None:
    with pytest.raises(ParamsError, match="unsupported dimension"):
        presets.derive("fast", dim)


@pytest.mark.parametrize("name", ["precise", "nope", ""])
def test_unknown_preset_refused(name: str) -> None:
    with pytest.raises(ParamsError, match="unknown preset"):
        presets.derive(name, 384)


def _file_with(tmp_path: Path, **override: object) -> Path:
    data = json.loads(JSON.read_text())
    data["presets"]["m0_depth2"].update(override)
    p = tmp_path / "p.json"
    p.write_text(json.dumps(data))
    return p


@pytest.mark.parametrize("level", ["HEStd_NotSet", "", "HEStd_64_classic"])
def test_below_floor_refused(tmp_path: Path, level: str) -> None:
    with pytest.raises(ParamsError, match="below 128 bits"):
        presets.derive("balanced", 384, _file_with(tmp_path, security_level=level))


def test_missing_security_level_refused(tmp_path: Path) -> None:
    data = json.loads(JSON.read_text())
    del data["presets"]["m0_depth2"]["security_level"]
    p = tmp_path / "p.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ParamsError, match="missing keys"):
        presets.derive("balanced", 384, p)


def test_every_shipped_preset_meets_floor() -> None:
    for preset in params.load_presets().values():
        assert preset["security_level"] in params.ACCEPTED_SECURITY_LEVELS


def test_matches_cpp_source() -> None:
    """The C++ server table is generated from the same JSON with these keys; check both."""
    cmake = CMAKE.read_text()
    assert params.PRESETS_PATH.resolve() == JSON.resolve()
    assert "../../core/params/ckks_presets.json" in cmake
    keys = re.findall(r"presets \$\{_name\} (\w+)\)", cmake)
    assert {"ring_dim", "batch_size", "first_mod_size", "scaling_mod_size"} <= set(keys)
    raw = json.loads(JSON.read_text())["presets"]
    for core, ring, batch in EXPECTED.values():
        assert raw[core]["ring_dim"] == ring
        assert raw[core]["batch_size"] == batch
        assert presets.derive(core, 384).values == raw[core]
