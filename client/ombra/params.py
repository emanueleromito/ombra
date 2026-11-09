"""Load CKKS parameter presets from core/params/ckks_presets.json.

All parameter numbers live in that JSON file (hard rule 1); this module only reads them. A preset
whose security level is not 128-bit-or-stronger is refused before any OpenFHE object is created.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PRESETS_PATH = Path(__file__).resolve().parents[2] / "core" / "params" / "ckks_presets.json"
SUPPORTED_FORMAT_VERSION = 1

# Names of OpenFHE security levels that are at least 128-bit. Anything else (HEStd_NotSet, a
# missing key, a typo) is refused.
ACCEPTED_SECURITY_LEVELS = frozenset(
    {
        "HEStd_128_classic",
        "HEStd_192_classic",
        "HEStd_256_classic",
        "HEStd_128_quantum",
        "HEStd_192_quantum",
        "HEStd_256_quantum",
    }
)

_REQUIRED_KEYS = (
    "security_level",
    "ring_dim",
    "multiplicative_depth",
    "scaling_mod_size",
    "first_mod_size",
    "batch_size",
    "scaling_technique",
    "key_switch_technique",
)


class ParamsError(ValueError):
    """The preset is missing, malformed or below the security floor."""


def load_presets(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Read the preset file and check its format version."""
    data = json.loads((path or PRESETS_PATH).read_text(encoding="utf-8"))
    if data.get("version") != SUPPORTED_FORMAT_VERSION:
        raise ParamsError(f"unsupported preset file version: {data.get('version')!r}")
    return data["presets"]


def validate_preset(preset: dict[str, Any]) -> None:
    """Raise ParamsError unless the preset is complete and at least 128-bit secure."""
    missing = [k for k in _REQUIRED_KEYS if k not in preset]
    if missing:
        raise ParamsError(f"preset is missing keys: {missing}")
    if preset["security_level"] not in ACCEPTED_SECURITY_LEVELS:
        raise ParamsError(
            f"security level {preset['security_level']!r} is below 128 bits; refusing the preset"
        )


def get_preset(name: str, path: Path | None = None) -> dict[str, Any]:
    presets = load_presets(path)
    if name not in presets:
        raise ParamsError(f"unknown preset {name!r}")
    return presets[name]


def make_ckks_params(preset: dict[str, Any]) -> Any:
    """Validate a preset dict and return a configured openfhe CCParamsCKKSRNS."""
    validate_preset(preset)
    import openfhe as ofhe

    params = ofhe.CCParamsCKKSRNS()
    params.SetSecurityLevel(getattr(ofhe.SecurityLevel, preset["security_level"]))
    params.SetRingDim(preset["ring_dim"])
    params.SetMultiplicativeDepth(preset["multiplicative_depth"])
    params.SetScalingModSize(preset["scaling_mod_size"])
    params.SetFirstModSize(preset["first_mod_size"])
    params.SetBatchSize(preset["batch_size"])
    params.SetScalingTechnique(getattr(ofhe.ScalingTechnique, preset["scaling_technique"]))
    params.SetKeySwitchTechnique(getattr(ofhe.KeySwitchTechnique, preset["key_switch_technique"]))
    # Optional: number of key-switching digits (HYBRID). Without it OpenFHE picks its default.
    # The digit count sets the size of the special modulus P, hence log2(QP).
    if "num_large_digits" in preset:
        params.SetNumLargeDigits(preset["num_large_digits"])
    # Optional, composite scaling only (T1.5): towers per rescale and the register word size that
    # bounds every prime (32 bit limbs). Absent for all FIXEDMANUAL presets.
    if "composite_degree" in preset:
        params.SetCompositeDegree(preset["composite_degree"])
    if "register_word_size" in preset:
        params.SetRegisterWordSize(preset["register_word_size"])
    return params


def make_context(name: str, path: Path | None = None) -> tuple[Any, Any]:
    """Return (params, crypto_context) for the named preset, with PKE, KEYSWITCH and LEVELEDSHE."""
    import openfhe as ofhe

    params = make_ckks_params(get_preset(name, path))
    cc = ofhe.GenCryptoContext(params)
    for feature in (
        ofhe.PKESchemeFeature.PKE,
        ofhe.PKESchemeFeature.KEYSWITCH,
        ofhe.PKESchemeFeature.LEVELEDSHE,
    ):
        cc.Enable(feature)
    return params, cc
