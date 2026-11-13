# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Shared block decryption for the client (PERF-A).

A served score block has 2 RNS towers (level 1 of a depth-2 preset). OpenFHE decodes a
multi-tower CKKS plaintext with big-integer CRT, which costs 4 to 8 times more than decoding a
single tower. So every ciphertext is reduced to its first tower before Decrypt.

Safety condition: the first modulus q0 must hold the value. Decrypting under q0 alone recovers
round(value * scale) correctly only while |value| * scale < q0 / 2. presets.filter_field_limit
bounds the largest value the client can produce (score 1 plus the filter band and the tombstone
penalty, lambda * (F + 1)) by 2^(first_mod - scale - 4), a factor 8 below that limit, and
tests/test_openfhe_decrypt.py checks the bound with real ciphertexts on both presets.

Validation (audit A-11, A-70): the server controls everything in a returned ciphertext, so before a
decryption `check_ciphertext` compares it with the shapes this collection can legitimately produce
(`ExpectedShapes`: key tag, two components, noise degree 1, level and scale of a stored column or
of a rescaled score block, and the serialized size of the level, which is the real tower count:
openfhe-python does not expose the towers directly). Anything else is refused before Decrypt.

Security: the secret key is only passed to OpenFHE's Decrypt; nothing here logs or writes
plaintext values or keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SIZE_TOLERANCE = 0.02  # serialized size vs the reference of the same level (a tower is >= 1/3)
SCALE_TOLERANCE = 1e-6  # relative


class CiphertextRefusedError(ValueError):
    """A ciphertext returned by the server does not have the shape this collection produces."""


@dataclass(frozen=True)
class ExpectedShapes:
    """What a ciphertext may look like: its key tag and, per allowed level, scale and byte size."""

    key_tag: str
    depth: int
    shapes: tuple[tuple[int, float, int], ...]  # (level, scale, serialized bytes)

    @classmethod
    def of_context(cls, cc: Any, pk: Any) -> ExpectedShapes:
        """Reference shapes made with the client's own keys: a fresh ciphertext (level 0, the
        stored columns) and a product with one rescale (level 1, the score blocks)."""
        import openfhe as ofhe

        fresh = cc.Encrypt(pk, cc.MakeCKKSPackedPlaintext([0.0]))
        shapes = [
            (fresh.GetLevel(), fresh.GetScalingFactor(), len(ofhe.Serialize(fresh, ofhe.BINARY)))
        ]
        if cc.GetMultiplicativeDepth() >= 1:
            prod = cc.Rescale(cc.EvalMult(fresh, fresh))
            shapes.append(
                (prod.GetLevel(), prod.GetScalingFactor(), len(ofhe.Serialize(prod, ofhe.BINARY)))
            )
        return cls(pk.GetKeyTag(), cc.GetMultiplicativeDepth(), tuple(shapes))


def check_ciphertext(expected: ExpectedShapes, ct: Any, nbytes: int) -> None:
    """Raise CiphertextRefusedError unless `ct` (serialized to `nbytes`) matches `expected`."""
    if ct.GetKeyTag() != expected.key_tag:
        raise CiphertextRefusedError("ciphertext is under another key")
    if len(ct.GetElements()) != 2 or ct.NumberCiphertextElements() != 2:
        raise CiphertextRefusedError("ciphertext does not have exactly two components")
    if ct.GetNoiseScaleDeg() != 1:
        raise CiphertextRefusedError("ciphertext has an unexpected noise degree")
    scale = ct.GetScalingFactor()
    for level, want_scale, want_bytes in expected.shapes:
        if (
            ct.GetLevel() == level
            and abs(scale - want_scale) <= SCALE_TOLERANCE * abs(want_scale)
            and abs(nbytes - want_bytes) <= SIZE_TOLERANCE * want_bytes
        ):
            return
    raise CiphertextRefusedError("ciphertext has an unexpected level, scale or tower count")


def tower_count(cc: Any, ct: Any) -> int:
    """RNS towers of a ciphertext: depth + 1 - level (FIXEDMANUAL, one tower dropped per level)."""
    return cc.GetMultiplicativeDepth() + 1 - ct.GetLevel()


def decrypt_values(cc: Any, sk: Any, ct: Any, slots: int) -> np.ndarray:
    """Decrypt `ct` to `slots` float64 values, dropping all but the first tower beforehand."""
    towers = tower_count(cc, ct)
    if towers > 1:
        ct = cc.LevelReduce(ct, None, towers - 1)
    dec = cc.Decrypt(ct, sk)
    dec.SetLength(slots)
    return np.asarray(dec.GetRealPackedValue(), dtype=np.float64)


def usable_cpus() -> int:
    """CPUs this process may use; os.sched_getaffinity exists on Linux only."""
    import os

    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1
