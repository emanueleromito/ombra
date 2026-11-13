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

Security: the secret key is only passed to OpenFHE's Decrypt; nothing here logs or writes
plaintext values or keys.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
