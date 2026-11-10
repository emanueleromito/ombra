"""T2.3 / decision 0019: how large a value a filtered score can hold and still decrypt.

With the additive penalty, a decrypted slot is `score + LAMBDA * c`, with |score| <= 1 and
c <= F (the number of filter fields), so |slot| <= 1 + LAMBDA * F. This test sweeps powers of two
at levels 0, 1 and 2 of each preset (a value v put through 0, 1 or 2 multiply-by-one and rescale
steps) and prints the largest 2^k that decrypts with an error below ERR_OK. The bound it asserts is
the one the client uses to refuse a schema: F <= (HEADROOM - 1) / LAMBDA, HEADROOM taken at the
level the filtered score is decrypted (level 1) and, to be safe, also at level 2.
Synthetic scalars only.
"""

from __future__ import annotations

import pytest
from ombra import params as ombra_params

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.openfhe

LAMBDA = 4
ERR_OK = 1e-3
PRESETS = ["m0_depth2", "m0_compact_n8192"]
# Largest value that the client allows in a decrypted filtered score, per preset, with a 4 bit
# safety margin below the measured cliff (set from the sweep in the T2.3 results note).
MAX_VALUE = {"m0_depth2": 2**6, "m0_compact_n8192": 2**11}


def _sweep(preset: str) -> dict[int, int]:
    _, cc = ombra_params.make_context(preset)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    slots = 8
    one = cc.MakeCKKSPackedPlaintext([1.0] * slots)
    best: dict[int, int] = {}
    for k in range(0, 80):
        v = float(2**k)
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([v, -v] + [0.0] * (slots - 2)))
        for level in (0, 1, 2):
            if level > 0:
                ct = cc.Rescale(cc.EvalMult(ct, one))
            try:
                dec = cc.Decrypt(ct, keys.secretKey)
                dec.SetLength(2)
                got = dec.GetRealPackedValue()
                ok = (
                    abs(got[0] - v) / max(v, 1.0) < ERR_OK
                    and abs(got[1] + v) / max(v, 1.0) < ERR_OK
                )
            except Exception:  # noqa: BLE001 - OpenFHE raises its own types past the cliff
                ok = False
            if ok:
                best[level] = k
    return best


@pytest.mark.parametrize("preset", PRESETS)
def test_filtered_score_headroom(preset: str) -> None:
    best = _sweep(preset)
    print(f"preset={preset} largest_ok_power_of_two_by_level={best}")
    limit = MAX_VALUE[preset]
    for level in (1, 2):
        assert 2 ** best.get(level, -1) >= 2 * limit, (preset, level, best)
    f_max = int((limit - 1) // LAMBDA)
    print(f"preset={preset} max_value={limit} lambda={LAMBDA} F_max={f_max}")
