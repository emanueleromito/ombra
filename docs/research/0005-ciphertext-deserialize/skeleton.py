# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Experiment 0005, part 2: is a serialized ciphertext a fixed skeleton plus the tower words?

    python docs/research/0005-ciphertext-deserialize/skeleton.py

Serializes several ciphertexts of the same shape under one key and reports the byte ranges where
they differ. If the differing ranges are exactly `components * towers` runs of `ring_dim * 8`
bytes, the words can be read by comparing the remaining bytes with a template and copying the
runs, with OpenFHE's own Deserialize as the fallback when anything else differs.
Prints sizes and offsets only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "client"))
SEED = 20261113


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 64)  # gaps under 64 bytes: equal bytes inside a run
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [(int(a), int(b - a + 1)) for a, b in zip(starts, ends)]


def main() -> None:
    import openfhe as ofhe
    from ombra import params as ombra_params

    out = []
    for preset in ("m0_depth2", "m0_compact_n8192"):
        _, cc = ombra_params.make_context(preset)
        n = cc.GetRingDimension()
        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        rng = np.random.default_rng(SEED)

        def fresh():
            v = rng.uniform(-1, 1, n // 2).tolist()
            return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(v))

        for kind, make in (("fresh", fresh), ("served", lambda: cc.Rescale(cc.EvalMult(fresh(), fresh())))):
            blobs = [np.frombuffer(ofhe.Serialize(make(), ofhe.BINARY), dtype=np.uint8) for _ in range(6)]
            same_len = len({b.size for b in blobs}) == 1
            diff = np.zeros(blobs[0].size, dtype=bool)
            for b in blobs[1:]:
                diff |= b != blobs[0]
            r = runs(diff)
            out.append(
                {
                    "preset": preset,
                    "ciphertext": kind,
                    "bytes": int(blobs[0].size),
                    "same_length": same_len,
                    "ring_dim": n,
                    "differing_runs": [{"offset": o, "length": ln} for o, ln in r],
                    "skeleton_bytes": int(blobs[0].size - sum(ln for _, ln in r)),
                }
            )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
