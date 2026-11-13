# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Experiment 0001: where the client's CPU time goes, per OpenFHE call.

    python docs/research/0001-cpu-client-profile/run.py --out docs/research/0001-cpu-client-profile/raw/local.json

Times, for each preset, the calls the client makes per query and per stored block: encode,
encrypt, serialize, deserialize, decrypt, reading the slot values. The "result" ciphertext is made
the way the server makes one (EvalMult with relinearization, then Rescale), with OpenFHE on the
CPU, so it has the towers and scale a stage-2 block has.

It also times one candidate (variant "drop_to_1_tower"): the same result with every tower but the
first dropped by the server before download (OpenFHE LevelReduce). That candidate touches the
level budget, so it is measured here only; nothing in the product uses it.

Synthetic data only (seeded unit vectors). The secret key stays in this process.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "client"))
sys.path.insert(0, str(HERE.parent / "tools"))

from envinfo import environment  # noqa: E402

PRESETS = ("m0_depth2", "m0_compact_n8192")
SEED = 20261113


def timed(fn, reps: int, warmup: int = 2) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "reps": reps,
    }


def profile(preset: str, reps: int, tmp: Path) -> dict:
    import openfhe as ofhe
    from ombra import params as ombra_params

    p, cc = ombra_params.make_context(preset)
    slots = ombra_params.get_preset(preset)["batch_size"]
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    rng = np.random.default_rng(SEED)
    a = rng.uniform(-1, 1, slots)
    b = rng.uniform(-1, 1, slots)
    a_list, b_list = a.tolist(), b.tolist()

    out: dict = {"preset": preset, "slots": slots, "ring_dim": cc.GetRingDimension(), "steps": {}}
    steps = out["steps"]

    steps["tolist"] = timed(lambda: a.tolist(), reps)
    steps["encode"] = timed(lambda: cc.MakeCKKSPackedPlaintext(a_list), reps)
    pt = cc.MakeCKKSPackedPlaintext(a_list)
    steps["encrypt_public"] = timed(lambda: cc.Encrypt(keys.publicKey, pt), reps)
    steps["encrypt_secret"] = timed(lambda: cc.Encrypt(keys.secretKey, pt), reps)
    ct_a = cc.Encrypt(keys.publicKey, pt)
    ct_b = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(b_list))
    fresh = tmp / "fresh.bin"
    steps["serialize_fresh_file"] = timed(
        lambda: ofhe.SerializeToFile(str(fresh), ct_a, ofhe.BINARY), reps
    )
    out["fresh_bytes"] = fresh.stat().st_size
    steps["deserialize_fresh_file"] = timed(
        lambda: ofhe.DeserializeCiphertext(str(fresh), ofhe.BINARY), reps
    )

    # A stage-2 result as the server produces it: multiply, relinearize, rescale.
    res = cc.Rescale(cc.EvalMult(ct_a, ct_b))
    variants = {"as_served": res, "drop_to_1_tower": cc.LevelReduce(res, None, 1)}
    expect = a * b
    for name, ct in variants.items():
        v: dict = {}
        f = tmp / f"{name}.bin"
        ofhe.SerializeToFile(str(f), ct, ofhe.BINARY)
        v["bytes"] = f.stat().st_size
        v["level"] = ct.GetLevel()
        v["deserialize_file"] = timed(lambda: ofhe.DeserializeCiphertext(str(f), ofhe.BINARY), reps)
        v["decrypt"] = timed(lambda: cc.Decrypt(ct, keys.secretKey), reps)
        dec = cc.Decrypt(ct, keys.secretKey)
        dec.SetLength(slots)
        v["get_values"] = timed(lambda: dec.GetRealPackedValue(), reps)
        v["to_numpy"] = timed(
            lambda: np.asarray(dec.GetRealPackedValue(), dtype=np.float64), reps
        )

        def whole() -> None:
            c, _ = ofhe.DeserializeCiphertext(str(f), ofhe.BINARY)
            d = cc.Decrypt(c, keys.secretKey)
            d.SetLength(slots)
            np.asarray(d.GetRealPackedValue(), dtype=np.float64)

        v["block_total"] = timed(whole, reps)
        got = np.asarray(dec.GetRealPackedValue(), dtype=np.float64)
        v["max_abs_error"] = float(np.max(np.abs(got - expect)))
        out[name] = v

    # Candidate without any server change: the client drops the tower itself, after
    # deserializing the block as served and before Decrypt.
    f = tmp / "as_served.bin"
    c: dict = {}
    c["level_reduce"] = timed(lambda: cc.LevelReduce(res, None, 1), reps)

    def client_drop() -> np.ndarray:
        ct, _ = ofhe.DeserializeCiphertext(str(f), ofhe.BINARY)
        d = cc.Decrypt(cc.LevelReduce(ct, None, 1), keys.secretKey)
        d.SetLength(slots)
        return np.asarray(d.GetRealPackedValue(), dtype=np.float64)

    c["block_total"] = timed(client_drop, reps)
    errs = [float(np.max(np.abs(client_drop() - expect))) for _ in range(20)]
    base = []
    for _ in range(20):
        d = cc.Decrypt(res, keys.secretKey)
        d.SetLength(slots)
        base.append(float(np.max(np.abs(np.asarray(d.GetRealPackedValue()) - expect))))
    c["max_abs_error_20_runs"] = max(errs)
    c["as_served_max_abs_error_20_runs"] = max(base)
    out["client_drop_to_1_tower"] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        rows = [profile(p, args.reps, Path(tmp)) for p in PRESETS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"experiment": "0001", "seed": SEED, "environment": environment(), "rows": rows}, indent=2)
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
