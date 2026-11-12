"""Serial vs parallel client decrypt + top-k on synthetic level-2 blocks.

CPU only. Prints the host CPU model, os.cpu_count() and the affinity, and the median of --runs runs
after a warmup. Synthetic data; only timings are printed.

    python core/bench/topk_parallel.py [--preset m0_depth2] [--blocks 25] [--runs 5]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "client"))
from ombra import params as ombra_params  # noqa: E402
from ombra import topk  # noqa: E402


def cpu_model() -> str:
    """CPU model from /proc/cpuinfo; some sandboxes hide the name, so also give family/model."""
    fields: dict[str, str] = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            fields.setdefault(key.strip(), val.strip())
    if "model name" in fields:
        return fields["model name"]
    keys = ("vendor_id", "cpu family", "model", "stepping", "Hardware", "CPU part")
    return ", ".join(f"{k}={fields[k]}" for k in keys if k in fields) or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="m0_depth2")
    ap.add_argument("--blocks", type=int, default=25)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import openfhe as ofhe

    _, cc = ombra_params.make_context(a.preset)
    keys = cc.KeyGen()
    slots = cc.GetRingDimension() // 2
    valid_last = slots * 2 // 3
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        # distinct scores 1e-4 apart, so the ranking is well defined despite decryption noise
        flat = rng.permutation(a.blocks * slots) * 1e-4 - 20.0
        for b in range(a.blocks):
            f = Path(tmp) / f"b{b}.bin"
            v = flat[b * slots : (b + 1) * slots]
            ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(v.tolist(), 1, 2))
            assert ofhe.SerializeToFile(str(f), ct, ofhe.BINARY)
            files.append(f)

        args = (files, cc, keys.secretKey, slots, valid_last, a.k)
        ref = topk.serial_topk(*args)  # warmup, also the identity check
        par = topk.parallel_topk(*args)
        assert np.array_equal(ref[1], par[1]) and np.max(np.abs(ref[0] - par[0])) < 1e-5

        def med(fn) -> float:
            ts = []
            for _ in range(a.runs):
                t = time.perf_counter()
                fn(*args)
                ts.append(time.perf_counter() - t)
            return statistics.median(ts)

        s = med(topk.serial_topk)
        p_oneshot = med(topk.parallel_topk)  # includes forking the pool on every call
        pool_runs = {}
        n_aff = len(os.sched_getaffinity(0))
        for w in sorted({4, 8, 16, n_aff}):
            if w > n_aff:
                continue
            with topk.ParallelTopk(cc, keys.secretKey, w) as pool:
                pool.run(files, slots, valid_last, a.k)  # warmup
                ts = []
                for _ in range(a.runs):
                    t = time.perf_counter()
                    r = pool.run(files, slots, valid_last, a.k)
                    ts.append(time.perf_counter() - t)
                assert np.array_equal(r[1], ref[1])
                pool_runs[w] = statistics.median(ts)
    best_w = min(pool_runs, key=pool_runs.get)
    res = {
        "preset": a.preset,
        "blocks": a.blocks,
        "slots": slots,
        "runs": a.runs,
        "serial_median_s": s,
        "parallel_oneshot_median_s": p_oneshot,
        "parallel_pool_median_s_by_workers": pool_runs,
        "speedup_oneshot": s / p_oneshot,
        "speedup_pool_best": s / pool_runs[best_w],
        "best_workers": best_w,
        "cpu_model": cpu_model(),
        "os_cpu_count": os.cpu_count(),
        "affinity": n_aff,
    }
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        Path(a.out, "parallel-decrypt.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
