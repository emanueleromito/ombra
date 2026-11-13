# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Timing of local mode: create time and search time at 10k synthetic vectors.

    python tests/bench_local_mode.py --d 768 [--n 10000] [--queries 5]

Run on Modal with `modal run modal_app.py::cpu_run --cmd "python tests/bench_local_mode.py --d 768"`
(4 CPUs) or `::cpu_run8` (8 CPUs). Prints one JSON line; `data_bytes` is everything the local
server wrote. OMBRA_SEEDED=0 stores plain column ciphertexts instead of seeded ones (0023).
Synthetic data only; nothing is logged
about the vectors.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import ombra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--queries", type=int, default=5)
    ap.add_argument("--preset", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((a.n, a.d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    with tempfile.TemporaryDirectory() as t:
        client = ombra.Client.local(Path(t) / "data")
        t0 = time.perf_counter()
        client.create_collection(
            "bench",
            x,
            [f"v{i}" for i in range(a.n)],
            key_file=Path(t) / "bench.key",
            passphrase="bench",  # noqa: S106
            preset=a.preset,
            scrypt_params=(2**14, 8, 1),
        )
        create_s = time.perf_counter() - t0
        info = client.get_collection("bench")
        times, ok = [], 0
        for i in range(a.queries + 1):  # the first search warms the caches and is not counted
            row = int(rng.integers(0, a.n))
            q = x[row] + 0.1 * rng.standard_normal(a.d) / np.sqrt(a.d)
            q /= np.linalg.norm(q)
            s0 = time.perf_counter()
            hits = client.search("bench", q, k=10)
            dt = time.perf_counter() - s0
            if i:
                times.append(dt)
                ok += hits[0].id == f"v{row}"
        data_bytes = sum(f.stat().st_size for f in (Path(t) / "data").rglob("*") if f.is_file())
        client.close()
    print(
        json.dumps(
            {
                "d": a.d,
                "n": a.n,
                "cpus": os.cpu_count(),
                "preset": info.preset,
                "clusters": info.cluster_count,
                "seeded_columns": os.environ.get("OMBRA_SEEDED", "1") != "0",
                "data_bytes": data_bytes,
                "create_s": round(create_s, 1),
                "search_s": [round(v, 2) for v in times],
                "search_median_s": round(statistics.median(times), 2),
                "top1_is_source": f"{ok}/{len(times)}",
            }
        )
    )


if __name__ == "__main__":
    main()
