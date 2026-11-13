# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Experiment 0006: table of the two search kernels from one column-search benchmark file.

    python docs/research/0006-gpu-lazy-search-kernel/analyze.py \
        docs/research/0006-gpu-lazy-search-kernel/raw/perf-column-search-a100.json
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    rec = json.load(open(sys.argv[1]))
    print(f"GPU: {rec['gpu_name']}; preset {rec['preset']}")
    print(
        "| d | blocks | kernel | median (ms) | min to max (ms) | vectors/s | GB/s read | lazy / mont |"
    )
    print("|" + " --- |" * 8)
    for r in rec["results"]:
        stored = r["blocks"] * r["dims"] * 2 * r["num_towers"] * r["ring_dim"] * 8
        gbps = stored / r["kernel_seconds_median"] / 1e9
        ratio = f"{r['speedup_vs_mont']:.3f}x" if "speedup_vs_mont" in r else ""
        print(
            f"| {r['dims']} | {r['blocks']} | {r['kernel']} | {r['kernel_seconds_median'] * 1e3:.2f} "
            f"| {r['kernel_seconds_min'] * 1e3:.2f} to {r['kernel_seconds_max'] * 1e3:.2f} "
            f"| {r['vectors_per_second']:.3e} | {gbps:.0f} | {ratio} |"
        )


if __name__ == "__main__":
    main()
