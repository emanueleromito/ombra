# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Experiment 0007: before/after table from two e2e-latency files of the same run.

    python docs/research/0007-expansion-leaf-sink/analyze.py \
        docs/research/0007-expansion-leaf-sink/raw/perf-e2e-before-a100.json \
        docs/research/0007-expansion-leaf-sink/raw/perf-e2e-after-a100.json
"""

from __future__ import annotations

import json
import sys

SERVER = (
    "server_deserialize_query",
    "expansion",
    "leaf_handoff_d2d",
    "stage1_gpu",
    "stage1_d2h",
    "stage2_gpu",
    "stage2_d2h",
)


def ms(case: dict, step: str) -> float:
    return case["steps_seconds"][step]["median"] * 1e3


def main() -> None:
    before = json.load(open(sys.argv[1]))
    after = json.load(open(sys.argv[2]))
    print(f"GPU: {after['gpu_name']}; host CPU: {after['host_cpu']}; preset {after['preset']}")
    print(
        "| d | N | searched | expansion + handoff before (ms) | after (ms) | stage 2 GPU before"
        " | after | server before (ms) | after (ms) | server speedup |"
    )
    print("|" + " --- |" * 10)
    for b, a in zip(before["cases"], after["cases"], strict=True):
        assert (b["d"], b["n_vectors"], b["fraction_searched"]) == (
            a["d"],
            a["n_vectors"],
            a["fraction_searched"],
        )
        eb = ms(b, "expansion") + ms(b, "leaf_handoff_d2d")
        ea = ms(a, "expansion") + ms(a, "leaf_handoff_d2d")
        sb = sum(ms(b, s) for s in SERVER)
        sa = sum(ms(a, s) for s in SERVER)
        print(
            f"| {b['d']} | {b['n_vectors']:,} | {b['fraction_searched']:.0%} | {eb:.2f} | {ea:.2f} "
            f"| {ms(b, 'stage2_gpu'):.2f} | {ms(a, 'stage2_gpu'):.2f} | {sb:.2f} | {sa:.2f} "
            f"| {sb / sa:.2f}x |"
        )


if __name__ == "__main__":
    main()
