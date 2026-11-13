# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Experiment 0008: one table over the GPU families from the per-family result files.

    python docs/research/0008-gpu-families/analyze.py docs/research/0008-gpu-families/raw

Reads fam-column-search-<gpu>.json and fam-expansion-<gpu>.json. Prices are Modal's per-second
GPU prices on 2026-11-13 (modal.com/pricing), GPU only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# requested name: (generation, compute capability, USD per second)
FAMILIES = {
    "t4": ("Turing", "sm_75", 0.000164),
    "a10": ("Ampere", "sm_86", 0.000306),
    "a100": ("Ampere", "sm_80", 0.000583),
    "l4": ("Ada", "sm_89", 0.000222),
    "l40s": ("Ada", "sm_89", 0.000542),
    "h100": ("Hopper", "sm_90", 0.001097),
}


def main() -> None:
    raw = Path(sys.argv[1])
    rows = []
    for name, (gen, sm, usd) in FAMILIES.items():
        cs = json.loads((raw / f"fam-column-search-{name}.json").read_text())
        ex = json.loads((raw / f"fam-expansion-{name}.json").read_text())
        search = {r["dims"]: r for r in cs["results"]}
        exp = {r["d"]: r["gpu_seconds"] * 1e3 for r in ex["results"]}
        rows.append((name, cs["gpu_name"], gen, sm, usd, search, exp))
    ref = next(r for r in rows if r[0] == "a10")
    print(
        "| GPU served | Generation | Build | Expansion d = 384 / 768 / 1536 (ms) | vs A10 "
        "| Search, vectors/s d = 384 / 1536 | vs A10 | USD/h | Expansions per USD (d = 768) |"
    )
    print("|" + " --- |" * 9)
    for _name, gpu, gen, sm, usd, search, exp in rows:
        e = "/".join(f"{exp[d]:.1f}" for d in (384, 768, 1536))
        s = " / ".join(f"{search[d]['vectors_per_second']:.2e}" for d in (384, 1536))
        print(
            f"| {gpu} | {gen} | {sm} | {e} | {ref[6][768] / exp[768]:.2f}x | {s} "
            f"| {search[384]['vectors_per_second'] / ref[5][384]['vectors_per_second']:.2f}x "
            f"| {usd * 3600:.2f} | {1 / (exp[768] / 1e3 * usd):,.0f} |"
        )


if __name__ == "__main__":
    main()
