# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Print a Markdown table from a .jsonl raw file (one JSON object per line).

    python docs/research/tools/jsonl_table.py <raw.jsonl> [--ratio base:new ...]

Every key becomes a column, in the order of the first line. `--ratio a:b` adds a column a/b
(for example `--ratio baseline_ms:fused_ms` is the speedup of `fused` over `baseline`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--ratio", action="append", default=[])
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.raw.read_text().splitlines() if line.strip()]
    flat = []
    for r in rows:
        if "rows" in r and isinstance(r["rows"], list):  # nested: one line per inner row
            outer = {k: v for k, v in r.items() if k != "rows"}
            flat += [{**outer, **inner} for inner in r["rows"]]
        else:
            flat.append(r)
    keys = list(flat[0])
    ratios = [x.split(":") for x in args.ratio]
    head = keys + [f"{a}/{b}" for a, b in ratios]
    print("| " + " | ".join(head) + " |")
    print("|" + " --- |" * len(head))
    for r in flat:
        cells = [str(r[k]).lower() if isinstance(r[k], bool) else str(r[k]) for k in keys]
        cells += [f"{r[a] / r[b]:.2f}x" for a, b in ratios]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
