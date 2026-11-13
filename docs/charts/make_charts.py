# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Make the SVG charts of the performance pages from the CSV files next to this script.

Every row of the CSV files names, in its `source` column, the file in docs/plan/results/ that its
numbers were copied from. Run from the repo root:

    python docs/charts/make_charts.py

It writes docs/site/performance/img/*.svg. The output is deterministic, so running it twice gives
the same bytes and `git diff` shows whether a chart is current.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "site" / "performance" / "img"
COLOR = {"balanced": "#3b5b92", "fast": "#d9822b"}
LABEL = {"balanced": "Balanced", "fast": "Fast"}
DIMS = ["384", "768", "1536"]

plt.rcParams.update(
    {
        "svg.hashsalt": "ombra",
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def read(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="") as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)


def latency() -> None:
    rows = read("latency.csv")
    cases = [
        ("100000", "5", "100k vectors, 5% searched"),
        ("1000000", "20", "1M vectors, 20% searched"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, (n, s, title) in zip(axes, cases, strict=True):
        for i, preset in enumerate(("balanced", "fast")):
            vals = [
                float(r["total_ms"])
                for d in DIMS
                for r in rows
                if r["d"] == d
                and r["n_vectors"] == n
                and r["searched_pct"] == s
                and r["preset"] == preset
            ]
            xs = [j + (i - 0.5) * 0.38 for j in range(len(DIMS))]
            bars = ax.bar(xs, vals, 0.36, color=COLOR[preset], label=LABEL[preset])
            ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
        ax.set_xticks(range(len(DIMS)), DIMS)
        ax.set_xlabel("vector dimension")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("total per query (ms)")
    axes[0].legend(frameon=False)
    fig.suptitle("Query latency on an A100, client included, no network", fontsize=10)
    fig.tight_layout()
    save(fig, "latency.svg")


def storage() -> None:
    rows = read("storage.csv")
    modes = [
        ("float32", "Plain float32", "#999999"),
        ("plaintext_db", "Plaintext-DB", "#d9822b"),
        ("encrypted", "Encrypted", "#3b5b92"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for i, (mode, label, color) in enumerate(modes):
        vals = [
            float(r["bytes_per_million"]) / 1e9
            for d in DIMS
            for r in rows
            if r["d"] == d and r["mode"] == mode
        ]
        xs = [j + (i - 1) * 0.27 for j in range(len(DIMS))]
        bars = ax.bar(xs, vals, 0.25, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(range(len(DIMS)), DIMS)
    ax.set_xlabel("vector dimension")
    ax.set_ylabel("GB per million vectors")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "storage.svg")


def cold() -> None:
    rows = [r for r in read("e2e.csv") if r["filter"] == "none"]
    labels = ["100k vectors\n(fits in the GPU cache)", "1M vectors\n(46 GiB on disk)"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for i, preset in enumerate(("balanced", "fast")):
        vals = [
            float(r["total_p50_ms"])
            for n in ("100k", "1M")
            for r in rows
            if r["preset"] == preset and r["n_vectors"] == n
        ]
        xs = [j + (i - 0.5) * 0.38 for j in range(2)]
        bars = ax.bar(xs, vals, 0.36, color=COLOR[preset], label=LABEL[preset])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_yscale("log")
    ax.set_xticks(range(2), labels)
    ax.set_ylabel("median total per query (ms, log scale)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "cold-cache.svg")


if __name__ == "__main__":
    latency()
    storage()
    cold()
