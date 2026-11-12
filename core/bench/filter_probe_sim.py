"""Plaintext simulation of three probe policies for filtered search (CPU only), a follow-up to the
filtered-recall benchmark.

    OMBRA_EMBED_DEVICE=cpu python core/bench/filter_probe_sim.py --out <dir> --datasets <cache dir>
        [--dataset arxiv-abstracts|banking77 ...] [--queries 1000] [--small]

The filtered-recall benchmark showed encrypted recall == plaintext recall under the same cluster
choices, so exact plaintext
scores with the same clusters, queries and filter cells answer the question without the GPU. The
loaders, embedding model, cluster counts, queries and filter cells are those of
core/bench/filtered_recall.py. Embeddings are cached under the (gitignored) dataset dir.

Policies (the same first-round floor as ceil(sqrt(clusters)) clusters), planned by
ombra.probe.ProbePlanner, the same code the client runs:
  E2 (Fast)   the shipped policy. Round 1: clusters in center-score order, zero expected matches
              skipped, add until the expected matches reach 3k (never fewer than the floor).
              Round 2: always one widening round, the next clusters with nonzero expected matches,
              2x round 1's real clusters, capped by what is left. Below 0.5% expected selectivity:
              scan all clusters (two rounds). Exactly 2 rounds for every filtered query.
  E2 (Decoy)  the same with round 1's real clusters capped at 2 * floor (Decoy's fixed padded size
              is 2 * floor requested clusters in round 1 and 4 * floor in round 2, so "clusters
              searched" for Decoy is min(C, 2f) + min(C - 2f, 4f), decoys included; recall counts
              the real plus the decoy clusters' results, as the client does).
  Strict      all clusters in two rounds: recall is exact by construction (not simulated).
  B    the original rule with the stop at 3k matches (first round = floor, widen 2x up to 8x).
  A0   (reference) the original rule, stop at k matches. Reproduces the filtered-recall
  benchmark's recall.
Superseded: "E" and "E+A" (the radius check never pruned; see the report).
For A0 and B (baselines) a query with fewer than k matches falls back to a full scan; E2 has no
fallback (fixed round count; `incomplete` instead). Latency is MODELLED, not measured (see the
report): a least-squares fit of the filtered-recall benchmark's measured per-cell latency on
clusters searched and
extra rounds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import filtered_recall as fr  # noqa: E402
from ombra.clustering import assign, cluster_vectors  # noqa: E402
from ombra.probe import ClusterStats, ProbePlanner, decoy_round_sizes  # noqa: E402

K = 10
POLICIES = {
    "A0 (basic, stop at k)": dict(policy="basic", stop=1),
    "B (basic, stop at 3k)": dict(policy="basic", stop=3),
    "E2 (Fast)": dict(policy="e2"),
    "E2 (Decoy)": dict(policy="e2", decoy=True),
}
FILTERED_RECALL_JSON = sorted(
    (HERE.parents[1] / "docs/plan/results").glob("*-filtered-recall.json")
)[-1]


def fit_latency(recall_json: dict[str, Any]) -> dict[str, float]:
    """latency_ms = base + per_cluster * clusters + per_round * extra rounds (least squares)."""
    rows, y = [], []
    for res in recall_json["results"]:
        for c in res["cells"]:
            rows.append([1.0, c["clusters_searched_mean"], c["extra_rounds_mean"]])
            y.append(c["latency_ms_mean"])
    a = np.asarray(rows)
    use = [0, 1, 2]
    while True:  # drop a term whose coefficient comes out negative (clusters and rounds correlate)
        sol, *_ = np.linalg.lstsq(a[:, use], np.asarray(y), rcond=None)
        coef = np.zeros(3)
        coef[use] = sol
        neg = [u for u in use if coef[u] < 0 and u != 0]
        if not neg:
            break
        use.remove(neg[0])
    resid = np.asarray(y) - a @ coef
    return {
        "base_ms": float(coef[0]),
        "per_cluster_ms": float(coef[1]),
        "per_extra_round_ms": float(coef[2]),
        "rmse_ms": float(np.sqrt((resid**2).mean())),
        "cells": len(y),
    }


def _nmatch(scores: list[np.ndarray]) -> int:
    return sum(len(a) for a in scores)


def simulate_cell(
    s_all: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    cs_all: np.ndarray,
    stats: ClusterStats,
    flt: dict[int, list[int]] | None,
    first: int,
    nc: int,
    pol: dict[str, Any],
    truth: list[set[int]],
) -> dict[str, float]:
    if flt is None and pol["policy"] != "basic":
        # unfiltered: round 1 = the floor, round 2 = the next 2 * floor (plain2); Fast would use
        # one round, but recall of an unfiltered query is not what this table is about
        pol = dict(policy="plain2", decoy=pol.get("decoy", False))
    members = [np.nonzero((labels == c) & mask)[0] for c in range(nc)]
    rec, clusters, rounds, strict = [], [], [], 0
    for qi in range(len(s_all)):
        cs = cs_all[qi]
        order = [int(c) for c in np.argsort(-cs, kind="stable")]
        plan = ProbePlanner(
            order,
            pol["policy"],
            K,
            first,
            stats=stats,
            filter=flt,
            cap1=2 * first if pol.get("decoy") else None,
            stop_matches=K * pol.get("stop", 1) if pol["policy"] == "basic" else None,
        )
        scores: list[np.ndarray] = []
        ids: list[np.ndarray] = []
        searched: set[int] = set()
        nrounds = 0

        def run(
            cl: list[int],
            searched: set[int] = searched,
            scores: list[np.ndarray] = scores,
            ids: list[np.ndarray] = ids,
            row: np.ndarray = s_all[qi],
        ) -> None:
            for c in cl:
                if c not in searched:
                    searched.add(c)
                    scores.append(row[members[c]])
                    ids.append(members[c])

        first_cl = plan.first_round()
        if first_cl is None:  # low expected selectivity: all clusters, two rounds
            run(order)
            strict += 1
            nrounds = 2
        elif pol["policy"] == "basic":
            run(first_cl)
            nrounds = 1
            while True:
                nxt = plan.next_round(_nmatch(scores))
                if not nxt:
                    break
                run(nxt)
                nrounds += 1
            if _nmatch(scores) < K and len(searched) < nc:  # the Strict fallback
                run(order)
                nrounds += 1
                strict += 1
        else:
            run(first_cl)
            if pol.get("decoy"):  # decoys' results count too; round sizes as the client pads them
                r1, r2 = decoy_round_sizes(first, nc)
                rng = np.random.default_rng(qi)
                rest = [c for c in order if c not in searched]
                run([int(c) for c in rng.permutation(rest)[: r1 - len(first_cl)]])
                nxt = plan.next_round(0, searched)
                run(nxt)
                rest = [c for c in order if c not in searched]
                run([int(c) for c in rng.permutation(rest)[: max(0, r2 - len(nxt))]])
            else:
                run(plan.next_round(0, searched))
            nrounds = 2
        cat = np.concatenate(scores) if scores else np.zeros(0)
        cid = np.concatenate(ids) if ids else np.zeros(0, dtype=int)
        top = set(cid[np.argsort(-cat, kind="stable")[:K]].tolist())
        rec.append(len(top & truth[qi]) / max(1, len(truth[qi])))
        clusters.append(len(searched))
        rounds.append(nrounds)
    n = len(rec)
    return {
        "recall_at_10": float(np.mean(rec)),
        "clusters_searched_mean": float(np.mean(clusters)),
        "rounds_mean": float(np.mean(rounds)),
        "rounds_max": int(max(rounds)),
        "strict_share": strict / n,
    }


def run_dataset(ds: dict[str, Any], lat: dict[str, float], nq: int) -> dict[str, Any]:
    x, fields, q = ds["x"], ds["fields"], ds["queries"][:nq]
    cl = cluster_vectors(x, ds["n_clusters"], seed=1)
    centers = fr._flush(cl.centers)
    labels = assign(
        x, centers
    )  # as the filtered-recall benchmark did (labels = assign(x, col.centers))
    nc = len(centers)
    first = math.ceil(math.sqrt(nc))
    stats = ClusterStats.build(labels, nc, fields, ds["cards"])
    s_all = q @ x.T
    cs_all = q @ centers.T
    filters = [None, *fr.build_filters(fields, ds["cards"], ds["field_names"], labels, nc)]
    cells = []
    for f in filters:
        if f is None:
            mask, flt = np.ones(len(x), dtype=bool), None
        else:
            mask = np.isin(fields[:, f["field"]], f["values"])
            flt = {f["field"]: [int(v) for v in f["values"]]}
        rows = np.nonzero(mask)[0]
        truth = [
            set(rows[np.argsort(-s_all[i, rows], kind="stable")[:K]].tolist())
            for i in range(len(q))
        ]
        out = {}
        for name, pol in POLICIES.items():
            r = simulate_cell(s_all, labels, mask, cs_all, stats, flt, first, nc, pol, truth)
            extra = r["rounds_mean"] - 1
            r["modelled_latency_ms"] = (
                lat["base_ms"]
                + lat["per_cluster_ms"] * r["clusters_searched_mean"]
                + lat["per_extra_round_ms"] * extra
            )
            out[name] = r
        label = "unfiltered" if f is None else f"{f['field_name']} {f['placement']}"
        share = 1.0 if f is None else f["share"]
        print(
            f"[{ds['name']}] {label} {share:.3f}: "
            + ", ".join(f"{n.split()[0]} {r['recall_at_10']:.3f}" for n, r in out.items()),
            flush=True,
        )
        cells.append({"filter": f, "share": share, "label": label, "policies": out})
    return {
        "dataset": ds["name"],
        "n_vectors": len(x),
        "clusters": nc,
        "first_round": first,
        "queries": len(q),
        "cells": cells,
    }


def _cache_embeddings(cache: Path) -> None:
    """Disk-cache fr.embed by the hash of its texts (the loaders call it once per text list)."""
    import hashlib

    orig = fr.embed

    def wrapped(texts: list[str]) -> np.ndarray:
        h = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
        p = cache / f"emb-{h}.npy"
        if p.exists():
            return np.load(p)
        v = orig(texts)
        np.save(p, v)
        return v

    fr.embed = wrapped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", type=Path, required=True)
    ap.add_argument("--dataset", nargs="*", default=list(fr.LOADERS))
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    nq = 20 if a.small else a.queries
    a.datasets.mkdir(parents=True, exist_ok=True)
    _cache_embeddings(a.datasets)
    a.out.mkdir(parents=True, exist_ok=True)
    recall_json = json.loads(FILTERED_RECALL_JSON.read_text())
    lat = fit_latency(recall_json)
    print("latency model", lat, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "task": "filtered-recall probe policies (plaintext simulation)",
        "k": K,
        "embedding_model": fr.MODEL,
        "latency_model": lat,
        "small": a.small,
        "results": [],
    }
    for name in a.dataset:
        ds = fr.LOADERS[name](a.datasets, a.small, nq)
        result["results"].append(run_dataset(ds, lat, nq))
        (a.out / f"filtered-recall-probe-policies{'-small' if a.small else ''}.json").write_text(
            json.dumps(result, indent=2)
        )


if __name__ == "__main__":
    main()
