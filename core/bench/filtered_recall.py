"""T3.11: filtered recall@10 on two real public datasets, through the M3 client path.

    python core/bench/filtered_recall.py --out <dir> --datasets <dir> [--small]
        [--dataset arxiv-abstracts|banking77 ...] [--queries 1000] [--preset fast]

One command on Modal (A100):  modal run modal_app.py::filtered_recall  (options: --dry-run, ...).

Datasets (real text with real categorical attributes; embeddings are computed here, so no dataset
ships vectors):
  * arxiv-abstracts: gfissore/arxiv-abstracts-2021 (CC0-1.0). Filter fields: primary arXiv
    category (top 63 values plus "other" = 64 values) and submission year (2-year bins, 16 values).
  * banking77: PolyAI/banking77 (CC-BY-4.0), real train/test split (test = queries). Filter field:
    the intent (77 values hashed to 64 buckets, intent % 64, as decision 0019 advises for more
    than 64 values; the ground truth uses the same buckets).
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0), 384 dimensions, normalized.

Path under test: ombra.collection.Collection.create + ombra.search.run_search against a real
ombra_server (GPU), preset "fast" (m0_compact_n8192), the probe rule of decision 0019: the first
round is the unfiltered probe count (ceil(sqrt(clusters))), widen 2x only below k matches, up to
8x, then the Strict fallback. Re-ranking at 3 x k (the client default).

Per cell (dataset, filter) the script reports: recall@10 against the brute-force filtered top-10
(numpy, float64, exact), the plaintext recall under the same pruning (the same probed clusters,
exact scores, so the loss from clusters and the stopping rule is separated from the loss from CKKS
noise), extra rounds (mean, max), Strict fallback rate, incomplete rate. The unfiltered cell gives
the reference. Filters per field: value sets built greedily to a target share of the vectors
(1%, 10%, 50%), in two orders: "clustered" (values whose vectors sit in few clusters first) and
"spread" (values spread over many clusters first). The effective number of clusters the matches
occupy is reported, so the reader can see how correlated each filter is with the clusters.

Only aggregates are written. Secret keys stay in this process (hard rule 2); no vectors, queries or
scores are printed or stored (hard rule 4; the data is public text, but the rule is kept).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import re
import statistics
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "client"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import e2e_client as ec  # noqa: E402
from ombra.clustering import assign  # noqa: E402
from ombra.collection import Collection  # noqa: E402

K = 10
VERSION = "test-openfhe"
SEED = 20261111
HF = "https://huggingface.co/datasets/"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TARGETS = (0.01, 0.10, 0.50)
GPU_BUDGET_MB = 30000
HOST_BUDGET_MB = 48000


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, device="cuda")
    model.max_seq_length = 128
    v = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    return _flush(np.asarray(v, dtype=np.float64))


FLUSH = 1e-9  # OpenFHE refuses to encode denormal-sized values ("Scaling factor too small")


def _flush(v: np.ndarray) -> np.ndarray:
    v = np.where(np.abs(v) < FLUSH, 0.0, v)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _patch_clustering() -> None:
    """Flush tiny entries of the k-means centers (the client does not, see the report)."""
    import ombra.clustering as oc
    import ombra.collection as ocol

    orig = oc.cluster_vectors

    def patched(*args: Any, **kw: Any) -> Any:
        c = orig(*args, **kw)
        return oc.Clustering(centers=_flush(c.centers), labels=c.labels)

    ocol.cluster_vectors = patched


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------------------------------------
# Datasets. Each returns dict(x, fields, cards, field_names, queries, info).


def load_banking77(cache: Path, small: bool, n_queries: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    base = f"{HF}PolyAI/banking77/resolve/main/data/"
    tr, te = cache / "banking77-train.parquet", cache / "banking77-test.parquet"
    _download(base + "train-00000-of-00001.parquet", tr)
    _download(base + "test-00000-of-00001.parquet", te)
    a, b = pq.read_table(tr).to_pydict(), pq.read_table(te).to_pydict()
    rng = np.random.default_rng(SEED)
    nq = min(n_queries, len(b["text"]))
    qi = rng.choice(len(b["text"]), nq, replace=False)
    texts = list(a["text"])
    if small:
        texts = texts[:2000]
    x = embed(texts)
    q = embed([b["text"][i] for i in qi])
    intent = np.asarray(a["label"][: len(texts)], dtype=np.int64)
    return {
        "name": "banking77",
        "x": x,
        "fields": (intent % 64)[:, None],
        "cards": (64,),
        "field_names": ("intent_bucket",),
        "queries": q,
        "n_clusters": 2 if small else 6,
        "info": {
            "source": HF + "PolyAI/banking77",
            "license": "CC-BY-4.0",
            "database": "train split",
            "queries": "random test-split utterances",
            "attribute": "intent (77 values) hashed to 64 buckets, intent % 64",
        },
    }


def load_arxiv(cache: Path, small: bool, n_queries: int) -> dict[str, Any]:
    n_db = 3000 if small else 40000
    need = n_db + n_queries
    stride = 40 if small else 48
    path = cache / f"arxiv-sample-{need}-{stride}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF}gfissore/arxiv-abstracts-2021/resolve/main/arxiv-abstracts.jsonl.gz"
        rows: list[dict[str, Any]] = []
        with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310
            with gzip.GzipFile(fileobj=r) as g:
                for i, line in enumerate(io.TextIOWrapper(g, encoding="utf-8")):
                    if i % stride:
                        continue
                    d = json.loads(line)
                    cats = d.get("categories") or []
                    if isinstance(cats, str):
                        cats = cats.split()
                    if not cats or not d.get("abstract"):
                        continue
                    ident = d["id"]
                    yy = int((ident.split("/")[1] if "/" in ident else ident)[:2])
                    year = 1900 + yy if yy >= 91 else 2000 + yy
                    text = " ".join((d["title"] + ". " + d["abstract"]).split())
                    rows.append({"text": text, "cat": cats[0], "year": year})
                    if len(rows) >= need:
                        break
        path.write_text(json.dumps(rows))
    rows = json.loads(path.read_text())
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    db_rows = [rows[i] for i in order[:n_db]]
    q_rows = [rows[i] for i in order[n_db : n_db + n_queries]]
    freq: dict[str, int] = {}
    for r in db_rows:
        freq[r["cat"]] = freq.get(r["cat"], 0) + 1
    top = [c for c, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:63]]
    cat_id = {c: i for i, c in enumerate(top)}
    cat = np.array([cat_id.get(r["cat"], 63) for r in db_rows])
    year = np.array([(min(max(r["year"], 1991), 2021) - 1991) // 2 for r in db_rows])
    return {
        "name": "arxiv-abstracts",
        "x": embed([r["text"] for r in db_rows]),
        "fields": np.stack([cat, year], axis=1),
        "cards": (64, 16),
        "field_names": ("primary_category", "year_bin"),
        "queries": embed([r["text"] for r in q_rows]),
        "n_clusters": 4 if small else 20,
        "info": {
            "source": HF + "gfissore/arxiv-abstracts-2021",
            "license": "CC0-1.0",
            "database": f"{n_db} abstracts, every {stride}th row of the file",
            "queries": "other abstracts from the same sample",
            "attribute": "primary arXiv category (top 63 + other), year of the arXiv id, 2-year bins",
        },
    }


# ---------------------------------------------------------------------------------------------
# Filters


def _eff_clusters(match: np.ndarray, labels: np.ndarray, n_clusters: int) -> float:
    cnt = np.bincount(labels[match], minlength=n_clusters).astype(float)
    p = cnt[cnt > 0] / cnt.sum()
    return float(math.exp(-(p * np.log(p)).sum()))


def build_filters(
    fields: np.ndarray, cards: tuple[int, ...], names: tuple[str, ...], labels: np.ndarray, nc: int
) -> list[dict[str, Any]]:
    """Greedy value sets (OR inside one field) at about 1%, 10% and 50% of the vectors."""
    n = len(fields)
    out: list[dict[str, Any]] = []
    for fi, card in enumerate(cards):
        cnt = np.bincount(fields[:, fi], minlength=card)
        present = [v for v in range(card) if cnt[v] > 0]
        spread = {v: _eff_clusters(fields[:, fi] == v, labels, nc) for v in present}
        for placement, rev in (("clustered", False), ("spread", True)):
            # clustered: few clusters first; spread: many clusters first
            vals = sorted(present, key=lambda v: (spread[v] * (-1 if rev else 1), v))
            for target in TARGETS:
                chosen: list[int] = []
                share = 0.0
                for v in vals:
                    s = cnt[v] / n
                    if share + s > target * 1.35:
                        continue
                    chosen.append(v)
                    share += s
                    if share >= target * 0.8:
                        break
                if not chosen or not (target * 0.5 <= share <= target * 1.5):
                    continue
                match = np.isin(fields[:, fi], chosen)
                out.append(
                    {
                        "field": fi,
                        "field_name": names[fi],
                        "placement": placement,
                        "target": target,
                        "values": chosen,
                        "share": float(match.mean()),
                        "matching": int(match.sum()),
                        "effective_clusters": _eff_clusters(match, labels, nc),
                    }
                )
    # drop a repeated (field, values) pair, which happens when both orders pick the same set
    seen: set[tuple[int, tuple[int, ...]]] = set()
    uniq = []
    for f in out:
        key = (f["field"], tuple(sorted(f["values"])))
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


# ---------------------------------------------------------------------------------------------


def run_cell(
    col: Any,
    x: np.ndarray,
    fields: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
    flt: dict[str, Any] | None,
) -> dict[str, Any]:
    ids = [b"v%07d" % i for i in range(len(x))]
    if flt is None:
        mask = np.ones(len(x), dtype=bool)
        spec = None
    else:
        mask = np.isin(fields[:, flt["field"]], flt["values"])
        spec = {flt["field"]: [int(v) for v in flt["values"]]}
    rows = np.nonzero(mask)[0]
    scores = queries @ x[rows].T
    rec, rec_plain, rounds, fb, inc, clusters, lat = [], [], [], 0, 0, [], []
    for qi, q in enumerate(queries):
        truth = {ids[rows[j]] for j in np.argsort(-scores[qi], kind="stable")[:K]}
        t0 = time.perf_counter()
        res = col.search(q, K, filter=spec)
        lat.append(time.perf_counter() - t0)
        rec.append(len({h.id for h in res.hits} & truth) / max(1, len(truth)))
        # plaintext under the same pruning: exact scores of matching vectors in the probed clusters
        keep = rows[np.isin(labels[rows], res.probed)]
        top = keep[np.argsort(-(x[keep] @ q), kind="stable")[:K]]
        rec_plain.append(len({ids[j] for j in top} & truth) / max(1, len(truth)))
        rounds.append(res.rounds - 1)
        fb += bool(res.strict_fallback)
        inc += bool(res.incomplete)
        clusters.append(len(res.probed))
    nq = len(queries)
    gaps = [a - b for a, b in zip(rec, rec_plain, strict=True)]
    return {
        "queries": nq,
        "recall_at_10": statistics.fmean(rec),
        "plaintext_same_pruning_recall_at_10": statistics.fmean(rec_plain),
        "gap_encrypted_minus_plaintext": statistics.fmean(gaps),
        "worst_query_gap": min(gaps),
        "extra_rounds_mean": statistics.fmean(rounds),
        "extra_rounds_max": max(rounds),
        "strict_fallback_rate": fb / nq,
        "incomplete_rate": inc / nq,
        "clusters_searched_mean": statistics.fmean(clusters),
        "latency_ms_mean": 1000 * statistics.fmean(lat),
        "latency_ms_p95": 1000 * sorted(lat)[int(0.95 * (nq - 1))],
    }


def run_dataset(ds: dict[str, Any], preset: str, n_queries: int, tmp: Path) -> dict[str, Any]:
    x, fields, q = ds["x"], ds["fields"], ds["queries"][:n_queries]
    for m in (x, q):
        assert np.isfinite(m).all() and np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-6)
    binary = ec.find_server_binary()
    if binary is None:
        raise SystemExit("ombra_server not found; build it first")
    pb, pbg = ec.load_stubs(tmp / "stubs")
    server = ec.ServerProcess(
        binary, tmp / "data", VERSION, tmp / "server.log", GPU_BUDGET_MB, HOST_BUDGET_MB
    )
    api = ec.OmbraClient(pb, pbg, server.address, "m0_compact_n8192", VERSION)
    col = None
    try:
        ids = [b"v%07d" % i for i in range(len(x))]
        from ombra.clustering import cluster_vectors

        cl = cluster_vectors(x, ds["n_clusters"], seed=1)
        print(
            f"[{ds['name']}] x finite {bool(np.isfinite(x).all())}, centers finite "
            f"{bool(np.isfinite(cl.centers).all())}, max|c| {float(np.abs(cl.centers).max()):.3f}, "
            f"sizes {np.bincount(cl.labels).tolist()}",
            flush=True,
        )
        t0 = time.perf_counter()
        col = Collection.create(
            pb,
            api.stub,
            "t311",
            x,
            ids,
            preset=preset,
            fields=fields,
            cards=ds["cards"],
            n_clusters=ds["n_clusters"],
            openfhe_version=VERSION,
            seed=1,
        )
        build_s = time.perf_counter() - t0
        labels = assign(x, col.centers)
        nc = col.cluster_count
        print(f"[{ds['name']}] collection built in {build_s:.0f} s, {nc} clusters", flush=True)
        filters = build_filters(fields, ds["cards"], ds["field_names"], labels, nc)
        cells = [{"filter": None, **run_cell(col, x, fields, labels, q, None)}]
        print(f"[{ds['name']}] unfiltered recall {cells[0]['recall_at_10']:.4f}", flush=True)
        for f in filters:
            r = run_cell(col, x, fields, labels, q, f)
            cells.append({"filter": f, **r})
            print(
                f"[{ds['name']}] {f['field_name']} {f['placement']} {f['share']:.3f}: "
                f"recall {r['recall_at_10']:.4f} rounds+ {r['extra_rounds_mean']:.2f}",
                flush=True,
            )
        return {
            "dataset": ds["name"],
            "info": ds["info"],
            "n_vectors": len(x),
            "dimension": x.shape[1],
            "clusters": nc,
            "first_round_probe": math.ceil(math.sqrt(nc)),
            "cards": list(ds["cards"]),
            "field_names": list(ds["field_names"]),
            "build_seconds": build_s,
            "cells": cells,
        }
    finally:
        if col is not None:
            col.close()
        api.close()
        server.stop()


LOADERS = {"arxiv-abstracts": load_arxiv, "banking77": load_banking77}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", type=Path, required=True, help="download cache (not committed)")
    ap.add_argument("--dataset", nargs="*", default=list(LOADERS))
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--small", action="store_true", help="dry run: tiny sizes, 5 queries")
    a = ap.parse_args()
    _patch_clustering()
    nq = 5 if a.small else a.queries
    a.out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "task": "T3.11",
        "preset": a.preset,
        "k": K,
        "embedding_model": MODEL,
        "small": a.small,
        "results": [],
    }
    for name in a.dataset:
        ds = LOADERS[name](a.datasets, a.small, nq)
        with tempfile.TemporaryDirectory(dir="/tmp") as t:  # noqa: S108
            result["results"].append(run_dataset(ds, a.preset, nq, Path(t)))
        (a.out / f"t3.11-filtered-recall{'-small' if a.small else ''}.json").write_text(
            json.dumps(result, indent=2)
        )
    print(re.sub(r"\s+", " ", "done"))


if __name__ == "__main__":
    main()
