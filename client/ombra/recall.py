# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Recall tool on your own data: what does encryption cost in recall@10?

    python -m ombra.recall vectors.npy [--filter-field f.npy] [--queries q.npy | --n-queries 100]
                           [--preset fast] [--security-level fast] [--k 10] [--n-probe N]

Builds a local-mode collection (ombra.Client.local, OpenFHE on the CPU) in a temporary directory,
runs the encrypted search, and for the same clusters the encrypted query chose it scores the
vectors in the clear. Prints recall@k of both against the exact top-k and the gap, and the verdict
of the recall rule in core/bench/recall_check.py. Nothing is written out: the
temporary directory, key file included, is removed at the end, and only aggregate numbers are
printed (never vectors, queries, ids or scores). Vectors must be unit length; queries too.

The rule: the gap (encrypted minus plaintext recall) must be within 0.005, over at least 1,000
queries. The rule asks for two datasets for a preset to pass; this tool judges your one corpus,
so PASS here means "within tolerance on this corpus with enough queries". NOT ELIGIBLE: within
tolerance but fewer than 1,000 queries. Recall is measured as the search returns it, with the
client re-ranking its candidates, against the exact top-k; the plaintext figure scans exactly the
clusters the encrypted search probed.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Same values as core/bench/recall_check.py (a test keeps them equal).
TOLERANCE = 0.005
MIN_QUERIES = 1000
PASS = "PASS"  # noqa: S105 - a verdict label, not a password
FAIL = "FAIL"
NOT_ELIGIBLE = "NOT ELIGIBLE"


def verdict(gap: float, n_queries: int) -> str:
    """The recall rule on one corpus: within tolerance, with enough queries, passes."""
    if abs(gap) > TOLERANCE + 1e-12:
        return FAIL
    return PASS if n_queries >= MIN_QUERIES else NOT_ELIGIBLE


def recall_at_k(found: Sequence[int], truth: Sequence[int], k: int) -> float:
    """Share of the exact top-k (`truth`, at most k ids) found; 1.0 when there is no truth."""
    t = set(truth)
    return 1.0 if not t else len(t & set(found)) / len(t)


def _exact_top(scores: np.ndarray, rows: np.ndarray, k: int) -> list[int]:
    if len(rows) == 0:
        return []
    top = rows[np.argsort(-scores[rows], kind="stable")[:k]]
    return [int(i) for i in top]


def measure(
    vectors: np.ndarray,
    queries: np.ndarray,
    *,
    fields: np.ndarray | None = None,
    preset: str = "fast",
    security_level: str = "fast",
    k: int = 10,
    n_probe: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run both searches; return aggregates only."""
    from ombra import Client
    from ombra.clustering import assign

    n = len(vectors)
    rng = np.random.default_rng(seed)
    n_values = int(fields.max()) + 1 if fields is not None else 0
    with tempfile.TemporaryDirectory(prefix="ombra-recall-") as tmp:
        client = Client.local(Path(tmp) / "data")
        try:
            meta = [{"f": int(v)} for v in fields] if fields is not None else None
            client.create_collection(
                "recall",
                vectors,
                [str(i) for i in range(n)],
                key_file=Path(tmp) / "recall.key",
                passphrase="recall-tool",  # noqa: S106 - throwaway key in a temp dir
                preset=preset,
                security_level=security_level,
                metadata=meta,
                filter_fields={"f": n_values} if fields is not None else None,
                scrypt_params=(2**14, 8, 1),
            )
            col, _ = client._core._get("recall")
            labels = assign(vectors, col.centers)
            enc, plain, incomplete = [], [], 0
            for q in queries:
                scores = vectors @ q
                allowed = np.arange(n)
                flt = None
                if fields is not None:
                    want = int(rng.integers(n_values))
                    allowed = np.nonzero(fields == want)[0]
                    flt = {0: want}
                truth = _exact_top(scores, allowed, k)
                res = col.search(np.asarray(q, dtype=np.float64), k, filter=flt, n_probe=n_probe)
                found = [int(h.id) for h in res.hits]
                incomplete += bool(res.incomplete)
                in_probed = allowed[np.isin(labels[allowed], res.probed)]
                enc.append(recall_at_k(found, truth, k))
                plain.append(recall_at_k(_exact_top(scores, in_probed, k), truth, k))
        finally:
            client.close()
    e, p = float(np.mean(enc)), float(np.mean(plain))
    return {
        "n_vectors": n,
        "n_queries": len(queries),
        "k": k,
        "preset": preset,
        "security_level": security_level,
        "filtered": fields is not None,
        "incomplete_queries": incomplete,
        "encrypted_recall": e,
        "plaintext_recall": p,
        "gap": e - p,
        "verdict": verdict(e - p, len(queries)),
    }


def format_report(r: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"vectors={r['n_vectors']} queries={r['n_queries']} k={r['k']} preset={r['preset']} "
            f"security_level={r['security_level']} filtered={r['filtered']}",
            f"plaintext recall@{r['k']} (same clusters): {r['plaintext_recall']:.4f}",
            f"encrypted recall@{r['k']}:                 {r['encrypted_recall']:.4f}",
            f"gap (encrypted - plaintext):              {r['gap']:+.4f}",
            f"recall rule (|gap| <= {TOLERANCE}, >= {MIN_QUERIES} queries): {r['verdict']}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ombra-recall", description="Recall@10 of encrypted vs plaintext search on your data."
    )
    p.add_argument("vectors", help=".npy file, (n, d) unit-length vectors")
    p.add_argument("--filter-field", help=".npy file, (n,) integer value of one filter field")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--queries", help=".npy file, (m, d) unit-length queries")
    g.add_argument("--n-queries", type=int, default=100, help="sample this many vectors as queries")
    p.add_argument("--preset", default="fast")
    p.add_argument("--security-level", default="fast", choices=["fast", "decoy", "strict"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-probe", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.k < 1 or (a.n_probe is not None and a.n_probe < 1):
        print("error: --k and --n-probe must be at least 1", file=sys.stderr)
        return 2
    x = np.load(a.vectors, allow_pickle=False).astype(np.float64)
    if x.ndim != 2:
        print("error: vectors must be a 2-D array", file=sys.stderr)
        return 2
    fields = None
    if a.filter_field:
        fields = np.load(a.filter_field, allow_pickle=False).astype(np.int64)
        if fields.shape != (len(x),) or fields.min() < 0:
            print("error: filter field must be (n,) non-negative integers", file=sys.stderr)
            return 2
    if a.queries:
        q = np.load(a.queries, allow_pickle=False).astype(np.float64)
    else:
        rng = np.random.default_rng(a.seed)
        pick = rng.choice(len(x), size=min(a.n_queries, len(x)), replace=False)
        # sampled corpus vectors, nudged so the top hit is not trivially the vector itself
        q = x[pick] + 0.3 * rng.standard_normal((len(pick), x.shape[1])) / np.sqrt(x.shape[1])
        q /= np.linalg.norm(q, axis=1, keepdims=True)
    if q.ndim != 2 or q.shape[1] != x.shape[1]:
        print("error: queries must be (m, d) with the vectors' d", file=sys.stderr)
        return 2
    r = measure(
        x,
        q,
        fields=fields,
        preset=a.preset,
        security_level=a.security_level,
        k=a.k,
        n_probe=a.n_probe,
        seed=a.seed,
    )
    print(format_report(r))
    return 1 if r["verdict"] == FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
