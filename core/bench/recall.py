"""T0.8 accuracy: recall@10 of the encrypted pipeline against plaintext search.

    python core/bench/recall.py --out <dir> --datasets <dir> [--small]

Real pipeline, preset m0_depth2, end to end. The client (this process) encrypts the column-packed
database blocks and the encrypted k-means centers with the PUBLIC key; each query goes through
`oq.encode_query`, GPU expansion and GPU column-packed search (C++ driver `ombra_recall_search`);
then OpenFHE Relinearize, Rescale, an all-ones encrypted mask at level 1, Rescale (scores at the
real final level 2); the client decrypts and ranks. Only public material (context, public
evaluation keys, ciphertexts) is written to disk; the secret key stays in this process and is never
serialized. Only aggregates are printed or stored: recall, times, counts, max errors. Never
vectors, queries, documents or scores (hard rule 4).

Settings (recall@10 against the exact plaintext top-10 over the same subset):
  * full scan (one cluster), with and without client re-ranking;
  * client-side faiss k-means with nlist clusters: stage 1 scores the encrypted query against the
    encrypted cluster centers (one extra stored block per nlist), the client picks the top nprobe
    clusters; stage 2 restricts the candidates to those clusters. To avoid repeating the expensive
    scan, the encrypted scan runs ONCE per query over all vectors and the decrypted scores are
    restricted to the probed clusters' vectors afterwards (the same scores a restricted scan would
    return, since every vector's score is independent of the others);
  * re-ranking: take the top RERANK_MULT x k candidates by encrypted score and re-score them
    exactly with the plaintext stored vectors (the client would fetch and decrypt them).
The plaintext recall of the same cluster settings (plaintext k-means, exact scoring) separates the
loss from clustering from the loss from CKKS noise.

Dataset: KShivendu/dbpedia-entities-openai-1M on Hugging Face (OpenAI text-embedding-ada-002,
1536 dimensions, 1M vectors, parquet), because ann-benchmarks.com does not host
dbpedia-openai-1000k-angular.hdf5 (HTTP 404). It has no query split: the database is a seeded random
subset of N vectors and the queries are further seeded random vectors that are NOT in the database.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "client"))

PRESET = "m0_depth2"
K = 10
RERANK_MULT = 3
SEED = 20260108
KMEANS_ITERS = 20
DEFAULT_NLISTS = (100, 316, 1000)
DEFAULT_NPROBES = (1, 2, 5, 10, 20, 50, 100)
ERROR_BOUND = 1e-6  # same agreed score bound as T0.2 to T0.6

DATASET_NAME = "KShivendu/dbpedia-entities-openai-1M"
HF_BASE = f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/data/"
# Shards spread over the 26 (about 38k rows each), so the subset is not just the start of the file.
SHARDS = {
    0: "train-00000-of-00026-3c7b99d1c7eda36e.parquet",
    7: "train-00007-of-00026-ff1a7a93f5f19199.parquet",
    13: "train-00013-of-00026-2036414050f9f044.parquet",
    20: "train-00020-of-00026-e6ff711af402609d.parquet",
}


def find_binary(name: str) -> Path:
    env = os.environ.get("OMBRA_BENCH_BIN_DIR")
    candidates = [
        Path(env) / name if env else None,
        Path("/root/ombra-build/core") / name,
        REPO_ROOT / "build" / "core" / name,
    ]
    for c in candidates:
        if c is not None and c.is_file():
            return c
    raise SystemExit(f"{name} not found; build with `make build` (or via modal_app.py)")


# ---------------------------------------------------------------------------------------------
# Dataset


def load_dataset(
    datasets_dir: Path, n: int, queries: int, shard_ids: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Download (idempotent) the shards and return normalized (db, queries), seeded, disjoint."""
    import pyarrow.parquet as pq

    folder = datasets_dir / "dbpedia-openai-1M"
    folder.mkdir(parents=True, exist_ok=True)
    parts = []
    for sid in shard_ids:
        path = folder / SHARDS[sid]
        if not path.exists():
            tmp = path.with_suffix(".part")
            print(f"downloading shard {sid}", flush=True)
            urllib.request.urlretrieve(HF_BASE + SHARDS[sid], tmp)  # noqa: S310
            tmp.rename(path)
        col = pq.read_table(path, columns=["openai"]).column("openai").combine_chunks()
        dim = len(col[0])
        parts.append(np.asarray(col.values, dtype=np.float32).reshape(-1, dim))
    data = np.concatenate(parts)
    if data.shape[0] < n + queries:
        raise SystemExit(f"only {data.shape[0]} vectors for {n} + {queries} requested")
    perm = np.random.default_rng(SEED).permutation(data.shape[0])
    db = data[np.sort(perm[:n])].astype(np.float64)
    qs = data[perm[n : n + queries]].astype(np.float64)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    return db, qs


# ---------------------------------------------------------------------------------------------
# Plaintext helpers


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, best first."""
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def kmeans(db: np.ndarray, nlist: int) -> tuple[np.ndarray, np.ndarray]:
    """faiss spherical k-means on CPU: (unit-length centers [nlist, d], assignment [N])."""
    import faiss

    x = np.ascontiguousarray(db, dtype=np.float32)
    km = faiss.Kmeans(x.shape[1], nlist, niter=KMEANS_ITERS, seed=SEED, spherical=True)
    km.train(x)
    _, assign = km.index.search(x, 1)
    centers = km.centroids.astype(np.float64)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return centers, assign[:, 0]


def recall_of(found: np.ndarray, truth: np.ndarray) -> float:
    return len(set(found.tolist()) & set(truth.tolist())) / len(truth)


def candidates_for(
    center_scores: np.ndarray, assign: np.ndarray, nlist: int, nprobe: int
) -> np.ndarray:
    selected = np.zeros(nlist, dtype=bool)
    selected[topk(center_scores, nprobe)] = True
    return selected[assign]  # boolean mask over the database


def search_result(
    scores: np.ndarray, cand: np.ndarray, db: np.ndarray, q: np.ndarray, rerank: bool
) -> np.ndarray:
    """Top-k ids among candidates by `scores`; with rerank, top RERANK_MULT*k re-scored exactly."""
    ids = np.flatnonzero(cand)
    m = RERANK_MULT * K if rerank else K
    top = ids[topk(scores[ids], m)]
    if rerank:
        top = top[topk(db[top] @ q, K)]
    return top


# ---------------------------------------------------------------------------------------------
# Encrypted pipeline


class GpuMemoryPoller:
    """Peak GPU memory used (MiB, whole device) sampled with nvidia-smi once per second."""

    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(  # noqa: S603
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],  # noqa: S607
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.split()
                self.peak = max([self.peak] + [int(x) for x in out])
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(1.0)

    def __enter__(self) -> GpuMemoryPoller:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def pack_blocks(matrix: np.ndarray, slots: int) -> list[np.ndarray]:
    """Split [rows, d] into blocks of `slots` rows (last one zero-padded)."""
    out = []
    for start in range(0, matrix.shape[0], slots):
        block = np.zeros((slots, matrix.shape[1]))
        part = matrix[start : start + slots]
        block[: part.shape[0]] = part
        out.append(block)
    return out


def encrypted_scan(
    db: np.ndarray,
    queries: np.ndarray,
    centers: dict[int, np.ndarray],
    work: Path,
    log: dict[str, Any],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, float]]:
    """Run the encrypted pipeline. Returns decrypted scores [Q, N], per-nlist center scores
    [Q, nlist] and max errors against numpy."""
    import openfhe as ofhe
    from ombra import params as ombra_params
    from ombra import query as oq

    n, d = db.shape
    nq = queries.shape[0]
    _, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    slots = ombra_params.get_preset(PRESET)["batch_size"]
    indices = oq.automorphism_indices(cc.GetRingDimension(), d)
    cc.ClearEvalAutomorphismKeys()  # the key store is static: hold exactly this d's indices
    cc.EvalAutomorphismKeyGen(keys.secretKey, indices)
    in_dir, out_dir = work / "in", work / "out"
    in_dir.mkdir(parents=True)
    out_dir.mkdir()
    assert ofhe.SerializeToFile(str(work / "context.bin"), cc, ofhe.BINARY)
    assert cc.SerializeEvalAutomorphismKey(str(work / "autokeys.bin"), ofhe.BINARY)

    # Blocks: database blocks first, then one centers block per nlist (nlist <= slots).
    db_blocks = pack_blocks(db, slots)
    center_blocks = []
    nlists = sorted(centers)
    for nlist in nlists:
        if nlist > slots:
            raise SystemExit("nlist must fit in one block")
        center_blocks.append(pack_blocks(centers[nlist], slots)[0])
    all_blocks = db_blocks + center_blocks
    t0 = time.time()
    for b, block in enumerate(all_blocks):
        for j in range(d):
            col = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(block[:, j].tolist()))
            assert ofhe.SerializeToFile(str(in_dir / f"stored_{b}_{j}.bin"), col, ofhe.BINARY)
        print(f"encrypted block {b + 1}/{len(all_blocks)}", flush=True)
    log["encrypt_database_seconds"] = time.time() - t0

    t0 = time.time()
    for i in range(nq):
        qct = oq.encode_query(cc, keys.publicKey, queries[i])
        assert ofhe.SerializeToFile(str(in_dir / f"query_{i}.bin"), qct, ofhe.BINARY)
    log["encode_queries_seconds"] = time.time() - t0

    stats_path = work / "driver_stats.json"
    t0 = time.time()
    with GpuMemoryPoller() as poller:
        proc = subprocess.run(  # noqa: S603
            [
                str(find_binary("ombra_recall_search")),
                str(work / "context.bin"),
                str(work / "autokeys.bin"),
                str(in_dir),
                str(out_dir),
                str(d),
                str(len(all_blocks)),
                str(nq),
                str(stats_path),
            ],
            capture_output=True,
            text=True,
        )
    log["gpu_search_driver_wall_seconds"] = time.time() - t0
    log["gpu_memory_peak_mib_whole_device"] = poller.peak
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise SystemExit(f"ombra_recall_search failed (exit {proc.returncode}): {proc.stderr}")
    log["driver"] = json.loads(stats_path.read_text())
    shutil.rmtree(in_dir)  # frees the disk: the encrypted database is the bulk of it

    # Decrypt: Relinearize, Rescale, all-ones mask at level 1, Rescale -> level 2.
    ones = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(np.ones(slots).tolist(), 1, 1))
    nb = len(db_blocks)
    scores = np.zeros((nq, n))
    cscores = {nl: np.zeros((nq, nl)) for nl in nlists}
    err_scan = err_centers = 0.0
    t0 = time.time()
    for i in range(nq):
        for b in range(len(all_blocks)):
            path = out_dir / f"scores_{i}_{b}.bin"
            ct, ok = ofhe.DeserializeCiphertext(str(path), ofhe.BINARY)
            assert ok
            path.unlink()
            score = cc.Rescale(cc.Relinearize(ct))
            final = cc.Rescale(cc.EvalMult(score, ones))
            assert final.GetLevel() == 2
            dec = cc.Decrypt(final, keys.secretKey)
            dec.SetLength(slots)
            got = np.array(dec.GetRealPackedValue())
            if b < nb:
                lo, hi = b * slots, min(n, (b + 1) * slots)
                scores[i, lo:hi] = got[: hi - lo]
            else:
                nl = nlists[b - nb]
                cscores[nl][i] = got[:nl]
        err_scan = max(err_scan, float(np.max(np.abs(scores[i] - db @ queries[i]))))
        for nl in nlists:
            err_centers = max(
                err_centers, float(np.max(np.abs(cscores[nl][i] - centers[nl] @ queries[i])))
            )
    log["decrypt_seconds"] = time.time() - t0
    log["final_level"] = 2
    return scores, cscores, {"scan": err_scan, "centers": err_centers}


# ---------------------------------------------------------------------------------------------
# Measurement


def measure(
    db: np.ndarray,
    queries: np.ndarray,
    nlists: tuple[int, ...],
    nprobes: tuple[int, ...],
    work: Path,
) -> dict[str, Any]:
    """Whole measurement; returns the result dict (aggregates only)."""
    t_start = time.time()
    n, d = db.shape
    nq = queries.shape[0]
    log: dict[str, Any] = {}

    truth = [topk(db @ queries[i], K) for i in range(nq)]

    t0 = time.time()
    clusters = {nl: kmeans(db, nl) for nl in nlists}
    log["kmeans_seconds"] = time.time() - t0

    scores, cscores, errs = encrypted_scan(
        db, queries, {nl: c[0] for nl, c in clusters.items()}, work, log
    )

    all_cand = np.ones(n, dtype=bool)
    exact = db @ queries.T  # [N, Q]

    def avg(vals: list[float]) -> float:
        return float(np.mean(vals))

    full = {"nprobe_all_clusters": True}
    for rerank in (False, True):
        key = "rerank" if rerank else "no_rerank"
        full[f"encrypted_{key}"] = avg(
            [
                recall_of(search_result(scores[i], all_cand, db, queries[i], rerank), truth[i])
                for i in range(nq)
            ]
        )
    full["plaintext"] = avg([recall_of(topk(exact[:, i], K), truth[i]) for i in range(nq)])
    full["encrypted_top10_equals_plaintext_top10_fraction"] = avg(
        [float(set(topk(scores[i], K).tolist()) == set(truth[i].tolist())) for i in range(nq)]
    )

    settings: dict[str, Any] = {}
    for nl in nlists:
        centers, assign = clusters[nl]
        per_nl: dict[str, Any] = {}
        for nprobe in sorted({min(p, nl) for p in nprobes}):
            row: dict[str, Any] = {}
            rec = {"enc_no": [], "enc_re": [], "plain": [], "plain_exactc": []}
            frac_enc: list[float] = []
            for i in range(nq):
                cand = candidates_for(cscores[nl][i], assign, nl, nprobe)
                frac_enc.append(float(cand.mean()))
                rec["enc_no"].append(
                    recall_of(search_result(scores[i], cand, db, queries[i], False), truth[i])
                )
                rec["enc_re"].append(
                    recall_of(search_result(scores[i], cand, db, queries[i], True), truth[i])
                )
                # Plaintext: exact center scores, exact vector scores (no CKKS anywhere).
                cand_p = candidates_for(centers @ queries[i], assign, nl, nprobe)
                rec["plain"].append(
                    recall_of(search_result(exact[:, i], cand_p, db, queries[i], False), truth[i])
                )
                # Encrypted vector scores with exact centers: isolates the stage-1 CKKS effect.
                rec["plain_exactc"].append(
                    recall_of(search_result(scores[i], cand_p, db, queries[i], False), truth[i])
                )
            row["encrypted_no_rerank"] = avg(rec["enc_no"])
            row["encrypted_rerank"] = avg(rec["enc_re"])
            row["plaintext_exact_scoring"] = avg(rec["plain"])
            row["encrypted_scores_exact_centers_no_rerank"] = avg(rec["plain_exactc"])
            row["mean_fraction_of_vectors_in_probed_clusters"] = avg(frac_enc)
            per_nl[str(nprobe)] = row
        settings[str(nl)] = per_nl

    return {
        "task": "T0.8",
        "preset": PRESET,
        "dataset": DATASET_NAME,
        "dataset_note": "ada-002; no query split: db and queries are disjoint seeded subsets",
        "n_vectors": n,
        "dims": d,
        "n_queries": nq,
        "k": K,
        "rerank_candidates": RERANK_MULT * K,
        "vectors_per_block": 8192,
        "kmeans": f"faiss spherical, {KMEANS_ITERS} iterations, seed {SEED}",
        "method_note": (
            "encrypted scan run once per query over all vectors; decrypted scores restricted to "
            "the probed clusters' vectors afterwards"
        ),
        "full_scan": full,
        "clustered": settings,
        "max_abs_score_error_scan": errs["scan"],
        "max_abs_score_error_center_scores": errs["centers"],
        "error_bound": ERROR_BOUND,
        "timing_and_memory": {
            **log,
            "total_wall_seconds": time.time() - t_start,
            "host_max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", type=Path, required=True)
    ap.add_argument("--small", action="store_true", help="small run: 10,000 vectors, 5 queries")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--scratch", type=Path, default=None, help="directory for encrypted data")
    args = ap.parse_args()

    if args.small:
        n, nq, shards = 10_000, 5, (0,)
        nlists, nprobes = (100,), (1, 5, 20)
        name = "t0.8-recall-small.json"
    else:
        n, nq, shards = args.n, args.queries, tuple(SHARDS)
        nlists, nprobes = DEFAULT_NLISTS, DEFAULT_NPROBES
        name = "t0.8-recall.json"
    db, qs = load_dataset(args.datasets, n, nq, shards)
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.scratch) as tmp:
        work = Path(tmp) / "work"
        work.mkdir()
        result = measure(db, qs, nlists, nprobes, work)
    (args.out / name).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out / name}")
    print(
        f"full scan recall@{K}: no_rerank={result['full_scan']['encrypted_no_rerank']:.4f} "
        f"rerank={result['full_scan']['encrypted_rerank']:.4f} "
        f"max_score_error={result['max_abs_score_error_scan']:.3e}"
    )


if __name__ == "__main__":
    main()
