# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Recall-based acceptance check of a preset (measurement and acceptance rule): recall@10 of the
encrypted pipeline against plaintext search.

    python core/bench/recall.py [--preset m0_depth2] [--dataset NAME ...|all] [--n 100000]
                                [--queries 1000] [--out <dir>] [--datasets <dir>] [--small]

One command on Modal (A100):  modal run modal_app.py::recall  (options: --preset, --dataset,
--n, --queries, --dry-run). Default datasets: one per dimension 1536, 384 and 768 (see DATASETS).

A preset PASSES if, on at least two datasets with at least 1,000 queries each, recall@10 under
encryption (no re-ranking) is within 0.005 of the plaintext recall at the same cluster settings, for
every setting (full scan and each nlist x nprobe). The score error is reported as a safety margin,
not as the rule. The rule is in recall_check.py. Exit code 0 for PASS or NOT ELIGIBLE (a run too
small to count, such as --small), 1 for FAIL. The vectors per block (slots) come from the preset.

Real pipeline end to end. The client (this process) encrypts the column-packed
database blocks and the encrypted k-means centers with the PUBLIC key; each query goes through
`oq.encode_query`, GPU expansion and GPU column-packed search (C++ driver `ombra_recall_search`);
then OpenFHE Relinearize, Rescale, an all-ones encrypted mask at level 1, Rescale (scores at the
real final level 2); the client decrypts and ranks. Only public material (context, public
evaluation keys, ciphertexts) is written to disk; the secret key stays in this process and is never
serialized. Only aggregates are printed or stored: recall, times, counts, max errors. Never
vectors, queries, documents or scores.

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
import csv
import gzip
import io
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import recall_check  # noqa: E402

PRESET = "m0_depth2"
FLUSH = 1e-9
K = 10
RERANK_MULT = 3
SEED = 20260108
KMEANS_ITERS = 20
DEFAULT_NLISTS = (100, 316, 1000)
DEFAULT_NPROBES = (1, 2, 5, 10, 20, 50, 100)
ERROR_BOUND = 1e-6  # same agreed score bound as the CPU baseline and the GPU tests

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


HF = "https://huggingface.co/datasets/"
SUPABASE = "Supabase/wikipedia-en-embeddings"

# Public text-embedding datasets, one entry each. `license` is the license on the dataset card.
# `pool_rows` is how many rows are read (the database and the queries are a seeded disjoint split of
# this pool); the stream readers stop after that many rows, so the pool is the start of the file.
DATASETS: dict[str, dict[str, Any]] = {
    "dbpedia-ada002-1536": {
        "kind": "parquet_shards",
        "folder": "dbpedia-openai-1M",
        "files": SHARDS,
        "column": "openai",
        "source_url": HF + DATASET_NAME,
        "model": "OpenAI text-embedding-ada-002",
        "dimension": 1536,
        "license": "MIT (dataset card)",
        "corpus": "DBpedia entities",
    },
    "wiki-simple-ada002-1536": {
        "kind": "ndjson_gz",
        "url": f"{HF}{SUPABASE}/resolve/main/wiki_openai.ndjson.gz",
        "key": "text-embedding-ada-002",
        "source_url": HF + SUPABASE,
        "model": "OpenAI text-embedding-ada-002",
        "dimension": 1536,
        "license": "MIT (dataset card; Wikipedia text under its own terms)",
        "corpus": "Simple English Wikipedia",
    },
    "wiki-simple-minilm-384": {
        "kind": "ndjson_gz",
        "url": f"{HF}{SUPABASE}/resolve/main/wiki_minilm.ndjson.gz",
        "key": "all-MiniLM-L6-v2",
        "source_url": HF + SUPABASE,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "license": "MIT (dataset card; Wikipedia text under its own terms)",
        "corpus": "Simple English Wikipedia",
    },
    "wiki-simple-gte-small-384": {
        "kind": "ndjson_gz",
        "url": f"{HF}{SUPABASE}/resolve/main/wiki_gte.ndjson.gz",
        "key": "embedding",
        "source_url": HF + SUPABASE,
        "model": "thenlper/gte-small",
        "dimension": 384,
        "license": "MIT (dataset card; Wikipedia text under its own terms)",
        "corpus": "Simple English Wikipedia",
    },
    "wiki-simple-cohere-768": {
        "kind": "csv",
        "url": f"{HF}timescale/wikipedia-22-12-simple-embeddings/resolve/main/wiki.csv",
        "column": "embedding",
        "source_url": HF + "timescale/wikipedia-22-12-simple-embeddings",
        "model": "Cohere multilingual-22-12 (as in Cohere/wikipedia-22-12-simple-embeddings)",
        "dimension": 768,
        "license": "Apache-2.0 (dataset card)",
        "corpus": "Simple English Wikipedia",
    },
    "arxiv-instructorxl-768": {
        "kind": "parquet_urls",
        "folder": "arxiv-instructorxl",
        "files": {i: f"part-0{i}-arxiv-titles-instructorxl.parquet" for i in range(3)},
        "base": f"{HF}Qdrant/arxiv-titles-instructorxl-embeddings/resolve/main/data/",
        "column": "vector",
        "source_url": HF + "Qdrant/arxiv-titles-instructorxl-embeddings",
        "model": "hkunlp/instructor-xl",
        "dimension": 768,
        "license": "not stated on the dataset card (titles from arXiv metadata)",
        "corpus": "arXiv paper titles",
    },
}
# One dataset per dimension: the default acceptance set (three datasets, at least two required).
DEFAULT_DATASETS = ("dbpedia-ada002-1536", "wiki-simple-minilm-384", "wiki-simple-cohere-768")


def dataset_meta(name: str) -> dict[str, Any]:
    """Name, source URL, model, dimension and license, recorded in every result file."""
    spec = DATASETS[name]
    keys = ("source_url", "model", "dimension", "license", "corpus")
    return {"name": name} | {k: spec[k] for k in keys}


def _download(url: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"downloading {path.name}", flush=True)
    urllib.request.urlretrieve(url, tmp)  # noqa: S310
    tmp.rename(path)


def _read_stream(spec: dict[str, Any], rows: int) -> np.ndarray:
    """Read the first `rows` embeddings of a streamed ndjson.gz or csv file (early stop)."""
    out: list[np.ndarray] = []
    with urllib.request.urlopen(spec["url"], timeout=120) as resp:  # noqa: S310
        if spec["kind"] == "ndjson_gz":
            with gzip.GzipFile(fileobj=resp) as gz:
                for line in gz:
                    out.append(np.asarray(json.loads(line)[spec["key"]], dtype=np.float32))
                    if len(out) >= rows:
                        break
        else:
            csv.field_size_limit(1 << 30)
            reader = csv.reader(io.TextIOWrapper(resp, encoding="utf-8", newline=""))
            col = next(reader).index(spec["column"])
            for row in reader:
                out.append(np.fromstring(row[col].strip("[]"), dtype=np.float32, sep=","))
                if len(out) >= rows:
                    break
    return np.stack(out)


def load_pool(
    name: str, datasets_dir: Path, rows: int, small: bool = False
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (float32 pool [R, d], dataset metadata). Downloads and caches idempotently."""
    spec = DATASETS[name]
    meta = dataset_meta(name)
    if spec["kind"] in ("parquet_shards", "parquet_urls"):
        import pyarrow.parquet as pq

        folder = datasets_dir / spec["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        ids = tuple(spec["files"])
        if small:
            ids = ids[:1]
        parts = []
        for sid in ids:
            path = folder / spec["files"][sid]
            if not path.exists():
                base = HF_BASE if spec["kind"] == "parquet_shards" else spec["base"]
                _download(base + spec["files"][sid], path)
            col = pq.read_table(path, columns=[spec["column"]]).column(spec["column"])
            col = col.combine_chunks()
            dim = len(col[0])
            parts.append(np.asarray(col.values, dtype=np.float32).reshape(-1, dim))
        pool = np.concatenate(parts)
    else:
        folder = datasets_dir / "stream-cache"
        folder.mkdir(parents=True, exist_ok=True)
        cache = folder / f"{name}-first{rows}.npy"
        if not cache.exists():
            print(f"streaming the first {rows} rows of {name}", flush=True)
            pool = _read_stream(spec, rows)
            with open(cache.with_suffix(".part"), "wb") as f:
                np.save(f, pool)
            cache.with_suffix(".part").rename(cache)
        pool = np.load(cache)
    if pool.shape[1] != spec["dimension"]:
        raise SystemExit(f"{name}: expected d={spec['dimension']}, found {pool.shape[1]}")
    return pool, meta


def load_dataset(
    name: str, datasets_dir: Path, n: int, queries: int, small: bool = False
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Normalized float64 (db, queries, metadata): a seeded disjoint split of the pool."""
    pool, meta = load_pool(name, datasets_dir, rows=2 * (n + queries), small=small)
    # Drop unusable rows: non-finite or (near) zero vectors cannot be normalized, and exact
    # duplicates would make top-10 ties that plaintext and encrypted search may break differently.
    norms = np.linalg.norm(pool, axis=1)
    keep = np.isfinite(norms) & (norms > 1e-6)
    seen: set[bytes] = set()
    for i in np.flatnonzero(keep):
        key = pool[i].tobytes()
        if key in seen:
            keep[i] = False
        seen.add(key)
    if not keep.all():
        print(f"{name}: dropped {int((~keep).sum())} of {len(keep)} rows (zero or duplicate)")
        pool = pool[keep]
    if pool.shape[0] < n + queries:
        raise SystemExit(f"{name}: only {pool.shape[0]} vectors for {n} + {queries} requested")
    perm = np.random.default_rng(SEED).permutation(pool.shape[0])
    db = pool[np.sort(perm[:n])].astype(np.float64)
    qs = pool[perm[n : n + queries]].astype(np.float64)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    # Some embeddings have a dimension that is zero up to rounding (about 1e-32 in every row for
    # all-MiniLM-L6-v2 here); OpenFHE refuses to encode such a column ("Scaling factor too
    # small"). Entries below 1e-9 move a unit-vector score by under 1e-9, far below CKKS noise.
    meta["entries_flushed_to_zero"] = int((np.abs(db) < FLUSH).sum() + (np.abs(qs) < FLUSH).sum())
    db[np.abs(db) < FLUSH] = 0.0
    qs[np.abs(qs) < FLUSH] = 0.0
    print(f"{name}: {meta['entries_flushed_to_zero']} tiny entries flushed to zero")
    return db, qs, meta


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
    centers[np.abs(centers) < FLUSH] = 0.0  # see load_dataset
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
    preset: str = PRESET,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, float]]:
    """Run the encrypted pipeline. Returns decrypted scores [Q, N], per-nlist center scores
    [Q, nlist] and max errors against numpy."""
    import openfhe as ofhe
    from ombra import params as ombra_params
    from ombra import query as oq

    n, d = db.shape
    nq = queries.shape[0]
    _, cc = ombra_params.make_context(preset)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    slots = ombra_params.get_preset(preset)["batch_size"]
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
            try:
                col = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(block[:, j].tolist()))
            except RuntimeError as e:  # aggregates of the failing column only, never its values
                v = block[:, j]
                raise RuntimeError(
                    f"{e} (block {b}, column {j}: finite={bool(np.isfinite(v).all())}, "
                    f"max_abs={float(np.nanmax(np.abs(v))):.3e}, nonzero={int((v != 0).sum())})"
                ) from None
            assert ofhe.SerializeToFile(str(in_dir / f"stored_{b}_{j}.bin"), col, ofhe.BINARY)
        print(f"encrypted block {b + 1}/{len(all_blocks)}", flush=True)
    log["encrypt_database_seconds"] = time.time() - t0

    t0 = time.time()
    for i in range(nq):
        qct = oq.encode_query(cc, keys.publicKey, queries[i])
        assert ofhe.SerializeToFile(str(in_dir / f"query_{i}.bin"), qct, ofhe.BINARY)
    log["encode_queries_seconds"] = time.time() - t0

    # The driver writes each query's results atomically (temporary name, then rename). The client
    # decrypts them while the driver works on later queries, and deletes each file after use, so the
    # disk holds the encrypted database plus a few queries' results instead of all of them.
    stats_path = work / "driver_stats.json"
    driver_log = work / "driver.log"
    ones = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(np.ones(slots).tolist(), 1, 1))
    nb = len(db_blocks)
    nbt = len(all_blocks)
    scores = np.zeros((nq, n))
    cscores = {nl: np.zeros((nq, nl)) for nl in nlists}
    err_scan = err_centers = 0.0
    decrypt_seconds = 0.0
    final_level = None
    t0 = time.time()
    with GpuMemoryPoller() as poller, open(driver_log, "w") as logf:
        proc = subprocess.Popen(  # noqa: S603
            [
                str(find_binary("ombra_recall_search")),
                str(work / "context.bin"),
                str(work / "autokeys.bin"),
                str(in_dir),
                str(out_dir),
                str(d),
                str(nbt),
                str(nq),
                str(stats_path),
            ],
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for i in range(nq):
                last = out_dir / f"scores_{i}_{nbt - 1}.bin"
                while not last.exists():  # present only once the whole query is written
                    if proc.poll() is not None and not last.exists():
                        raise SystemExit(
                            f"ombra_recall_search exited ({proc.returncode}) before query {i}:\n"
                            + driver_log.read_text()[-2000:]
                        )
                    time.sleep(0.05)
                t1 = time.time()
                for b in range(nbt):
                    path = out_dir / f"scores_{i}_{b}.bin"
                    ct, ok = ofhe.DeserializeCiphertext(str(path), ofhe.BINARY)
                    assert ok
                    path.unlink()
                    score = cc.Rescale(cc.Relinearize(ct))
                    final = cc.Rescale(cc.EvalMult(score, ones))
                    if final_level is None:
                        final_level = final.GetLevel()
                    assert final.GetLevel() == final_level
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
                        err_centers,
                        float(np.max(np.abs(cscores[nl][i] - centers[nl] @ queries[i]))),
                    )
                decrypt_seconds += time.time() - t1
                if (i + 1) % 100 == 0:
                    print(f"decrypted {i + 1}/{nq} queries", flush=True)
            code = proc.wait()
        except BaseException:
            proc.kill()
            raise
    log["gpu_search_and_decrypt_wall_seconds"] = time.time() - t0
    log["gpu_memory_peak_mib_whole_device"] = poller.peak
    print(driver_log.read_text(), end="")
    if code != 0:
        raise SystemExit(f"ombra_recall_search failed (exit {code})")
    log["driver"] = json.loads(stats_path.read_text())
    shutil.rmtree(in_dir)  # frees the disk: the encrypted database is the bulk of it
    log["decrypt_seconds"] = decrypt_seconds
    log["final_level"] = final_level
    return scores, cscores, {"scan": err_scan, "centers": err_centers}


# ---------------------------------------------------------------------------------------------
# Measurement


def measure(
    db: np.ndarray,
    queries: np.ndarray,
    nlists: tuple[int, ...],
    nprobes: tuple[int, ...],
    work: Path,
    preset: str = PRESET,
) -> dict[str, Any]:
    """Whole measurement; returns the result dict (aggregates only)."""
    from ombra import params as ombra_params

    slots = ombra_params.get_preset(preset)["batch_size"]
    t_start = time.time()
    n, d = db.shape
    nq = queries.shape[0]
    log: dict[str, Any] = {}

    truth = [topk(db @ queries[i], K) for i in range(nq)]

    t0 = time.time()
    clusters = {nl: kmeans(db, nl) for nl in nlists}
    log["kmeans_seconds"] = time.time() - t0

    scores, cscores, errs = encrypted_scan(
        db, queries, {nl: c[0] for nl, c in clusters.items()}, work, log, preset
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
        "task": "recall-acceptance",
        "preset": preset,
        "dataset_note": "no query split: db and queries are disjoint seeded subsets of a pool",
        "n_vectors": n,
        "dims": d,
        "n_queries": nq,
        "k": K,
        "rerank_candidates": RERANK_MULT * K,
        "vectors_per_block": slots,
        "kmeans": f"faiss spherical, {KMEANS_ITERS} iterations, seed {SEED}",
        "method_note": (
            "encrypted scan run once per query over all vectors; decrypted scores restricted to "
            "the probed clusters' vectors afterwards"
        ),
        "full_scan": full,
        "clustered": settings,
        "max_abs_score_error_scan": errs["scan"],
        "max_abs_score_error_center_scores": errs["centers"],
        "reference_error_bound_not_pass_fail": ERROR_BOUND,
        "timing_and_memory": {
            **log,
            "total_wall_seconds": time.time() - t_start,
            "host_max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--preset", default=PRESET, help="preset name from core/params (default: m0_depth2)"
    )
    ap.add_argument(
        "--dataset",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help=f"dataset names, or 'all' for every configured one: {', '.join(DATASETS)}",
    )
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "recall-results")
    ap.add_argument("--datasets", type=Path, default=REPO_ROOT / "data" / "datasets")
    ap.add_argument("--small", action="store_true", help="dry run: 10,000 vectors, 5 queries")
    ap.add_argument("--n", type=int, default=100_000, help="database size")
    ap.add_argument("--queries", type=int, default=recall_check.MIN_QUERIES)
    ap.add_argument("--tolerance", type=float, default=recall_check.TOLERANCE)
    ap.add_argument("--min-datasets", type=int, default=recall_check.MIN_DATASETS)
    ap.add_argument("--scratch", type=Path, default=None, help="directory for encrypted data")
    args = ap.parse_args()

    names = list(DATASETS) if args.dataset == ["all"] else args.dataset
    unknown = [x for x in names if x not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; choose from {list(DATASETS)} or 'all'")
    if args.small:
        n, nq = 10_000, 5
        nlists, nprobes = (100,), (1, 5, 20)
        suffix = "-small"
    else:
        n, nq = args.n, args.queries
        nlists, nprobes = DEFAULT_NLISTS, DEFAULT_NPROBES
        suffix = ""
    args.out.mkdir(parents=True, exist_ok=True)

    checks: dict[str, dict[str, Any]] = {}
    margins: dict[str, float] = {}
    for name in names:
        print(f"=== {name} (preset {args.preset}, N={n}, {nq} queries)", flush=True)
        db, qs, meta = load_dataset(name, args.datasets, n, nq, small=args.small)
        with tempfile.TemporaryDirectory(dir=args.scratch) as tmp:
            work = Path(tmp) / "work"
            work.mkdir()
            result = measure(db, qs, nlists, nprobes, work, args.preset)
        check = recall_check.check_dataset(result, args.tolerance)
        result["dataset"] = meta
        result["acceptance"] = check
        path = args.out / f"recall-{args.preset}-{name}{suffix}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        checks[name] = check
        margins[name] = result["max_abs_score_error_scan"]
        print(f"wrote {path}")
        one = recall_check.check_overall({name: check}, min_datasets=1)
        print(recall_check.format_report({name: check}, one, margins))
        del db, qs, result

    overall = recall_check.check_overall(checks, args.min_datasets)
    summary = {
        "task": "recall-acceptance",
        "preset": args.preset,
        "n_vectors": n,
        "tolerance": args.tolerance,
        "datasets": {name: dataset_meta(name) for name in names},
        "per_dataset": {
            name: {k: v for k, v in c.items() if k != "settings"} for name, c in checks.items()
        },
        "max_abs_score_error_scan": margins,
        "overall": overall,
    }
    summary_path = args.out / f"recall-{args.preset}-summary{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"wrote {summary_path}")
    print(recall_check.format_report(checks, overall, margins))
    sys.exit(1 if overall["verdict"] == recall_check.FAIL else 0)


if __name__ == "__main__":
    main()
