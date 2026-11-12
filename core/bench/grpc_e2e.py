"""End-to-end latency, recall and storage through the real gRPC server (A100, Modal).

    python core/bench/run.py --task grpc-e2e --out <dir> [--presets-grpc ...] [--sizes-grpc ...]
        writes <dir>/grpc-e2e-<preset>-n<N>-a100.json, one file per (preset, size)

This process is the CLIENT (secret key held here, never written or sent). It starts `ombra_server`
on localhost like tests/test_gpu_e2e_grpc.py, drives it with tests/e2e_client.py, and measures
every step of a query. Synthetic data only (a Gaussian mixture, as in filter_measurements.py); only
aggregates are printed or stored: times, byte counts, recalls. Never vectors, queries or scores.

Setup per (preset, N): d = 384, two filter fields (8 values each, so 400 columns per block),
client-side spherical k-means (Lloyd, a few iterations) into C = round(N / (2 * slots)) clusters,
so a cluster holds about two blocks (2 to 3 with the usual imbalance) and a probe of nprobe
clusters reads about 2 * nprobe blocks. Blocks are encrypted by a pool of forked worker processes
(public key only; the same pool decrypts the answers, as in topk_parallel.py, and holds the secret
key in its
memory like client/ombra/topk.py). Collection is created, public keys uploaded, blocks inserted.

Per query (timed, after a warm-up) the steps are:
  encrypt     client: build and encrypt the query ciphertext (leaves = d + 16)
  stage1      server: from sending the query to receiving the center scores (expansion, scoring of
              the centers block, and the loopback transfer; the API has no server-time field, so
              server time and transfer are one number on localhost)
  dec1        client: decrypt the center scores and choose the clusters (top nprobe)
  stage2      from sending SearchChosen (nprobe = 2 clusters) to the last SearchScores message
              (server column-packed search over the chosen clusters' blocks + transfer);
              also the arrival of the first message
  decrypt     client: decrypt all score blocks of stage 2 in parallel processes and band-
              decode
  topk        client: merge the per-block top-10 lists
A second SearchChosen of the next 2 clusters follows on the same stream (not part of the per-query
total; it lets recall be computed for nprobe 1, 2 and 4 from one encrypted scan). Both a query
without a filter and one with a filter (both fields set) are run, Q queries each.

Recall: recall@10 of the encrypted pipeline (no re-ranking) against the exact top-10 (over the
vectors that match the filter, for the filtered variant), and the plaintext recall of the same
cluster setting (same centers, exact centers scores, exact scores), per nprobe in (1, 2, 4); the
gap is encrypted - plaintext (the recall rule uses 0.005). With 20 to 50 queries this is a sanity
check of the pipeline, not the rule's verdict (the rule needs 1,000 queries on 2 datasets).
"""

from __future__ import annotations

import faulthandler
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")  # the pool forks with a gRPC channel open
os.environ["OMP_NUM_THREADS"] = "1"  # one thread per forked worker (no oversubscription)

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "client"))
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e2e_client as ec  # noqa: E402
from ombra import params as ombra_params  # noqa: E402
from ombra.file_format import Kind, read_envelope, write_envelope  # noqa: E402
from recall import recall_of, topk  # noqa: E402

VERSION = "test-openfhe"
SEED = 20261110
K = 10
D = 384
CARDS = [8, 8]
WANTS = [2, 5]  # the filtered variant: one value per field (1 in 64 vectors)
LAMBDA = ec.LAMBDA
NOISE = 0.7  # mixture noise relative to unit-length component centers
BLOCKS_PER_CLUSTER = 2
NPROBE_TIMED = 2  # first round of every query
NPROBES = (1, 2, 4)  # recall settings; round 2 adds the clusters 3 and 4
GPU_BUDGET_MB = 24000
HOST_BUDGET_MB = 8192
BYTES_PER_VECTOR_COLUMN = 96  # ciphertext bytes per slot per column, both presets (3 limbs)
WARMUP = 3
LLOYD_ITERS = 5

# Set in the parent before the pool forks; children read their copy (never pickled).
_ST: dict[str, Any] = {}


# -------------------------------------------------------------------------------------------------
# Workers (forked: they inherit the crypto context, keys and data from the parent's memory)


def _shm_path(tag: str) -> str:
    base = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()  # noqa: S108
    return f"{base}/grpc-e2e-{os.getpid()}-{tag}.bin"


def _ct_envelope(ct: Any) -> bytes:
    import openfhe as ofhe

    st = _ST
    path = _shm_path("enc")
    try:
        assert ofhe.SerializeToFile(path, ct, ofhe.BINARY)
        raw = Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)
    return write_envelope(Kind.CIPHERTEXT, st["preset"], VERSION, raw)


class _InlinePool:
    """workers = 1: run in this process (a way to tell a pool hang from an encryption bug)."""

    class _Done:
        def __init__(self, v: Any) -> None:
            self._v = v

        def get(self) -> Any:
            return self._v

    def apply_async(self, fn: Any, args: tuple) -> Any:
        return self._Done(fn(*args))

    def map(self, fn: Any, items: Any, chunksize: int = 1) -> list:
        return [fn(x) for x in items]

    def close(self) -> None:
        pass

    def terminate(self) -> None:
        pass

    def join(self) -> None:
        pass


def _enc_block(
    task: tuple[int, bool, int, int],
) -> tuple[int, bool, list[bytes], list[bytes], float]:
    """Encrypt the columns of one block: (cluster, is_center, ids, columns, cpu seconds)."""
    t0 = time.process_time()
    cluster_id, is_center, lo, hi = task
    st = _ST
    cc, pk, slots = st["cc"], st["pk"], st["slots"]
    if is_center:
        vecs = st["centers"].astype(np.float64)
        ids = [b"center%d" % c for c in range(len(vecs))]
        fields = None
    else:
        rows = st["order"][lo:hi]
        vecs = st["db"][rows].astype(np.float64)
        ids = [b"v%07d" % int(r) for r in rows]
        fields = st["fields"][rows]
    n = len(vecs)
    cols = []
    for j in range(vecs.shape[1]):
        col = np.zeros(slots)
        col[:n] = vecs[:, j]
        cols.append(col)
    if fields is not None:
        for fi, card in enumerate(CARDS):
            for v in range(card):
                col = np.zeros(slots)
                col[:n] = (fields[:, fi] == v).astype(np.float64)
                cols.append(col)
    out = []
    for col in cols:
        pt = cc.MakeCKKSPackedPlaintext(col.tolist())
        out.append(_ct_envelope(cc.Encrypt(pk, pt)))
    return cluster_id, is_center, ids, out, time.process_time() - t0


def _dec_block(task: tuple[bytes, int]) -> tuple[np.ndarray, np.ndarray]:
    """Decrypt one score block, band-decode, top-K of live matching slots (local index)."""
    import openfhe as ofhe

    raw, count = task
    st = _ST
    env = read_envelope(raw, Kind.CIPHERTEXT, st["preset"], VERSION)
    path = _shm_path("dec")
    try:
        Path(path).write_bytes(env.payload)
        ct, ok = ofhe.DeserializeCiphertext(path, ofhe.BINARY)
    finally:
        Path(path).unlink(missing_ok=True)
    if not ok:
        raise RuntimeError("cannot deserialize a score block")
    dec = st["cc"].Decrypt(ct, st["sk"])
    dec.SetLength(st["slots"])
    vals = np.asarray(dec.GetRealPackedValue(), dtype=np.float64)[:count]
    c = np.floor((vals + 2.0) / LAMBDA).astype(np.int64)
    sc = vals - LAMBDA * c
    keep = np.nonzero(c == len(CARDS))[0]  # live and every field matches (free fields count)
    o = keep[np.lexsort((keep, -sc[keep]))][:K]
    return sc[o], o.astype(np.int64)


# -------------------------------------------------------------------------------------------------
# Data


def host_info() -> dict[str, Any]:
    model, n_proc = None, 0
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                n_proc += 1
                model = model or line.split(":", 1)[1].strip()
    except OSError:
        pass
    cgroup = None
    for p in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            cgroup = Path(p).read_text().strip()
            break
        except OSError:
            continue
    gpu = None
    try:
        gpu = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    commit = None
    try:
        commit = subprocess.run(  # noqa: S603
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "cpu_model_per_proc_cpuinfo": model,
        "note": (
            "Modal virtualizes the CPU: /proc/cpuinfo may not name the real host CPU, and the "
            "visible logical CPUs may exceed the container's quota (cgroup cpu.max below). The "
            "client (this process and its workers) and the server run in the same container, so "
            "they share this one host description."
        ),
        "logical_cpus_in_proc_cpuinfo": n_proc,
        "os_cpu_count": os.cpu_count(),
        "sched_getaffinity": len(os.sched_getaffinity(0)),
        "cgroup_cpu_max": cgroup,
        "gpu": gpu,
        "git_commit": commit,
    }


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def gen_mixture(rng: np.random.Generator, n: int, nq: int, comps: int):
    """Gaussian mixture on the unit sphere, float32 (chunked: 1M x 384 is 1.5 GB)."""
    mu = _unit(rng.standard_normal((comps, D)).astype(np.float32))

    def draw(m: int) -> np.ndarray:
        out = np.empty((m, D), dtype=np.float32)
        for s in range(0, m, 50000):
            e = min(m, s + 50000)
            lab = rng.integers(0, comps, size=e - s)
            noise = rng.standard_normal((e - s, D), dtype=np.float32) * (NOISE / math.sqrt(D))
            out[s:e] = _unit(mu[lab] + noise)
        return out

    return draw(n), draw(nq)


def assign_to(db: np.ndarray, centers: np.ndarray) -> np.ndarray:
    out = np.empty(len(db), dtype=np.int32)
    for s in range(0, len(db), 100000):
        out[s : s + 100000] = np.argmax(db[s : s + 100000] @ centers.T, axis=1)
    return out


def lloyd(rng: np.random.Generator, db: np.ndarray, c: int) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means from random data points; empty clusters are dropped and renumbered."""
    centers = db[rng.choice(len(db), c, replace=False)].copy()
    assign = assign_to(db, centers)
    for _ in range(LLOYD_ITERS):
        order = np.argsort(assign, kind="stable")
        counts = np.bincount(assign, minlength=len(centers))
        starts = np.concatenate([[0], np.cumsum(counts)])
        for i in range(len(centers)):
            if counts[i]:
                centers[i] = _unit(db[order[starts[i] : starts[i + 1]]].mean(axis=0))
        assign = assign_to(db, centers)
    counts = np.bincount(assign, minlength=len(centers))
    keep = np.nonzero(counts)[0]
    remap = np.full(len(centers), -1, dtype=np.int32)
    remap[keep] = np.arange(len(keep), dtype=np.int32)
    return centers[keep], remap[assign]


# -------------------------------------------------------------------------------------------------
# Measurement


def _pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def _summ(xs: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(xs)), "p50": _pct(xs, 0.5), "p95": _pct(xs, 0.95)}


class Runner:
    def __init__(self, cl: Any, pool: Any, slots: int) -> None:
        self.cl, self.pool, self.slots = cl, pool, slots

    def fetch(self, stream: Any, chosen: list[int]):
        """SearchChosen round: messages, first/last arrival (s), response bytes."""
        t0 = time.perf_counter()
        stream.send(self.cl.pb.SearchRequest(chosen=self.cl.pb.SearchChosen(cluster_ids=chosen)))
        msgs, t_first, nbytes = [], None, 0
        while True:
            m = stream.next().scores
            if t_first is None:
                t_first = time.perf_counter() - t0
            nbytes += sum(len(b) for b in m.score_blocks) + sum(len(i) for i in m.ids)
            msgs.append(m)
            if m.last:
                break
        return msgs, t_first, time.perf_counter() - t0, nbytes

    def decrypt(self, msgs: list[Any]):
        """Parallel decrypt of all blocks; per cluster (rows, scores) of at most K, and times."""
        tasks, owner = [], []
        for m in msgs:
            ids = list(m.ids)
            for i, blk in enumerate(m.score_blocks):
                n = len(ids[i * self.slots : (i + 1) * self.slots])
                tasks.append((bytes(blk), n))
                owner.append((int(m.cluster_id), ids, i))
        t0 = time.perf_counter()
        parts = self.pool.map(_dec_block, tasks, chunksize=1)
        t_dec = time.perf_counter() - t0
        t1 = time.perf_counter()
        per: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for (cid, ids, i), (sc, loc) in zip(owner, parts, strict=True):
            rows = np.array([int(ids[i * self.slots + j][1:]) for j in loc], dtype=np.int64)
            per.setdefault(cid, []).append((rows, sc))
        cand = {}
        for cid, lst in per.items():
            rows = np.concatenate([r for r, _ in lst])
            sc = np.concatenate([s for _, s in lst])
            o = np.lexsort((rows, -sc))[:K]
            cand[cid] = (rows[o], sc[o])
        all_rows = np.concatenate([r for r, _ in cand.values()])
        all_sc = np.concatenate([s for _, s in cand.values()])
        o = np.lexsort((all_rows, -all_sc))[:K]
        _ = all_rows[o]  # the answer of nprobe = 2; recall is evaluated from `cand`
        return cand, t_dec, time.perf_counter() - t1, len(tasks)

    def query(self, name: str, q: np.ndarray, wants: list[int | None]) -> dict[str, Any]:
        cl, rec = self.cl, {}
        t = time.perf_counter()
        qb = cl.encrypt_query(cl.ext_query(q, CARDS, wants))
        rec["encrypt"] = time.perf_counter() - t
        rec["query_bytes"] = len(qb)
        t = time.perf_counter()
        stream, centers = cl.open_search(name, qb, sum(CARDS))
        rec["stage1"] = time.perf_counter() - t
        try:
            cs = centers.centers
            rec["stage1_bytes"] = sum(len(b) for b in cs.score_blocks)
            t = time.perf_counter()
            cscore = cl.decrypt_envelope(cs.score_blocks[0])[: cs.center_count]
            order = [int(i) for i in np.argsort(-cscore, kind="stable")]
            rec["dec1"] = time.perf_counter() - t
            msgs, t_first, t_last, nb = self.fetch(stream, order[:NPROBE_TIMED])
            rec["stage2"], rec["stage2_first_message"], rec["stage2_bytes"] = t_last, t_first, nb
            cand, rec["decrypt"], rec["topk"], rec["blocks"] = self.decrypt(msgs)
            rec["blocks_per_probe"] = rec["blocks"] / max(1, len(msgs))
            rec["total"] = sum(
                rec[k] for k in ("encrypt", "stage1", "dec1", "stage2", "decrypt", "topk")
            )
            if len(order) > NPROBE_TIMED:  # round 2: clusters 3 and 4, for recall only
                msgs2, f2, l2, nb2 = self.fetch(stream, order[NPROBE_TIMED : max(NPROBES)])
                cand2, d2, tk2, nblk2 = self.decrypt(msgs2)
                cand.update(cand2)
                rec["round2"] = {
                    "stage2": l2,
                    "decrypt": d2,
                    "topk": tk2,
                    "bytes": nb2,
                    "blocks": nblk2,
                }
        finally:
            stream.close()
        rec["order"], rec["cand"], rec["cscore"] = order, cand, cscore
        return rec


def run_case(
    out: Path,
    preset: str,
    n: int,
    nq: int,
    workers: int,
    rerun: bool,
    name_prefix: str = "grpc-e2e",
) -> None:
    dest = out / f"{name_prefix}-{preset}-n{n}-a100.json"
    if dest.exists() and not rerun:
        print(f"[grpc-e2e] {dest.name} exists, skipped (resumable at this granularity)", flush=True)
        return
    p = ombra_params.get_preset(preset)
    slots = int(p["batch_size"])
    n_clusters = max(max(NPROBES), round(n / (BLOCKS_PER_CLUSTER * slots)))
    n_cols = D + sum(CARDS)
    est_bytes = int(n * n_cols * BYTES_PER_VECTOR_COLUMN * 1.15)
    tmpdir = Path(tempfile.mkdtemp(prefix="grpc-e2e-"))
    free = shutil.disk_usage(tmpdir).free
    result: dict[str, Any] = {
        "task": "grpc-e2e",
        "preset": preset,
        "n_vectors": n,
        "d": D,
        "filter_fields": CARDS,
        "filter_wants": WANTS,
        "slots": slots,
        "target_blocks_per_cluster": BLOCKS_PER_CLUSTER,
        "k": K,
        "n_queries_per_variant": nq,
        "nprobe_timed": NPROBE_TIMED,
        "nprobes_recall": list(NPROBES),
        "host": host_info(),
        "disk_free_bytes_at_start": free,
        "disk_estimate_bytes": est_bytes,
    }
    if free < est_bytes:
        result["status"] = "not_run"
        result["reason"] = f"disk: need about {est_bytes} bytes, free {free}"
        dest.write_text(json.dumps(result, indent=2))
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"[grpc-e2e] {preset} n={n}: NOT RUN ({result['reason']})", flush=True)
        return
    try:
        _run_case(result, tmpdir, preset, n, nq, workers, slots, n_clusters)
        result["status"] = "done"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    dest.write_text(json.dumps(result, indent=2))
    print(f"[grpc-e2e] wrote {dest}", flush=True)


def _run_case(
    result: dict[str, Any],
    tmp: Path,
    preset: str,
    n: int,
    nq: int,
    workers: int,
    slots: int,
    n_clusters: int,
) -> None:
    rng = np.random.default_rng(SEED + n)
    t0 = time.perf_counter()
    db, queries = gen_mixture(rng, n, 2 * nq + WARMUP, max(4, n_clusters // 4))
    fields = np.stack([rng.integers(0, c, size=n) for c in CARDS], axis=1).astype(np.uint8)
    centers, assign = lloyd(rng, db, n_clusters)
    c_count = len(centers)
    counts = np.bincount(assign, minlength=c_count)
    if c_count > slots:
        raise SystemExit(f"{c_count} clusters do not fit one centers block of {slots}")
    order = np.argsort(assign, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)])
    blocks = []
    for c in range(c_count):
        for a in range(int(starts[c]), int(starts[c + 1]), slots):
            blocks.append((c, a, min(int(starts[c + 1]), a + slots)))
    match = np.all(fields == WANTS, axis=1)
    result["data"] = {
        "generator": "gaussian mixture on the unit sphere, noise 0.7, comps = max(4, C // 4)",
        "clusters": c_count,
        "cluster_size_min_mean_max": [int(counts.min()), float(counts.mean()), int(counts.max())],
        "blocks_total": len(blocks),
        "blocks_per_cluster_mean": len(blocks) / c_count,
        "filter_matching_vectors": int(match.sum()),
        "generate_and_cluster_s": time.perf_counter() - t0,
    }

    binary = ec.find_server_binary()
    if binary is None:
        raise SystemExit("ombra_server not found; build with `make build` (or via modal_app)")
    pb, pbg = ec.load_stubs(tmp / "stubs")
    data_dir = tmp / "data"
    omp = os.environ.pop("OMP_NUM_THREADS", None)  # the server keeps its own thread default
    try:
        server = ec.ServerProcess(
            binary, data_dir, VERSION, tmp / "server.log", GPU_BUDGET_MB, HOST_BUDGET_MB
        )
    finally:
        if omp is not None:
            os.environ["OMP_NUM_THREADS"] = omp
    cl = ec.OmbraClient(pb, pbg, server.address, preset, VERSION)
    pool = None
    faulthandler.dump_traceback_later(900, repeat=True)  # a hang shows where, in the log
    try:
        t = time.perf_counter()
        cl.make_keys(D + sum(CARDS))
        result["keygen_s"] = time.perf_counter() - t
        print(f"[grpc-e2e {preset} n={n}] keys generated in {result['keygen_s']:.0f}s", flush=True)
        files = cl.key_files()
        result["key_bytes"] = {str(k): len(v) for k, v in files.items()}
        cl.create_collection(cl.make_spec("bench", D, CARDS, c_count))
        t = time.perf_counter()
        cl.check(cl.upload_keys("bench", files).error)
        result["keys_upload_s"] = time.perf_counter() - t

        _ST.update(
            cc=cl.cc, pk=cl.keys.publicKey, sk=cl.keys.secretKey, preset=preset, slots=slots,
            db=db, fields=fields, order=order, centers=centers,
        )  # fmt: skip
        n_workers = workers or len(os.sched_getaffinity(0))
        pool = _InlinePool() if n_workers == 1 else mp.get_context("fork").Pool(n_workers)
        result["client_workers"] = n_workers
        tasks = len(blocks) + 1
        print(f"[grpc-e2e {preset} n={n}] pool of {n_workers} workers, {tasks} tasks", flush=True)

        # ---- ingest: encrypt in the pool, upload from this process ----
        tasks = [(c, False, a, b) for c, a, b in blocks] + [(c_count, True, 0, 0)]
        t_ingest = time.perf_counter()
        t_upload = cpu_enc = 0.0
        sent = 0
        pending: deque = deque()
        n_up = [0]

        def upload(res: tuple) -> None:
            nonlocal t_upload, cpu_enc, sent
            cid, is_center, ids, cols, cpu = res
            blk = pb.InsertBlock(
                api_version=1, collection="bench", cluster_id=cid, ids=ids, columns=cols,
                is_center=is_center,
            )  # fmt: skip
            t1 = time.perf_counter()
            cl.insert([blk])
            t_upload += time.perf_counter() - t1
            cpu_enc += cpu
            sent += sum(len(c) for c in cols)
            n_up[0] += 1
            if n_up[0] % 10 == 1:
                print(f"[grpc-e2e {preset} n={n}] uploaded {n_up[0]} blocks", flush=True)

        for task in tasks:
            while len(pending) >= n_workers + 2:
                upload(pending.popleft().get())
            pending.append(pool.apply_async(_enc_block, (task,)))
        while pending:
            upload(pending.popleft().get())
        wall = time.perf_counter() - t_ingest
        info = cl.get_collection("bench")
        assert info.vector_count == n, (info.vector_count, n)
        files_on_disk = [f for f in data_dir.rglob("*") if f.is_file()]
        result["storage"] = {
            "ingest_wall_s": wall,
            "encrypt_cpu_s_summed_over_workers": cpu_enc,
            "upload_call_s_summed": t_upload,
            "ciphertext_bytes_uploaded": sent,
            "bytes_on_disk_apparent": sum(f.stat().st_size for f in files_on_disk),
            "bytes_on_disk_allocated": sum(f.stat().st_blocks * 512 for f in files_on_disk),
            "files": len(files_on_disk),
            "bytes_per_vector_on_disk": sum(f.stat().st_size for f in files_on_disk) / n,
            "ingest_vectors_per_s": n / wall,
        }
        print(
            f"[grpc-e2e {preset} n={n}] ingest {wall:.0f}s, "
            f"{result['storage']['bytes_on_disk_apparent'] / 2**30:.1f} GiB on disk",
            flush=True,
        )

        # ---- queries ----
        run = Runner(cl, pool, slots)
        variants = {"no_filter": [None, None], "filter": list(WANTS)}
        for _ in range(WARMUP):
            run.query("bench", queries[0].astype(np.float64), variants["no_filter"])
            run.query("bench", queries[0].astype(np.float64), variants["filter"])
        result["variants"] = {}
        for vname, wants in variants.items():
            allowed = np.ones(n, bool) if vname == "no_filter" else match
            ids_ok = np.nonzero(allowed)[0]
            recs: list[dict[str, Any]] = []
            enc_r = {p_: [] for p_ in NPROBES}
            pla_r = {p_: [] for p_ in NPROBES}
            same_choice = {p_: [] for p_ in NPROBES}
            for i in range(nq):
                q = queries[WARMUP + i].astype(np.float64)
                r = run.query("bench", q, wants)
                recs.append(r)
                exact = db @ q.astype(np.float32)
                truth = ids_ok[topk(exact[ids_ok], K)]
                for p_ in NPROBES:
                    chosen = r["order"][:p_]
                    rows = [r["cand"][c][0] for c in chosen if c in r["cand"]]
                    sc = [r["cand"][c][1] for c in chosen if c in r["cand"]]
                    rows_a, sc_a = np.concatenate(rows), np.concatenate(sc)
                    enc_r[p_].append(
                        recall_of(rows_a[topk(sc_a, K)], truth) if len(rows_a) else 0.0
                    )
                    pc = topk(centers @ q.astype(np.float32), p_)
                    ids_p = np.nonzero(np.isin(assign, pc) & allowed)[0]
                    best = ids_p[topk(exact[ids_p], K)] if len(ids_p) else ids_p
                    pla_r[p_].append(recall_of(best, truth))
                    same_choice[p_].append(float(set(pc.tolist()) == set(chosen)))
            steps = ["encrypt", "stage1", "dec1", "stage2", "decrypt", "topk", "total"]
            result["variants"][vname] = {
                "latency_s": {s: _summ([r[s] for r in recs]) for s in steps},
                "stage2_first_message_s": _summ([r["stage2_first_message"] for r in recs]),
                "round2_extra_s": _summ(
                    [
                        r["round2"]["stage2"] + r["round2"]["decrypt"] + r["round2"]["topk"]
                        for r in recs
                        if "round2" in r
                    ]
                )
                if any("round2" in r for r in recs)
                else None,
                "bytes_per_query": {
                    "query_up": recs[0]["query_bytes"],
                    "stage1_down": recs[0]["stage1_bytes"],
                    "stage2_down_mean": float(np.mean([r["stage2_bytes"] for r in recs])),
                },
                "stage2_blocks_per_query_mean": float(np.mean([r["blocks"] for r in recs])),
                "recall_at_10": {
                    str(p_): {
                        "encrypted_no_rerank": float(np.mean(enc_r[p_])),
                        "plaintext_same_clusters": float(np.mean(pla_r[p_])),
                        "gap": float(np.mean(enc_r[p_]) - np.mean(pla_r[p_])),
                        "same_cluster_choice_fraction": float(np.mean(same_choice[p_])),
                    }
                    for p_ in NPROBES
                },
                "n_queries": nq,
                "recall_rule_note": "tolerance 0.005; few queries, no verdict",
            }
            lat = result["variants"][vname]["latency_s"]["total"]
            print(
                f"[grpc-e2e {preset} n={n}] {vname}: total p50 {lat['p50'] * 1000:.0f} ms "
                f"p95 {lat['p95'] * 1000:.0f} ms, recall gaps "
                + ", ".join(
                    f"{p_}:{np.mean(enc_r[p_]) - np.mean(pla_r[p_]):+.3f}" for p_ in NPROBES
                ),
                flush=True,
            )
        # Server cache statistics: not in this PR (/metrics is separate); only what the log has.
        log = (tmp / "server.log").read_text(errors="replace").splitlines()
        result["server_log"] = {
            "lines": len(log),
            "cache_lines": [ln[:200] for ln in log if "cache" in ln.lower()][-20:],
            "note": "GPU cache hit rate (/metrics) not available; these log lines are all there is",
        }
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        _ST.clear()
        cl.close()
        server.stop()


def run_grpc(
    out: Path,
    presets: tuple[str, ...],
    sizes: tuple[int, ...],
    nq: int,
    workers: int,
    rerun: bool = False,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for preset in presets:
        for n in sizes:
            run_case(out, preset, n, nq, workers, rerun)
