"""The filter measurements, through the real gRPC server (A100, Modal).

    # writes <dir>/filter-measurements-<preset>-a100.json
    python core/bench/run.py --task filter-measurements --out <dir>

This process is the CLIENT (it holds the secret key, never written or sent). It starts the C++
server `ombra_server` on localhost like tests/test_gpu_e2e_grpc.py and drives it with
tests/e2e_client.py (reused, not copied): create collection, upload public keys, insert encrypted
column blocks, search, decrypt, band-decode. Synthetic data only; only aggregates
are printed or stored (errors, recalls, counts, times), never vectors, queries or scores.

Three measurements per preset (`m0_depth2`, `m0_compact_n8192`):

 1. Score error with filter columns in the sum. d = 384, one block (slots vectors, unit-norm
    random), filter fields of cardinality 4, F = 0, 1, 4 and F_max (F_max found by asking the
    server which F it accepts, and compared with the formula of server/search/layout.cpp). Field
    values are planted so that the number c of matching fields is spread over 0..F and the query
    matches some vectors on every field. Reported per F: max |decoded - numpy| over the vectors
    matching every field and over all vectors, the smallest distance of a decoded value to the
    decoder's band boundary (4m - 2), the observed gap between the bands, whether every c decoded
    exactly, and the ratio of the F > 0 error to the F = 0 error.
 2. The recall rule with the extra columns (F = 2 fields of 8 values): same method as
    the recall acceptance checks (recall.py / recall_check.py): recall@10 of the encrypted pipeline
    without re-ranking against the plaintext recall of the same cluster setting (exact centers,
    exact scores), gap within 0.005 for the full scan and every nprobe. One encrypted scan of all
    clusters per query, restricted to the probed clusters afterwards (as recall.py does). Query with
    the filter matching one value in each field, and with 'no filter' (all-ones).
 3. Filtered top-10 recall against brute-force filtered ground truth at selectivity 1%, 10% and
    50%, with the selective-filter client loop: first round p clusters, each
    extra round doubles the number of clusters, up to max_probe = mult * p clusters, then a Strict
    fallback (all remaining clusters), else `incomplete`. Gaussian-mixture data, d = 128, 8
    clusters by client k-means, k = 10, p = 2; filter placements `independent` of the clusters and
    `correlated` (the matching vectors are packed into whole clusters, so pruning matters).
    mult = 8 is the default (it covers all 8 clusters, so Strict is never separate);
    mult = 2 is added so the Strict fallback is exercised.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "client"))
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e2e_client as ec  # noqa: E402
import recall_check  # noqa: E402
from ombra import params as ombra_params  # noqa: E402
from recall import kmeans, recall_of, topk  # noqa: E402

LAMBDA = ec.LAMBDA
VERSION = "test-openfhe"
SEED = 20261110
K = 10
GPU_BUDGET_MB = 4096
HOST_BUDGET_MB = 2048

P1_D = 384
P1_CARD = 4
P1_QUERIES = 3
P2_FIELDS = [8, 8]
P2_WANTS = [2, 5]
P2_NPROBES = (1, 2, 4, 8)
P3_N = 16000
P3_D = 128
P3_CLUSTERS = 8
P3_QUERIES = 50
P3_NPROBE = 2
P3_SELS = (0.01, 0.1, 0.5)
P3_PLACEMENTS = ("independent", "correlated")
P3_MULTS = (8, 2)
# (first-round clusters p, max_probe multiplier): p is what an unfiltered query would probe
# ; the loop never probes fewer, widening only below k matches.
P3_SETTINGS = ((2, 8), (2, 2), (4, 8))
P3_WARMUP = 3
NOISE = 0.7  # mixture noise relative to the unit-length component centers


def _ident(i: int) -> bytes:
    return b"v%06d" % i


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def f_max_formula(preset: str) -> int:
    """server/search/layout.cpp max_filter_fields(); only a cross-check of the server's answer."""
    p = ombra_params.get_preset(preset)
    first, scale = p["first_mod_size"], p["scaling_mod_size"]
    if first < scale + 5:
        return 0
    allowed = 1 << (first - scale - 1 - 3)
    if allowed < 1 + 2 * int(LAMBDA):
        return 0
    return min((allowed - 1) // int(LAMBDA) - 1, 16)


class Env:
    """Server process + client for one preset."""

    def __init__(self, tmp: Path, preset: str) -> None:
        binary = ec.find_server_binary()
        if binary is None:
            raise SystemExit("ombra_server not found; build with `make build` (or via modal_app)")
        self.preset = preset
        self.pb, pbg = ec.load_stubs(tmp / "stubs")
        self.server = ec.ServerProcess(
            binary, tmp / "data", VERSION, tmp / "server.log", GPU_BUDGET_MB, HOST_BUDGET_MB
        )
        self.cl = ec.OmbraClient(self.pb, pbg, self.server.address, preset, VERSION)
        self.slots = self.cl.slots
        self.files: dict[int, bytes] = {}

    def close(self) -> None:
        self.cl.close()
        self.server.stop()

    def f_max_from_server(self) -> int:
        """Largest F the server accepts at CreateCollection (cardinality 2, no keys needed)."""
        best = 0
        for f in range(1, 18):
            spec = self.cl.make_spec(f"probe{f}", 8, [2] * f, 1, key_tag="probe")
            try:
                self.cl.create_collection(spec)
            except ec.ApiError as e:
                if e.code != self.pb.ERROR_INVALID_FILTER_LAYOUT:
                    raise
                break
            best = f
            self.cl.drop_collection(f"probe{f}")
        return best

    def keys(self, n_leaves: int) -> None:
        self.cl.make_keys(n_leaves)
        self.files = self.cl.key_files()

    def new_collection(self, name: str, d: int, cards: list[int], clusters: int) -> None:
        self.cl.create_collection(self.cl.make_spec(name, d, cards, clusters))
        self.cl.check(self.cl.upload_keys(name, self.files).error)

    def enc_columns(self, cols: list[np.ndarray]) -> list[bytes]:
        return [self.cl.ct_envelope(self.cl.encrypt(c)) for c in cols]

    def vector_columns(self, vecs: np.ndarray) -> list[bytes]:
        """d column envelopes of one block (rows = slots, zero padded)."""
        cols = []
        for j in range(vecs.shape[1]):
            col = np.zeros(self.slots)
            col[: len(vecs)] = vecs[:, j]
            cols.append(col)
        return self.enc_columns(cols)

    def onehot_columns(self, fields: np.ndarray, cards: list[int]) -> list[bytes]:
        cols = []
        for fi, card in enumerate(cards):
            for v in range(card):
                col = np.zeros(self.slots)
                col[: len(fields)] = (fields[:, fi] == v).astype(float)
                cols.append(col)
        return self.enc_columns(cols)

    def insert_block(
        self,
        name: str,
        cluster_id: int,
        ids: list[bytes],
        columns: list[bytes],
        is_center: bool = False,
    ) -> None:
        blk = self.pb.InsertBlock(
            api_version=1,
            collection=name,
            cluster_id=cluster_id,
            ids=ids,
            columns=columns,
            is_center=is_center,
        )
        self.cl.insert([blk])

    # ---- search -------------------------------------------------------------------------------
    def open(self, name: str, q: np.ndarray, cards: list[int], wants: list[int | None]):
        ext = self.cl.ext_query(q, cards, wants)
        stream, centers = self.cl.open_search(name, self.cl.encrypt_query(ext), sum(cards))
        cs = centers.centers
        scores = self.cl.decrypt_envelope(cs.score_blocks[0])[: cs.center_count]
        return stream, scores

    def fetch(self, stream: Any, chosen: list[int]) -> dict[int, tuple[list[bytes], np.ndarray]]:
        """SearchChosen on an open stream; raw decrypted slot values per cluster (live ids only)."""
        stream.send(self.pb.SearchRequest(chosen=self.pb.SearchChosen(cluster_ids=chosen)))
        out: dict[int, tuple[list[bytes], np.ndarray]] = {}
        while True:
            msg = stream.next().scores
            vals = []
            for i, bl in enumerate(msg.score_blocks):
                n = len(msg.ids[i * self.slots : (i + 1) * self.slots])
                vals.append(self.cl.decrypt_envelope(bl)[:n])
            out[msg.cluster_id] = (
                [bytes(x) for x in msg.ids],
                np.concatenate(vals) if vals else np.zeros(0),
            )
            if msg.last:
                break
        return out


# -------------------------------------------------------------------------------------------------
# 1. Score error with filter columns


def run_part1(env: Env, f_max: int) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    slots = env.slots
    d = P1_D
    fs = sorted({0, 1, 4, f_max})
    env.keys(d + P1_CARD * f_max)
    assert math.ceil(math.log2(2 * (d + P1_CARD * f_max))) == math.ceil(math.log2(2 * d))
    vecs = _unit(rng.standard_normal((slots, d)))
    queries = _unit(rng.standard_normal((P1_QUERIES, d)))
    center = _unit(rng.standard_normal((1, d)))
    vec_cols = env.vector_columns(vecs)
    center_cols = env.vector_columns(center)
    truth = vecs @ queries.T  # [slots, Q]
    rows: dict[str, Any] = {}
    base_err = None
    for f in fs:
        cards = [P1_CARD] * f
        t = rng.integers(0, f + 1, size=slots)  # number of matching fields per vector
        fields = rng.integers(1, P1_CARD, size=(slots, f))  # non-matching values 1..3
        if f:
            fields[np.argsort(rng.random((slots, f)), axis=1) < t[:, None]] = 0  # 0 = the wanted
        name = f"p1_f{f}"
        env.new_collection(name, d, cards, 1)
        cols = vec_cols + env.onehot_columns(fields, cards)
        env.insert_block(name, 0, [_ident(i) for i in range(slots)], cols)
        env.insert_block(name, 1, [b"center0"], center_cols, is_center=True)
        err_match = err_all = 0.0
        min_boundary = math.inf
        gap_obs = math.inf
        wrong_c = 0
        n_match = int((t == f).sum())
        t_search = []
        for qi in range(P1_QUERIES):
            t0 = time.perf_counter()
            stream, _ = env.open(name, queries[qi], cards, [0] * f)
            try:
                ids, raw = env.fetch(stream, [0])[0]
            finally:
                stream.close()
            t_search.append(time.perf_counter() - t0)
            assert len(ids) == slots
            c_dec, sc = env.cl.decode_band(raw)
            wrong_c += int((c_dec != t).sum())
            err_all = max(err_all, float(np.max(np.abs(sc - truth[:, qi]))))
            m = t == f
            err_match = max(err_match, float(np.max(np.abs(sc[m] - truth[m, qi]))))
            u = (raw + 2.0) / LAMBDA
            frac = u - np.floor(u)
            min_boundary = min(min_boundary, float(LAMBDA * np.min(np.minimum(frac, 1 - frac))))
            for c in range(f):
                lo, hi = raw[t == c], raw[t == c + 1]
                if len(lo) and len(hi):
                    gap_obs = min(gap_obs, float(hi.min() - lo.max()))
        env.cl.drop_collection(name)
        if f == 0:
            base_err = err_match
        rows[str(f)] = {
            "F": f,
            "D": d + P1_CARD * f,
            "vectors": slots,
            "vectors_matching_all_fields": n_match,
            "queries": P1_QUERIES,
            "max_abs_error_matching_vectors": err_match,
            "max_abs_error_all_vectors": err_all,
            "min_distance_to_band_boundary": min_boundary,
            "min_distance_expected_at_least": LAMBDA / 2 - 1.0,
            "observed_gap_between_bands": None if gap_obs == math.inf else gap_obs,
            "band_gap_expected": LAMBDA - 2.0,
            "wrong_c_decodes": wrong_c,
            "all_vectors_classified_correctly": wrong_c == 0,
            "error_ratio_to_F0": (err_match / base_err) if base_err else None,
            "error_ratio_estimate_sqrt_1_plus_16F": math.sqrt(1 + 16 * f),
            "search_wall_seconds_mean": statistics.mean(t_search),
        }
        print(f"[part1 {env.preset}] F={f} err={err_match:.3e} wrong_c={wrong_c}", flush=True)
    return {"d": d, "cardinality": P1_CARD, "lambda": LAMBDA, "per_F": rows}


# -------------------------------------------------------------------------------------------------
# Shared synthetic data for parts 2 and 3


def mixture(rng: np.random.Generator, n: int, d: int, comps: int, nq: int):
    mu = _unit(rng.standard_normal((comps, d)))

    def draw(m: int) -> np.ndarray:
        lab = rng.integers(0, comps, size=m)
        return _unit(mu[lab] + NOISE * rng.standard_normal((m, d)) / math.sqrt(d))

    return draw(n), draw(nq)


class Data:
    """Vectors, k-means clusters and the encrypted vector columns of every block (per key set)."""

    def __init__(self, env: Env, rng: np.random.Generator, n_queries: int) -> None:
        self.db, self.queries = mixture(rng, P3_N, P3_D, P3_CLUSTERS, n_queries)
        self.centers, self.assign = kmeans(self.db, P3_CLUSTERS)
        self.rows = [np.nonzero(self.assign == c)[0] for c in range(P3_CLUSTERS)]
        sizes = [len(r) for r in self.rows]
        if min(sizes) == 0 or max(sizes) > env.slots:
            raise SystemExit(f"k-means cluster sizes {sizes} do not fit a block of {env.slots}")
        self.sizes = sizes
        self.vec_cols = [env.vector_columns(self.db[r]) for r in self.rows]
        self.center_cols = env.vector_columns(self.centers)

    def load(self, env: Env, name: str, cards: list[int], fields: np.ndarray) -> None:
        env.new_collection(name, P3_D, cards, P3_CLUSTERS)
        for c, r in enumerate(self.rows):
            cols = self.vec_cols[c] + env.onehot_columns(fields[r], cards)
            env.insert_block(name, c, [_ident(int(i)) for i in r], cols)
        env.insert_block(
            name,
            P3_CLUSTERS,
            [b"center%d" % c for c in range(P3_CLUSTERS)],
            self.center_cols,
            is_center=True,
        )


def candidates(
    env: Env, got: dict[int, tuple[list[bytes], np.ndarray]], n_fields: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per cluster: (row indices, decoded scores) of the vectors with c == n_fields."""
    out = {}
    for cid, (ids, raw) in got.items():
        c, sc = env.cl.decode_band(raw)
        keep = c == n_fields
        idx = np.array([int(ids[j][1:]) for j in np.nonzero(keep)[0]], dtype=np.int64)
        out[cid] = (idx, sc[keep])
    return out


# -------------------------------------------------------------------------------------------------
# 2. Recall rule with the extra columns


def run_part2(env: Env, data: Data, nq: int) -> dict[str, Any]:
    rng = np.random.default_rng(SEED + 2)
    fields = np.stack([rng.integers(0, c, size=P3_N) for c in P2_FIELDS], axis=1)
    data.load(env, "p2", P2_FIELDS, fields)
    match = np.all(fields == P2_WANTS, axis=1)
    out: dict[str, Any] = {
        "fields": P2_FIELDS,
        "wants": "one value per field (fixed)",
        "matching_vectors": int(match.sum()),
        "n_vectors": P3_N,
        "n_queries": nq,
        "k": K,
        "nlist": P3_CLUSTERS,
        "tolerance": recall_check.TOLERANCE,
    }
    for variant, wants in (("no_filter", [None, None]), ("filter", list(P2_WANTS))):
        allowed = np.ones(P3_N, bool) if variant == "no_filter" else match
        enc = {p: [] for p in P2_NPROBES}
        plain = {p: [] for p in P2_NPROBES}
        err = 0.0
        for i in range(nq):
            q = data.queries[i]
            exact = data.db @ q
            ids_ok = np.nonzero(allowed)[0]
            truth = ids_ok[topk(exact[ids_ok], K)]
            stream, cscore = env.open("p2", q, P2_FIELDS, wants)
            try:
                got = env.fetch(stream, list(range(P3_CLUSTERS)))
            finally:
                stream.close()
            cand = candidates(env, got, len(P2_FIELDS))
            for _, (idx, sc) in cand.items():
                if len(idx):
                    err = max(err, float(np.max(np.abs(sc - exact[idx]))))
            for p in P2_NPROBES:
                chosen = topk(cscore, p)
                idx = np.concatenate([cand[int(c)][0] for c in chosen])
                sc = np.concatenate([cand[int(c)][1] for c in chosen])
                enc[p].append(recall_of(idx[topk(sc, K)], truth) if len(idx) else 0.0)
                pc = topk(data.centers @ q, p)
                m = np.isin(data.assign, pc) & allowed
                ids_p = np.nonzero(m)[0]
                best = ids_p[topk(exact[ids_p], K)] if len(ids_p) else ids_p
                plain[p].append(recall_of(best, truth))
        result = {
            "n_queries": nq,
            "full_scan": {
                "encrypted_no_rerank": float(np.mean(enc[P3_CLUSTERS])),
                "plaintext": float(np.mean(plain[P3_CLUSTERS])),
            },
            "clustered": {
                str(P3_CLUSTERS): {
                    str(p): {
                        "encrypted_no_rerank": float(np.mean(enc[p])),
                        "plaintext_exact_scoring": float(np.mean(plain[p])),
                    }
                    for p in P2_NPROBES
                }
            },
        }
        check = recall_check.check_dataset(result)
        check["headroom_to_tolerance"] = check["tolerance"] - abs(check["worst_gap"])
        check["max_abs_score_error_matching"] = err
        out[variant] = {"recalls": result, "rule": check}
        print(
            f"[part2 {env.preset}] {variant}: verdict={check['verdict']} "
            f"worst_gap={check['worst_gap']:+.4f} err={err:.2e}",
            flush=True,
        )
    env.cl.drop_collection("p2")
    return out


# -------------------------------------------------------------------------------------------------
# 3. Filtered recall with the selective-filter loop


def placements(rng: np.random.Generator, assign: np.ndarray, sel: float, kind: str) -> np.ndarray:
    """Field values (0 = the filter value, 1 = other) with exactly round(sel * n) vectors at 0."""
    n = len(assign)
    want = int(round(sel * n))
    hit = np.zeros(n, bool)
    if kind == "independent":
        hit[rng.choice(n, want, replace=False)] = True
    else:  # value tied to the cluster: fill whole clusters in a random order
        left = want
        for c in rng.permutation(P3_CLUSTERS):
            rows = np.nonzero(assign == c)[0]
            take = min(left, len(rows))
            hit[rng.choice(rows, take, replace=False)] = True
            left -= take
            if left == 0:
                break
    return np.where(hit, 0, 1)


def selective_search(
    env: Env,
    name: str,
    q: np.ndarray,
    cards: list[int],
    wants: list[int | None],
    mult: int,
    p: int = P3_NPROBE,
) -> dict[str, Any]:
    """Selective-filter client loop; returns found {row: score}, rounds, strict flag, incomplete."""
    nf = len(cards)
    stream, cscore = env.open(name, q, cards, wants)
    found: dict[int, float] = {}
    try:
        order = [int(c) for c in np.argsort(-cscore, kind="stable")]
        total = len(order)
        max_probe = min(mult * p, total)
        pos, size, rounds, strict = 0, p, 0, False
        while pos < total:
            if pos < max_probe:
                n = min(size, max_probe - pos)
                size *= 2
            elif not strict:
                n, strict = total - pos, True  # Strict fallback: every cluster left
            else:
                break
            cand = candidates(env, env.fetch(stream, order[pos : pos + n]), nf)
            pos += n
            rounds += 1
            for idx, sc in cand.values():
                found.update(zip(idx.tolist(), sc.tolist(), strict=True))
            if len(found) >= K:
                break
    finally:
        stream.close()
    return {
        "found": found,
        "rounds": rounds,
        "strict": strict,
        "incomplete": len(found) < K,
        "clusters": pos,
    }


def run_part3(env: Env, data: Data, nq: int) -> dict[str, Any]:
    rng = np.random.default_rng(SEED + 3)
    cols = [(s, k) for s in P3_SELS for k in P3_PLACEMENTS]
    fields = np.stack([placements(rng, data.assign, s, k) for s, k in cols], axis=1)
    cards = [2] * len(cols)
    data.load(env, "p3", cards, fields)
    out: dict[str, Any] = {
        "n_vectors": P3_N,
        "d": P3_D,
        "clusters": P3_CLUSTERS,
        "cluster_sizes": data.sizes,
        "n_queries": nq,
        "k": K,
        "initial_nprobe": P3_NPROBE,
        "settings_p_mult": [list(x) for x in P3_SETTINGS],
        "fields": len(cols),
        "max_probe_multipliers": list(P3_MULTS),
        "results": {},
    }
    for fi, (sel, kind) in enumerate(cols):
        wants: list[int | None] = [None] * len(cols)
        wants[fi] = 0
        allowed = fields[:, fi] == 0
        ids_ok = np.nonzero(allowed)[0]
        for _ in range(P3_WARMUP):  # untimed: first touches of the blocks on the GPU
            selective_search(env, "p3", data.queries[0], cards, wants, P3_MULTS[0])
        for p, mult in P3_SETTINGS:
            recs, rounds, strict, inc, clus, lat = [], [], [], [], [], []
            for i in range(nq):
                q = data.queries[i]
                truth = ids_ok[topk((data.db @ q)[ids_ok], K)]
                t0 = time.perf_counter()
                r = selective_search(env, "p3", q, cards, wants, mult, p)
                lat.append(time.perf_counter() - t0)
                rows = np.array(list(r["found"]), dtype=np.int64)
                sc = np.array(list(r["found"].values()))
                best = rows[topk(sc, K)] if len(rows) else rows
                recs.append(recall_of(best, truth))
                rounds.append(r["rounds"] - 1)
                strict.append(float(r["strict"]))
                inc.append(float(r["incomplete"]))
                clus.append(r["clusters"])
            lat_ms = sorted(x * 1000 for x in lat)
            key = f"sel{sel}_{kind}_p{p}_mult{mult}"
            out["results"][key] = {
                "selectivity": sel,
                "matching_vectors": int(allowed.sum()),
                "placement": kind,
                "first_round_clusters": p,
                "max_probe_multiplier": mult,
                "recall_at_10": float(np.mean(recs)),
                "mean_extra_rounds": float(np.mean(rounds)),
                "fraction_strict_fallback": float(np.mean(strict)),
                "fraction_incomplete": float(np.mean(inc)),
                "mean_clusters_searched": float(np.mean(clus)),
                "latency_ms_mean": float(np.mean(lat_ms)),
                "latency_ms_p50": lat_ms[len(lat_ms) // 2],
                "latency_ms_p95": lat_ms[min(len(lat_ms) - 1, int(0.95 * len(lat_ms)))],
            }
            print(
                f"[part3 {env.preset}] {key}: recall={np.mean(recs):.3f} "
                f"extra={np.mean(rounds):.2f} strict={np.mean(strict):.2f} "
                f"incomplete={np.mean(inc):.2f} lat={np.mean(lat_ms):.0f}ms",
                flush=True,
            )
    env.cl.drop_collection("p3")
    return out


# -------------------------------------------------------------------------------------------------


def run_preset(out_dir: Path, preset: str, queries: int, scratch: Path | None = None) -> Path:
    t_start = time.time()
    with tempfile.TemporaryDirectory(dir=scratch) as t:
        env = Env(Path(t), preset)
        try:
            f_server = env.f_max_from_server()
            f_formula = f_max_formula(preset)
            print(f"[{preset}] F_max server={f_server} formula={f_formula}", flush=True)
            p1 = run_part1(env, f_server)
            env.keys(P3_D + 16)  # parts 2 and 3: D = 144 and 140 share the expansion depth
            assert math.ceil(math.log2(2 * (P3_D + 16))) == math.ceil(math.log2(2 * (P3_D + 12)))
            data = Data(env, np.random.default_rng(SEED + 1), max(queries, P3_QUERIES))
            p2 = run_part2(env, data, queries)
            p3 = run_part3(env, data, P3_QUERIES)
        finally:
            env.close()
    result = {
        "task": "filter-measurements",
        "preset": preset,
        "slots": ombra_params.get_preset(preset)["batch_size"],
        "lambda": LAMBDA,
        "f_max_from_server": f_server,
        "f_max_formula": f_formula,
        "server_budgets_mb": {"gpu": GPU_BUDGET_MB, "host": HOST_BUDGET_MB},
        "synthetic_data_note": "synthetic only; aggregates only",
        "part1_score_error": p1,
        "part2_recall_rule": p2,
        "part3_filtered_recall": p3,
        "total_wall_seconds": time.time() - t_start,
    }
    path = out_dir / f"filter-measurements-{preset}-a100.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {path}", flush=True)
    return path


def run_filter(out_dir: Path, presets: tuple[str, ...], queries: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for preset in presets:
        run_preset(out_dir, preset, queries)
