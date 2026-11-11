"""Two-stage search with filters, security levels and exact re-ranking (T3.6).

Flow on one Search stream (decisions 0004, 0019): the query carries the d query values followed by
lambda * one-hot entries for each filter field (all lambda for a field that is not filtered, so the
server cannot tell filtered from unfiltered queries). Stage 1 returns encrypted center scores; the
client decrypts them and picks clusters. Each round's slot value is score + lambda * c, with c the
number of fields the vector matches; the client decodes the band, keeps the vectors with c = F,
and counts them. Fewer than k matches: widen 2x per round on the same stream up to max_probe, then
fall back to Strict if allowed, else report `incomplete`. The candidates are re-ranked with the
exact vectors from their record blobs (only the best `rerank_factor * k` are fetched).

What the server learns: the cluster choices (real, plus decoys at the Decoy level) and how many
rounds were needed. No plaintext, score, secret key or filter value leaves the client.
"""

from __future__ import annotations

import math
import queue
import random
import secrets
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ombra.blobstore import KIND_RECORD
from ombra.file_format import Kind, read_envelope
from ombra.presets import FILTER_LAMBDA
from ombra.probe import FIELDS_KEY, POLICIES, ProbePlanner
from ombra.topk import ParallelTopk

API_VERSION = 1
MAX_CLUSTERS_PER_ROUND = 4096  # LIMIT_MAX_CLUSTERS_PER_ROUND
LEVELS = ("fast", "decoy", "strict")


@dataclass(frozen=True)
class Hit:
    id: bytes
    score: float = field(repr=False)  # exact cosine score from the record's vector
    metadata: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class SearchResult:
    hits: list[Hit] = field(repr=False)
    incomplete: bool  # fewer than k matches although everything allowed was searched
    probed: list[int]  # clusters chosen for real, in order (decoys are not listed)
    rounds: int
    matches: int  # candidates that passed the filter
    strict_fallback: bool  # the Strict fallback was used


class _Stream:
    """One Search stream; requests come from a queue so rounds can follow each other."""

    def __init__(self, stub: Any, first: Any) -> None:
        self._q: queue.Queue = queue.Queue()
        self._q.put(first)
        self._call = stub.Search(self._requests())

    def _requests(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item

    def send(self, request: Any) -> None:
        self._q.put(request)

    def next(self) -> Any:
        from ombra.collection import CollectionError

        try:
            resp = next(self._call)
        except StopIteration:
            raise CollectionError(0, "search stream ended early") from None
        if resp.WhichOneof("step") == "error":
            self._q.put(None)
            raise CollectionError(resp.error.code, resp.error.message)
        return resp

    def close(self) -> None:
        self._q.put(None)  # half-close
        try:
            for _ in self._call:
                pass
        except Exception:  # noqa: S110  # the stream already ended with an error
            pass


def decode_band(slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(c, score) per slot: c = floor((slot + 2) / lambda), score = slot - lambda * c."""
    c = np.floor((slots + 2.0) / FILTER_LAMBDA).astype(np.int64)
    return c, slots - FILTER_LAMBDA * c


def filter_query(
    query: np.ndarray, cards: Sequence[int], filter: Mapping[int, int | Sequence[int]] | None
) -> np.ndarray:
    """The d query values, then lambda * one-hot per field (all lambda for a field left free)."""
    wants = dict(filter or {})
    if any(not 0 <= f < len(cards) for f in wants):
        raise ValueError("filter names a field the collection does not have")
    parts = [np.asarray(query, dtype=np.float64)]
    for fi, card in enumerate(cards):
        if fi not in wants:
            parts.append(np.full(card, float(FILTER_LAMBDA)))
            continue
        w = wants[fi]
        vals = [w] if isinstance(w, int | np.integer) else list(w)
        if not vals or any(not 0 <= int(v) < card for v in vals):
            raise ValueError("a filter value is outside its field's cardinality")
        oh = np.zeros(card)
        oh[[int(v) for v in vals]] = FILTER_LAMBDA
        parts.append(oh)
    return np.concatenate(parts)


def _chunks(seq: list[int]) -> list[list[int]]:
    return [seq[i : i + MAX_CLUSTERS_PER_ROUND] for i in range(0, len(seq), MAX_CLUSTERS_PER_ROUND)]


def _decrypt_round(col: Any, envelopes: list[bytes], counts: list[int], workers: int) -> list:
    """Decrypt score blocks: in parallel (T2.1) when `workers` > 1, else in this process."""
    if workers <= 1 or len(envelopes) < 2:
        return [col.kit.decrypt_column(e)[:n] for e, n in zip(envelopes, counts, strict=True)]
    kit = col.kit
    if getattr(col, "_pool", None) is None:
        col._pool = ParallelTopk(kit.cc, kit._keys.secretKey, workers)
    with tempfile.TemporaryDirectory() as t:
        files = []
        for i, e in enumerate(envelopes):
            env = read_envelope(e, Kind.CIPHERTEXT, kit.derived.preset, kit.openfhe_version)
            p = Path(t) / f"b{i}.bin"
            p.write_bytes(env.payload)
            files.append(p)
        return col._pool.decrypt(files, counts, col.slots)


def run_search(
    col: Any,
    query: np.ndarray,
    k: int = 10,
    *,
    filter: Mapping[int, int | Sequence[int]] | None = None,
    level: str | None = None,
    decoys: int | None = None,
    n_probe: int | None = None,
    max_probe_factor: int = 8,
    probe: str = "e_a",
    max_rounds: int = 4,
    strict_fallback: bool = True,
    rerank_factor: int = 3,
    workers: int = 1,
    seed: int | None = None,
) -> SearchResult:
    """Search `col` for the k nearest vectors that pass `filter`.

    `filter` maps a field index to one allowed value or a list of them (OR inside a field, AND
    across fields). `level` overrides the collection's security level for this query. `n_probe`
    is the first round's cluster count (default ceil(sqrt(clusters))); a filtered query starts
    with the same count, so it probes at least what the unfiltered query would. At the Decoy
    level, `decoys` random clusters are added to each round (default: as many as the real ones;
    0 is refused, since it would make Decoy the same as Fast). `rerank_factor * k` record blobs
    are fetched for re-ranking (default 3, the 2-3x k of the requirements).

    `probe` selects the cluster policy of a filtered query (ombra.probe): "e_a" (default) uses the
    client's per-cluster value counts to skip clusters with no matches, probes until about 3k
    matches are expected, then widens while an unsearched cluster could still beat the k-th match
    (at most `max_rounds` rounds); "e" is the same without the stopping check; "basic" is the
    original 0019 rule (stop at k matches). Unfiltered queries, and collections without stored
    statistics, always use "basic".
    """
    pb = col.pb
    level = level or col.security_level
    if level not in LEVELS:
        raise ValueError(f"security level must be one of {list(LEVELS)}")
    if probe not in POLICIES:
        raise ValueError(f"probe must be one of {list(POLICIES)}")
    if k < 1 or rerank_factor < 1 or max_probe_factor < 1:
        raise ValueError("k, rerank_factor and max_probe_factor must be positive")
    if decoys is not None and (decoys < 1 if level == "decoy" else decoys < 0):
        raise ValueError("the Decoy level needs at least one decoy per round")
    q = np.asarray(query, dtype=np.float64)
    if q.shape != (col.derived.dimension,):
        raise ValueError("query has the wrong dimension")
    if abs(float(np.linalg.norm(q)) - 1.0) > 1e-6:
        raise ValueError("query must be normalized to unit length before encryption")
    n_fields = len(col.cards)
    ext = filter_query(q, col.cards, filter)
    total = col.cluster_count
    sizes: dict[int, list[int]] = {}
    listing = col.stub.ListBlocks(
        pb.ListBlocksRequest(api_version=API_VERSION, collection=col.name)
    )
    for b in listing.blocks:
        sizes.setdefault(b.cluster_id, []).append(b.n_vectors)  # catalog order = result order
    # decoys must be unpredictable; a seed is for tests only
    rng = secrets.SystemRandom() if seed is None else random.Random(seed)  # noqa: S311

    start = pb.SearchStart(
        collection=col.name,
        key_tag=col.kit.key_tag,
        query=col.kit.encrypt_query(ext),
        filter_leaves=sum(col.cards),
    )
    stream = _Stream(col.stub, pb.SearchRequest(start=start))
    cand: dict[bytes, float] = {}
    probed: list[int] = []
    asked_all: set[int] = set()  # every cluster requested on this stream, real or decoy
    rounds = 0

    def run_round(chosen: list[int]) -> None:
        nonlocal rounds
        # A cluster is never requested twice on one stream: a repeat would tell the server which
        # earlier choices were decoys. Results of a cluster asked earlier are already in `cand`.
        asked = [c for c in dict.fromkeys(chosen) if c not in asked_all]
        asked_all.update(asked)
        if not asked:
            rounds += 1
            return
        stream.send(pb.SearchRequest(chosen=pb.SearchChosen(cluster_ids=asked)))
        msgs = [stream.next().scores for _ in asked]
        envs, counts, owners = [], [], []
        for m in msgs:
            blocks = sizes.get(m.cluster_id, [])
            if len(m.score_blocks) != len(blocks) or len(m.ids) != sum(blocks):
                raise RuntimeError("block layout changed during the search")
            pos = 0
            for env, n in zip(m.score_blocks, blocks, strict=True):
                envs.append(bytes(env))
                counts.append(n)
                owners.append([bytes(i) for i in m.ids[pos : pos + n]])
                pos += n
        for vals, ids in zip(_decrypt_round(col, envs, counts, workers), owners, strict=True):
            c, score = decode_band(vals)
            for j, ident in enumerate(ids):
                if c[j] == n_fields:  # live (c >= 0) and every field matches
                    cand[ident] = float(score[j])
        rounds += 1

    fell_back = False
    try:
        centers = stream.next().centers
        cs = col.kit.decrypt_column(bytes(centers.score_blocks[0]))[: centers.center_count]
        order = [int(i) for i in np.argsort(-cs, kind="stable")]
        first = max(1, math.ceil(math.sqrt(total))) if n_probe is None else max(1, n_probe)
        first = min(first, total)
        if level == "strict":
            for part in _chunks(order):
                run_round(part)
            probed = list(order)
        else:
            policy = probe if (filter and col.stats is not None) else "basic"
            plan = ProbePlanner(
                order,
                policy,
                k,
                first,
                stats=col.stats,
                filter=filter,
                center_scores=cs,
                max_rounds=max_rounds,
                max_probe_factor=max_probe_factor,
            )
            real = plan.first_round()
            if real is None:  # expected selectivity too low: a full scan is the better plan
                for part in _chunks(order):
                    run_round(part)
                probed = list(order)
            else:
                while real:
                    probed += real
                    extra: list[int] = []
                    if level == "decoy":
                        rest = [
                            c for c in range(total) if c not in asked_all and c not in set(real)
                        ]
                        n_decoys = len(real) if decoys is None else decoys
                        extra = rng.sample(rest, min(n_decoys, len(rest)))
                    run_round(real + extra)  # decoy results are valid candidates too
                    kth = (
                        sorted(cand.values(), reverse=True)[k - 1] if len(cand) >= k else -math.inf
                    )
                    real = plan.next_round(kth, len(cand), asked_all)
                if len(cand) < k and strict_fallback and len(set(probed)) < total:
                    rest_all = [c for c in order if c not in set(probed)]
                    for part in _chunks(rest_all):
                        run_round(part)
                    probed += rest_all
                    fell_back = True
    finally:
        stream.close()

    top = sorted(cand.items(), key=lambda kv: (-kv[1], kv[0]))[: rerank_factor * k]
    return SearchResult(
        hits=_rerank(col, q, top, k),
        incomplete=len(cand) < k,
        probed=probed,
        rounds=rounds,
        matches=len(cand),
        strict_fallback=fell_back,
    )


def _rerank(col: Any, q: np.ndarray, top: list[tuple[bytes, float]], k: int) -> list[Hit]:
    scored = []
    for ident, _ in top:
        rec = col.codec.open_record(ident, col.blobs.get(col.name, KIND_RECORD, ident))
        v = np.frombuffer(rec.vector, dtype="<f4").astype(np.float64)
        meta = {k: m for k, m in rec.metadata.items() if k != FIELDS_KEY}
        scored.append(Hit(ident, float(v @ q), meta))
    scored.sort(key=lambda h: (-h.score, h.id))
    return scored[:k]
