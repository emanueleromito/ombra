"""Client-side cluster statistics and the probe policy for filtered search (T3.11, T3.6).

Decision 0019's first probe rule (stop at k matches in the first round) lost recall at about 1%
selectivity (docs/plan/results/2026-11-11-t3.11-filtered-recall.md). This module holds the fix, as
pure numpy logic with no OpenFHE and no server, so it is unit tested and also drives the plaintext
simulation in core/bench/filter_probe_sim.py:

* `ClusterStats`: per cluster the number of live vectors, the number per value of each filter field
  and `cos_min`, the smallest cosine between a member and its center. The client builds it at
  create, updates it on insert and delete, and rebuilds it on re-clustering. It is persisted only
  as an encrypted blob (`seal`/`unseal`, under the reserved id `STATS_ID`), never in plaintext.
* `ProbePlanner`: which clusters to search in each round. Policies:
  - "e_a" (default for filtered queries): order clusters by center score, skip clusters with zero
    expected matches, add clusters until the expected match count reaches 3k (never fewer than the
    unfiltered first round); then keep widening while some remaining cluster could still hold a
    vector that beats the current k-th match (the stopping check A), up to `max_rounds`.
  - "e": the same first round, then widen only while there are fewer than k matches.
  - "basic": decision 0019's rule: first round = unfiltered count, widen 2x below `stop_matches`
    (k by default), up to `max_probe_factor` times the first round.

Score bound used by A. For a cluster with center c and a member m, cos(m, c) >= cos_min, so the
angle theta_mc <= theta_r = arccos(cos_min). For a query q with center score s = cos(q, c),
theta_qc = arccos(s), and by the triangle inequality on angles theta_qm >= max(0, theta_qc -
theta_r), so score(q, m) <= cos(max(0, theta_qc - theta_r)). Deletes keep `cos_min` (still a valid
bound); inserts lower it; re-clustering recomputes it.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

RESERVED_PREFIX = b"\x00ombra/"
STATS_ID = RESERVED_PREFIX + b"stats"
FIELDS_KEY = "\x00ombra/fields"  # record metadata key: the filter values of a vector (for delete)
POLICIES = ("e_a", "e", "basic")
SCORE_SLACK = 1e-4  # CKKS noise on center scores is about 1e-5; the bound gets this much room
_MAGIC = b"OSTA"
_VERSION = 1


def check_user_id(ident: bytes) -> None:
    if bytes(ident).startswith(RESERVED_PREFIX):
        raise ValueError("ids starting with b'\\x00ombra/' are reserved")


@dataclass
class ClusterStats:
    cards: tuple[int, ...]
    sizes: np.ndarray  # (C,) live vectors per cluster
    counts: list[np.ndarray]  # per field, (C, card): live vectors per value
    cos_min: np.ndarray  # (C,) min cosine member-to-center (1.0 for an empty cluster)

    @classmethod
    def empty(cls, n_clusters: int, cards: Sequence[int]) -> ClusterStats:
        return cls(
            tuple(cards),
            np.zeros(n_clusters, dtype=np.int64),
            [np.zeros((n_clusters, c), dtype=np.int64) for c in cards],
            np.ones(n_clusters),
        )

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        fields: np.ndarray,
        cards: Sequence[int],
    ) -> ClusterStats:
        s = cls.empty(len(centers), cards)
        s.add(vectors, labels, centers, fields)
        return s

    @property
    def n_clusters(self) -> int:
        return len(self.sizes)

    def add(
        self, vectors: np.ndarray, labels: np.ndarray, centers: np.ndarray, fields: np.ndarray
    ) -> None:
        labels = np.asarray(labels, dtype=np.int64)
        if len(labels) == 0:
            return
        self.sizes += np.bincount(labels, minlength=self.n_clusters)
        for fi in range(len(self.cards)):
            np.add.at(self.counts[fi], (labels, np.asarray(fields)[:, fi]), 1)
        cos = np.einsum("ij,ij->i", np.asarray(vectors, dtype=np.float64), centers[labels])
        np.minimum.at(self.cos_min, labels, cos)

    def remove(self, labels: np.ndarray, fields: np.ndarray | None) -> None:
        """Delete vectors. Without their field values only the sizes can be updated."""
        labels = np.asarray(labels, dtype=np.int64)
        if len(labels) == 0:
            return
        self.sizes = np.maximum(self.sizes - np.bincount(labels, minlength=self.n_clusters), 0)
        if fields is not None:
            for fi in range(len(self.cards)):
                np.subtract.at(self.counts[fi], (labels, np.asarray(fields)[:, fi]), 1)
                np.maximum(self.counts[fi], 0, out=self.counts[fi])

    def expected(self, filter: Mapping[int, int | Sequence[int]] | None) -> np.ndarray:
        """Expected matches per cluster: size times the product of per-field match fractions."""
        out = self.sizes.astype(np.float64)
        if not filter:
            return out
        denom = np.maximum(self.sizes, 1).astype(np.float64)
        for fi, w in filter.items():
            vals = [int(w)] if isinstance(w, int | np.integer) else [int(v) for v in w]
            out = out * (self.counts[fi][:, vals].sum(axis=1) / denom)
        return out

    def upper_bounds(self, center_scores: np.ndarray) -> np.ndarray:
        """Per cluster: an upper bound on the score of any member (see the module docstring)."""
        th_q = np.arccos(np.clip(center_scores, -1.0, 1.0))
        th_r = np.arccos(np.clip(self.cos_min, -1.0, 1.0))
        return np.cos(np.maximum(0.0, th_q - th_r))

    # ---- persistence: sealed by the caller's DocumentCodec, never stored in plaintext --------
    def to_bytes(self) -> bytes:
        head = struct.pack("<4sHIH", _MAGIC, _VERSION, self.n_clusters, len(self.cards))
        head += struct.pack(f"<{len(self.cards)}I", *self.cards)
        body = self.sizes.astype("<i8").tobytes() + self.cos_min.astype("<f8").tobytes()
        body += b"".join(c.astype("<i8").tobytes() for c in self.counts)
        packed = zlib.compress(body, 6)
        return head + struct.pack("<I", len(packed)) + packed

    @classmethod
    def from_bytes(cls, raw: bytes) -> ClusterStats:
        h = struct.Struct("<4sHIH")
        magic, version, n, nf = h.unpack_from(raw)
        if magic != _MAGIC or version != _VERSION:
            raise ValueError("stats blob has an unknown format or version")
        cards = struct.unpack_from(f"<{nf}I", raw, h.size)
        o = h.size + 4 * nf
        (plen,) = struct.unpack_from("<I", raw, o)
        body = zlib.decompress(raw[o + 4 : o + 4 + plen])
        sizes = np.frombuffer(body, "<i8", n, 0).astype(np.int64)
        cos_min = np.frombuffer(body, "<f8", n, 8 * n).astype(np.float64)
        off, counts = 16 * n, []
        for c in cards:
            counts.append(np.frombuffer(body, "<i8", n * c, off).reshape(n, c).astype(np.int64))
            off += 8 * n * c
        return cls(tuple(cards), sizes, counts, cos_min)

    def seal(self, codec: Any) -> bytes:
        raw = self.to_bytes()
        raw += b"\0" * (-len(raw) % 4)  # a record's vector is float32 bytes; the tail is ignored
        return codec.seal_record(STATS_ID, raw, {})

    @classmethod
    def unseal(cls, codec: Any, blob: bytes) -> ClusterStats:
        return cls.from_bytes(codec.open_record(STATS_ID, blob).vector)


class ProbePlanner:
    """Chooses the clusters of each round. The caller runs the rounds and reports the results."""

    def __init__(
        self,
        order: Sequence[int],
        policy: str,
        k: int,
        first: int,
        *,
        stats: ClusterStats | None = None,
        filter: Mapping[int, int | Sequence[int]] | None = None,
        center_scores: np.ndarray | None = None,
        max_rounds: int = 4,
        expect_factor: int = 3,
        min_selectivity: float = 0.005,
        stop_matches: int | None = None,
        max_probe_factor: int = 8,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"probe policy must be one of {list(POLICIES)}")
        self.order = [int(c) for c in order]
        self.policy, self.k, self.first = policy, k, max(1, min(first, len(self.order)))
        self.max_rounds, self.expect_factor = max_rounds, expect_factor
        self.min_selectivity = min_selectivity
        self.stop_matches = k if stop_matches is None else stop_matches
        self.cap = min(len(self.order), self.first * max_probe_factor)
        self.taken: set[int] = set()
        self.rounds = 0
        self.n1 = self.first
        if policy != "basic":
            if stats is None or center_scores is None:
                raise ValueError("this policy needs cluster stats and center scores")
            self.expected = stats.expected(filter)
            self.bounds = stats.upper_bounds(np.asarray(center_scores, dtype=np.float64))
            self.total = float(stats.sizes.sum())

    def strict_by_selectivity(self) -> bool:
        return (
            self.policy != "basic"
            and self.total > 0
            and float(self.expected.sum()) / self.total < self.min_selectivity
        )

    def first_round(self) -> list[int] | None:
        """The first round, or None when a full (Strict) scan is the better plan."""
        self.rounds = 1
        if self.policy == "basic":
            real = self.order[: self.first]
        else:
            if self.strict_by_selectivity():
                return None
            real, acc = [], 0.0
            for c in self.order:
                if self.expected[c] <= 0:
                    continue
                if acc >= self.expect_factor * self.k and len(real) >= self.first:
                    break
                real.append(c)
                acc += self.expected[c]
            self.n1 = max(1, len(real))
        self.taken.update(real)
        return real

    def next_round(self, kth: float, n_matches: int, searched: set[int] | None = None) -> list[int]:
        """The next round's clusters, [] to stop. `kth` is the k-th best match score so far
        (-inf below k matches); `searched` are clusters already asked, e.g. as decoys."""
        skip = self.taken | (searched or set())
        if self.policy == "basic":
            if n_matches >= self.stop_matches or len(self.taken) >= self.cap:
                return []
            size = self.first * 2**self.rounds
            real = [c for c in self.order[: self.cap] if c not in self.taken][:size]
        else:
            if self.rounds >= self.max_rounds:
                return []
            rest = [c for c in self.order if c not in skip and self.expected[c] > 0]
            if self.policy == "e":
                rest = rest if n_matches < self.k else []
            else:
                rest = [c for c in rest if self.bounds[c] + SCORE_SLACK > kth]
            real = rest[: self.n1 * 2**self.rounds]
        if real:
            self.rounds += 1
            self.taken.update(real)
        return real
