# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Client-side cluster statistics and the probe policy for filtered search.

Stopping at k matches in the first round lost recall at about 1% selectivity, so the client plans
probes from per-cluster statistics. Pure numpy logic with no OpenFHE and no server.

* `ClusterStats`: per cluster the number of live vectors and the number per value of each filter
  field. The client builds it at create, updates it on insert and delete, and rebuilds it on
  re-clustering. It is persisted only as an encrypted blob (`seal`/`unseal`,
  under the reserved id `STATS_ID`), never in plaintext.
* `ProbePlanner`: which clusters to search in each round. Every planned query has exactly two
  rounds, so the round count does not depend on the filter. Policies:
  - "e2" (filtered queries with statistics): round 1 orders clusters by center score, skips
    clusters with zero expected matches and adds clusters until the expected match count reaches
    3k (never fewer than the unfiltered first round). Round 2 is always one widening round: the
    next clusters with nonzero expected matches, 2x as many as round 1's real clusters, capped by
    what is left. Below 0.5% expected selectivity `first_round` returns None: scan everything
    (the caller splits it into two rounds by cluster id).
  - "plain2" (unfiltered queries at Decoy/Strict, filtered queries without statistics): round 1 =
    the first `first` clusters by center score, round 2 = the next 2 * `first`.
  - "basic": the first rule, first round = unfiltered count, widen 2x below `stop_matches`
    (k by default), up to `max_probe_factor` times the first round. Only for unfiltered Fast
    queries, which may use a different round count.
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
POLICIES = ("e2", "plain2", "basic")
_MAGIC = b"OSTA"
_VERSION = 3  # 2 also held the radii of the dropped check A; refused on load


def check_user_id(ident: bytes) -> None:
    if bytes(ident).startswith(RESERVED_PREFIX):
        raise ValueError("ids starting with b'\\x00ombra/' are reserved")


@dataclass
class ClusterStats:
    cards: tuple[int, ...]
    sizes: np.ndarray  # (C,) live vectors per cluster
    counts: list[np.ndarray]  # per field, (C, card): live vectors per value

    @classmethod
    def empty(cls, n_clusters: int, cards: Sequence[int]) -> ClusterStats:
        return cls(
            tuple(cards),
            np.zeros(n_clusters, dtype=np.int64),
            [np.zeros((n_clusters, c), dtype=np.int64) for c in cards],
        )

    @classmethod
    def build(
        cls,
        labels: np.ndarray,
        n_clusters: int,
        fields: np.ndarray,
        cards: Sequence[int],
    ) -> ClusterStats:
        s = cls.empty(n_clusters, cards)
        s.add(labels, fields)
        return s

    @property
    def n_clusters(self) -> int:
        return len(self.sizes)

    def add(self, labels: np.ndarray, fields: np.ndarray) -> None:
        labels = np.asarray(labels, dtype=np.int64)
        if len(labels) == 0:
            return
        self.sizes += np.bincount(labels, minlength=self.n_clusters)
        for fi in range(len(self.cards)):
            np.add.at(self.counts[fi], (labels, np.asarray(fields)[:, fi]), 1)

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

    # ---- persistence: sealed by the caller's DocumentCodec, never stored in plaintext --------
    def to_bytes(self) -> bytes:
        head = struct.pack("<4sHIH", _MAGIC, _VERSION, self.n_clusters, len(self.cards))
        head += struct.pack(f"<{len(self.cards)}I", *self.cards)
        body = self.sizes.astype("<i8").tobytes()
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
        off, counts = 8 * n, []
        for c in cards:
            counts.append(np.frombuffer(body, "<i8", n * c, off).reshape(n, c).astype(np.int64))
            off += 8 * n * c
        return cls(tuple(cards), sizes, counts)

    def seal(self, codec: Any) -> bytes:
        raw = self.to_bytes()
        raw += b"\0" * (-len(raw) % 4)  # a record's vector is float32 bytes; the tail is ignored
        return codec.seal_record(STATS_ID, raw, {})

    @classmethod
    def unseal(cls, codec: Any, blob: bytes) -> ClusterStats:
        return cls.from_bytes(codec.open_record(STATS_ID, blob).vector)


def strict_halves(n_clusters: int) -> list[list[int]]:
    """Strict: every cluster once, in two rounds, in a fixed data-independent order (cluster id)."""
    h = (n_clusters + 1) // 2
    return [list(range(h)), list(range(h, n_clusters))]


def decoy_round_sizes(first: int, n_clusters: int) -> tuple[int, int]:
    """Requested clusters (real plus decoys) in Decoy rounds 1 and 2: 2*first, then 4*first.

    Depends only on the first-round count and the collection size, never on the filter. Round 1's
    real clusters are capped at 2*first and round 2's real ones at 2x round 1's, so they always
    fit. When fewer clusters are left, the round asks for all of those (also filter-independent)."""
    r1 = min(n_clusters, 2 * first)
    return r1, min(n_clusters - r1, 4 * first)


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
        cap1: int | None = None,
        expect_factor: int = 3,
        min_selectivity: float = 0.005,
        stop_matches: int | None = None,
        max_probe_factor: int = 8,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"probe policy must be one of {list(POLICIES)}")
        self.order = [int(c) for c in order]
        self.policy, self.k, self.first = policy, k, max(1, min(first, len(self.order)))
        self.expect_factor, self.min_selectivity = expect_factor, min_selectivity
        self.cap1 = len(self.order) if cap1 is None else max(1, cap1)  # max real clusters, round 1
        self.stop_matches = k if stop_matches is None else stop_matches
        self.cap = min(len(self.order), self.first * max_probe_factor)
        self.taken: set[int] = set()
        self.rounds = 0
        self.n1 = self.first
        if policy == "e2":
            if stats is None:
                raise ValueError("policy e2 needs cluster stats")
            self.expected = stats.expected(filter)
            self.total = float(stats.sizes.sum())

    def strict_by_selectivity(self) -> bool:
        return (
            self.policy == "e2"
            and self.total > 0
            and float(self.expected.sum()) / self.total < self.min_selectivity
        )

    def first_round(self) -> list[int] | None:
        """The first round, or None when a full (Strict) scan is the better plan."""
        self.rounds = 1
        if self.policy == "e2":
            if self.strict_by_selectivity():
                return None
            real, acc = [], 0.0
            for c in self.order:
                if len(real) >= self.cap1:
                    break
                if self.expected[c] <= 0:
                    continue
                if acc >= self.expect_factor * self.k and len(real) >= self.first:
                    break
                real.append(c)
                acc += self.expected[c]
            self.n1 = max(1, len(real))
        else:
            real = self.order[: self.first]
        self.taken.update(real)
        return real

    def next_round(self, n_matches: int = 0, searched: set[int] | None = None) -> list[int]:
        """The next round's clusters, [] to stop. `searched` are clusters already asked, e.g. as
        decoys. e2 and plain2 always have exactly one more round (possibly empty if nothing is
        left); basic stops at `stop_matches` matches or its cap."""
        skip = self.taken | (searched or set())
        if self.policy == "basic":
            if n_matches >= self.stop_matches or len(self.taken) >= self.cap:
                return []
            size = self.first * 2**self.rounds
            real = [c for c in self.order[: self.cap] if c not in self.taken][:size]
        else:
            if self.rounds >= 2:
                return []
            rest = [c for c in self.order if c not in skip]
            if self.policy == "e2":
                rest = [c for c in rest if self.expected[c] > 0]
            real = rest[: 2 * self.n1]
        self.rounds += 1
        self.taken.update(real)
        return real
