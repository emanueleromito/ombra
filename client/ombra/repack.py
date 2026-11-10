"""Planning of block repacking (T3.5). Pure logic, no I/O, no crypto.

A deleted vector is one slot in every column of its block, so only the client can drop deleted
slots: decrypt, drop, re-encrypt, ReplaceBlock. ReplaceBlock turns 1..N blocks of one cluster into
one block, so merging half-empty blocks is what makes the stored bytes fall.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_THRESHOLD = 0.30


@dataclass(frozen=True)
class BlockStat:
    segment: str
    cluster: int
    n_vectors: int  # occupied slots, deleted ones included
    n_deleted: int

    @property
    def live(self) -> int:
        return self.n_vectors - self.n_deleted

    @property
    def deleted_fraction(self) -> float:
        return self.n_deleted / self.n_vectors if self.n_vectors else 0.0


def plan_repack(
    blocks: Sequence[BlockStat],
    slots: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[list[BlockStat]]:
    """Groups of blocks (one cluster each) to merge into one block each.

    Data blocks only (the caller leaves out the center block). A cluster is touched only if one of
    its blocks has at least `threshold` of its slots deleted. Its blocks with a live vector are
    packed first-fit-decreasing into groups of at most `slots` live vectors; a group is rewritten
    if it holds a flagged block or merges several blocks. Blocks with no live vector are dropped by
    the server itself and are left out. Deterministic.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    by_cluster: dict[int, list[BlockStat]] = {}
    for b in blocks:
        if b.live > 0:
            by_cluster.setdefault(b.cluster, []).append(b)

    def flagged(b: BlockStat) -> bool:
        return b.deleted_fraction >= threshold

    plan: list[list[BlockStat]] = []
    for cluster in sorted(by_cluster):
        group = by_cluster[cluster]
        if not any(flagged(b) for b in group):
            continue
        bins: list[list[BlockStat]] = []
        for b in sorted(group, key=lambda x: (-x.live, x.segment)):
            for bin_ in bins:
                if sum(m.live for m in bin_) + b.live <= slots:
                    bin_.append(b)
                    break
            else:
                bins.append([b])
        plan.extend(bin_ for bin_ in bins if len(bin_) > 1 or flagged(bin_[0]))
    return plan
