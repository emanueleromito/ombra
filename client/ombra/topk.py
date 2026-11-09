"""Decrypt stage-2 result blocks and pick the top-k, serially or across processes (T2.1).

Each block is one serialized OpenFHE ciphertext file holding `slots` scores. A block is
deserialized, decrypted, and reduced to its local top-k with global indices; the per-block lists
are then merged. Both paths use the same order (score descending, global index ascending), so their
results are identical, ties included.

Parallel path: worker processes are forked (start method "fork"), so they inherit the crypto
context and the secret key from the parent's memory. The key is never pickled, passed as an
argument or written to disk. Only block indices go in, and (scores, indices) of at most k entries
per block come out. Nothing here logs scores.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np

# Set in the parent before the pool forks; children read their copy. Never sent through pickle.
_STATE: dict[str, Any] = {}


def _order(scores: np.ndarray, indices: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The k best entries, ordered by score descending then index ascending."""
    o = np.lexsort((indices, -scores))[:k]
    return scores[o], indices[o]


def _decrypt_block(path: str, index: int, count: int, slots: int, k: int):
    import openfhe as ofhe

    st = _STATE
    ct, ok = ofhe.DeserializeCiphertext(path, ofhe.BINARY)
    if not ok:
        raise RuntimeError(f"cannot deserialize block {index}")
    dec = st["cc"].Decrypt(ct, st["sk"])
    dec.SetLength(slots)
    vals = np.asarray(dec.GetRealPackedValue(), dtype=np.float64)[:count]
    idx = np.arange(count, dtype=np.int64) + index * slots
    return _order(vals, idx, k)


def _task(args: tuple[str, int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    return _decrypt_block(*args)


def _merge(parts: list[tuple[np.ndarray, np.ndarray]], k: int) -> tuple[np.ndarray, np.ndarray]:
    return _order(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), k)


def _tasks(files: list[Path | str], slots: int, valid_last: int, k: int) -> list[tuple]:
    if not files:
        raise ValueError("no blocks")
    if not 0 < valid_last <= slots:
        raise ValueError("valid_last must be in 1..slots")
    last = len(files) - 1
    return [(str(f), i, valid_last if i == last else slots, slots, k) for i, f in enumerate(files)]


def serial_topk(
    files: list[Path | str], cc: Any, sk: Any, slots: int, valid_last: int, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, global indices) of the k best entries over all blocks, one process."""
    tasks = _tasks(files, slots, valid_last, k)
    _STATE.update(cc=cc, sk=sk)
    try:
        return _merge([_task(t) for t in tasks], k)
    finally:
        _STATE.clear()


class ParallelTopk:
    """A pool of forked workers that inherit `cc` and `sk` from this process's memory.

    Create it once (the fork costs about as much as a few blocks) and call `run` per query. The
    workers hold the secret key in their own memory only for the pool's lifetime; use `close()` or
    a `with` block.
    """

    def __init__(self, cc: Any, sk: Any, workers: int | None = None) -> None:
        _STATE.update(cc=cc, sk=sk)
        try:
            self._pool = mp.get_context("fork").Pool(workers or len(os.sched_getaffinity(0)))
        finally:
            _STATE.clear()  # the children keep their copy; the parent's registry is emptied

    def run(
        self, files: list[Path | str], slots: int, valid_last: int, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        tasks = _tasks(files, slots, valid_last, k)
        return _merge(self._pool.map(_task, tasks, chunksize=1), k)

    def close(self) -> None:
        self._pool.terminate()
        self._pool.join()

    def __enter__(self) -> ParallelTopk:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def parallel_topk(
    files: list[Path | str],
    cc: Any,
    sk: Any,
    slots: int,
    valid_last: int,
    k: int,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Same result as serial_topk, with blocks decrypted by forked worker processes."""
    n = max(1, min(workers or len(os.sched_getaffinity(0)), len(files)))
    with ParallelTopk(cc, sk, n) as pool:
        return pool.run(files, slots, valid_last, k)
