# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Spherical k-means for the coarse stage of the two-stage search.

Vectors are unit length, so cosine similarity is a dot product and the centers are unit vectors
too. faiss does the k-means (it is fast and standard; sklearn would pull a far larger dependency).
Clustering runs on the client on plaintext vectors the client already holds; the server only ever
sees the encrypted centers.

faiss runs in a short-lived child process (ombra/_kmeans_worker.py, a fresh interpreter that
imports only numpy and faiss), and this process never imports faiss. On macOS faiss and
openfhe-python each ship their own libomp, and libomp aborts when one process loads both. The
child runs on every platform, so every test run exercises the same code. The vectors pass only
through the pipe between the two local processes: nothing is written to disk or logged.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_WORKER = Path(__file__).with_name("_kmeans_worker.py")


@dataclass(frozen=True)
class Clustering:
    centers: np.ndarray  # (k, d) unit vectors; every cluster has at least one vector
    labels: np.ndarray  # (n,) cluster of each input vector


def cluster_vectors(
    vectors: np.ndarray, n_clusters: int, *, seed: int = 0, niter: int = 20
) -> Clustering:
    """Cluster unit vectors into at most `n_clusters` non-empty clusters (deterministic by seed)."""
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("vectors must be a non-empty 2-D array")
    k = max(1, min(int(n_clusters), len(x)))
    params = np.array([k, niter, seed, max(256, len(x))], dtype=np.int64)
    centers, labels = _run_worker(params, x)
    return Clustering(centers=centers, labels=labels)


class _Stream:
    """Only read and write, so numpy streams the .npy data instead of seeking (pipes cannot)."""

    def __init__(self, f) -> None:
        self.read, self.write = getattr(f, "read", None), getattr(f, "write", None)


def _run_worker(params: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # -P: do not put ombra/ on sys.path, so no ombra module can shadow one numpy or faiss imports.
    proc = subprocess.Popen(  # noqa: S603 (our own script, run by this interpreter)
        [sys.executable, "-P", str(_WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None

    def send() -> None:
        try:
            out = _Stream(proc.stdin)
            np.lib.format.write_array(out, params, allow_pickle=False)
            np.lib.format.write_array(out, x, allow_pickle=False)
        except BrokenPipeError:
            pass  # the child failed; its exit code and stderr say why
        finally:
            proc.stdin.close()

    # Write from a thread while this one reads, so neither side blocks on a full pipe.
    writer = threading.Thread(target=send, daemon=True)
    writer.start()
    try:
        src = _Stream(proc.stdout)
        centers = np.lib.format.read_array(src, allow_pickle=False)
        labels = np.lib.format.read_array(src, allow_pickle=False)
    except (EOFError, ValueError):
        centers = labels = None
    writer.join()
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0 or centers is None or labels is None:
        raise RuntimeError(f"k-means child process failed (exit {proc.returncode}): {err[-2000:]}")
    return centers, labels


def assign(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest center (largest dot product) of each vector."""
    return np.argmax(np.asarray(vectors, dtype=np.float64) @ np.asarray(centers).T, axis=1)
