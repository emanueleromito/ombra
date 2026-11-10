"""Spherical k-means for the coarse stage of the two-stage search (T3.5).

Vectors are unit length, so cosine similarity is a dot product and the centers are unit vectors
too. faiss does the k-means (it is fast and standard; sklearn would pull a far larger dependency).
Clustering runs on the client on plaintext vectors the client already holds; the server only ever
sees the encrypted centers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Clustering:
    centers: np.ndarray  # (k, d) unit vectors; every cluster has at least one vector
    labels: np.ndarray  # (n,) cluster of each input vector


def _unit(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def cluster_vectors(
    vectors: np.ndarray, n_clusters: int, *, seed: int = 0, niter: int = 20
) -> Clustering:
    """Cluster unit vectors into at most `n_clusters` non-empty clusters (deterministic by seed)."""
    import faiss

    x = np.ascontiguousarray(vectors, dtype=np.float32)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("vectors must be a non-empty 2-D array")
    k = max(1, min(int(n_clusters), len(x)))
    km = faiss.Kmeans(
        x.shape[1],
        k,
        niter=niter,
        seed=seed,
        spherical=True,
        verbose=False,
        min_points_per_centroid=1,
        max_points_per_centroid=max(256, len(x)),
    )
    km.train(x)
    centers = _unit(np.asarray(km.centroids, dtype=np.float64))
    labels = np.argmax(np.asarray(x, dtype=np.float64) @ centers.T, axis=1)
    used = np.unique(labels)  # drop empty clusters and renumber, so every cluster has blocks
    remap = np.full(k, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Clustering(centers=centers[used], labels=remap[labels])


def assign(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest center (largest dot product) of each vector."""
    return np.argmax(np.asarray(vectors, dtype=np.float64) @ np.asarray(centers).T, axis=1)
