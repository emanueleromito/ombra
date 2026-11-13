# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""The faiss k-means child process of ombra.clustering. Imports only numpy and faiss.

Run as a script, not as a module of the ombra package, so the child never imports ombra or
openfhe-python. faiss and openfhe-python each ship their own libomp on macOS, and libomp aborts
when a process loads two copies.

Protocol, over stdin and stdout, as .npy arrays (no pickle): the parent sends
params = [k, niter, seed, max_points_per_centroid] (int64) and then x (float32, n by d). The child
sends back centers (float64, k' by d, unit length) and labels (int64, n), with empty clusters
dropped and the clusters renumbered. Nothing is written to disk or printed.
"""

import sys

import faiss
import numpy as np


def kmeans(x: np.ndarray, k: int, niter: int, seed: int, max_ppc: int):
    km = faiss.Kmeans(
        x.shape[1],
        k,
        niter=niter,
        seed=seed,
        spherical=True,
        verbose=False,
        min_points_per_centroid=1,
        max_points_per_centroid=max_ppc,
    )
    km.train(x)
    centers = np.asarray(km.centroids, dtype=np.float64)
    centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    labels = np.argmax(np.asarray(x, dtype=np.float64) @ centers.T, axis=1)
    used = np.unique(labels)  # drop empty clusters and renumber, so every cluster has blocks
    remap = np.full(k, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return centers[used], remap[labels]


class _Stream:
    """Only read and write, so numpy streams the .npy data instead of seeking (pipes cannot)."""

    def __init__(self, f) -> None:
        self.read, self.write = getattr(f, "read", None), getattr(f, "write", None)


def main() -> None:
    src, dst = _Stream(sys.stdin.buffer), _Stream(sys.stdout.buffer)
    params = np.lib.format.read_array(src, allow_pickle=False)
    x = np.lib.format.read_array(src, allow_pickle=False)
    k, niter, seed, max_ppc = (int(v) for v in params)
    centers, labels = kmeans(x, k, niter, seed, max_ppc)
    np.lib.format.write_array(dst, centers, allow_pickle=False)
    np.lib.format.write_array(dst, labels, allow_pickle=False)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
