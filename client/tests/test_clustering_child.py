# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""cluster_vectors runs faiss in a child process and gives the same result as in-process faiss.

The reference below is the in-process k-means cluster_vectors used before the child process.
Synthetic data only.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

np = pytest.importorskip("numpy")
faiss = pytest.importorskip("faiss")

import ombra  # noqa: E402
from ombra.clustering import cluster_vectors  # noqa: E402


def _reference(vectors, n_clusters, *, seed=0, niter=20):
    x = np.ascontiguousarray(vectors, dtype=np.float32)
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
    centers = np.asarray(km.centroids, dtype=np.float64)
    centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    labels = np.argmax(np.asarray(x, dtype=np.float64) @ centers.T, axis=1)
    used = np.unique(labels)
    remap = np.full(k, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return centers[used], remap[labels]


def _unit_vectors(n, d, seed):
    x = np.random.default_rng(seed).standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.mark.parametrize(
    ("n", "d", "k", "seed"), [(300, 16, 8, 0), (1000, 64, 20, 7), (5, 8, 9, 1)]
)
def test_child_matches_in_process_kmeans(n, d, k, seed):
    x = _unit_vectors(n, d, seed)
    got = cluster_vectors(x, k, seed=seed)
    centers, labels = _reference(x, k, seed=seed)
    np.testing.assert_array_equal(got.centers, centers)
    np.testing.assert_array_equal(got.labels, labels)
    assert got.labels.dtype == np.int64 and got.centers.dtype == np.float64


def test_parent_never_imports_faiss():
    code = textwrap.dedent(
        """
        import sys
        import numpy as np
        from ombra.clustering import cluster_vectors
        x = np.random.default_rng(0).standard_normal((200, 16))
        c = cluster_vectors(x / np.linalg.norm(x, axis=1, keepdims=True), 4)
        assert len(c.labels) == 200, c.labels.shape
        assert "faiss" not in sys.modules, "faiss was imported in the parent"
        """
    )
    root = os.path.dirname(os.path.dirname(ombra.__file__))  # the folder that holds ombra/
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([root, os.environ.get("PYTHONPATH", "")])}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)  # noqa: S603


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        cluster_vectors(np.zeros((0, 4)), 2)
