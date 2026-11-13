# 0026. Run faiss k-means in a child process

- **Status:** proposed
- **Date:** 2026-11-13

## Context

On macOS the faiss-cpu wheel and the ombra-openfhe wheel each bundle their own libomp. The client imports openfhe-python for keys and encryption, and faiss for the k-means of a collection build. Once both are in one process, libomp aborts with "OMP: Error #15". So every collection build on macOS crashed. On Linux each bundled copy gets a unique name, and the two coexist.

## Options considered

- **A. faiss k-means in a short-lived child process.** The client never loads faiss. Full speed on every platform. Costs one interpreter start (about a second) per clustering.
- **B. ombra-openfhe on macOS without OpenMP.** The simplest packaging, but encryption and local mode on macOS would run on one thread.
- **C. Our own numpy k-means, dropping faiss.** One dependency less, but clustering results change everywhere and the recall numbers would need re-checking.
- **D. `KMP_DUPLICATE_LIB_OK=TRUE`.** Rejected: libomp documents it as unsafe, with possible crashes or wrong results.

## Decision

A, on every platform, so there is one code path and every test run exercises it. `ombra.clustering.cluster_vectors` starts `ombra/_kmeans_worker.py` in a fresh interpreter (`python -P`) that imports only numpy and faiss. Collection create, re-clustering and repack all use it. The vectors and the results pass only through the pipes between the two local processes, as `.npy` arrays without pickle. Nothing is written to disk or logged.

This is a fresh interpreter started with `subprocess`, not `multiprocessing` with the spawn method. Spawn runs the parent's `__main__` module again in the child. A user script that imports openfhe at the top would then load openfhe in the child too, and the crash would come back.

## Consequences

- The client process can load openfhe-python on macOS without loading a second libomp.
- Tests check that the result is identical to the earlier in-process k-means with the same seed (`client/tests/test_clustering_child.py`). They also check that a collection build loads openfhe but never faiss in the client process (`tests/test_openfhe_no_faiss_in_parent.py`, and `tests/wheel_check.py` in every wheel job, macOS included).
- Each clustering pays one interpreter start and one copy of the vectors through a pipe.
- Later we may replace faiss with our own numpy k-means (option C). That drops both the dependency and the child process, but needs a recall re-check.
