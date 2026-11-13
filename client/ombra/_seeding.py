# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Seeded stored columns (docs/decisions/0023), client side: worker processes around ombra._native.

The compiled module is never imported by the client process (it holds the secret key through the
openfhe-python wheel, and one process must not hold two OpenFHE copies). A small pool of spawned
worker processes imports `ombra._native` and builds one `Seeder` each, from the crypto context,
the public key and the secret key as bytes. The keys travel once through the pool's pipe to child
processes of this client and live in their memory only for the pool's lifetime; they are never
written to disk, logged or sent anywhere else. Only column values go in and kind-4 payloads or
decrypted column values come out; nothing here logs them.

`available()` says whether the client writes seeded columns: the compiled module is installed and
the environment variable OMBRA_SEEDED is not "0" (the opt-out for users and for tests of the
old format).
"""

from __future__ import annotations

import importlib.util
import os
import weakref
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

_NATIVE_HINT = (
    "this collection holds seeded columns (envelope kind 4), which need the compiled module "
    "ombra._native (build client/ with OpenFHE 1.6.1, see client/CMakeLists.txt)"
)


class SeededUnavailableError(RuntimeError):
    """A seeded column was met but ombra._native is not available."""

    def __init__(self) -> None:
        super().__init__(_NATIVE_HINT)


def native_installed() -> bool:
    """True when ombra._native can be found (it is not imported here)."""
    try:
        return importlib.util.find_spec("ombra._native") is not None
    except (ImportError, ValueError):
        return False


def available() -> bool:
    """Write seeded columns? Native module present and OMBRA_SEEDED not "0"."""
    return os.environ.get("OMBRA_SEEDED", "1") != "0" and native_installed()


# State of a worker process (empty in the parent).
_STATE: dict[str, Any] = {}


def _init(cc_bytes: bytes, pk_bytes: bytes, sk_bytes: bytes) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    import ctypes

    try:
        ctypes.CDLL("libgomp.so.1").omp_set_num_threads(1)
    except (OSError, AttributeError):
        pass
    from ombra import _native

    if getattr(_native, "SEEDED_API", 0) < 1:
        raise RuntimeError("ombra._native has no seeded-column API; rebuild it")
    _STATE["seeder"] = _native.Seeder(cc_bytes, pk_bytes, sk_bytes)


def _encrypt(values: np.ndarray) -> bytes:
    return bytes(_STATE["seeder"].encrypt_column(values))


def _decrypt(args: tuple[bytes, int]) -> np.ndarray:
    return np.asarray(_STATE["seeder"].decrypt_column(*args), dtype=np.float64)


def _shutdown(pool: ProcessPoolExecutor) -> None:
    pool.shutdown(wait=False, cancel_futures=True)


class SeedPool:
    """Spawned workers, each loaded once with the context and the key pair (see module doc)."""

    def __init__(
        self, cc_bytes: bytes, pk_bytes: bytes, sk_bytes: bytes, workers: int | None = None
    ):
        import multiprocessing as mp

        n = workers or max(1, min(4, len(os.sched_getaffinity(0))))
        self._workers = n
        self._pool = ProcessPoolExecutor(
            max_workers=n,
            mp_context=mp.get_context("spawn"),
            initializer=_init,
            initargs=(cc_bytes, pk_bytes, sk_bytes),
        )
        self._finalizer = weakref.finalize(self, _shutdown, self._pool)

    def encrypt_columns(self, columns: list[np.ndarray]) -> list[bytes]:
        """Kind-4 payloads, one per padded column (one float64 per slot)."""
        chunk = max(1, len(columns) // (4 * self._workers))
        return list(self._pool.map(_encrypt, columns, chunksize=chunk))

    def decrypt_columns(self, payloads: list[bytes], slots: int) -> list[np.ndarray]:
        chunk = max(1, len(payloads) // (4 * self._workers))
        return list(self._pool.map(_decrypt, [(p, slots) for p in payloads], chunksize=chunk))

    def close(self) -> None:
        self._finalizer()
