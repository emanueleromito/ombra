# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Install check: runs against the INSTALLED package (`pip install ./client[local]`) on a
machine without a GPU: creates a tiny collection with Client.local and finds a vector. Synthetic
data only. Run by `modal run modal_app.py::test_local_install`."""

import tempfile
import time
from pathlib import Path

import numpy as np
import ombra

assert "site-packages" in ombra.__file__, ombra.__file__
rng = np.random.default_rng(0)
x = rng.standard_normal((12, 384))
x /= np.linalg.norm(x, axis=1, keepdims=True)
with tempfile.TemporaryDirectory() as t:
    t0 = time.time()
    client = ombra.Client.local(Path(t) / "data")
    client.create_collection(
        "tiny",
        x,
        [f"v{i}" for i in range(12)],
        key_file=Path(t) / "tiny.key",
        passphrase="pw",  # noqa: S106
        preset="fast",
        scrypt_params=(2**4, 8, 1),
    )
    t1 = time.time()
    hits = client.search("tiny", x[5], k=3)
    t2 = time.time()
    client.close()
assert hits[0].id == "v5" and abs(hits[0].score - 1.0) < 1e-5, hits
print(f"local install check OK: create {t1 - t0:.1f}s, search {t2 - t1:.1f}s, top hit {hits[0].id}")
