# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""A collection build loads openfhe-python but never faiss in the client process.

faiss runs in its own child process (ombra.clustering), because on macOS faiss and openfhe-python
each ship a libomp and libomp aborts when one process loads two. The build runs in a fresh
interpreter, so no other test can have imported faiss there first. Synthetic data only.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

for name in ("numpy", "openfhe", "grpc_tools", "faiss"):
    if importlib.util.find_spec(name) is None:
        pytest.skip(f"{name} is not installed", allow_module_level=True)
pytest.importorskip("ombra._native", reason="the compiled extension is not built")

import ombra  # noqa: E402

pytestmark = pytest.mark.openfhe

CODE = textwrap.dedent(
    """
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np
    import ombra

    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 64))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    with tempfile.TemporaryDirectory() as t:
        client = ombra.Client.local(Path(t) / "data")
        try:
            client.create_collection(
                "c", x, [f"v{i}" for i in range(len(x))], key_file=Path(t) / "c.key",
                passphrase="pw", preset="fast", n_clusters=4, scrypt_params=(2**4, 8, 1),
            )
            hits = client.search("c", x[7], k=1)
            assert hits[0].id == "v7", hits
        finally:
            client.close()
    assert "openfhe" in sys.modules, "the build did not use openfhe-python"
    assert "faiss" not in sys.modules, "faiss was imported in the client process"
    """
)


def test_collection_build_never_loads_faiss_in_the_client_process():
    root = os.path.dirname(os.path.dirname(ombra.__file__))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([root, os.environ.get("PYTHONPATH", "")])}
    subprocess.run([sys.executable, "-c", CODE], check=True, env=env, timeout=600)  # noqa: S603
