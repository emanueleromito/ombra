# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Check of an INSTALLED ombra wheel, run by cibuildwheel as test-command (client/pyproject.toml).

It proves, with synthetic data only, that:
  1. the wheel carries the compiled extension, and the local-mode child process loads it and not
     openfhe-python (so the expansion is the C++ OpenFHE tree of core/, not a Python one);
  2. a tiny encrypted search in local mode finds the source vector;
  3. the score blocks come back at ciphertext level 1, the level the GPU server's search leaves
     them at (the expansion uses no level): same number of levels as the server.

The same level is asserted against the server's by tests/test_gpu_local_parity.py (Modal, GPU).
Needs openfhe-python 1.6.1 in the environment (it is the client's own dependency for keys).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

SERVER_RESULT_LEVEL = 1  # a fresh ciphertext at level 0, one rescale after the dot product
D = 384


def main() -> None:
    # Imported here: the local server child is spawned and re-imports this module, and it must
    # not import openfhe-python (the check below).
    import numpy as np
    import ombra
    import openfhe as ofhe
    from ombra._local import LocalTransport
    from ombra.collection import Collection
    from ombra.file_format import Kind, read_envelope

    assert "site-packages" in ombra.__file__, f"not the installed package: {ombra.__file__}"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((12, D))
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)

        # 1 and 2: the Client API in local mode.
        client = ombra.Client.local(tmp / "data")
        try:
            client.create_collection(
                "tiny",
                x,
                [f"v{i}" for i in range(len(x))],
                key_file=tmp / "tiny.key",
                passphrase="pw",  # noqa: S106
                preset="fast",
                scrypt_params=(2**4, 8, 1),
            )
            mods = client._core._t.modules_loaded()
            assert mods == {"openfhe_loaded": False, "native_loaded": True}, mods
            hits = client.search("tiny", x[5], k=3)
            assert hits[0].id == "v5" and abs(hits[0].score - 1.0) < 1e-5, hits
        finally:
            client.close()

        # 3: the level of the score blocks, read from the serialized ciphertext.
        transport = LocalTransport(tmp / "local")
        try:
            pb, stub = transport.pb, transport.stub
            col = Collection.create(
                pb,
                stub,
                "lv",
                x,
                [b"v%d" % i for i in range(len(x))],
                preset="fast",
                openfhe_version=transport.openfhe_version,
            )
            start = pb.SearchStart(
                collection="lv",
                key_tag=col.kit.key_tag,
                query=col.kit.encrypt_query(x[2]),
                filter_leaves=0,
            )
            stream = stub.Search(iter([pb.SearchRequest(start=start)]))
            centers = next(stream).centers
            stream.close()
            env = read_envelope(
                bytes(centers.score_blocks[0]), Kind.CIPHERTEXT, "m0_compact_n8192", "1.6.1"
            )
            p = tmp / "c.bin"
            p.write_bytes(env.payload)
            ct, ok = ofhe.DeserializeCiphertext(str(p), ofhe.BINARY)
            assert ok and ct.GetLevel() == SERVER_RESULT_LEVEL, ct.GetLevel()
            col.close()
        finally:
            transport.close()

    print("wheel check OK: compiled expansion in use, result level", SERVER_RESULT_LEVEL)


if __name__ == "__main__":  # the local server child is spawned: the module must import cleanly
    main()
