# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Runs one scripted client session against a server; the upgrade tests start it in a subprocess.

    python driver.py write ADDR WORKDIR      # create, keys, insert, delete, replace, blobs
    python driver.py read ADDR WORKDIR       # reopen, check everything, then write once more

PYTHONPATH decides which `ombra` client runs it (the current one or the previous release's). It
uses only the public Client API plus the collection object behind it (block list, blobs), which
the smoke test in tests/test_server_cpu.py uses too. WORKDIR holds `key.bin` and `state.json`
(counts and ids of synthetic data only). The last line of stdout is one JSON object:
{"ok": true, ...} or {"ok": false, "error": "...", "where": "..."}. Synthetic data only.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

D = 384  # the smallest supported dimension
N = 40
NAME = "upgrade"
PASSPHRASE = "pw"  # noqa: S105 - synthetic
NO_GPU = "without GPU support"


def vectors(n: int = N, seed: int = 7) -> np.ndarray:
    x = np.random.default_rng(seed).normal(size=(n, D))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _collection(client):
    return client._core._get(NAME)[0]


def _live(col) -> list[str]:
    return sorted(i.decode() for b in col.block_stats() for i in col.read_live(b.segment)[0])


def _expect_no_gpu(client, x) -> None:
    """Search starts and fails with the CPU build's own error, not with a protocol error."""
    try:
        client.search(NAME, x[35], k=3)
    except Exception as e:  # noqa: BLE001
        if NO_GPU in str(e):
            return
        raise AssertionError(f"search failed with another error: {type(e).__name__}: {e}") from e
    raise AssertionError("search succeeded on a server without a GPU")


def _info(client) -> list[int]:
    i = client.get_collection(NAME)
    return [i.vector_count, i.deleted_count]


def write(client, work: Path) -> dict:
    from ombra.blobstore import BUCKET_SIZES, KIND_RECORD

    x = vectors()
    ids = [f"v{i}" for i in range(N)]
    client.create_collection(
        NAME,
        x,
        ids,
        key_file=work / "key.bin",
        passphrase=PASSPHRASE,
        preset="fast",
        n_clusters=1,
        scrypt_params=(2**4, 8, 1),
    )
    client.insert(NAME, x[:5], [f"w{i}" for i in range(5)])
    client.delete(NAME, ids[:30])
    col = _collection(client)
    col.repack(threshold=0.1)  # ReplaceBlock
    client.delete(NAME, ["w0"])  # a delete after the replace: a tombstone on the new block
    blob = bytes(range(256)) * (BUCKET_SIZES[0] // 256)
    col.blobs.put(col.name, KIND_RECORD, b"rec1", blob)
    col.blobs.delete(col.name, KIND_RECORD, [b"rec1"])
    col.blobs.put(col.name, KIND_RECORD, b"rec2", blob)
    _expect_no_gpu(client, x)
    state = {"live": _live(col), "info": _info(client)}
    (work / "state.json").write_text(json.dumps(state))
    return state


def read(client, work: Path) -> dict:
    from ombra.blobstore import BUCKET_SIZES, KIND_RECORD

    state = json.loads((work / "state.json").read_text())
    x = vectors()
    assert client.list_collections() == [NAME], client.list_collections()
    client.open_collection(NAME, key_file=work / "key.bin", passphrase=PASSPHRASE)
    assert _info(client) == state["info"], (_info(client), state["info"])
    col = _collection(client)
    assert _live(col) == state["live"], "live ids differ from what was written"
    blob = bytes(range(256)) * (BUCKET_SIZES[0] // 256)
    assert col.blobs.get(col.name, KIND_RECORD, b"rec2") == blob, "blob differs"
    _expect_no_gpu(client, x)
    # writes work on the opened data
    client.insert(NAME, x[:1], ["z0"])
    victim = state["live"][0]
    client.delete(NAME, [victim])
    live = _live(col)
    assert live == sorted([*(set(state["live"]) - {victim}), "z0"]), "writes did not apply"
    # later reads (and the next phase of the test) start from what this phase left
    state = {"live": live, "info": _info(client)}
    (work / "state.json").write_text(json.dumps(state))
    return {"info": state["info"]}


def main() -> int:
    phase, addr, work = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    from ombra import Client

    client = Client(addr, insecure=True)
    try:
        result = {"write": write, "read": read}[phase](client, work)
    except BaseException as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}"[:1500],
                    "where": f"{phase} ({Path(tb.filename).name}:{tb.lineno})",
                }
            )
        )
        return 1
    finally:
        client.close()
    print(json.dumps({"ok": True, **result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
