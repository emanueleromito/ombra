# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Writes a small data directory with the CURRENT code (CPU server, current client).

    python tests/upgrade/make_fixture.py OUT_DIR

OUT_DIR then holds `data/` (the server's data dir), `key.bin` (the client key file, passphrase
"pw", cheap scrypt, synthetic keys), `state.json` (what a reader must find) and `VERSION`
(the Ombra and OpenFHE versions that wrote it). The data is synthetic: d = 384 (the smallest
supported dimension), preset fast, 40 vectors in one cluster, plus inserts, deletes, a block
replacement and two blobs, exactly the driver's `write` phase.

Freeze one at every release: run this at the release commit with
`OUT_DIR = tests/fixtures/datadir-<version>` and commit the result (see docs/release.md). Every
later version must keep opening every frozen fixture (tests/upgrade/test_upgrade.py).
If a fixture is too large for git (over 5 MB), the test generates the current one at test time
and caches it instead; frozen ones then have to be kept as release assets.
Needs `make build-cpu` and openfhe-python (run it from the repository root).
"""

from __future__ import annotations

import sys
from pathlib import Path

from common import REPO_ROOT, run_driver

sys.path.insert(0, str(REPO_ROOT / "client"))
from server_harness import ServerProcess  # noqa: E402


def make(out: Path) -> Path:
    import ombra
    import openfhe

    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "_server"  # the harness puts server.log next to data/
    with ServerProcess(scratch) as server:
        run_driver(REPO_ROOT / "client", "write", server.address, out)
        server.stop()
    (scratch / "data").rename(out / "data")
    for leftover in scratch.iterdir():
        leftover.unlink()
    scratch.rmdir()
    (out / "VERSION").write_text(f"ombra {ombra.__version__}\nopenfhe {openfhe.__version__}\n")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(make(Path(sys.argv[1])))
