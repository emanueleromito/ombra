"""The GPU cluster cache pages blocks without changing search results.

Python generates keys and seeded synthetic data (both presets), packs the stored vectors column-
wise (one ciphertext per dimension) and the pre-expanded query, encrypts them, and writes only
public material (context, ciphertexts) to disk. The C++ driver `ombra_cache_search` loads each
block through the cache loader (OpenFHE ciphertexts -> device RNS words), runs the unchanged column
search kernel over every block twice in random orders with a GPU budget of one third of the
collection, and compares every RNS word of every result with the all-resident run (exact equality,
non-zero exit on mismatch). It also checks that a leased block survives eviction pressure and that
a budget below one block is refused. This test asserts the driver's verdicts and that evictions
happened with a hit rate strictly between 0 and 1. The secret key stays in this process and is
never written; nothing prints values.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
from ombra import params as ombra_params

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.gpu

PRESETS = ["m0_depth2", "m0_compact_n8192"]
SEED = 20261110
DIMS = 4
BLOCKS = 9  # GPU budget = 3 blocks: room for the leased block, the current one and the prefetch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _driver() -> Path:
    candidates = [
        os.environ.get("OMBRA_CACHE_SEARCH_BIN", ""),
        "/root/ombra-build/server/cache/ombra_cache_search",
        str(REPO_ROOT / "build" / "server" / "cache" / "ombra_cache_search"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail("ombra_cache_search not found; build with `make build` or run `make test-gpu`")


@pytest.mark.parametrize("preset", PRESETS)
def test_cache_paging_matches_resident(tmp_path: Path, preset: str) -> None:
    import openfhe as ofhe

    _, cc = ombra_params.make_context(preset)
    keys = cc.KeyGen()
    slots = ombra_params.get_preset(preset)["batch_size"]
    rng = np.random.default_rng(SEED)

    assert ofhe.SerializeToFile(str(tmp_path / "context.bin"), cc, ofhe.BINARY)
    for j in range(DIMS):
        q = cc.MakeCKKSPackedPlaintext(np.full(slots, rng.standard_normal() / 2).tolist())
        ct = cc.Encrypt(keys.publicKey, q)
        assert ofhe.SerializeToFile(str(tmp_path / f"query_{j}.bin"), ct, ofhe.BINARY)
        for b in range(BLOCKS):
            col = cc.MakeCKKSPackedPlaintext((rng.standard_normal(slots) / 2).tolist())
            ct = cc.Encrypt(keys.publicKey, col)
            assert ofhe.SerializeToFile(str(tmp_path / f"stored_{b}_{j}.bin"), ct, ofhe.BINARY)

    proc = subprocess.run(  # noqa: S603
        [str(_driver()), str(tmp_path / "context.bin"), str(tmp_path), str(DIMS), str(BLOCKS), "7"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"driver failed (exit {proc.returncode}): {proc.stderr}"
    out = proc.stdout
    assert "cache paging exact match: PASS" in out
    assert "cache pinned block survives eviction pressure: PASS" in out
    assert "cache refuses a block larger than the GPU budget: PASS" in out

    m = re.search(r"hits=(\d+) misses=(\d+) evictions=(\d+) hit_rate=([0-9.]+)", out)
    assert m, out
    hits, misses, evictions, hit_rate = int(m[1]), int(m[2]), int(m[3]), float(m[4])
    print(f"{preset}: hits={hits} misses={misses} evictions={evictions} hit_rate={hit_rate:.3f}")
    assert evictions > 0
    assert hits + misses == 2 * BLOCKS
    assert 0.0 < hit_rate < 1.0
    # the collection really was larger than the GPU budget
    sizes = re.search(r"collection_bytes=(\d+) gpu_budget_bytes=(\d+)", out)
    assert sizes
    assert int(sizes[1]) >= 3 * int(sizes[2]) - 3 * 256
