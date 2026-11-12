# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Parallel decrypt + top-k gives the same result as the serial path.

Synthetic data only. Blocks are real CKKS ciphertexts at level 2 with a few forced ties.
"""

from __future__ import annotations

import pytest
from ombra import params as ombra_params
from ombra import topk

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.openfhe

PRESET = "m0_compact_n8192"
K = 10


def _write_blocks(tmp_path, cc, pk, slots, valid_last, nblocks, rng):
    import openfhe as ofhe

    # Distinct values 1e-3 apart: OpenFHE's CKKS decryption adds fresh noise on every call, so
    # exactly equal true scores can swap places between two decryptions of the same ciphertext.
    # Separated scores make the ranking well defined; exact tie order is tested on _order below.
    total = (nblocks - 1) * slots + valid_last
    flat = rng.permutation(total) * 1e-3 - 10.0
    flat = np.concatenate([flat, np.zeros(slots - valid_last)])
    files, allvals = [], []
    for b in range(nblocks):
        vals = flat[b * slots : (b + 1) * slots]
        allvals.append(vals)
        ct = cc.Encrypt(pk, cc.MakeCKKSPackedPlaintext(vals.tolist(), 1, 2))
        f = tmp_path / f"b{b}.bin"
        assert ofhe.SerializeToFile(str(f), ct, ofhe.BINARY)
        files.append(f)
    return files, allvals


def test_parallel_equals_serial(tmp_path) -> None:
    _, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()
    slots = cc.GetRingDimension() // 2
    valid_last = slots // 3
    rng = np.random.default_rng(7)
    files, allvals = _write_blocks(tmp_path, cc, keys.publicKey, slots, valid_last, 5, rng)

    s_sc, s_ix = topk.serial_topk(files, cc, keys.secretKey, slots, valid_last, K)
    p_sc, p_ix = topk.parallel_topk(files, cc, keys.secretKey, slots, valid_last, K, workers=3)

    assert np.array_equal(s_ix, p_ix)
    assert np.max(np.abs(s_sc - p_sc)) < 1e-4  # same values up to fresh decryption noise
    assert len(s_ix) == K
    assert s_ix.max() < 4 * slots + valid_last  # padding of the short last block is excluded
    # the ids match an exact plaintext ranking up to CKKS noise
    flat = np.concatenate(allvals)
    assert np.max(np.abs(flat[s_ix] - s_sc)) < 1e-4
    assert set(s_ix) == set(np.argsort(-flat[: 4 * slots + valid_last])[:K])


def test_order_breaks_ties_by_index() -> None:
    scores = np.array([1.0, 2.0, 2.0, 0.5, 2.0])
    idx = np.array([10, 7, 3, 1, 5])
    sc, ix = topk._order(scores, idx, 4)
    assert list(ix) == [3, 5, 7, 10]
    assert list(sc) == [2.0, 2.0, 2.0, 1.0]
