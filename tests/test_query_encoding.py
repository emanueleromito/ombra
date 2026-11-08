"""T0.5: client query encoding (client/ombra/query.py).

CPU-only checks of the numpy side (embedding, indices, expansion tree on plain polynomials) run
under `make test`. The `openfhe` test checks that OpenFHE's complex MakeCKKSPackedPlaintext accepts
the embedding and returns it on decode. That round trip does not depend on the slot convention, so
the coefficient-level check (which does) is in test_gpu_query_expansion.py via a C++ driver, because
openfhe-python exposes no access to the encoded polynomial.
"""

from __future__ import annotations

import pytest
from ombra import params as ombra_params
from ombra import query as oq

np = pytest.importorskip("numpy")

SLOT_TOLERANCE = 1e-9


def test_indices_and_depth() -> None:
    assert oq.expansion_depth(1) == 0
    assert oq.expansion_depth(13) == 4
    assert oq.expansion_depth(16) == 4
    assert oq.expansion_depth(17) == 5
    assert oq.automorphism_indices(16384, 13) == [16385, 8193, 4097, 2049]
    assert oq.automorphism_indices(16384, 1) == []


def test_embedding_matches_direct_evaluation() -> None:
    n = 32
    rng = np.random.default_rng(1)
    q = rng.standard_normal(6)
    z = oq.canonical_embedding(q, n)
    zeta = np.exp(2j * np.pi / (2 * n))
    want = [sum(q[j] * zeta ** (j * int(e)) for j in range(q.size)) for e in oq.slot_exponents(n)]
    assert float(np.max(np.abs(z - np.array(want)))) < SLOT_TOLERANCE


def _auto(poly: np.ndarray, t: int) -> np.ndarray:
    n = poly.size
    out = np.zeros(n)
    for j, c in enumerate(poly):
        pos = (j * t) % (2 * n)
        if pos >= n:
            out[pos - n] -= c
        else:
            out[pos] += c
    return out


@pytest.mark.parametrize("d", [1, 2, 13, 16, 31])
def test_plain_polynomial_tree_gives_constants(d: int) -> None:
    """The tree of decision 0010 on plain polynomials: leaf r is the constant 2^L * q_r."""
    n = 64
    q = np.random.default_rng(2).standard_normal(d)
    depth = oq.expansion_depth(d)
    nodes = {0: np.concatenate([q, np.zeros(n - d)])}
    for level in range(depth):
        step = 1 << level
        nxt = {}
        for r, c in nodes.items():
            sub = _auto(c, n // step + 1)
            nxt[r] = c + sub
            if r + step < d:
                shifted = np.roll(c - sub, -step)
                shifted[n - step :] *= -1
                nxt[r + step] = shifted
        nodes = nxt
    assert sorted(nodes) == list(range(d))
    for r, c in nodes.items():
        want = np.zeros(n)
        want[0] = q[r] * (1 << depth)
        assert float(np.max(np.abs(c - want))) < SLOT_TOLERANCE


def test_encode_rejects_unnormalized() -> None:
    class Ctx:
        @staticmethod
        def GetRingDimension() -> int:  # noqa: N802
            return 16

    with pytest.raises(ValueError):
        oq.encode_query(Ctx(), None, np.array([2.0, 0.0]))


@pytest.mark.openfhe
def test_complex_plaintext_slots_round_trip() -> None:
    _, cc = ombra_params.make_context("m0_depth2")
    rng = np.random.default_rng(3)
    q = rng.standard_normal(16)
    q /= np.linalg.norm(q)
    slots = oq.canonical_embedding(q, cc.GetRingDimension())
    pt = cc.MakeCKKSPackedPlaintext(slots)
    got = np.array(pt.GetCKKSPackedValue())
    assert got.size == slots.size
    assert float(np.max(np.abs(got - slots))) < SLOT_TOLERANCE
