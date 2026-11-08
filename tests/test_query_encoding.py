"""T0.5: client query encoding (client/ombra/query.py).

CPU-only checks of the numpy side (symmetric coefficients, embedding, indices, expansion tree on
plain polynomials) run under `make test`. The `openfhe` tests check, through OpenFHE, that the real
MakeCKKSPackedPlaintext accepts the embedding and, via the C++ driver `ombra_encoding_check`
(openfhe-python cannot reach the encoded polynomial), that the encoded coefficients are q_j * scale
at index j and -q_j * scale at index N - j.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from ombra import params as ombra_params
from ombra import query as oq

np = pytest.importorskip("numpy")

SLOT_TOLERANCE = 1e-9
# Max |coefficient / scale - target| over all N coefficients of OpenFHE's encoded polynomial, in
# units of the scaling factor 2^50. Error sources: double-precision FFT (about 1e-15 relative) and
# rounding to integers (0.5 / 2^50 = 4e-16). 1e-12 leaves a wide margin.
ENCODING_TOLERANCE = 1e-12
SEED = 20260108
PRESET = "m0_depth2"
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_indices_and_depth() -> None:
    assert oq.expansion_depth(1) == 1
    assert oq.expansion_depth(13) == 5
    assert oq.expansion_depth(16) == 5
    assert oq.expansion_depth(17) == 6
    assert oq.expansion_depth(384) == 10
    assert oq.expansion_depth(768) == 11
    assert oq.expansion_depth(1536) == 12
    assert oq.automorphism_indices(16384, 13) == [16385, 8193, 4097, 2049, 1025]
    assert oq.automorphism_indices(16384, 1) == [16385]


def test_symmetric_coefficients() -> None:
    n = 32
    q = np.array([0.5, 1.0, 2.0, 3.0])
    c = oq.symmetric_coefficients(q, n)
    want = np.zeros(n)
    want[:4] = q
    want[n - 1], want[n - 2], want[n - 3] = -1.0, -2.0, -3.0  # index N - j holds -q_j
    assert np.array_equal(c, want)
    assert np.array_equal(oq.symmetric_coefficients(np.array([0.7]), n), np.eye(n)[0] * 0.7)


@pytest.mark.parametrize("d", [1, 13, 16])
def test_symmetric_slots_are_real_and_match_cosine_form(d: int) -> None:
    n = 64
    q = np.random.default_rng(4).standard_normal(d)
    z = oq.canonical_embedding(oq.symmetric_coefficients(q, n), n)
    assert float(np.max(np.abs(z.imag))) < oq.IMAG_TOLERANCE
    # p(zeta^e) = q_0 + 2 sum_j q_j cos(pi j e / N)
    e = oq.slot_exponents(n)
    want = q[0] + sum(2 * q[j] * np.cos(np.pi * j * e / n) for j in range(1, d))
    assert float(np.max(np.abs(oq.symmetric_slots(q, n) - want))) < SLOT_TOLERANCE


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


@pytest.mark.parametrize("d", [1, 2, 13, 16, 31, 32])
def test_plain_polynomial_tree_gives_constants(d: int) -> None:
    """The tree of decision 0010 on plain symmetric polynomials: leaf r is the constant 2^L q_r."""
    n = 64
    q = np.random.default_rng(2).standard_normal(d)
    depth = oq.expansion_depth(d)
    assert (1 << depth) >= 2 * d
    nodes = {0: oq.symmetric_coefficients(q, n)}
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


def _binary(name: str) -> Path:
    candidates = [
        os.environ.get(f"OMBRA_{name.upper()}_BIN", ""),
        f"/root/ombra-build/core/{name}",
        str(REPO_ROOT / "build" / "core" / name),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.fail(f"{name} not found; build it (`make test-openfhe` builds the CPU drivers)")


@pytest.mark.openfhe
def test_real_plaintext_slots_round_trip() -> None:
    _, cc = ombra_params.make_context(PRESET)
    q = np.random.default_rng(3).standard_normal(16)
    q /= np.linalg.norm(q)
    slots = oq.symmetric_slots(q, cc.GetRingDimension())
    pt = cc.MakeCKKSPackedPlaintext(slots.tolist())
    got = np.array(pt.GetRealPackedValue())
    assert got.size == slots.size
    assert float(np.max(np.abs(got - slots))) < SLOT_TOLERANCE


@pytest.mark.openfhe
@pytest.mark.parametrize("d", [1, 13, 16])
def test_encoded_coefficients_are_symmetric_query(tmp_path: Path, d: int) -> None:
    """OpenFHE's encoded polynomial has q_j*scale at j, -q_j*scale at N-j, zero elsewhere."""
    import openfhe as ofhe

    _, cc = ombra_params.make_context(PRESET)
    ring = cc.GetRingDimension()
    q = np.random.default_rng(SEED + d).standard_normal(d)
    q /= np.linalg.norm(q)
    assert ofhe.SerializeToFile(str(tmp_path / "context.bin"), cc, ofhe.BINARY)
    (tmp_path / "slots.bin").write_bytes(oq.symmetric_slots(q, ring).astype("<f8").tobytes())
    (tmp_path / "q.bin").write_bytes(q.astype("<f8").tobytes())
    proc = subprocess.run(  # noqa: S603
        [
            str(_binary("ombra_encoding_check")),
            str(tmp_path / "context.bin"),
            str(tmp_path / "slots.bin"),
            str(tmp_path / "q.bin"),
            str(d),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"driver failed (exit {proc.returncode}): {proc.stderr}"
    print(f"encoding d={d}: {proc.stdout.strip()}")
    prefix = "max_coeff_error_in_scale_units="
    value = float(proc.stdout.split(prefix)[1].split()[0])
    assert value < ENCODING_TOLERANCE
