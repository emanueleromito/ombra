"""Coefficient-packed query encoding for server-side expansion (decision 0010, T0.5).

The query q (d normalized values) becomes the polynomial q(X) = sum_j q_j X^j in R = Z[X]/(X^N+1).
CKKS slots hold the canonical embedding of that polynomial, so we compute the embedding with numpy
and hand the slot values to OpenFHE's own `MakeCKKSPackedPlaintext` (complex overload) and
`Encrypt`. Nothing here implements encryption or touches key material beyond passing the public key
to OpenFHE.

Convention, read from OpenFHE v1.6.1: slot k of a full-slot plaintext is
z_k = m(zeta^(5^k mod 2N)) with zeta = exp(2*pi*i / 2N) (DiscreteFourierTransform::PrecomputedValues,
core/lib/math/dftransform.cpp:60-64 powers of five, :70-72 positive-angle roots; the decode
pairing of coefficient idx with idx + N/2 is pke/lib/encoding/ckkspackedencoding.cpp:315-377).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Vectors must be unit length to within this much (client/CLAUDE.md: normalize before encryption).
NORM_TOLERANCE = 1e-6


def expansion_depth(d: int) -> int:
    """L = ceil(log2 d): the number of tree levels needed to isolate d coefficients."""
    if d < 1:
        raise ValueError("d must be at least 1")
    return (d - 1).bit_length()


def automorphism_indices(ring_dim: int, d: int) -> list[int]:
    """Galois elements N/2^k + 1 for k in [0, L), L = ceil(log2 d), in level order.

    `ring_dim` must come from the crypto context (`cc.GetRingDimension()`), never a constant.
    """
    depth = expansion_depth(d)
    if d > ring_dim // 2:
        raise ValueError("d must be at most N/2")
    return [ring_dim // (1 << k) + 1 for k in range(depth)]


def slot_exponents(ring_dim: int) -> np.ndarray:
    """Exponents 5^k mod 2N for the N/2 slots, in OpenFHE's slot order."""
    modulus = 2 * ring_dim
    out = np.empty(ring_dim // 2, dtype=np.int64)
    e = 1
    for k in range(out.size):
        out[k] = e
        e = (e * 5) % modulus
    return out


def canonical_embedding(coeffs: np.ndarray, ring_dim: int) -> np.ndarray:
    """Slot values z_k = sum_j coeffs[j] * zeta^(j * 5^k), zeta = exp(2*pi*i / 2N).

    Computed as one length-2N inverse FFT (which evaluates at all 2N-th roots of unity) followed by
    a gather at the powers of five.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim != 1 or coeffs.size > ring_dim // 2:
        raise ValueError("coeffs must be one-dimensional with at most N/2 entries")
    full = np.fft.ifft(coeffs, n=2 * ring_dim) * (2 * ring_dim)
    return full[slot_exponents(ring_dim)]


def encode_query(cc: Any, public_key: Any, query: np.ndarray) -> Any:
    """Encrypt the coefficient-packed query with OpenFHE and return the ciphertext.

    `query` must be a normalized vector of d values. The result is a single ciphertext whose
    plaintext polynomial has q_j in coefficient j.
    """
    q = np.asarray(query, dtype=np.float64)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("query must be a non-empty one-dimensional vector")
    if abs(float(np.linalg.norm(q)) - 1.0) > NORM_TOLERANCE:
        raise ValueError("query must be normalized to unit length before encryption")
    ring_dim = cc.GetRingDimension()
    slots = canonical_embedding(q, ring_dim)  # complex128, N/2 entries (a full-slot plaintext)
    pt = cc.MakeCKKSPackedPlaintext(slots)  # openfhe-python bindings.cpp:536, complex overload
    return cc.Encrypt(public_key, pt)
