"""Symmetric real query encoding for server-side expansion.

The query q (d normalized values) becomes the polynomial p(X) = q_0 + sum_{j>=1} q_j (X^j + X^-j)
in R = Z[X]/(X^N+1), where X^-j = -X^(N-j): coefficient j is q_j, coefficient N-j is -q_j, all
others are 0. Such a p has real slots, which is what OpenFHE's default REAL mode keeps. The slot
values are the real canonical embedding of p, computed with numpy and handed to OpenFHE's own
`MakeCKKSPackedPlaintext` (real overload) and `Encrypt`. Nothing here implements encryption or
touches key material beyond passing the public key to OpenFHE.

Convention, read from OpenFHE v1.6.1: slot k of a full-slot plaintext is
z_k = m(zeta^(5^k mod 2N)) with zeta = exp(2*pi*i / 2N)
(DiscreteFourierTransform::PrecomputedValues, core/lib/math/dftransform.cpp:60-64 powers of five,
:70-72 positive-angle roots). Because p is
symmetric (p(X) = p(X^-1) as a function on roots of unity), its slots are real and do not depend on
the sign convention of the root.

Pre-scaling: the server's expansion tree multiplies every value by 2^L, L = expansion_depth(d), and
does no EvalMult and no scale change to undo it. The client therefore encodes
q * 2^-L (exact: a power of two only changes the exponent), so each expanded leaf decrypts to q_j.
The norm check is done on the unscaled q.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Vectors must be unit length to within this much (vectors are normalized before encryption).
NORM_TOLERANCE = 1e-6
# The embedding of a symmetric p is real up to floating-point error; its imaginary part must be
# below this (absolute) or the encoding is refused.
IMAG_TOLERANCE = 1e-9


def expansion_depth(d: int) -> int:
    """L = ceil(log2(2d)): 2^L >= 2d, so the tree separates index j from N - j."""
    if d < 1:
        raise ValueError("d must be at least 1")
    return (2 * d - 1).bit_length()


def automorphism_indices(ring_dim: int, d: int) -> list[int]:
    """Galois elements N/2^k + 1 for k in [0, L), L = ceil(log2(2d)), in level order.

    `ring_dim` must come from the crypto context (`cc.GetRingDimension()`), never a constant.
    """
    if d > ring_dim // 2:
        raise ValueError("d must be at most N/2")
    return [ring_dim // (1 << k) + 1 for k in range(expansion_depth(d))]


def symmetric_coefficients(query: np.ndarray, ring_dim: int) -> np.ndarray:
    """Coefficients of p(X) = q_0 + sum_{j>=1} q_j (X^j + X^-j): q_j at j, -q_j at N - j."""
    q = np.asarray(query, dtype=np.float64)
    if q.ndim != 1 or q.size == 0 or q.size > ring_dim // 2:
        raise ValueError("query must be one-dimensional with 1 to N/2 entries")
    coeffs = np.zeros(ring_dim)
    coeffs[: q.size] = q
    coeffs[ring_dim - q.size + 1 :] = -q[1:][::-1]  # index N - j holds -q_j for j >= 1
    return coeffs


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
    if coeffs.ndim != 1 or coeffs.size > ring_dim:
        raise ValueError("coeffs must be one-dimensional with at most N entries")
    full = np.fft.ifft(coeffs, n=2 * ring_dim) * (2 * ring_dim)
    return full[slot_exponents(ring_dim)]


def symmetric_slots(query: np.ndarray, ring_dim: int) -> np.ndarray:
    """Real slot values (N/2 doubles) of the symmetric polynomial for `query`."""
    z = canonical_embedding(symmetric_coefficients(query, ring_dim), ring_dim)
    if float(np.max(np.abs(z.imag))) > IMAG_TOLERANCE:
        raise ValueError("symmetric query embedding is not real; refusing to encode")
    return np.ascontiguousarray(z.real)


def prescale_query(query: np.ndarray) -> np.ndarray:
    """Return query * 2^-L, L = expansion_depth(len(query)), exactly (np.ldexp, no rounding)."""
    q = np.asarray(query, dtype=np.float64)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("query must be a non-empty one-dimensional vector")
    return np.ldexp(q, -expansion_depth(q.size))


def encode_query(cc: Any, public_key: Any, query: np.ndarray) -> Any:
    """Encrypt the symmetric coefficient-packed query with OpenFHE and return the ciphertext.

    `query` must be a normalized vector of d values. It is pre-scaled by 2^-L (`prescale_query`)
    before encoding, so the plaintext polynomial has q_j * 2^-L in coefficient j and
    -q_j * 2^-L in coefficient N - j, and the server's tree restores q_j.
    """
    q = np.asarray(query, dtype=np.float64)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("query must be a non-empty one-dimensional vector")
    if abs(float(np.linalg.norm(q)) - 1.0) > NORM_TOLERANCE:
        raise ValueError("query must be normalized to unit length before encryption")
    slots = symmetric_slots(prescale_query(q), cc.GetRingDimension())
    pt = cc.MakeCKKSPackedPlaintext(slots.tolist())  # real overload: no CKKSDataType change
    return cc.Encrypt(public_key, pt)
