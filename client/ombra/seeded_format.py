# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Payload of envelope kind 4 (seeded ciphertext, docs/decisions/0023), pure Python.

Mirrors core/include/ombra/core/seeded_ciphertext.hpp. The client normally gets these bytes from
ombra._native; this module reads and writes the layout without OpenFHE, for tests and tools.
Little-endian: u16 payload version (1) | u16 generator (1) | 32 seed | u32 N | u16 T | u32 level |
u32 noise scale degree | u32 slots | f64 scaling factor | u16 K + key tag | T * u64 moduli |
T * N * u64 words of c0'.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

PAYLOAD_VERSION = 1
GENERATOR_BLAKE2XB = 1
SEED_BYTES = 32
_FIXED = struct.Struct("<HH32sIHIIId")  # up to and including the scaling factor


class SeededFormatError(ValueError):
    """Raised for a malformed seeded payload."""


@dataclass(frozen=True)
class SeededPayload:
    seed: bytes
    ring_dim: int
    towers: int
    level: int
    noise_scale_deg: int
    slots: int
    scaling_factor: float
    key_tag: str
    moduli: tuple[int, ...]
    c0: tuple[int, ...]  # [tower][coeff] flattened
    payload_version: int = PAYLOAD_VERSION
    generator: int = GENERATOR_BLAKE2XB


def write_payload(p: SeededPayload) -> bytes:
    tag = p.key_tag.encode()
    if len(tag) > 65535 or len(p.seed) != SEED_BYTES:
        raise SeededFormatError("bad key tag or seed length")
    if len(p.moduli) != p.towers or len(p.c0) != p.towers * p.ring_dim:
        raise SeededFormatError("inconsistent shape")
    head = _FIXED.pack(
        p.payload_version,
        p.generator,
        p.seed,
        p.ring_dim,
        p.towers,
        p.level,
        p.noise_scale_deg,
        p.slots,
        p.scaling_factor,
    )
    return (
        head
        + struct.pack("<H", len(tag))
        + tag
        + struct.pack(f"<{p.towers}Q", *p.moduli)
        + struct.pack(f"<{len(p.c0)}Q", *p.c0)
    )


def read_payload(data: bytes, *, check_words: bool = True) -> SeededPayload:
    """Parse and validate; sizes are checked against the bytes present before anything is built."""
    if len(data) < _FIXED.size + 2:
        raise SeededFormatError("truncated payload")
    ver, gen, seed, n, t, level, nsd, slots, sf = _FIXED.unpack_from(data)
    if ver != PAYLOAD_VERSION:
        raise SeededFormatError(f"unknown payload version {ver}")
    if gen != GENERATOR_BLAKE2XB:
        raise SeededFormatError(f"unknown generator {gen}")
    if n == 0 or n & (n - 1):
        raise SeededFormatError("ring dimension not a power of two")
    if t == 0:
        raise SeededFormatError("zero towers")
    pos = _FIXED.size
    (k,) = struct.unpack_from("<H", data, pos)
    pos += 2
    if len(data) - pos < k:
        raise SeededFormatError("truncated payload")
    tag = data[pos : pos + k].decode("utf-8", errors="replace")
    pos += k
    want = 8 * t + 8 * t * n
    if len(data) - pos < want:
        raise SeededFormatError("payload size does not match N and T")
    if len(data) - pos > want:
        raise SeededFormatError("trailing bytes after payload")
    moduli = struct.unpack_from(f"<{t}Q", data, pos)
    pos += 8 * t
    for q in moduli:
        if q < 2 or q % 2 == 0 or q >= 1 << 60:
            raise SeededFormatError("modulus must be odd and in [2, 2^60)")
    words = struct.unpack_from(f"<{t * n}Q", data, pos)
    if check_words:
        for i, w in enumerate(words):
            if w >= moduli[i // n]:
                raise SeededFormatError("word not reduced modulo its modulus")
    return SeededPayload(seed, n, t, level, nsd, slots, sf, tag, moduli, words, ver, gen)


def fixture_payload() -> SeededPayload:
    """The synthetic payload of tests/fixtures/envelope_v2_seeded.bin (not a real ciphertext)."""
    return SeededPayload(
        seed=bytes(range(32)),
        ring_dim=8,
        towers=2,
        level=0,
        noise_scale_deg=1,
        slots=4,
        scaling_factor=2.0**40,
        key_tag="fixture-tag",
        moduli=(97, 193),
        c0=tuple(range(8)) + tuple(100 + i for i in range(8)),
    )
