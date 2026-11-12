"""Ombra file envelope v1: a small header around an OpenFHE BINARY payload.

Little-endian layout, mirrored by core/include/ombra/core/file_envelope.hpp:
  magic "OMBR" | u16 version (1) | u16 kind | u16 preset length | preset |
  u16 OpenFHE version length | OpenFHE version | u64 payload length | payload (file ends here)
There is no kind for a secret key (the secret key never leaves the client). No OpenFHE dependency.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass

MAGIC = b"OMBR"
FORMAT_VERSION = 1
MAX_NAME = 255


class Kind(enum.IntEnum):
    CONTEXT = 1
    CIPHERTEXT = 2
    PUBLIC_EVAL_KEY = 3


class EnvelopeError(ValueError):
    """Raised when a file is not a valid, expected Ombra envelope."""


@dataclass(frozen=True)
class Envelope:
    kind: Kind
    preset: str
    openfhe_version: str
    payload: bytes


def write_envelope(kind: Kind, preset: str, openfhe_version: str, payload: bytes) -> bytes:
    names = [preset.encode(), openfhe_version.encode()]
    if any(not 1 <= len(n) <= MAX_NAME for n in names):
        raise EnvelopeError("preset and OpenFHE version must be 1..255 bytes")
    return (
        MAGIC
        + struct.pack("<HHH", FORMAT_VERSION, int(Kind(kind)), len(names[0]))
        + names[0]
        + struct.pack("<H", len(names[1]))
        + names[1]
        + struct.pack("<Q", len(payload))
        + bytes(payload)
    )


def read_envelope(
    data: bytes, expected_kind: Kind, expected_preset: str, expected_openfhe_version: str
) -> Envelope:
    """Parse and validate; the payload length is checked against the bytes present first."""
    pos = 0

    def take(n: int) -> bytes:
        nonlocal pos
        if len(data) - pos < n:
            raise EnvelopeError("truncated file")
        out = data[pos : pos + n]
        pos += n
        return out

    if take(4) != MAGIC:
        raise EnvelopeError("bad magic, not an Ombra file")
    (version,) = struct.unpack("<H", take(2))
    if version != FORMAT_VERSION:
        raise EnvelopeError(f"unknown format version {version}")
    (kind,) = struct.unpack("<H", take(2))
    if kind != int(expected_kind):
        raise EnvelopeError(f"wrong kind {kind}")

    def name(what: str) -> str:
        (n,) = struct.unpack("<H", take(2))
        if not 1 <= n <= MAX_NAME:
            raise EnvelopeError(f"bad {what} length")
        return take(n).decode("utf-8", errors="replace")

    preset = name("preset")
    if preset != expected_preset:
        raise EnvelopeError(f"preset mismatch (file has {preset!r})")
    ofhe = name("OpenFHE version")
    if ofhe != expected_openfhe_version:
        raise EnvelopeError(f"OpenFHE version mismatch (file has {ofhe!r})")
    (length,) = struct.unpack("<Q", take(8))
    if length > len(data) - pos:
        raise EnvelopeError("payload length exceeds file")
    if length < len(data) - pos:
        raise EnvelopeError("trailing bytes after payload")
    return Envelope(Kind(kind), preset, ofhe, take(length))
