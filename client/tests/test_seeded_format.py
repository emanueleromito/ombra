# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Seeded payload format (kind 4) and the v2 envelope fixture. Synthetic bytes; no OpenFHE."""

from __future__ import annotations

import dataclasses
import struct
from pathlib import Path

import pytest
from ombra.file_format import Kind, read_envelope, write_envelope
from ombra.seeded_format import (
    SeededFormatError,
    fixture_payload,
    read_payload,
    write_payload,
)

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "envelope_v2_seeded.bin"
P = fixture_payload()
GOOD = write_payload(P)
TAG_AT = 2 + 2 + 32 + 4 + 2 + 4 + 4 + 4 + 8  # offset of the key tag length


def test_round_trip() -> None:
    assert read_payload(GOOD) == P


def test_cross_language_fixture() -> None:
    """ctest file_envelope reads the same file with the C++ reader."""
    expected = write_envelope(Kind.SEEDED_CIPHERTEXT, "fixture_preset", "0.0.0-fixture", GOOD)
    assert FIXTURE.read_bytes() == expected
    env = read_envelope(expected, Kind.SEEDED_CIPHERTEXT, "fixture_preset", "0.0.0-fixture")
    assert read_payload(env.payload) == P


def _patched(off: int, fmt: str, value: int) -> bytes:
    b = bytearray(GOOD)
    struct.pack_into(fmt, b, off, value)
    return bytes(b)


def test_refusals() -> None:
    with pytest.raises(SeededFormatError, match="payload version"):
        read_payload(_patched(0, "<H", 2))
    with pytest.raises(SeededFormatError, match="generator"):
        read_payload(_patched(2, "<H", 2))
    with pytest.raises(SeededFormatError, match="power of two"):
        read_payload(_patched(36, "<I", 0))
    with pytest.raises(SeededFormatError, match="power of two"):
        read_payload(_patched(36, "<I", 12))
    with pytest.raises(SeededFormatError, match="zero towers"):
        read_payload(_patched(40, "<H", 0))
    with pytest.raises(SeededFormatError, match="does not match"):
        read_payload(_patched(40, "<H", 3))
    with pytest.raises(SeededFormatError, match="does not match"):
        read_payload(_patched(36, "<I", 16))
    with pytest.raises(SeededFormatError, match="trailing"):
        read_payload(GOOD + b"\0")
    with pytest.raises(SeededFormatError):
        read_payload(GOOD[:-1])
    with pytest.raises(SeededFormatError, match="does not match|truncated"):
        read_payload(_patched(TAG_AT, "<H", 65535))  # key tag longer than the bytes present
    mod_at = TAG_AT + 2 + len(P.key_tag)
    for bad in (0, 1, 98, 1 << 60, (1 << 60) + 1, (1 << 63) + 1):
        with pytest.raises(SeededFormatError, match="modulus"):
            read_payload(_patched(mod_at, "<Q", bad))
    word_at = mod_at + 16
    with pytest.raises(SeededFormatError, match="reduced"):
        read_payload(_patched(word_at, "<Q", 97))
    assert read_payload(_patched(word_at, "<Q", 97), check_words=False)  # header check only
    for n in range(len(GOOD)):
        with pytest.raises(SeededFormatError):
            read_payload(GOOD[:n])
    with pytest.raises(SeededFormatError):
        write_payload(dataclasses.replace(P, key_tag="x" * 65536))
