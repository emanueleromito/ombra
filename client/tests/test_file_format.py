"""File envelope. Synthetic bytes only; no OpenFHE."""

from __future__ import annotations

from pathlib import Path

import pytest
from ombra.file_format import EnvelopeError, Kind, read_envelope, write_envelope

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "envelope_v1_ciphertext.bin"
PAYLOAD = bytes([0, 1, 2, 3, 250, 255])


def _good() -> bytes:
    return write_envelope(Kind.CIPHERTEXT, "m0_depth2", "1.2.3", PAYLOAD)


def _read(b: bytes, kind: Kind = Kind.CIPHERTEXT, preset: str = "m0_depth2", v: str = "1.2.3"):
    return read_envelope(b, kind, preset, v)


def test_round_trip() -> None:
    e = _read(_good())
    assert (e.kind, e.preset, e.openfhe_version, e.payload) == (
        Kind.CIPHERTEXT,
        "m0_depth2",
        "1.2.3",
        PAYLOAD,
    )


def test_refusals() -> None:
    b = _good()
    with pytest.raises(EnvelopeError, match="magic"):
        _read(b"X" + b[1:])
    with pytest.raises(EnvelopeError, match="version"):
        _read(b[:4] + b"\x02" + b[5:])
    with pytest.raises(EnvelopeError, match="kind"):
        _read(b, Kind.CONTEXT)
    with pytest.raises(EnvelopeError, match="preset"):
        _read(b, preset="other")
    with pytest.raises(EnvelopeError, match="OpenFHE"):
        _read(b, v="9.9")
    with pytest.raises(EnvelopeError, match="trailing"):
        _read(b + b"\0")
    huge = b[:-14] + b"\xff" * 8 + b[-6:]
    with pytest.raises(EnvelopeError, match="exceeds"):
        _read(huge)
    for n in range(len(b)):
        with pytest.raises(EnvelopeError):
            _read(b[:n])


def test_cross_language_fixture() -> None:
    """The C++ test (ctest file_envelope) checks the same file against its own writer."""
    expected = write_envelope(Kind.CIPHERTEXT, "fixture_preset", "0.0.0-fixture", bytes(range(10)))
    assert FIXTURE.read_bytes() == expected
    assert _read(expected, preset="fixture_preset", v="0.0.0-fixture").payload == bytes(range(10))
