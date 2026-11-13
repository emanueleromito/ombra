# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Writes tests/fixtures/envelope_v2_seeded.bin (synthetic content). Run from the repo root with
PYTHONPATH=client; client/tests/test_seeded_format.py checks the file against the same writer."""

from pathlib import Path

from ombra.file_format import Kind, write_envelope
from ombra.seeded_format import fixture_payload, write_payload

out = Path(__file__).with_name("envelope_v2_seeded.bin")
out.write_bytes(
    write_envelope(
        Kind.SEEDED_CIPHERTEXT, "fixture_preset", "0.0.0-fixture", write_payload(fixture_payload())
    )
)
print(out, out.stat().st_size)
