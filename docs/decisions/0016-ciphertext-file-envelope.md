# 0016. Ciphertext and key file envelope

- **Status:** proposed
- **Date:** 2026-11-09

## Context

Hard rule 6 requires a version header on every ciphertext and key file. Today those files are raw OpenFHE serialization, whose per-class version tracks OpenFHE's layout, not Ombra's format, preset or scale conventions (review finding F8, task T1.16). All current files are temporary files in temp dirs, but M2 storage would build on them.

## Options considered

- **A. Envelope now, drivers unchanged.** Define and test a small envelope in both languages; M2 storage must use it; test drivers keep raw serialization. Smallest change, leaves a gap for fixtures.
- **B. Wire every driver now.** Fully meets rule 6 everywhere, but touches many Modal-only drivers and tests for files that live seconds in a temp dir.
- **C. Decision only.** No code; nothing enforces the rule when M2 starts.

## Decision

Option A. Every ciphertext, context and public evaluation key file written by Ombra storage is an envelope around an OpenFHE BINARY payload. Little-endian layout, version 1:

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | magic `OMBR` |
| 4 | 2 | format version, u16 (1) |
| 6 | 2 | kind, u16: 1 context, 2 ciphertext, 3 public eval key |
| 8 | 2 | preset name length P, u16, 1 to 255 |
| 10 | P | preset name |
| . | 2 | OpenFHE version length V, u16, 1 to 255 |
| . | V | OpenFHE version string |
| . | 8 | payload length L, u64 |
| . | L | payload; the file ends here |

There is no kind for a secret key: the server never handles one (hard rule 2), and the client's passphrase-protected key file format is a separate matter. Readers refuse: wrong magic, unknown version, wrong kind, a preset or OpenFHE version different from the caller's expected value, truncated files, trailing bytes, and a payload length larger than the bytes present (checked before allocating). Implementations: `core/include/ombra/core/file_envelope.hpp` and `client/ombra/file_format.py`, tested by ctest `file_envelope`, `client/tests/test_file_format.py` and the fixture `tests/fixtures/envelope_v1_ciphertext.bin`. Any layout change bumps the format version.

The temporary files the test drivers write in temp dirs stay raw OpenFHE serialization for now. They are test fixtures, not an Ombra format. Stated plainly: this leaves hard rule 6 unmet for those fixtures until the user accepts this reading.

## Consequences

M2 storage must use the envelope for every ciphertext and key file. Drivers are not rewired, so their files are unversioned. The preset and OpenFHE version are in the header, so a mismatch is refused before OpenFHE deserializes anything. Cost: a few dozen bytes per file.
