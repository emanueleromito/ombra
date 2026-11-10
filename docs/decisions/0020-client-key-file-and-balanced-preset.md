# 0020. Client key file format and the "balanced" preset

- **Status:** proposed
- **Date:** 2026-11-10

## Context

T3.2 needs a passphrase-protected key file; decision 0016 deliberately has no envelope kind for a secret key and leaves the client's key file "a separate matter". T3.3 needs product preset names; 0013 makes `m0_compact_n8192` "fast", but no decision names "balanced" or "precise".

## Options considered

- **Key file A. Own format** (`client/ombra/keyfile.py`): magic `OMBK`, u16 version 1, scrypt (default N = 2^17, r = 8, p = 1, bounded on read), 16-byte salt, AES-256-GCM over master key + secret key blob, whole header as associated data. Keeps secret keys out of the server-facing 0016 format.
- **Key file B. New 0016 kind.** One format, but puts a secret-key kind into a format the server parses.
- **Presets:** "balanced" = `m0_depth2` (the baseline, default); "precise" refused until a preset exists.

## Decision

Key file A; "balanced" = `m0_depth2` as the default; "precise" refused.

## Consequences

Export refuses to overwrite an existing key file unless asked. `cryptography` becomes a client dependency. A "precise" preset, if wanted, needs its own decision and recall check (0012).
