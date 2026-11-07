# client/ rules

Python 3.11+. The client holds the secret key and does everything security-sensitive: key generation, encryption, decryption, clustering, top-k and re-ranking. Cryptography goes through openfhe-python only.

## Rules

- **Never send the secret key anywhere.** Not to the server, not in logs, not in exceptions, not in `repr()`. Key objects must not print their contents.
- **Key files are passphrase-protected** and carry a format version. Never write an unprotected key to disk, except in tests using a temporary directory.
- **Parameters come from presets** ("fast", "balanced", "precise"), derived from the preset and the vector dimension. Refuse anything below 128-bit security, including in expert mode.
- **Normalize vectors** before encryption so cosine similarity works.
- **No plaintext in logs:** no vectors, queries, scores or documents.
- **Sync and async APIs** expose the same methods when the client API exists.

## Tooling

- `ruff` for formatting and linting, `pytest` for tests, type hints on public functions.
- CPU tests run locally with `make test`; tests needing GPU results go through `make test-gpu`.
