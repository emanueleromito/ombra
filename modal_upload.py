# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""What `modal_app.py` uploads into the image: an allowlist, minus secrets and build output.

Pure Python with no `modal` import, so a CPU test can run it. `modal_app.py` calls it locally at
image build. This file and `.gitignore` are still uploaded, because pytest collects their test in
the container.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import PurePosixPath

# Only these top-level entries are uploaded.
ALLOWED_TOP_LEVEL = (
    "core",
    "client",
    "server",
    "proto",
    "tests",
    "CMakeLists.txt",
    "pyproject.toml",
    "conftest.py",
    "Makefile",
    "modal_app.py",
    "modal_upload.py",
    ".gitignore",
)

# Credential-like names, matched against every path component. Keep in step with the secrets
# section of .gitignore (tests/test_modal_upload.py fails if a pattern there is not covered).
SECRET_PATTERNS = (
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "*.keyfile",
    "*.ombrakey",
    "keys",
    "secrets",
    "test-keys",
    ".modal.toml",
)

# Build output, caches and data, even inside allowed folders.
EXCLUDED_PATTERNS = (
    ".git",
    ".claude",
    ".venv",
    "venv",
    "data",
    "datasets",
    "build",
    "build-*",
    "cmake-build-*",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "*.ctxt",
    "*.ombra",
    "*.npy",
    "*.npz",
    "*.parquet",
    "*.bin",
    "*.sqlite",
)

# The only .bin files that are uploaded: the committed test fixtures (listed in .gitignore too).
ALLOWED_BINARY = (
    "tests/fixtures/envelope_v1_ciphertext.bin",
    "tests/fixtures/envelope_v2_seeded.bin",
)


def _normalise(path: str) -> tuple[str, ...]:
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    return tuple(p for p in parts if p not in ("", "."))


def is_unsafe(path: str) -> bool:
    """True if a path (relative to any folder) matches a secret, build-output or data pattern.
    For uploads of a single folder, where the top-level allowlist does not apply."""
    parts = _normalise(path)
    if "/".join(parts) in ALLOWED_BINARY:
        return False
    return any(fnmatch(p, pat) for p in parts for pat in SECRET_PATTERNS + EXCLUDED_PATTERNS)


def is_excluded(path: str) -> bool:
    """True if `path` (relative to the repo root) must NOT be uploaded."""
    parts = _normalise(path)
    if not parts or parts[0] not in ALLOWED_TOP_LEVEL:
        return True
    return is_unsafe(path)
