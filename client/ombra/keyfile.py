# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Passphrase-protected key file: OpenFHE secret key blob plus the document master key.

The file envelope deliberately has no kind for a secret key, so this file has its own magic
"OMBK" and version, laid out like the envelope (little-endian):
  magic "OMBK" | u16 version (1) | u8 kdf (1 = scrypt) | u32 n | u32 r | u32 p | salt (16) |
  nonce (12) | AES-256-GCM ciphertext+tag
The whole header (everything before the ciphertext, so the KDF parameters and salt too) is the
GCM associated data. Plaintext: master key (32 bytes) | secret key blob. Key bytes and the
passphrase never appear in errors or reprs. No OpenFHE dependency.
"""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"OMBK"
KEYFILE_VERSION = 1
KDF_SCRYPT = 1
MASTER_KEY_LEN = 32
SALT_LEN = 16
NONCE_LEN = 12
_HEADER = struct.Struct("<4sHBIII")
_HEADER_LEN = _HEADER.size + SALT_LEN + NONCE_LEN
# OWASP-level scrypt (128 MiB); tests pass smaller values.
DEFAULT_SCRYPT = (2**17, 8, 1)
# Bounds on read and write. scrypt needs 128 * n * r bytes: a hostile file must not make us allocate
# more than 1 GiB (audit A-41). Weak n is refused unless the environment variable below is set,
# which only tests and throwaway-key tools do (a key file of a real collection is never weak).
_MAX_N, _MAX_R, _MAX_P = 2**20, 32, 16
_MAX_SCRYPT_BYTES = 1 << 30
_MIN_N = 2**14
ALLOW_WEAK_ENV = "OMBRA_ALLOW_WEAK_SCRYPT"


def _check_kdf_params(n: int, r: int, p: int) -> None:
    if n < 2 or n & (n - 1) or not (n <= _MAX_N and 1 <= r <= _MAX_R and 1 <= p <= _MAX_P):
        raise KeyFileError("key derivation parameters out of range")
    if 128 * n * r > _MAX_SCRYPT_BYTES:
        raise KeyFileError("key derivation parameters out of range (memory above 1 GiB)")
    if n < _MIN_N and os.environ.get(ALLOW_WEAK_ENV) != "1":
        raise KeyFileError(f"scrypt n below {_MIN_N} is too weak for a key file")


_EXISTS = "key file already exists; pass overwrite=True to replace it"


class KeyFileError(ValueError):
    """The key file is invalid, corrupt, or the passphrase is wrong. Never carries key material."""


@dataclass(frozen=True)
class KeyMaterial:
    """Secret key blob (opaque OpenFHE serialization) and the 32-byte document master key."""

    secret_key: bytes = b""
    master_key: bytes = b""

    def __post_init__(self) -> None:
        if len(self.master_key) != MASTER_KEY_LEN:
            raise KeyFileError(f"master key must be {MASTER_KEY_LEN} bytes")
        if not self.secret_key:
            raise KeyFileError("secret key is empty")

    def __repr__(self) -> str:
        return f"KeyMaterial(secret_key=<{len(self.secret_key)} bytes>, master_key=<redacted>)"

    __str__ = __repr__

    @classmethod
    def generate(cls, secret_key: bytes) -> KeyMaterial:
        """Wrap an OpenFHE secret key blob with a fresh random master key."""
        return cls(bytes(secret_key), os.urandom(MASTER_KEY_LEN))


def _derive(passphrase: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    if not passphrase:
        raise KeyFileError("passphrase must not be empty")
    return hashlib.scrypt(
        passphrase.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        maxmem=129 * n * r + 2**26,
        dklen=32,
    )


def export_key_bytes(
    material: KeyMaterial, passphrase: str, *, scrypt_params: tuple[int, int, int] = DEFAULT_SCRYPT
) -> bytes:
    n, r, p = scrypt_params
    _check_kdf_params(n, r, p)
    salt, nonce = os.urandom(SALT_LEN), os.urandom(NONCE_LEN)
    header = _HEADER.pack(MAGIC, KEYFILE_VERSION, KDF_SCRYPT, n, r, p) + salt + nonce
    key = _derive(passphrase, salt, n, r, p)
    ct = AESGCM(key).encrypt(nonce, material.master_key + material.secret_key, header)
    return header + ct


def import_key_bytes(data: bytes, passphrase: str) -> KeyMaterial:
    if len(data) < _HEADER_LEN + 16 + MASTER_KEY_LEN + 1:
        raise KeyFileError("truncated key file")
    magic, version, kdf, n, r, p = _HEADER.unpack_from(data)
    if magic != MAGIC:
        raise KeyFileError("bad magic, not an Ombra key file")
    if version != KEYFILE_VERSION:
        raise KeyFileError(f"unknown key file version {version}")
    if kdf != KDF_SCRYPT:
        raise KeyFileError(f"unknown key derivation {kdf}")
    _check_kdf_params(n, r, p)
    header = data[:_HEADER_LEN]
    salt = header[_HEADER.size : _HEADER.size + SALT_LEN]
    nonce = header[_HEADER.size + SALT_LEN :]
    key = _derive(passphrase, salt, n, r, p)
    try:
        plain = AESGCM(key).decrypt(nonce, data[_HEADER_LEN:], header)
    except InvalidTag:
        raise KeyFileError("wrong passphrase or corrupted key file") from None
    return KeyMaterial(plain[MASTER_KEY_LEN:], plain[:MASTER_KEY_LEN])


def export_key_file(
    path: str | os.PathLike[str],
    material: KeyMaterial,
    passphrase: str,
    *,
    scrypt_params: tuple[int, int, int] = DEFAULT_SCRYPT,
    overwrite: bool = False,
) -> None:
    """Write atomically (temp file in the same directory, mode 0600, then rename).

    Refuses to replace an existing file unless ``overwrite`` is set: it may hold the only copy
    of a secret key. Without ``overwrite`` the file is put in place with a hard link, which fails
    if the name exists, so a concurrent writer is never overwritten (audit A-46).
    """
    target = Path(path)
    if target.exists() and not overwrite:
        raise KeyFileError(_EXISTS)
    blob = export_key_bytes(material, passphrase, scrypt_params=scrypt_params)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        if hasattr(os, "fchmod"):  # not on Windows; mkstemp already creates the file 0600
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        if overwrite:
            os.replace(tmp, target)
        else:
            try:
                os.link(tmp, target)  # atomic, fails if `target` appeared in the meantime
            except FileExistsError:
                raise KeyFileError(_EXISTS) from None
            os.unlink(tmp)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def import_key_file(path: str | os.PathLike[str], passphrase: str) -> KeyMaterial:
    return import_key_bytes(Path(path).read_bytes(), passphrase)
