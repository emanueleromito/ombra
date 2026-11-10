"""Encrypted record and document blobs (T3.4).

Each blob has its own fresh AES-256-GCM key, wrapped with the document master key
(KeyMaterial.master_key). Layout, version 1 (all integers little-endian); the stored blob is
exactly one bucket size (LIMIT_BLOB_BUCKET_*), the smallest that fits:
  magic "OMBD" | u16 version (1) | wrap_nonce (12) | wrapped_key (32 + 16 tag) |
  data_nonce (12) | AES-256-GCM ciphertext+tag of the padded plaintext
Padded plaintext: u8 mode | u64 true_length | payload | zero padding up to the bucket.
  mode 1 = text body (UTF-8), 2 = bytes body, 3 = reference (JSON {uri, sha256, size}),
  4 = record (u32 vector_byte_length | float32 vector | JSON metadata).
The true length and the padding are inside the ciphertext, so only the bucket size is visible.
Associated data of the key wrap: magic | version | u8 kind | u16 len | collection | u16 len | id.
The body's associated data is that plus the whole header (wrapped key and nonces). So a blob
moved to another id, collection or kind, or with any byte changed, fails authentication.

Reference mode: a body larger than `inline_threshold` (default 1 MiB) is not stored; the blob
holds an encrypted reference to the customer's own storage, and `fetch_document` checks the sha256.
Errors and reprs never carry plaintext, keys or ids (hard rule 4).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .blobstore import BUCKET_SIZES, KIND_DOCUMENT, KIND_RECORD

MAGIC = b"OMBD"
BLOB_VERSION = 1
DEFAULT_INLINE_THRESHOLD = 1 << 20
MODE_TEXT, MODE_BYTES, MODE_REFERENCE, MODE_RECORD = 1, 2, 3, 4

_HEAD = struct.Struct("<4sH")
_NONCE = 12
_WRAPPED = 32 + 16
_PREFIX = _HEAD.size + _NONCE + _WRAPPED + _NONCE
_OVERHEAD = _PREFIX + 16  # prefix plus the body GCM tag
_INNER = struct.Struct("<BQ")


class DocumentError(ValueError):
    """Sealing or opening failed. Messages never include ids, contents or keys."""


@dataclass(frozen=True)
class Reference:
    uri: str = field(repr=False)
    sha256: str = ""
    size: int = 0


@dataclass(frozen=True)
class Document:
    """An opened document: `body` is str or bytes, or None when `reference` is set."""

    body: str | bytes | None = field(default=None, repr=False)
    reference: Reference | None = None


@dataclass(frozen=True)
class Record:
    vector: bytes = field(repr=False)  # float32, little-endian
    metadata: dict[str, Any] = field(repr=False, default_factory=dict)


def _aad(kind: int, collection: str, blob_id: bytes) -> bytes:
    c = collection.encode()
    return (
        _HEAD.pack(MAGIC, BLOB_VERSION)
        + bytes([kind])
        + struct.pack("<H", len(c))
        + c
        + struct.pack("<H", len(blob_id))
        + bytes(blob_id)
    )


def _bucket_for(inner_len: int) -> int:
    for b in BUCKET_SIZES:
        if _OVERHEAD + inner_len <= b:
            return b
    raise DocumentError("content too large for the largest blob bucket; use reference mode")


class DocumentCodec:
    def __init__(
        self,
        master_key: bytes,
        collection: str,
        inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
    ) -> None:
        if len(master_key) != 32:
            raise DocumentError("master key must be 32 bytes")
        self._wrap = AESGCM(bytes(master_key))
        self._collection = collection
        self.inline_threshold = inline_threshold

    def __repr__(self) -> str:
        return "DocumentCodec(<redacted>)"

    def _seal(self, kind: int, blob_id: bytes, mode: int, payload: bytes) -> bytes:
        bucket = _bucket_for(_INNER.size + len(payload))
        plain = _INNER.pack(mode, len(payload)) + payload
        plain += b"\0" * (bucket - _OVERHEAD - len(plain))
        aad = _aad(kind, self._collection, blob_id)
        doc_key = AESGCM.generate_key(bit_length=256)
        wrap_nonce, data_nonce = os.urandom(_NONCE), os.urandom(_NONCE)
        wrapped = self._wrap.encrypt(wrap_nonce, doc_key, aad)
        prefix = _HEAD.pack(MAGIC, BLOB_VERSION) + wrap_nonce + wrapped + data_nonce
        blob = prefix + AESGCM(doc_key).encrypt(data_nonce, plain, aad + prefix)
        if len(blob) != bucket:
            raise DocumentError("internal size error")
        return blob

    def seal_record(self, blob_id: bytes, vector: bytes, metadata: dict[str, Any]) -> bytes:
        """`vector` is the float32 little-endian bytes of the vector (e.g. numpy tobytes())."""
        if len(vector) % 4:
            raise DocumentError("vector bytes must be float32")
        meta = json.dumps(metadata, sort_keys=True).encode()
        payload = struct.pack("<I", len(vector)) + vector + meta
        return self._seal(KIND_RECORD, blob_id, MODE_RECORD, payload)

    def seal_document(self, blob_id: bytes, body: str | bytes, *, uri: str | None = None) -> bytes:
        """Inline when the body fits the threshold, else reference mode (needs `uri`)."""
        raw = body.encode() if isinstance(body, str) else bytes(body)
        if len(raw) <= self.inline_threshold:
            mode = MODE_TEXT if isinstance(body, str) else MODE_BYTES
            return self._seal(KIND_DOCUMENT, blob_id, mode, raw)
        if uri is None:
            raise DocumentError("body above the inline threshold needs a reference uri")
        return self.seal_reference(blob_id, uri, hashlib.sha256(raw).hexdigest(), len(raw))

    def seal_reference(self, blob_id: bytes, uri: str, sha256: str, size: int) -> bytes:
        ref = json.dumps({"uri": uri, "sha256": sha256, "size": size}, sort_keys=True)
        return self._seal(KIND_DOCUMENT, blob_id, MODE_REFERENCE, ref.encode())

    def _open(self, kind: int, blob_id: bytes, blob: bytes) -> tuple[int, bytes]:
        if len(blob) not in BUCKET_SIZES or blob[:4] != MAGIC:
            raise DocumentError("not a valid blob")
        if _HEAD.unpack_from(blob)[1] != BLOB_VERSION:
            raise DocumentError("unsupported blob version")
        aad = _aad(kind, self._collection, blob_id)
        o = _HEAD.size
        try:
            doc_key = self._wrap.decrypt(
                blob[o : o + _NONCE], blob[o + _NONCE : o + _NONCE + _WRAPPED], aad
            )
            plain = AESGCM(doc_key).decrypt(
                blob[_PREFIX - _NONCE : _PREFIX], blob[_PREFIX:], aad + blob[:_PREFIX]
            )
        except InvalidTag:
            raise DocumentError("blob failed authentication") from None
        mode, length = _INNER.unpack_from(plain)
        if _INNER.size + length > len(plain):
            raise DocumentError("blob failed authentication")
        return mode, plain[_INNER.size : _INNER.size + length]

    def open_record(self, blob_id: bytes, blob: bytes) -> Record:
        mode, payload = self._open(KIND_RECORD, blob_id, blob)
        if mode != MODE_RECORD:
            raise DocumentError("not a record blob")
        (n,) = struct.unpack_from("<I", payload)
        return Record(payload[4 : 4 + n], json.loads(payload[4 + n :]))

    def open_document(self, blob_id: bytes, blob: bytes) -> Document:
        mode, payload = self._open(KIND_DOCUMENT, blob_id, blob)
        if mode == MODE_TEXT:
            return Document(body=payload.decode())
        if mode == MODE_BYTES:
            return Document(body=payload)
        if mode == MODE_REFERENCE:
            d = json.loads(payload)
            return Document(reference=Reference(d["uri"], d["sha256"], int(d["size"])))
        raise DocumentError("not a document blob")


def fetch_document(doc: Document, fetcher: Callable[[str], bytes]) -> bytes:
    """Return the body of an opened document, fetching reference mode through `fetcher(uri)`.

    The fetched bytes must match the reference's size and sha256, else DocumentError.
    """
    if doc.reference is None:
        body = doc.body
        return body.encode() if isinstance(body, str) else bytes(body or b"")
    data = fetcher(doc.reference.uri)
    if len(data) != doc.reference.size or hashlib.sha256(data).hexdigest() != doc.reference.sha256:
        raise DocumentError("fetched document does not match its reference")
    return data
