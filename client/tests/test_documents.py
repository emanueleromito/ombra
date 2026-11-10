"""T3.4 client: encrypted record/document blobs. Synthetic data only; no OpenFHE, no server."""

from __future__ import annotations

import hashlib
import os

import pytest
from ombra.blobstore import (
    BUCKET_SIZES,
    KIND_DOCUMENT,
    KIND_RECORD,
    BlobStoreError,
    InMemoryBlobStore,
)
from ombra.documents import DocumentCodec, DocumentError, fetch_document

COL = "col"
TEXT = "synthetic-secret-sentence-xyzzy"
META = {"owner": "plugh-owner-value", "n": 7}
VEC = bytes(range(64)) * 2  # 128 bytes, distinctive float32 stand-in


@pytest.fixture
def codec() -> DocumentCodec:
    return DocumentCodec(os.urandom(32), COL)


def test_round_trip_inline_and_record(codec: DocumentCodec) -> None:
    store = InMemoryBlobStore()
    store.put(COL, KIND_DOCUMENT, b"d1", codec.seal_document(b"d1", TEXT))
    store.put(COL, KIND_DOCUMENT, b"d2", codec.seal_document(b"d2", b"\x00\x01bytes"))
    store.put(COL, KIND_RECORD, b"r1", codec.seal_record(b"r1", VEC, META))
    assert codec.open_document(b"d1", store.get(COL, KIND_DOCUMENT, b"d1")).body == TEXT
    assert codec.open_document(b"d2", store.get(COL, KIND_DOCUMENT, b"d2")).body == b"\x00\x01bytes"
    rec = codec.open_record(b"r1", store.get(COL, KIND_RECORD, b"r1"))
    assert rec.vector == VEC and rec.metadata == META
    assert store.delete(COL, KIND_DOCUMENT, [b"d1", b"missing"]) == 1


def test_reference_mode_above_threshold_and_sha_check() -> None:
    codec = DocumentCodec(os.urandom(32), COL, inline_threshold=100)
    body = os.urandom(5000)
    with pytest.raises(DocumentError):
        codec.seal_document(b"d", body)  # no uri, too big to inline
    blob = codec.seal_document(b"d", body, uri="s3://bucket/obj")
    doc = codec.open_document(b"d", blob)
    assert doc.body is None and doc.reference is not None
    assert doc.reference.sha256 == hashlib.sha256(body).hexdigest()
    assert doc.reference.size == len(body)
    assert fetch_document(doc, lambda uri: body) == body
    tampered = bytearray(body)
    tampered[0] ^= 1
    with pytest.raises(DocumentError):
        fetch_document(doc, lambda uri: bytes(tampered))
    with pytest.raises(DocumentError):
        fetch_document(doc, lambda uri: body + b"x")
    small = codec.open_document(b"s", codec.seal_document(b"s", "tiny", uri="u"))
    assert small.body == "tiny"


def test_default_threshold_is_1_mib_and_huge_body_needs_reference() -> None:
    codec = DocumentCodec(os.urandom(32), COL)
    assert codec.inline_threshold == 1 << 20
    big = b"a" * ((1 << 20) + 1)
    with pytest.raises(DocumentError):
        codec.seal_document(b"d", big)
    ref = codec.open_document(b"d", codec.seal_document(b"d", big, uri="file:///x"))
    assert ref.reference is not None
    # a threshold above the largest bucket still cannot inline an oversized body
    c2 = DocumentCodec(os.urandom(32), COL, inline_threshold=1 << 30)
    with pytest.raises(DocumentError):
        c2.seal_document(b"d", b"a" * BUCKET_SIZES[-1], uri="u")


def test_any_changed_byte_is_refused(codec: DocumentCodec) -> None:
    blob = codec.seal_document(b"d", TEXT)
    assert codec.open_document(b"d", blob).body == TEXT
    for i in [*range(120), len(blob) // 2, len(blob) - 1]:
        bad = bytearray(blob)
        bad[i] ^= 1
        with pytest.raises(DocumentError):
            codec.open_document(b"d", bytes(bad))


def test_moved_or_wrong_key_blob_is_refused(codec: DocumentCodec) -> None:
    doc = codec.seal_document(b"d", TEXT)
    rec = codec.seal_record(b"d", VEC, META)
    with pytest.raises(DocumentError):
        codec.open_document(b"other", doc)  # id
    with pytest.raises(DocumentError):
        codec.open_record(b"d", doc)  # kind
    with pytest.raises(DocumentError):
        codec.open_document(b"d", rec)  # kind
    with pytest.raises(DocumentError):
        DocumentCodec(os.urandom(32), COL).open_document(b"d", doc)  # other master key


def test_moved_to_other_collection_is_refused() -> None:
    key = os.urandom(32)
    a, b = DocumentCodec(key, "a"), DocumentCodec(key, "b")
    blob = a.seal_document(b"d", TEXT)
    assert a.open_document(b"d", blob).body == TEXT
    with pytest.raises(DocumentError):
        b.open_document(b"d", blob)


def test_stored_bytes_hold_no_plaintext_and_sizes_are_buckets(codec: DocumentCodec) -> None:
    store = InMemoryBlobStore()
    store.put(COL, KIND_DOCUMENT, b"d", codec.seal_document(b"d", TEXT))
    store.put(COL, KIND_RECORD, b"r", codec.seal_record(b"r", VEC, META))
    store.put(COL, KIND_DOCUMENT, b"big", codec.seal_document(b"big", os.urandom(100_000)))
    store.put(COL, KIND_DOCUMENT, b"ref", codec.seal_reference(b"ref", "s3://x/y", "0" * 64, 5))
    for needle in (TEXT.encode(), b"plugh-owner-value", VEC[:32], b"s3://x/y"):
        assert all(needle not in blob for blob in store.blobs.values())
    assert {len(b) for b in store.blobs.values()} <= set(BUCKET_SIZES)
    assert len(codec.seal_document(b"a", "x")) == len(codec.seal_document(b"b", "x" * 5000))
    assert len(codec.seal_document(b"c", os.urandom(100_000))) == BUCKET_SIZES[1]


def test_fresh_key_per_blob_and_no_leak_in_repr_or_errors(codec: DocumentCodec) -> None:
    assert codec.seal_document(b"d", TEXT) != codec.seal_document(b"d", TEXT)
    assert TEXT not in repr(codec.open_document(b"d", codec.seal_document(b"d", TEXT)))
    assert "xyzzy" not in repr(codec.open_record(b"r", codec.seal_record(b"r", VEC, META)))
    with pytest.raises(DocumentError) as e:
        codec.open_document(b"secret-id", b"junk")
    assert "secret-id" not in str(e.value)


def test_store_enforces_bucket_size() -> None:
    with pytest.raises(BlobStoreError):
        InMemoryBlobStore().put(COL, KIND_DOCUMENT, b"d", b"short")


def test_bucket_sizes_match_proto():
    import pathlib
    import re

    from ombra.blobstore import BUCKET_SIZES

    proto = pathlib.Path(__file__).resolve().parents[2] / "proto/ombra/v1/ombra.proto"
    found = re.findall(r"LIMIT_BLOB_BUCKET_\d+ = (\d+);", proto.read_text())
    assert tuple(int(v) for v in found) == BUCKET_SIZES
