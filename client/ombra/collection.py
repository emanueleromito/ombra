"""Collection creation, re-clustering and repacking (T3.5), client side.

The client holds the secret key (never sent, never printed). It clusters the vectors, encrypts
the data blocks and the cluster centers, generates the evaluation keys, uploads them once per
collection and creates the collection on the server. Parameters come only from
`ombra.presets.derive` (hard rule 1).

Layout (docs/architecture.md, decision 0019): a data block holds up to `slots` vectors of one
cluster as D column ciphertexts (slot i of column j = component j of vector i) followed by one
one-hot column per filter value; the center block holds the centers as D columns, under cluster id
`cluster_count` with `is_center` set.

Re-clustering. The server has no way to move a vector between clusters or to replace the center
block (ReplaceBlock keeps the cluster; center blocks cannot be replaced). So `recluster` builds a
new collection (generation g + 1) from the live vectors of the old one, under the SAME key pair
(so the user's key file stays valid; the public keys are uploaded again to the new collection),
copies the encrypted record and document blobs byte for byte (their associated data binds the
base name, not the generation, so they need no re-encryption), and leaves the old one untouched
and searchable; the caller switches to the new one and calls `retire()` on the old one. Dropped
slots never reach the new collection.

The transport is a protobuf messages module `pb` and a gRPC stub of service ombra.v1.Ombra,
passed in by the caller (the generated modules are built from proto/ombra/v1/ombra.proto).
"""

from __future__ import annotations

import os
import re
import struct
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ombra import presets
from ombra import query as oq
from ombra.blobstore import KIND_DOCUMENT, KIND_RECORD, BlobStore, BlobStoreError, GrpcBlobStore
from ombra.clustering import assign, cluster_vectors
from ombra.documents import DocumentCodec
from ombra.file_format import Kind, read_envelope, write_envelope
from ombra.keyfile import DEFAULT_SCRYPT, KeyMaterial, export_key_file, import_key_file
from ombra.params import make_ckks_params
from ombra.probe import FIELDS_KEY, STATS_ID, ClusterStats, check_user_id
from ombra.repack import DEFAULT_THRESHOLD, BlockStat, plan_repack

KEY_CHUNK = 4_000_000  # below LIMIT_MAX_CHUNK_BYTES (4 MiB)
API_VERSION = 1
FLUSH = 1e-9  # OpenFHE refuses to encode denormal-sized values ("Scaling factor too small")
_LEVELS = {
    "fast": "SECURITY_LEVEL_FAST",
    "decoy": "SECURITY_LEVEL_DECOY",
    "strict": "SECURITY_LEVEL_STRICT",
}


class CollectionError(Exception):
    """The server answered with an ombra.v1.Error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"error code {code}: {message}")
        self.code = code
        self.message = message


def _check(err: Any) -> None:
    if err.code != 0:
        raise CollectionError(err.code, err.message)


class KeyKit:
    """Context, key pair and evaluation keys of one collection. The secret key stays inside."""

    def __init__(
        self,
        derived: presets.Derived,
        openfhe_version: str,
        n_leaves: int,
        *,
        master_key: bytes | None = None,
        secret_blob: bytes | None = None,
    ) -> None:
        """New keys, or (with `secret_blob` from `KeyKit.load`) the keys of an earlier kit."""
        import openfhe as ofhe

        self.derived = derived
        self.openfhe_version = openfhe_version
        self.n_leaves = n_leaves
        self.slots = derived.batch_size
        self.master_key = master_key if master_key is not None else os.urandom(32)
        cc = ofhe.GenCryptoContext(make_ckks_params(derived.values))
        for feature in (
            ofhe.PKESchemeFeature.PKE,
            ofhe.PKESchemeFeature.KEYSWITCH,
            ofhe.PKESchemeFeature.LEVELEDSHE,
        ):
            cc.Enable(feature)
        self.cc = cc
        self._keys = cc.KeyGen() if secret_blob is None else self._restore(secret_blob)
        cc.EvalMultKeyGen(self._keys.secretKey)
        cc.ClearEvalAutomorphismKeys()  # the key store is static: hold exactly this layout's keys
        cc.EvalAutomorphismKeyGen(
            self._keys.secretKey, oq.automorphism_indices(cc.GetRingDimension(), n_leaves)
        )
        self.key_tag = self._keys.publicKey.GetKeyTag()

    # ---- persistence (T3.2 key file) ------------------------------------------------------
    _BLOB_MAGIC = b"OKSB"
    _BLOB_VERSION = 1

    def _secret_blob(self) -> bytes:
        import openfhe as ofhe

        sk = self._via_file(lambda p: ofhe.SerializeToFile(p, self._keys.secretKey, ofhe.BINARY))
        pk = self._via_file(lambda p: ofhe.SerializeToFile(p, self._keys.publicKey, ofhe.BINARY))
        return self._BLOB_MAGIC + struct.pack("<HI", self._BLOB_VERSION, len(sk)) + sk + pk

    def _restore(self, blob: bytes) -> Any:
        import openfhe as ofhe

        head = struct.Struct("<4sHI")
        if len(blob) < head.size:
            raise ValueError("key blob is truncated")
        magic, version, n = head.unpack_from(blob)
        if magic != self._BLOB_MAGIC or version != self._BLOB_VERSION:
            raise ValueError("key blob has an unknown format or version")
        sk_raw, pk_raw = blob[head.size : head.size + n], blob[head.size + n :]

        def load(raw: bytes, fn: Any) -> Any:
            with tempfile.TemporaryDirectory() as t:
                p = Path(t) / "k.bin"
                p.write_bytes(raw)
                obj, ok = fn(str(p), ofhe.BINARY)
            if not ok:
                raise ValueError("key could not be deserialized")
            return obj

        return SimpleNamespace(
            secretKey=load(sk_raw, ofhe.DeserializePrivateKey),
            publicKey=load(pk_raw, ofhe.DeserializePublicKey),
        )

    def save(
        self,
        path: str | os.PathLike[str],
        passphrase: str,
        *,
        scrypt_params: tuple[int, int, int] = DEFAULT_SCRYPT,
        overwrite: bool = False,
    ) -> None:
        """Write the secret key and the document master key to a passphrase-protected key file."""
        material = KeyMaterial(self._secret_blob(), self.master_key)
        export_key_file(
            path, material, passphrase, scrypt_params=scrypt_params, overwrite=overwrite
        )

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        passphrase: str,
        derived: presets.Derived,
        openfhe_version: str,
        n_leaves: int,
    ) -> KeyKit:
        """Reopen a saved kit. The evaluation keys are generated again from the secret key."""
        m = import_key_file(path, passphrase)
        return cls(
            derived, openfhe_version, n_leaves, master_key=m.master_key, secret_blob=m.secret_key
        )

    def encrypt_query(self, ext: np.ndarray) -> bytes:
        """The symmetric coefficient-packed query (decision 0010), encrypted, as an envelope."""
        import openfhe as ofhe

        slots = oq.symmetric_slots(oq.prescale_query(ext), self.cc.GetRingDimension())
        ct = self.cc.Encrypt(self._keys.publicKey, self.cc.MakeCKKSPackedPlaintext(slots.tolist()))
        raw = self._via_file(lambda p: ofhe.SerializeToFile(p, ct, ofhe.BINARY))
        return write_envelope(Kind.CIPHERTEXT, self.derived.preset, self.openfhe_version, raw)

    def __repr__(self) -> str:
        return f"KeyKit(preset={self.derived.preset!r}, key_tag={self.key_tag!r})"

    @property
    def public_key(self) -> Any:
        return self._keys.publicKey

    @staticmethod
    def _via_file(write: Any) -> bytes:
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.bin"
            if not write(str(p)):
                raise RuntimeError("OpenFHE serialization failed")
            return p.read_bytes()

    def key_files(self, pb: Any) -> dict[int, bytes]:
        """Public key files in 0016 envelopes, by KeyKind. Nothing secret."""
        import openfhe as ofhe

        d, v = self.derived.preset, self.openfhe_version
        raw = {
            pb.KEY_KIND_CONTEXT: (
                Kind.CONTEXT,
                self._via_file(lambda p: ofhe.SerializeToFile(p, self.cc, ofhe.BINARY)),
            ),
            pb.KEY_KIND_EVAL_MULT: (
                Kind.PUBLIC_EVAL_KEY,
                self._via_file(
                    lambda p: ofhe.CryptoContext.SerializeEvalMultKey(p, ofhe.BINARY, self.key_tag)
                ),
            ),
            pb.KEY_KIND_AUTOMORPHISM: (
                Kind.PUBLIC_EVAL_KEY,
                self._via_file(lambda p: self.cc.SerializeEvalAutomorphismKey(p, ofhe.BINARY)),
            ),
        }
        return {k: write_envelope(kind, d, v, b) for k, (kind, b) in raw.items()}

    def encrypt_column(self, values: np.ndarray) -> bytes:
        """One column (one value per slot, zero padded) as a ciphertext envelope."""
        import openfhe as ofhe

        padded = np.zeros(self.slots)
        padded[: len(values)] = values
        padded[np.abs(padded) < FLUSH] = 0.0
        ct = self.cc.Encrypt(
            self._keys.publicKey, self.cc.MakeCKKSPackedPlaintext([float(v) for v in padded])
        )
        raw = self._via_file(lambda p: ofhe.SerializeToFile(p, ct, ofhe.BINARY))
        return write_envelope(Kind.CIPHERTEXT, self.derived.preset, self.openfhe_version, raw)

    def decrypt_column(self, envelope: bytes) -> np.ndarray:
        """Decrypt a ciphertext envelope to its `slots` values."""
        import openfhe as ofhe

        env = read_envelope(envelope, Kind.CIPHERTEXT, self.derived.preset, self.openfhe_version)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "ct.bin"
            p.write_bytes(env.payload)
            ct, ok = ofhe.DeserializeCiphertext(str(p), ofhe.BINARY)
        if not ok:
            raise ValueError("ciphertext could not be deserialized")
        dec = self.cc.Decrypt(ct, self._keys.secretKey)
        dec.SetLength(self.slots)
        return np.array(dec.GetRealPackedValue(), dtype=np.float64)


@dataclass(frozen=True)
class RepackReport:
    groups: int  # ReplaceBlock calls made
    blocks_before: int  # blocks read
    blocks_after: int  # blocks written
    bytes_before: int  # ciphertext bytes of the blocks read
    bytes_after: int  # ciphertext bytes written


class Collection:
    """One collection on the server, with the client's keys for it."""

    def __init__(
        self,
        pb: Any,
        stub: Any,
        *,
        base_name: str,
        generation: int,
        derived: presets.Derived,
        cards: Sequence[int],
        kit: KeyKit,
        centers: np.ndarray,
        blobs: BlobStore | None = None,
        security_level: str = "fast",
    ) -> None:
        self.pb, self.stub = pb, stub
        self.blobs: BlobStore = blobs if blobs is not None else GrpcBlobStore(stub, pb)
        self.security_level = security_level
        # blob associated data binds the base name, so blobs survive a re-clustering unchanged
        self.codec = DocumentCodec(kit.master_key, base_name)
        self.base_name, self.generation = base_name, generation
        self.derived, self.cards, self.kit = derived, tuple(cards), kit
        self.centers = centers
        self.slots = derived.batch_size
        # cluster and value counts for the probe policy; None if missing or old version
        self.stats: ClusterStats | None = None

    @property
    def name(self) -> str:
        """Name on the server: the base name for generation 0, then `<base>.g<n>`."""
        return self.base_name if self.generation == 0 else f"{self.base_name}.g{self.generation}"

    @property
    def cluster_count(self) -> int:
        return len(self.centers)

    @property
    def n_leaves(self) -> int:
        return self.derived.dimension + sum(self.cards)

    # ---- create ---------------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        pb: Any,
        stub: Any,
        name: str,
        vectors: np.ndarray,
        ids: Sequence[bytes],
        *,
        preset: str = presets.DEFAULT_PRESET,
        fields: np.ndarray | None = None,
        cards: Sequence[int] = (),
        n_clusters: int | None = None,
        openfhe_version: str | None = None,
        seed: int = 0,
        generation: int = 0,
        kit: KeyKit | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
        documents: Sequence[str | bytes | None] | None = None,
        security_level: str = "fast",
        store_records: bool = True,
        blobs: BlobStore | None = None,
    ) -> Collection:
        """Cluster, create the collection, upload the keys once, insert centers and data."""
        x = np.asarray(vectors, dtype=np.float64)
        derived = presets.derive(preset, x.shape[1])  # hard rule 1: parameters only from here
        if len(ids) != len(x):
            raise ValueError("one id per vector")
        if not np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-6):
            raise ValueError("vectors must be normalized before they are encrypted")
        cards = tuple(cards)
        presets.check_filter_schema(derived, cards)  # decision 0019: refuse above F_max
        if security_level not in _LEVELS:
            raise ValueError(f"security level must be one of {sorted(_LEVELS)}")
        f = _fields(fields, len(x), cards)
        k = n_clusters or max(1, round(len(x) / derived.batch_size))
        if k > derived.batch_size:
            raise ValueError("more clusters than slots: the centers must fit in one block")
        for ident in ids:
            check_user_id(ident)
        clustering = cluster_vectors(x, k, seed=seed)
        # k-means from faiss can leave denormal-sized center entries; OpenFHE cannot encode them
        centers = np.where(np.abs(clustering.centers) < FLUSH, 0.0, clustering.centers)
        if openfhe_version is None:
            import openfhe as ofhe

            openfhe_version = ofhe.__version__
        n_leaves = derived.dimension + sum(cards)
        if kit is None:
            kit = KeyKit(derived, openfhe_version, n_leaves)
        elif kit.derived.preset != derived.preset or kit.n_leaves != n_leaves:
            raise ValueError("the key kit was made for another preset or layout")
        self = cls(
            pb,
            stub,
            base_name=name,
            generation=generation,
            derived=derived,
            cards=cards,
            kit=kit,
            centers=centers,
            blobs=blobs,
            security_level=security_level,
        )
        spec = pb.CollectionSpec(
            api_version=API_VERSION,
            name=self.name,
            preset=derived.preset,
            dimension=derived.dimension,
            security_level=getattr(pb, _LEVELS[security_level]),
            key_tag=kit.key_tag,
            cluster_count=self.cluster_count,
        )
        for i, c in enumerate(cards):
            spec.filter_fields.add(field_id=i, cardinality=c)
        r = stub.CreateCollection(pb.CreateCollectionRequest(spec=spec))
        if r.WhichOneof("result") == "error":
            raise CollectionError(r.error.code, r.error.message)
        try:
            self._upload_keys()
            self._insert_centers()
            for c in range(self.cluster_count):
                rows = np.nonzero(clustering.labels == c)[0]
                self._insert_cluster(c, x[rows], f[rows], [ids[i] for i in rows])
            if store_records:
                self._put_records(x, ids, metadata, documents, f)
            self.stats = ClusterStats.build(clustering.labels, len(centers), f, cards)
            self._save_stats()
        except Exception:
            stub.DropCollection(pb.DropCollectionRequest(name=self.name))
            raise
        return self

    @classmethod
    def open(
        cls, pb: Any, stub: Any, name: str, kit: KeyKit, *, blobs: BlobStore | None = None
    ) -> Collection:
        """Reopen an existing collection with a kit loaded from the key file (new process).

        The centers are read back from the server's center block and decrypted.
        """
        r = stub.GetCollection(pb.GetCollectionRequest(name=name))
        if r.WhichOneof("result") == "error":
            raise CollectionError(r.error.code, r.error.message)
        spec = r.info.spec
        derived = presets.derive(spec.preset, spec.dimension)
        cards = tuple(f.cardinality for f in sorted(spec.filter_fields, key=lambda f: f.field_id))
        if kit.key_tag != spec.key_tag or kit.n_leaves != derived.dimension + sum(cards):
            raise ValueError("the key kit does not belong to this collection")
        m = re.fullmatch(r"(.*)\.g(\d+)", name)
        base, gen = (m.group(1), int(m.group(2))) if m else (name, 0)
        level = {v: k for k, v in _LEVELS.items()}.get(pb.SecurityLevel.Name(spec.security_level))
        self = cls(
            pb,
            stub,
            base_name=base,
            generation=gen,
            derived=derived,
            cards=cards,
            kit=kit,
            centers=np.zeros((spec.cluster_count, derived.dimension)),
            blobs=blobs,
            security_level=level or "fast",
        )
        lst = stub.ListBlocks(pb.ListBlocksRequest(api_version=API_VERSION, collection=name))
        _check(lst.error)
        seg = [b.segment for b in lst.blocks if b.cluster_id == spec.cluster_count]
        if len(seg) != 1:
            raise CollectionError(0, "the center block is missing")
        header, cols = self._get_block(seg[0])
        n = spec.cluster_count
        c = np.stack([kit.decrypt_column(col)[:n] for col in cols], axis=1)
        self.centers = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
        try:
            self.stats = ClusterStats.unseal(
                self.codec, self.blobs.get(self.name, KIND_RECORD, STATS_ID)
            )
        except (BlobStoreError, ValueError):
            self.stats = None  # an older collection or stats version: no filter-aware probing
        return self

    def search(self, query: np.ndarray, k: int = 10, **options: Any) -> Any:
        """Top-k search with exact re-ranking; see ombra.search.run_search for the options."""
        from ombra.search import run_search

        return run_search(self, query, k, **options)

    def close(self) -> None:
        """Stop the parallel decrypt workers, which hold a copy of the secret key in memory."""
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.close()
            self._pool = None

    def _upload_keys(self) -> None:
        pb = self.pb
        files = self.kit.key_files(pb)

        def chunks():
            first = True
            for kind, data in files.items():
                for off in range(0, len(data), KEY_CHUNK):
                    c = pb.UploadKeysChunk(
                        api_version=API_VERSION,
                        kind=kind,
                        offset=off,
                        data=data[off : off + KEY_CHUNK],
                        last=off + KEY_CHUNK >= len(data),
                    )
                    if first:
                        c.collection = self.name
                        first = False
                    yield c

        _check(self.stub.UploadKeys(chunks()).error)

    def _insert_centers(self) -> None:
        pb, d = self.pb, self.derived.dimension
        cols = [self.kit.encrypt_column(self.centers[:, j]) for j in range(d)]
        block = pb.InsertBlock(
            api_version=API_VERSION,
            collection=self.name,
            cluster_id=self.cluster_count,
            ids=[b"center%d" % c for c in range(self.cluster_count)],
            columns=cols,
            is_center=True,
        )
        self._insert([block])

    def _columns(self, vecs: np.ndarray, fields: np.ndarray) -> list[bytes]:
        cols = [self.kit.encrypt_column(vecs[:, j]) for j in range(self.derived.dimension)]
        for fi, card in enumerate(self.cards):
            for v in range(card):
                cols.append(self.kit.encrypt_column((fields[:, fi] == v).astype(np.float64)))
        return cols

    def _insert_cluster(
        self, cluster: int, vecs: np.ndarray, fields: np.ndarray, ids: Sequence[bytes]
    ) -> None:
        blocks = []
        for s in range(0, len(vecs), self.slots):
            part = slice(s, s + self.slots)
            blocks.append(
                self.pb.InsertBlock(
                    api_version=API_VERSION,
                    collection=self.name,
                    cluster_id=cluster,
                    ids=list(ids[part]),
                    columns=self._columns(vecs[part], fields[part]),
                )
            )
        self._insert(blocks)

    def _insert(self, blocks: list[Any]) -> int:
        r = self.stub.Insert(iter(blocks))
        _check(r.error)
        return r.inserted

    def insert_vectors(
        self,
        vectors: np.ndarray,
        ids: Sequence[bytes],
        fields: np.ndarray | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
        documents: Sequence[str | bytes | None] | None = None,
    ) -> None:
        """Assign each vector to its nearest center, upload one new block per cluster, and put the
        encrypted record (exact vector copy and metadata) and document of each vector."""
        x = np.asarray(vectors, dtype=np.float64)
        if len(ids) != len(x):
            raise ValueError("one id per vector")
        if not np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-6):
            raise ValueError("vectors must be normalized before they are encrypted")
        for ident in ids:
            check_user_id(ident)
        f = _fields(fields, len(x), self.cards)
        labels = assign(x, self.centers)
        for c in np.unique(labels):
            rows = np.nonzero(labels == c)[0]
            self._insert_cluster(int(c), x[rows], f[rows], [ids[i] for i in rows])
        self._put_records(x, ids, metadata, documents, f)
        if self.stats is not None:
            self.stats.add(labels, f)
            self._save_stats()

    def _put_records(
        self,
        x: np.ndarray,
        ids: Sequence[bytes],
        metadata: Sequence[dict[str, Any]] | None,
        documents: Sequence[str | bytes | None] | None,
        fields: np.ndarray | None = None,
    ) -> None:
        for name, seq in (("metadata", metadata), ("documents", documents)):
            if seq is not None and len(seq) != len(ids):
                raise ValueError(f"one {name} entry per vector")
        v32 = np.asarray(x, dtype="<f4")
        for i, blob_id in enumerate(ids):
            meta = dict(metadata[i]) if metadata is not None else {}
            if fields is not None and fields.shape[1]:
                meta[FIELDS_KEY] = [int(v) for v in fields[i]]  # lets delete() update the stats
            sealed = self.codec.seal_record(blob_id, v32[i].tobytes(), meta)
            self.blobs.put(self.name, KIND_RECORD, blob_id, sealed)
            if documents is not None and documents[i] is not None:
                doc = self.codec.seal_document(blob_id, documents[i])
                self.blobs.put(self.name, KIND_DOCUMENT, blob_id, doc)

    def delete(self, ids: Sequence[bytes]) -> int:
        """Delete vectors (tombstones on the server) and their record and document blobs."""
        self._stats_for_delete(ids)
        r = self.stub.Delete(
            self.pb.DeleteRequest(api_version=API_VERSION, collection=self.name, ids=list(ids))
        )
        _check(r.error)
        for kind in (KIND_RECORD, KIND_DOCUMENT):
            self.blobs.delete(self.name, kind, ids)
        return r.deleted

    def _save_stats(self) -> None:
        """Store the cluster statistics as an encrypted blob under the reserved id (never plain)."""
        if self.stats is not None:
            self.blobs.put(self.name, KIND_RECORD, STATS_ID, self.stats.seal(self.codec))

    def _stats_for_delete(self, ids: Sequence[bytes]) -> None:
        """Lower the cluster counts for vectors about to be deleted (found via their records)."""
        if self.stats is None:
            return
        vecs, fields = [], []
        for ident in ids:
            try:
                rec = self.codec.open_record(ident, self.blobs.get(self.name, KIND_RECORD, ident))
            except (BlobStoreError, ValueError):
                continue  # no record: the vector was never stored through this client
            vecs.append(np.frombuffer(rec.vector, dtype="<f4").astype(np.float64))
            fields.append(rec.metadata.get(FIELDS_KEY))
        if not vecs:
            return
        labels = assign(np.stack(vecs), self.centers)
        known = all(f is not None for f in fields) and (
            fields[0] is None or len(fields[0]) == len(self.cards)
        )
        self.stats.remove(
            labels, np.asarray(fields, dtype=np.int64) if known and self.cards else None
        )
        self._save_stats()

    def retire(self) -> None:
        """Drop this collection from the server (after the caller switched to a newer one)."""
        r = self.stub.DropCollection(self.pb.DropCollectionRequest(name=self.name))
        _check(r.error)

    # ---- blocks ---------------------------------------------------------------------------
    def block_stats(self) -> list[BlockStat]:
        """Data blocks only; the center block is left out."""
        pb = self.pb
        r = self.stub.ListBlocks(
            pb.ListBlocksRequest(api_version=API_VERSION, collection=self.name)
        )
        _check(r.error)
        return [
            BlockStat(b.segment, b.cluster_id, b.n_vectors, b.n_deleted)
            for b in r.blocks
            if b.cluster_id != self.cluster_count
        ]

    def _get_block(self, segment: str) -> tuple[Any, list[bytes]]:
        pb = self.pb
        stream = self.stub.GetBlock(
            pb.GetBlockRequest(api_version=API_VERSION, collection=self.name, segment=segment)
        )
        header, cols = None, []
        for m in stream:
            part = m.WhichOneof("part")
            if part == "error":
                raise CollectionError(m.error.code, m.error.message)
            if part == "header":
                header = m.header
            else:
                cols.append(bytes(m.column))
        if header is None or len(cols) != header.n_columns:
            raise CollectionError(0, "incomplete block stream")
        return header, cols

    def read_live(self, segment: str) -> tuple[list[bytes], np.ndarray, np.ndarray, int]:
        """Decrypt a block: (live ids, live vectors, live field values, bytes read)."""
        header, cols = self._get_block(segment)
        keep = np.array([not d for d in header.deleted], dtype=bool)
        n = len(header.ids)
        d = self.derived.dimension
        plain = np.stack([self.kit.decrypt_column(c)[:n] for c in cols], axis=1)[keep]
        vecs = plain[:, :d]
        vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        fields = np.zeros((len(vecs), len(self.cards)), dtype=np.int64)
        pos = d
        for fi, card in enumerate(self.cards):
            fields[:, fi] = np.argmax(plain[:, pos : pos + card], axis=1)
            pos += card
        ids = [bytes(i) for i, k in zip(header.ids, keep, strict=True) if k]
        return ids, vecs, fields, sum(len(c) for c in cols)

    # ---- repack ---------------------------------------------------------------------------
    def repack(self, threshold: float = DEFAULT_THRESHOLD) -> RepackReport:
        """Rewrite clusters that have a block with `threshold` or more of its slots deleted.

        For each such cluster: GetBlock, decrypt, drop the deleted slots, merge blocks that fit in
        one block, re-encrypt, ReplaceBlock. Blocks with no live slot are left to the server.
        """
        pb = self.pb
        groups = plan_repack(self.block_stats(), self.slots, threshold)
        before = after = n_before = n_after = 0
        for group in groups:
            parts = [self.read_live(b.segment) for b in group]
            ids = [i for p in parts for i in p[0]]
            vecs = np.concatenate([p[1] for p in parts])
            fields = np.concatenate([p[2] for p in parts])
            before += sum(p[3] for p in parts)
            cols = self._columns(vecs, fields)
            after += sum(len(c) for c in cols)
            segments = [b.segment for b in group]

            def msgs(segments=segments, ids=ids, cols=cols):
                yield pb.ReplaceBlockRequest(
                    api_version=API_VERSION,
                    header=pb.ReplaceBlockHeader(collection=self.name, segments=segments, ids=ids),
                )
                for c in cols:
                    yield pb.ReplaceBlockRequest(api_version=API_VERSION, column=c)

            _check(self.stub.ReplaceBlock(msgs()).error)
            n_before += len(group)
            n_after += 1
        return RepackReport(len(groups), n_before, n_after, before, after)

    # ---- re-clustering --------------------------------------------------------------------
    def recluster(
        self, n_clusters: int | None = None, *, seed: int = 1, openfhe_version: str | None = None
    ) -> Collection:
        """Build the next generation from the live vectors of this one; this one is untouched.

        Search keeps working on this collection throughout. When the new one is ready the caller
        switches to it and calls `retire()` on this one. Deleted slots are not carried over, so
        the new blocks are dense. The new collection uses the same key pair (its public keys are
        uploaded again) and gets a byte-for-byte copy of the record and document blobs.
        """
        ids: list[bytes] = []
        vecs, fields = [], []
        for b in self.block_stats():
            if b.live == 0:
                continue
            i, v, f, _ = self.read_live(b.segment)
            ids += i
            vecs.append(v)
            fields.append(f)
        if not ids:
            raise ValueError("no live vectors to cluster")
        new = Collection.create(
            self.pb,
            self.stub,
            self.base_name,
            np.concatenate(vecs),
            ids,
            preset=self.derived.preset,  # a core preset name; presets.derive accepts it too
            fields=np.concatenate(fields),
            cards=self.cards,
            n_clusters=n_clusters or self.cluster_count,
            openfhe_version=openfhe_version or self.kit.openfhe_version,
            seed=seed,
            generation=self.generation + 1,
            kit=self.kit,
            security_level=self.security_level,
            store_records=False,  # the exact vectors are copied below, not re-derived
            blobs=self.blobs,
        )
        try:
            self._copy_blobs(new.name, ids)
        except Exception:
            new.retire()
            raise
        return new

    def _copy_blobs(self, target: str, ids: Sequence[bytes]) -> None:
        """Copy the encrypted blobs of `ids` (opaque bytes) to `target`, skipping missing ones."""
        pb = self.pb
        for kind in (pb.BLOB_KIND_RECORD, pb.BLOB_KIND_DOCUMENT):
            for blob_id in ids:
                req = pb.GetBlobRequest(
                    api_version=API_VERSION, collection=self.name, kind=kind, id=blob_id
                )
                data, size = bytearray(), None
                for r in self.stub.GetBlob(req):
                    part = r.WhichOneof("part")
                    if part == "error":
                        if r.error.code == pb.ERROR_BLOB_NOT_FOUND:
                            break
                        raise CollectionError(r.error.code, r.error.message)
                    if part == "size":
                        size = r.size
                    else:
                        data += r.data
                if size is None:
                    continue
                if len(data) != size:
                    raise CollectionError(pb.ERROR_INTERNAL, "incomplete blob stream")
                header = pb.PutBlobHeader(collection=target, kind=kind, id=blob_id, size=size)

                def msgs(header=header, data=bytes(data)):
                    yield pb.PutBlobRequest(api_version=API_VERSION, header=header)
                    for o in range(0, len(data), pb.LIMIT_MAX_CHUNK_BYTES):
                        chunk = data[o : o + pb.LIMIT_MAX_CHUNK_BYTES]
                        yield pb.PutBlobRequest(api_version=API_VERSION, data=chunk)

                _check(self.stub.PutBlob(msgs()).error)


def _fields(fields: np.ndarray | None, n: int, cards: Sequence[int]) -> np.ndarray:
    if not cards:
        return np.zeros((n, 0), dtype=np.int64)
    if fields is None:
        raise ValueError("fields are required when the collection has filter fields")
    f = np.asarray(fields, dtype=np.int64)
    if f.shape != (n, len(cards)):
        raise ValueError("fields must have shape (n, number of filter fields)")
    if np.any(f < 0) or np.any(f >= np.asarray(cards)):
        raise ValueError("a field value is outside its cardinality")
    return f
