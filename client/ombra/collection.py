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
new collection (generation g + 1, with its own keys upload) from the live vectors of the old one
and leaves the old one untouched and searchable; the caller switches to the new one and calls
`retire()` on the old one. Dropped slots never reach the new collection.

The transport is a protobuf messages module `pb` and a gRPC stub of service ombra.v1.Ombra,
passed in by the caller (the generated modules are built from proto/ombra/v1/ombra.proto).
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ombra import presets
from ombra import query as oq
from ombra.clustering import assign, cluster_vectors
from ombra.file_format import Kind, read_envelope, write_envelope
from ombra.params import make_ckks_params
from ombra.repack import DEFAULT_THRESHOLD, BlockStat, plan_repack

KEY_CHUNK = 4_000_000  # below LIMIT_MAX_CHUNK_BYTES (4 MiB)
API_VERSION = 1


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

    def __init__(self, derived: presets.Derived, openfhe_version: str, n_leaves: int) -> None:
        import openfhe as ofhe

        self.derived = derived
        self.openfhe_version = openfhe_version
        self.slots = derived.batch_size
        cc = ofhe.GenCryptoContext(make_ckks_params(derived.values))
        for feature in (
            ofhe.PKESchemeFeature.PKE,
            ofhe.PKESchemeFeature.KEYSWITCH,
            ofhe.PKESchemeFeature.LEVELEDSHE,
        ):
            cc.Enable(feature)
        self.cc = cc
        self._keys = cc.KeyGen()
        cc.EvalMultKeyGen(self._keys.secretKey)
        cc.ClearEvalAutomorphismKeys()  # the key store is static: hold exactly this layout's keys
        cc.EvalAutomorphismKeyGen(
            self._keys.secretKey, oq.automorphism_indices(cc.GetRingDimension(), n_leaves)
        )
        self.key_tag = self._keys.publicKey.GetKeyTag()

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
    ) -> None:
        self.pb, self.stub = pb, stub
        self.base_name, self.generation = base_name, generation
        self.derived, self.cards, self.kit = derived, tuple(cards), kit
        self.centers = centers
        self.slots = derived.batch_size

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
    ) -> Collection:
        """Cluster, create the collection, upload the keys once, insert centers and data."""
        x = np.asarray(vectors, dtype=np.float64)
        derived = presets.derive(preset, x.shape[1])  # hard rule 1: parameters only from here
        if len(ids) != len(x):
            raise ValueError("one id per vector")
        if not np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-6):
            raise ValueError("vectors must be normalized before they are encrypted")
        cards = tuple(cards)
        f = _fields(fields, len(x), cards)
        k = n_clusters or max(1, round(len(x) / derived.batch_size))
        if k > derived.batch_size:
            raise ValueError("more clusters than slots: the centers must fit in one block")
        clustering = cluster_vectors(x, k, seed=seed)
        if openfhe_version is None:
            import openfhe as ofhe

            openfhe_version = ofhe.__version__
        kit = KeyKit(derived, openfhe_version, derived.dimension + sum(cards))
        self = cls(
            pb,
            stub,
            base_name=name,
            generation=generation,
            derived=derived,
            cards=cards,
            kit=kit,
            centers=clustering.centers,
        )
        spec = pb.CollectionSpec(
            api_version=API_VERSION,
            name=self.name,
            preset=derived.preset,
            dimension=derived.dimension,
            security_level=pb.SECURITY_LEVEL_FAST,
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
        except Exception:
            stub.DropCollection(pb.DropCollectionRequest(name=self.name))
            raise
        return self

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
    ) -> None:
        """Assign each vector to its nearest center and upload one new block per cluster."""
        x = np.asarray(vectors, dtype=np.float64)
        f = _fields(fields, len(x), self.cards)
        labels = assign(x, self.centers)
        for c in np.unique(labels):
            rows = np.nonzero(labels == c)[0]
            self._insert_cluster(int(c), x[rows], f[rows], [ids[i] for i in rows])

    def delete(self, ids: Sequence[bytes]) -> int:
        r = self.stub.Delete(
            self.pb.DeleteRequest(api_version=API_VERSION, collection=self.name, ids=list(ids))
        )
        _check(r.error)
        return r.deleted

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
        the new blocks are dense. The new collection has its own keys and key upload.
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
        return Collection.create(
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
        )


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
