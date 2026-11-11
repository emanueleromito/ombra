"""Local mode: an in-process CPU server behind the same API as the gRPC server (T3.7).

`LocalServer` has the method names and message types of the gRPC stub of service ombra.v1.Ombra
(streams are Python iterators), so `Collection`, `run_search` and `GrpcBlobStore` run on it
unchanged. It is the CPU twin of the C++ server in server/: every computation on ciphertexts is
OpenFHE (openfhe-python), no CUDA and no kernel of ours. It never sees a secret key: like the real
server it gets a context, the EvalMult key and the automorphism keys, as bytes in 0016 envelopes.
It refuses what the real server refuses (api version, name and id sizes, preset and key tag,
filter layout above F_max, sizes of blobs, block layouts); the error codes are the same.

Search, as in the C++ server (decisions 0010 and 0019): expand the query ciphertext into one
ciphertext per entry (the same constants q_j in every slot), then for each block the dot product
sum_j leaf_j * column_j (EvalMultNoRelin, one relinearization, one rescale), and for a tombstoned
slot add the plaintext penalty -lambda * (F + 1).

Expansion. openfhe-python has no exact multiplication by X^-k, which the SealPIR-style tree of the
C++ server needs. The same leaf is obtained from the symmetric query polynomial p with a real
plaintext: with C_r = X^r + X^-r, the coefficients of p * C_r at the multiples of 2^L (L =
ceil(log2(2 * leaves))) are 2 q_r, and the subgroup trace prod_l (1 + tau_l), tau_l = the
automorphism with Galois element N / 2^l + 1, keeps exactly those and multiplies them by 2^L (the
client pre-scaled the query by 2^-L). So leaf_r = trace(p * cos-plaintext_r), using the same
automorphism keys. It costs L key switches per leaf instead of about two per leaf for the tree, one
extra level for the plaintext product, and the same results to about 1e-6.

Speed limits (CPU, one thread of Python, measured on Modal, 4 vCPU): see docs/local-mode.md.
Files are kept under the directory given to `Client.local(path)` (format "ombra-local 1").
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ombra import params as oparams
from ombra import presets
from ombra import query as oq
from ombra._proto import load_stubs
from ombra.file_format import EnvelopeError, Kind, read_envelope, write_envelope

FORMAT = b"ombra-local 1\n"
API = 1
MAX_CHUNK = 4194304
MAX_NAME = 128
MAX_ID = 256
MAX_IDS = 100000
MAX_CLUSTERS_ROUND = 4096
MAX_CLUSTER_COUNT = 1 << 20
MAX_KEY_FILE = 1 << 30
BUCKETS = (16384, 262144, 2097152, 16777216)
LAMBDA = 4.0
KEY_KINDS = {1: "context", 2: "evalmult", 3: "automorphism"}
CACHE_BLOCKS = 16


class Refuse(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


class LocalTransport:
    """What `ClientCore` needs: pb, stub, openfhe_version, close()."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        import openfhe

        self.pb, _ = load_stubs(grpc_module=False)
        self.openfhe_version = openfhe.__version__
        self.server = LocalServer(path, self.pb, self.openfhe_version)
        self.stub = self.server

    def __repr__(self) -> str:
        return "LocalTransport()"

    def close(self) -> None:
        self.server.close()


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def dlog5(galois: int, ring: int) -> int:
    """r with 5^r = galois mod 2N (rotation index of an automorphism of the slots)."""
    e = 1
    for r in range(ring // 2):
        if e == galois:
            return r
        e = (e * 5) % (2 * ring)
    raise ValueError("Galois element is not a slot rotation")


class _Coll:
    def __init__(self, spec: Any, root: Path) -> None:
        self.spec, self.root = spec, root
        self.keys: dict[str, bytes] = {}
        self.segments: dict[str, Any] = {}  # name -> InsertBlock (ids + column envelopes)
        self.order: list[str] = []
        self.tombstones: set[bytes] = set()
        self.next_seg = 0
        self.ctx: Any = None  # (cc, ring, batch) after keys are installed


class LocalServer:
    def __init__(self, path: str | os.PathLike[str], pb: Any, openfhe_version: str) -> None:
        self.pb, self.version = pb, openfhe_version
        self.root = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / "FORMAT"
        if marker.exists():
            if marker.read_bytes() != FORMAT:
                raise ValueError("unknown local store format or version")
        else:
            _atomic(marker, FORMAT)
        self._tmp = Path(tempfile.mkdtemp(prefix="ombra-local-"))
        self._cols: dict[tuple[str, str], list[Any]] = {}  # (collection, segment) -> ciphertexts
        self._colls: dict[str, _Coll] = {}
        for d in sorted((self.root / "c").glob("*")) if (self.root / "c").exists() else []:
            spec = pb.CollectionSpec.FromString((d / "spec.pb").read_bytes())
            self._colls[spec.name] = self._load(spec, d)

    def close(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)
        self._cols.clear()

    def __repr__(self) -> str:
        return "LocalServer()"

    # ---- storage ----------------------------------------------------------------------------
    def _dir(self, name: str) -> Path:
        return self.root / "c" / name.encode().hex()

    def _load(self, spec: Any, d: Path) -> _Coll:
        c = _Coll(spec, d)
        for k in KEY_KINDS.values():
            if (d / "keys" / k).exists():
                c.keys[k] = (d / "keys" / k).read_bytes()
        if (d / "seg_counter").exists():
            c.next_seg = int((d / "seg_counter").read_text())
        for f in sorted((d / "blocks").glob("*.pb")) if (d / "blocks").exists() else []:
            c.segments[f.stem] = self.pb.InsertBlock.FromString(f.read_bytes())
            c.order.append(f.stem)
        if (d / "tombstones").exists():
            c.tombstones = {bytes.fromhex(x) for x in (d / "tombstones").read_text().split()}
        if len(c.keys) == 3:
            self._install(c)
        return c

    def _save_tombstones(self, c: _Coll) -> None:
        _atomic(c.root / "tombstones", "\n".join(sorted(t.hex() for t in c.tombstones)).encode())

    def _coll(self, name: str) -> _Coll:
        c = self._colls.get(name)
        if c is None:
            raise Refuse(self.pb.ERROR_COLLECTION_NOT_FOUND, "collection not found")
        return c

    def _err(self, e: Refuse) -> Any:
        return self.pb.Error(code=e.code, message=e.message)

    def _version(self, v: int) -> None:
        if v != API:
            raise Refuse(self.pb.ERROR_UNSUPPORTED_VERSION, "unsupported api_version")

    def _envelope(self, c: _Coll, raw: bytes, kind: Kind) -> bytes:
        try:
            return read_envelope(bytes(raw), kind, c.spec.preset, self.version).payload
        except EnvelopeError as ex:
            msg = str(ex)
            if "unknown format version" in msg:
                raise Refuse(
                    self.pb.ERROR_UNSUPPORTED_VERSION, "unsupported envelope version"
                ) from None
            if "preset mismatch" in msg:
                raise Refuse(self.pb.ERROR_PRESET_MISMATCH, "envelope preset differs") from None
            raise Refuse(self.pb.ERROR_INVALID_ARGUMENT, "invalid envelope") from None

    # ---- collections ------------------------------------------------------------------------
    def _validate(self, s: Any) -> None:
        pb = self.pb
        self._version(s.api_version)
        if not s.name:
            raise Refuse(pb.ERROR_INVALID_ARGUMENT, "collection name is empty")
        if len(s.name.encode()) > MAX_NAME:
            raise Refuse(pb.ERROR_SIZE_LIMIT, "collection name is too long")
        try:
            derived = presets.derive(s.preset, s.dimension or 384)
        except oparams.ParamsError:
            raise Refuse(pb.ERROR_INVALID_ARGUMENT, "unknown preset") from None
        if s.dimension == 0:
            raise Refuse(pb.ERROR_INVALID_ARGUMENT, "dimension must be positive")
        if not s.key_tag:
            raise Refuse(pb.ERROR_INVALID_ARGUMENT, "key_tag is empty")
        if not 1 <= s.cluster_count <= MAX_CLUSTER_COUNT:
            raise Refuse(pb.ERROR_INVALID_ARGUMENT, "cluster_count out of range")
        if not 0 <= s.security_level <= 3:
            raise Refuse(pb.ERROR_INVALID_ARGUMENT, "unknown security level")
        fmax = min(presets.filter_field_limit(derived), presets.MAX_FILTER_FIELDS)
        bad = pb.ERROR_INVALID_FILTER_LAYOUT
        if len(s.filter_fields) > fmax:
            raise Refuse(bad, "layout: too many filter fields for this preset")
        for i, f in enumerate(s.filter_fields):
            if f.field_id != i:
                raise Refuse(bad, "layout: filter field ids must be 0, 1, ...")
            if not 1 <= f.cardinality <= presets.MAX_FILTER_CARDINALITY:
                raise Refuse(bad, "layout: filter cardinality out of range")
        cards = sum(f.cardinality for f in s.filter_fields)
        if s.dimension + cards > derived.ring_dim // 2:
            raise Refuse(bad, "layout: d + sum(cardinality) exceeds N/2")

    def CreateCollection(self, req: Any) -> Any:
        try:
            self._validate(req.spec)
            if req.spec.name in self._colls:
                raise Refuse(self.pb.ERROR_COLLECTION_EXISTS, "collection exists")
        except Refuse as e:
            return self.pb.CreateCollectionResponse(error=self._err(e))
        d = self._dir(req.spec.name)
        _atomic(d / "spec.pb", req.spec.SerializeToString())
        self._colls[req.spec.name] = _Coll(req.spec, d)
        return self.pb.CreateCollectionResponse(spec=req.spec)

    def _info(self, c: _Coll) -> Any:
        n = sum(len(b.ids) for b in c.segments.values() if b.cluster_id != c.spec.cluster_count)
        return self.pb.CollectionInfo(
            spec=c.spec,
            vector_count=n,
            deleted_count=len(c.tombstones),
            keys_uploaded=len(c.keys) == 3,
        )

    def GetCollection(self, req: Any) -> Any:
        try:
            return self.pb.GetCollectionResponse(info=self._info(self._coll(req.name)))
        except Refuse as e:
            return self.pb.GetCollectionResponse(error=self._err(e))

    def ListCollections(self, req: Any) -> Any:
        r = self.pb.ListCollectionsResponse()
        for n in sorted(self._colls):
            r.collections.append(self._info(self._colls[n]))
        return r

    def DropCollection(self, req: Any) -> Any:
        try:
            c = self._coll(req.name)
        except Refuse as e:
            return self.pb.DropCollectionResponse(error=self._err(e))
        shutil.rmtree(c.root, ignore_errors=True)
        del self._colls[req.name]
        for k in [k for k in self._cols if k[0] == req.name]:
            del self._cols[k]
        return self.pb.DropCollectionResponse()

    # ---- keys -------------------------------------------------------------------------------
    def UploadKeys(self, chunks: Iterable[Any]) -> Any:
        pb = self.pb
        try:
            name, cur, buf, staged, n = "", 0, bytearray(), {}, 0
            for c in chunks:
                n += 1
                self._version(c.api_version)
                if n == 1:
                    if not c.collection:
                        raise Refuse(pb.ERROR_INVALID_ARGUMENT, "collection is required")
                    name = c.collection
                    self._coll(name)
                elif c.collection and c.collection != name:
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, "collection changed")
                if c.kind not in KEY_KINDS:
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, "unknown key kind")
                if cur and c.kind != cur:
                    raise Refuse(pb.ERROR_BAD_SEQUENCE, "key kind changed before last")
                if not cur:
                    if c.offset != 0:
                        raise Refuse(pb.ERROR_BAD_SEQUENCE, "offset must start at 0")
                    if KEY_KINDS[c.kind] in staged:
                        raise Refuse(pb.ERROR_INVALID_ARGUMENT, "key kind uploaded twice")
                    cur, buf = c.kind, bytearray()
                if c.offset != len(buf):
                    raise Refuse(pb.ERROR_BAD_SEQUENCE, "chunk offset does not follow")
                if len(c.data) > MAX_CHUNK:
                    raise Refuse(pb.ERROR_SIZE_LIMIT, "chunk is larger than the limit")
                if len(buf) + len(c.data) > MAX_KEY_FILE:
                    raise Refuse(pb.ERROR_SIZE_LIMIT, "key file is too large")
                buf += c.data
                if c.last:
                    staged[KEY_KINDS[cur]] = bytes(buf)
                    cur = 0
            if n == 0:
                raise Refuse(pb.ERROR_BAD_SEQUENCE, "no chunks")
            if cur:
                raise Refuse(pb.ERROR_BAD_SEQUENCE, "stream ended before last chunk")
            if set(staged) != set(KEY_KINDS.values()):
                raise Refuse(pb.ERROR_INVALID_ARGUMENT, "an upload needs all three key kinds")
            c = self._coll(name)
            self._check_keys(c, staged)
            for k, v in staged.items():
                _atomic(c.root / "keys" / k, v)
            c.keys = staged
            self._install(c)
        except Refuse as e:
            return pb.UploadKeysResponse(error=self._err(e))
        return pb.UploadKeysResponse()

    def _check_keys(self, c: _Coll, files: dict[str, bytes]) -> None:
        self._envelope(c, files["context"], Kind.CONTEXT)
        tag = c.spec.key_tag.encode()
        for k in ("evalmult", "automorphism"):
            payload = self._envelope(c, files[k], Kind.PUBLIC_EVAL_KEY)
            if tag not in payload:  # the key tag is stored in every evaluation key
                raise Refuse(
                    self.pb.ERROR_KEY_TAG_MISMATCH,
                    "evaluation keys were made under another key than the collection's",
                )

    def _file(self, payload: bytes, name: str = "x.bin") -> str:
        p = self._tmp / name
        p.write_bytes(payload)
        return str(p)

    def _install(self, c: _Coll) -> None:
        import openfhe as ofhe

        pl = self._envelope(c, c.keys["context"], Kind.CONTEXT)
        cc, ok = ofhe.DeserializeCryptoContext(self._file(pl), ofhe.BINARY)
        if not ok:
            raise Refuse(self.pb.ERROR_INVALID_ARGUMENT, "cannot read the context")
        mult = self._file(self._envelope(c, c.keys["evalmult"], Kind.PUBLIC_EVAL_KEY))
        auto = self._file(
            self._envelope(c, c.keys["automorphism"], Kind.PUBLIC_EVAL_KEY), "auto.bin"
        )
        try:
            ok_mult = cc.DeserializeEvalMultKey(mult, ofhe.BINARY)
        except RuntimeError as ex:
            # OpenFHE's key store is process-wide: when the client lives in this process, its own
            # (identical, same key tag) public keys are already there.
            if "there is a key vector for the given keyTag" not in str(ex):
                raise
            ok_mult = True
        if not ok_mult or not cc.DeserializeEvalAutomorphismKey(auto, ofhe.BINARY):
            raise Refuse(self.pb.ERROR_INVALID_ARGUMENT, "cannot read the evaluation keys")
        c.ctx = (cc, cc.GetRingDimension(), cc.GetEncodingParams().GetBatchSize())

    # ---- insert, delete ---------------------------------------------------------------------
    def Insert(self, blocks: Iterable[Any]) -> Any:
        pb, n, name, c = self.pb, 0, "", None
        try:
            count = 0
            for b in blocks:
                self._version(b.api_version)
                if count == 0:
                    name = b.collection
                    c = self._coll(name)
                    if c.ctx is None:
                        raise Refuse(pb.ERROR_KEYS_MISSING, "evaluation keys are not uploaded")
                elif b.collection != name:
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, "collection changed in one stream")
                count += 1
                spec = c.spec
                if b.cluster_id > spec.cluster_count:
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, "cluster id out of range")
                if b.is_center != (b.cluster_id == spec.cluster_count):
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, "is_center must match the center id")
                if len(b.ids) == 0:
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, "block has no ids")
                if len(b.ids) > c.ctx[2]:
                    raise Refuse(pb.ERROR_SIZE_LIMIT, "block has more ids than slots")
                known = {i for s in c.segments.values() for i in s.ids}
                seen: set[bytes] = set()
                for i in b.ids:
                    if not i:
                        raise Refuse(pb.ERROR_INVALID_ARGUMENT, "empty id")
                    if len(i) > MAX_ID:
                        raise Refuse(pb.ERROR_SIZE_LIMIT, "id is too long")
                    if i in seen or i in known:
                        raise Refuse(pb.ERROR_INVALID_ARGUMENT, "duplicate id")
                    seen.add(bytes(i))
                cards = sum(f.cardinality for f in spec.filter_fields)
                want = spec.dimension + (0 if b.is_center else cards)
                if len(b.columns) != want:
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, "wrong number of columns")
                for col in b.columns:
                    self._envelope(c, col, Kind.CIPHERTEXT)
                self._put_segment(c, b)
                n += len(b.ids)
            if count == 0:
                raise Refuse(pb.ERROR_BAD_SEQUENCE, "no blocks")
        except Refuse as e:
            return pb.InsertResponse(inserted=n, error=self._err(e))
        return pb.InsertResponse(inserted=n)

    def _put_segment(self, c: _Coll, b: Any) -> str:
        seg = f"s{c.next_seg:08d}"
        c.next_seg += 1
        _atomic(c.root / "blocks" / f"{seg}.pb", b.SerializeToString())
        _atomic(c.root / "seg_counter", str(c.next_seg).encode())
        c.segments[seg] = b
        c.order.append(seg)
        return seg

    def _drop_segment(self, c: _Coll, seg: str) -> None:
        (c.root / "blocks" / f"{seg}.pb").unlink(missing_ok=True)
        c.segments.pop(seg, None)
        c.order.remove(seg)
        self._cols.pop((c.spec.name, seg), None)

    def _purge(self, c: _Coll) -> None:
        """Drop blocks whose vectors are all deleted, and tombstones of ids no block holds."""
        for seg in list(c.order):
            b = c.segments[seg]
            if b.cluster_id != c.spec.cluster_count and all(i in c.tombstones for i in b.ids):
                self._drop_segment(c, seg)
        held = {i for s in c.segments.values() for i in s.ids}
        c.tombstones &= held
        self._save_tombstones(c)

    def Delete(self, req: Any) -> Any:
        pb = self.pb
        try:
            self._version(req.api_version)
            if len(req.ids) > MAX_IDS:
                raise Refuse(pb.ERROR_SIZE_LIMIT, "too many ids")
            c = self._coll(req.collection)
            for i in req.ids:
                if not i or len(i) > MAX_ID:
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, "invalid id")
            data = {
                i
                for s in c.segments.values()
                if s.cluster_id != c.spec.cluster_count
                for i in s.ids
            }
            todo = {bytes(i) for i in req.ids} & data - c.tombstones
            c.tombstones |= todo
            self._purge(c)
            return pb.DeleteResponse(deleted=len(todo))
        except Refuse as e:
            return pb.DeleteResponse(error=self._err(e))

    # ---- blocks -----------------------------------------------------------------------------
    def _stat(self, c: _Coll, seg: str) -> Any:
        b = c.segments[seg]
        return self.pb.BlockInfo(
            segment=seg,
            cluster_id=b.cluster_id,
            n_vectors=len(b.ids),
            n_deleted=sum(1 for i in b.ids if i in c.tombstones),
        )

    def ListBlocks(self, req: Any) -> Any:
        pb = self.pb
        try:
            self._version(req.api_version)
            c = self._coll(req.collection)
            r = pb.ListBlocksResponse(api_version=API)
            for s in c.order:
                r.blocks.append(self._stat(c, s))
            return r
        except Refuse as e:
            return pb.ListBlocksResponse(api_version=API, error=self._err(e))

    def GetBlock(self, req: Any) -> Iterator[Any]:
        pb = self.pb
        try:
            self._version(req.api_version)
            c = self._coll(req.collection)
            b = c.segments.get(req.segment)
            if b is None:
                raise Refuse(pb.ERROR_BLOCK_NOT_FOUND, "block not found")
        except Refuse as e:
            yield pb.GetBlockResponse(api_version=API, error=self._err(e))
            return
        h = pb.BlockHeader(cluster_id=b.cluster_id, n_columns=len(b.columns))
        for i in b.ids:
            h.ids.append(i)
            h.deleted.append(i in c.tombstones)
        yield pb.GetBlockResponse(api_version=API, header=h)
        for col in b.columns:
            yield pb.GetBlockResponse(api_version=API, column=col)

    def ReplaceBlock(self, msgs: Iterable[Any]) -> Any:
        pb = self.pb
        try:
            header, cols = None, []
            for m in msgs:
                self._version(m.api_version)
                part = m.WhichOneof("part")
                if part == "header":
                    if header is not None:
                        raise Refuse(pb.ERROR_BAD_SEQUENCE, "second header")
                    header = m.header
                    if not 1 <= len(header.segments) <= MAX_CLUSTERS_ROUND:
                        raise Refuse(pb.ERROR_INVALID_ARGUMENT, "1 to 4096 blocks must be listed")
                    if len(header.ids) > MAX_IDS:
                        raise Refuse(pb.ERROR_SIZE_LIMIT, "too many ids")
                    if any(not i or len(i) > MAX_ID for i in header.ids):
                        raise Refuse(pb.ERROR_INVALID_ARGUMENT, "invalid id")
                elif part == "column":
                    if header is None:
                        raise Refuse(pb.ERROR_BAD_SEQUENCE, "a column before the header")
                    cols.append(bytes(m.column))
                else:
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, "empty message")
            if header is None:
                raise Refuse(pb.ERROR_BAD_SEQUENCE, "no header")
            c = self._coll(header.collection)
            if len(header.ids) > c.ctx[2] if c.ctx else False:
                raise Refuse(pb.ERROR_SIZE_LIMIT, "block has more ids than slots")
            segs = list(header.segments)
            if any(s not in c.segments for s in segs):
                raise Refuse(pb.ERROR_BLOCK_NOT_FOUND, "block not found")
            if len(set(segs)) != len(segs):
                raise Refuse(pb.ERROR_BLOCK_MISMATCH, "a block is listed twice")
            cluster = c.segments[segs[0]].cluster_id
            live: list[bytes] = []
            for s in segs:
                b = c.segments[s]
                if b.cluster_id == c.spec.cluster_count:
                    raise Refuse(pb.ERROR_BLOCK_MISMATCH, "center blocks cannot be replaced")
                if b.cluster_id != cluster:
                    raise Refuse(pb.ERROR_BLOCK_MISMATCH, "cluster differs from the old block's")
                if len(cols) != len(b.columns):
                    raise Refuse(pb.ERROR_BLOCK_MISMATCH, "column count differs")
                live += [bytes(i) for i in b.ids if i not in c.tombstones]
            ids = [bytes(i) for i in header.ids]
            if not ids:
                raise Refuse(
                    pb.ERROR_BLOCK_MISMATCH, "no ids: a block with no live vector is dropped"
                )
            if len(set(ids)) != len(ids) or set(ids) != set(live) or len(ids) != len(live):
                raise Refuse(pb.ERROR_BLOCK_MISMATCH, "ids are not exactly the live ids")
            for col in cols:
                self._envelope(c, col, Kind.CIPHERTEXT)
            for s in segs:
                self._drop_segment(c, s)
            new = pb.InsertBlock(api_version=API, collection=c.spec.name, cluster_id=cluster)
            new.ids.extend(ids)
            new.columns.extend(cols)
            seg = self._put_segment(c, new)
            self._purge(c)
            return pb.ReplaceBlockResponse(api_version=API, segment=seg)
        except Refuse as e:
            return pb.ReplaceBlockResponse(api_version=API, error=self._err(e))

    # ---- blobs ------------------------------------------------------------------------------
    def _blob_path(self, c: _Coll, kind: int, blob_id: bytes) -> Path:
        return c.root / "blobs" / str(kind) / bytes(blob_id).hex()

    def _blob_kind(self, kind: int) -> None:
        if kind not in (self.pb.BLOB_KIND_RECORD, self.pb.BLOB_KIND_DOCUMENT):
            raise Refuse(self.pb.ERROR_INVALID_ARGUMENT, "unknown blob kind")

    def PutBlob(self, msgs: Iterable[Any]) -> Any:
        pb = self.pb
        try:
            head, data = None, bytearray()
            for m in msgs:
                self._version(m.api_version)
                part = m.WhichOneof("part")
                if part == "header":
                    if head is not None:
                        raise Refuse(pb.ERROR_BAD_SEQUENCE, "second header")
                    head = m.header
                    if not head.collection or len(head.collection.encode()) > MAX_NAME:
                        raise Refuse(pb.ERROR_INVALID_ARGUMENT, "invalid collection name")
                    if not head.id or len(head.id) > MAX_ID:
                        raise Refuse(pb.ERROR_INVALID_ARGUMENT, "invalid id")
                    self._blob_kind(head.kind)
                    if head.size not in BUCKETS:
                        raise Refuse(pb.ERROR_SIZE_LIMIT, "blob size is not a bucket size")
                elif part == "data":
                    if head is None:
                        raise Refuse(pb.ERROR_BAD_SEQUENCE, "data before the header")
                    if len(m.data) > MAX_CHUNK:
                        raise Refuse(pb.ERROR_SIZE_LIMIT, "chunk too large")
                    data += m.data
                    if len(data) > head.size:
                        raise Refuse(pb.ERROR_INVALID_ARGUMENT, "more data than the size")
                else:
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, "empty message")
            if head is None:
                raise Refuse(pb.ERROR_BAD_SEQUENCE, "no header")
            if len(data) != head.size:
                raise Refuse(pb.ERROR_INVALID_ARGUMENT, "less data than the size")
            c = self._coll(head.collection)
            _atomic(self._blob_path(c, head.kind, head.id), bytes(data))
        except Refuse as e:
            return pb.PutBlobResponse(api_version=API, error=self._err(e))
        return pb.PutBlobResponse(api_version=API)

    def GetBlob(self, req: Any) -> Iterator[Any]:
        pb = self.pb
        try:
            self._version(req.api_version)
            self._blob_kind(req.kind)
            if not req.id or len(req.id) > MAX_ID:
                raise Refuse(pb.ERROR_INVALID_ARGUMENT, "invalid id")
            p = self._blob_path(self._coll(req.collection), req.kind, req.id)
            if not p.exists():
                raise Refuse(pb.ERROR_BLOB_NOT_FOUND, "blob not found")
        except Refuse as e:
            yield pb.GetBlobResponse(api_version=API, error=self._err(e))
            return
        raw = p.read_bytes()
        yield pb.GetBlobResponse(api_version=API, size=len(raw))
        for o in range(0, len(raw), MAX_CHUNK):
            yield pb.GetBlobResponse(api_version=API, data=raw[o : o + MAX_CHUNK])

    def DeleteBlobs(self, req: Any) -> Any:
        pb = self.pb
        try:
            self._version(req.api_version)
            self._blob_kind(req.kind)
            if len(req.ids) > MAX_IDS:
                raise Refuse(pb.ERROR_SIZE_LIMIT, "too many ids")
            if any(not i or len(i) > MAX_ID for i in req.ids):
                raise Refuse(pb.ERROR_INVALID_ARGUMENT, "invalid id")
            c = self._coll(req.collection)
            n = 0
            for i in req.ids:
                p = self._blob_path(c, req.kind, i)
                if p.exists():
                    p.unlink()
                    n += 1
            return pb.DeleteBlobsResponse(deleted=n)
        except Refuse as e:
            return pb.DeleteBlobsResponse(error=self._err(e))

    # ---- search -----------------------------------------------------------------------------
    def _ct(self, payload: bytes) -> Any:
        import openfhe as ofhe

        ct, ok = ofhe.DeserializeCiphertext(self._file(payload, "ct.bin"), ofhe.BINARY)
        if not ok:
            raise Refuse(self.pb.ERROR_INVALID_ARGUMENT, "cannot read a ciphertext")
        return ct

    def _expand(self, c: _Coll, query: Any, n_leaves: int) -> list[Any]:
        """leaf_r = trace(p * plaintext_r) for r < n_leaves; see the module docstring."""
        cc, ring, _ = c.ctx
        depth = oq.expansion_depth(n_leaves)
        rots = [dlog5(g, ring) for g in oq.automorphism_indices(ring, n_leaves)]
        leaves = []
        for r in range(n_leaves):
            e = np.zeros(n_leaves)
            e[r] = 1.0 if r == 0 else 0.5
            slots = oq.symmetric_slots(e, ring)  # C_r / 2 (1 for r = 0)
            acc = cc.EvalMult(query, cc.MakeCKKSPackedPlaintext(slots.tolist()))
            for rot in rots[:depth]:
                acc = cc.EvalAdd(acc, cc.EvalAtIndex(acc, rot))
            leaves.append(cc.Rescale(acc))
        return leaves

    def _columns(self, c: _Coll, seg: str) -> list[Any]:
        key = (c.spec.name, seg)
        cols = self._cols.get(key)
        if cols is None:
            b = c.segments[seg]
            cols = [self._ct(self._envelope(c, col, Kind.CIPHERTEXT)) for col in b.columns]
            if len(self._cols) >= CACHE_BLOCKS:
                self._cols.pop(next(iter(self._cols)))
            self._cols[key] = cols
        return cols

    def _score(self, c: _Coll, seg: str, leaves: list[Any], tomb: bool) -> bytes:
        import openfhe as ofhe

        cc, _, batch = c.ctx
        b = c.segments[seg]
        cols = self._columns(c, seg)
        acc = None
        for leaf, col in zip(leaves, cols, strict=True):
            t = cc.EvalMultNoRelin(leaf, col)
            acc = t if acc is None else cc.EvalAdd(acc, t)
        res = cc.Rescale(cc.Relinearize(acc))
        if tomb:
            n_fields = len(c.spec.filter_fields)
            pen = np.zeros(batch)
            hit = False
            for i, ident in enumerate(b.ids):
                if ident in c.tombstones:
                    pen[i], hit = -LAMBDA * (n_fields + 1), True
            if hit:
                res = cc.EvalAdd(res, cc.MakeCKKSPackedPlaintext(pen.tolist(), 1, res.GetLevel()))
        p = self._tmp / "out.bin"
        if not ofhe.SerializeToFile(str(p), res, ofhe.BINARY):
            raise Refuse(self.pb.ERROR_INTERNAL, "internal error")
        return write_envelope(Kind.CIPHERTEXT, c.spec.preset, self.version, p.read_bytes())

    def Search(self, requests: Iterable[Any]) -> Iterator[Any]:
        pb = self.pb
        it = iter(requests)

        def error(e: Refuse) -> Any:
            return pb.SearchResponse(error=self._err(e))

        try:
            first = next(it, None)
            if first is None:
                return
            if first.WhichOneof("step") != "start":
                raise Refuse(pb.ERROR_BAD_SEQUENCE, "the first message must be SearchStart")
            st = first.start
            c = self._coll(st.collection)
            if st.key_tag != c.spec.key_tag:
                raise Refuse(pb.ERROR_KEY_TAG_MISMATCH, "request key tag differs")
            if c.ctx is None:
                raise Refuse(pb.ERROR_KEYS_MISSING, "evaluation keys are not uploaded")
            if st.filter_leaves != sum(f.cardinality for f in c.spec.filter_fields):
                raise Refuse(pb.ERROR_INVALID_FILTER_LAYOUT, "filter_leaves differs")
            if not st.query:
                raise Refuse(pb.ERROR_INVALID_ARGUMENT, "query is empty")
            query = self._ct(self._envelope(c, st.query, Kind.CIPHERTEXT))
            if query.GetKeyTag() != c.spec.key_tag:
                raise Refuse(pb.ERROR_KEY_TAG_MISMATCH, "query made under another key")
            if query.GetLevel() != 0 or query.GetNoiseScaleDeg() != 1:
                raise Refuse(pb.ERROR_INVALID_ARGUMENT, "query has an unexpected level or scale")
            d = c.spec.dimension
            leaves = self._expand(c, query, d + st.filter_leaves)
            center = c.spec.cluster_count
            out = pb.SearchCenterScores()
            for seg in c.order:
                if c.segments[seg].cluster_id == center:
                    out.score_blocks.append(self._score(c, seg, leaves[:d], False))
                    out.center_count += len(c.segments[seg].ids)
            yield pb.SearchResponse(centers=out)
            for req in it:
                if req.WhichOneof("step") != "chosen":
                    raise Refuse(pb.ERROR_BAD_SEQUENCE, "expected SearchChosen")
                ids = list(req.chosen.cluster_ids)
                if not ids:
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, "no clusters chosen")
                if len(ids) > MAX_CLUSTERS_ROUND:
                    raise Refuse(pb.ERROR_SIZE_LIMIT, "too many clusters in one round")
                if any(i >= center for i in ids):
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, "cluster id out of range")
                clusters = list(dict.fromkeys(ids))
                for n, cl in enumerate(clusters):
                    msg = pb.SearchScores(cluster_id=cl, last=n + 1 == len(clusters))
                    for seg in c.order:
                        b = c.segments[seg]
                        if b.cluster_id == cl:
                            msg.score_blocks.append(self._score(c, seg, leaves, True))
                            msg.ids.extend(b.ids)
                    yield pb.SearchResponse(scores=msg)
        except Refuse as e:
            yield error(e)
