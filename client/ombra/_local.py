"""Local mode: a CPU server in a child process behind the same API as the gRPC server.

`LocalTransport` (client process) starts the child with the `spawn` method and talks to it over a
local socket. The child runs `LocalServer`, which has the method names and message types of the gRPC
stub of service ombra.v1.Ombra (streams are Python iterators), so `Collection`, `run_search` and
`GrpcBlobStore` run on it unchanged, in both modes. It is the CPU twin of the C++ server in
server/. The child receives only the API messages: ciphertexts, the crypto context and public
evaluation keys, as bytes in 0016 envelopes (the child never gets the secret key, and its
code has no way to ask for one). It deserializes the context and the keys itself.

All OpenFHE work of the child is C++: `ombra._native` (client/native/, built against OpenFHE 1.6.1)
holds the context and keys of a collection, runs the server's exact expansion tree (decision 0010,
core/include/ombra/core/expansion_tree.hpp with the CPU operations of
core/src/openfhe_expand_ops.hpp; decision 0022) and the column dot product, so local mode uses the
same levels as the server. The child never imports openfhe-python: the client process holds one
OpenFHE copy (the wheel, for keys, encryption and decryption), the child another, never both in one
process. The child refuses envelopes whose OpenFHE version is not 1.6.1, and checks the key tag
against the keys it deserialized.

Search, as in the C++ server (decisions 0010 and 0019): expand the query ciphertext into one
ciphertext per entry (the same constants q_j in every slot), then for each block the dot product
sum_j leaf_j * column_j (EvalMultNoRelin, one relinearization, one rescale), and for a tombstoned
slot add the plaintext penalty -lambda * (F + 1).

It refuses what the real server refuses (api version, name and id sizes, preset and key tag,
filter layout above F_max, sizes of blobs, block layouts); the error codes are the same.
Files are kept under the directory given to `Client.local(path)` (format "ombra-local 1").
"""

from __future__ import annotations

import atexit
import multiprocessing
import multiprocessing.connection as mpc
import os
import secrets
import shutil
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ombra import params as oparams
from ombra import presets
from ombra._proto import load_stubs
from ombra.file_format import EnvelopeError, Kind, read_envelope, write_envelope

FORMAT = b"ombra-local 1\n"
OPENFHE_VERSION = "1.6.1"  # the server's OpenFHE; envelopes of another version are refused
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
CACHE_BYTES = 4 << 30  # deserialized columns kept between searches (least recently used out)


class Refuse(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def _native() -> Any:
    try:
        from ombra import _native
    except ImportError as exc:
        raise ImportError(
            "local mode needs the compiled extension ombra._native (OpenFHE 1.6.1 C++, Linux "
            "x86-64, Python 3.12): install a wheel of ombra, or build client/ with OpenFHE "
            "1.6.1 installed (docs/local-mode.md)"
        ) from exc
    return _native


# ---- the client side: start the child, speak the API to it ---------------------------------------
class ChildError(RuntimeError):
    """The local server process failed (no data is included in the message)."""


def _method(pb: Any, name: str) -> tuple[bool, bool, Any, Any]:
    """(client streams, server streams, request class, response class) from the service."""
    from google.protobuf import message_factory

    m = pb.DESCRIPTOR.services_by_name["Ombra"].methods_by_name[name]
    return (
        m.client_streaming,
        m.server_streaming,
        message_factory.GetMessageClass(m.input_type),
        message_factory.GetMessageClass(m.output_type),
    )


class _Channel:
    """One connection to the child; a call owns it until its response is fully read."""

    def __init__(self, conn: Any) -> None:
        self.conn, self.ok = conn, True


class _Stub:
    """Method names of the gRPC stub; each call runs on its own connection to the child."""

    def __init__(self, owner: LocalTransport) -> None:
        self._o = owner

    def __getattr__(self, method: str) -> Any:
        if (
            method.startswith("_")
            or method not in self._o.pb.DESCRIPTOR.services_by_name["Ombra"].methods_by_name
        ):
            raise AttributeError(method)
        return lambda request: self._o._call(method, request)


class LocalTransport:
    """What `ClientCore` needs: pb, stub, openfhe_version, close()."""

    default_preset = "fast"  # m0_compact_n8192: the preset local mode is sized for

    def __init__(self, path: str | os.PathLike[str]) -> None:
        import openfhe

        if ".".join(openfhe.__version__.split(".")[:3]) != OPENFHE_VERSION:
            raise RuntimeError(
                f"local mode needs openfhe-python {OPENFHE_VERSION}, found {openfhe.__version__}"
            )
        self.pb, _ = load_stubs(grpc_module=False)
        self._lock = threading.Lock()
        self._free: list[_Channel] = []
        self.closed = False
        self._authkey = secrets.token_bytes(32)
        self._sock = os.path.join(tempfile.mkdtemp(prefix="ombra-local-sock-"), "s")
        ctx = multiprocessing.get_context("spawn")
        self._proc = ctx.Process(
            target=_child_main,
            args=(str(Path(path)), self._sock, self._authkey, os.getpid()),
            daemon=True,
        )
        self._proc.start()
        atexit.register(self.close)
        deadline = time.monotonic() + 120
        while not os.path.exists(self._sock):
            if not self._proc.is_alive() or time.monotonic() > deadline:
                raise ChildError("the local server process did not start")
            time.sleep(0.02)
        self.stub = _Stub(self)
        self.openfhe_version = self._control("hello")["openfhe_version"]
        if self.openfhe_version != OPENFHE_VERSION:
            self.close()
            raise RuntimeError(
                f"ombra._native is built against OpenFHE {self.openfhe_version}, not "
                f"{OPENFHE_VERSION}"
            )

    def __repr__(self) -> str:
        return "LocalTransport()"

    # connections
    def _acquire(self) -> _Channel:
        with self._lock:
            if self._free:
                return self._free.pop()
        return _Channel(mpc.Client(self._sock, family="AF_UNIX", authkey=self._authkey))

    def _release(self, ch: _Channel) -> None:
        if ch.ok and not self.closed:
            with self._lock:
                self._free.append(ch)
        else:
            ch.conn.close()

    def _control(self, what: str) -> dict[str, Any]:
        ch = self._acquire()
        try:
            ch.conn.send(("control", what))
            out: dict[str, Any] = ch.conn.recv()
            return out
        except BaseException:
            ch.ok = False
            raise
        finally:
            self._release(ch)

    def modules_loaded(self) -> dict[str, Any]:
        """For tests: what the child process has imported (never the openfhe-python wheel)."""
        return self._control("modules")

    def _call(self, method: str, request: Any) -> Any:
        streams_in, streams_out, _, resp_cls = _method(self.pb, method)
        ch = self._acquire()
        try:
            if streams_in:
                ch.conn.send(("call", method, None))
                requests = iter(request)
            else:
                ch.conn.send(("call", method, request.SerializeToString()))
                requests = iter(())
        except BaseException:
            ch.ok = False
            self._release(ch)
            raise
        gen = self._drive(ch, requests, resp_cls)
        if streams_out and method == "Search":
            return gen  # interactive: the caller's next request depends on the last response
        out = list(gen)
        if streams_out:
            return iter(out)
        if len(out) != 1:
            raise ChildError("the local server gave no single response")
        return out[0]

    def _drive(self, ch: _Channel, requests: Iterator[Any], resp_cls: Any) -> Iterator[Any]:
        done = False
        try:
            while True:
                frame = ch.conn.recv_bytes()
                tag, body = frame[:1], frame[1:]
                if tag == b"R":
                    yield resp_cls.FromString(body)
                elif tag == b"N":
                    try:
                        msg = next(requests)
                    except StopIteration:
                        ch.conn.send_bytes(b"E")
                    else:
                        ch.conn.send_bytes(b"D" + msg.SerializeToString())
                elif tag == b"E":
                    done = True
                    return
                elif tag == b"X":
                    raise ChildError(f"the local server failed: {body.decode(errors='replace')}")
                else:
                    raise ChildError("the local server process failed")
        finally:
            if not done:
                ch.ok = False  # abandoned or failed: do not reuse the connection
            self._release(ch)

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._control("shutdown")
        except Exception:  # noqa: S110 - the child may be gone already
            pass
        self.closed = True
        for ch in self._free:
            ch.conn.close()
        self._free.clear()
        self._proc.join(10)
        if self._proc.is_alive():
            self._proc.kill()
        shutil.rmtree(os.path.dirname(self._sock), ignore_errors=True)


# ---- the child side ------------------------------------------------------------------------------
def _child_main(path: str, sock: str, authkey: bytes, parent_pid: int) -> None:
    """Entry point of the child process: serve `LocalServer` on a local socket until told to stop,
    or until the client process is gone."""
    if os.environ.get("OMBRA_LOCAL_THREADS"):  # threads of the server's OpenFHE work (before load)
        os.environ["OMP_NUM_THREADS"] = os.environ["OMBRA_LOCAL_THREADS"]
    if os.environ.get("OMBRA_LOCAL_DEBUG") == "1":
        import faulthandler

        faulthandler.dump_traceback_later(240, repeat=True, file=sys.stderr)
    pb, _ = load_stubs(grpc_module=False)
    server = LocalServer(path, pb, _native().openfhe_version())
    listener = mpc.Listener(sock, family="AF_UNIX", authkey=authkey)
    stop = threading.Event()

    def watch_parent() -> None:
        while not stop.wait(1.0):
            if os.getppid() != parent_pid:
                os._exit(0)

    threading.Thread(target=watch_parent, daemon=True).start()

    def handle(conn: Any) -> None:
        try:
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    return
                if msg[0] == "control":
                    if msg[1] == "hello":
                        conn.send({"openfhe_version": server.version})
                    elif msg[1] == "modules":
                        conn.send(
                            {
                                "openfhe_loaded": "openfhe" in sys.modules,
                                "native_loaded": "ombra._native" in sys.modules,
                            }
                        )
                    else:  # shutdown
                        conn.send({})
                        stop.set()
                        return
                    continue
                _serve_call(server, pb, conn, msg[1], msg[2])
        except (EOFError, OSError):
            return
        finally:
            conn.close()

    def accept_loop() -> None:
        while not stop.is_set():
            try:
                conn = listener.accept()
            except (OSError, EOFError, mpc.AuthenticationError):
                continue
            threading.Thread(target=handle, args=(conn,), daemon=True).start()

    threading.Thread(target=accept_loop, daemon=True).start()
    stop.wait()
    server.close()
    os._exit(0)


def _serve_call(server: LocalServer, pb: Any, conn: Any, method: str, body: bytes | None) -> None:
    streams_in, streams_out, req_cls, _ = _method(pb, method)
    fn = getattr(server, method)

    def requests() -> Iterator[Any]:
        while True:
            conn.send_bytes(b"N")
            frame = conn.recv_bytes()
            if frame[:1] != b"D":
                return
            yield req_cls.FromString(frame[1:])

    try:
        result = fn(requests()) if streams_in else fn(req_cls.FromString(body or b""))
        if streams_out:
            for r in result:
                conn.send_bytes(b"R" + r.SerializeToString())
        else:
            conn.send_bytes(b"R" + result.SerializeToString())
        conn.send_bytes(b"E")
    except (EOFError, ConnectionError):
        raise  # the socket to the client is gone: nothing to report to
    except Exception as exc:
        # Includes OSError of the server's own work (disk full, ...): the client must hear of it,
        # not see a closed connection. Only the exception type name goes back (no payloads, hard
        # rule 4). OMBRA_LOCAL_DEBUG=1 also prints the traceback (for developers).
        if os.environ.get("OMBRA_LOCAL_DEBUG") == "1":
            import traceback

            print("local server error:", type(exc).__name__, file=sys.stderr)
            traceback.print_exc()
        conn.send_bytes(b"X" + type(exc).__name__.encode()[:100])


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class _Coll:
    def __init__(self, spec: Any, root: Path) -> None:
        self.spec, self.root = spec, root
        self.keys: dict[str, bytes] = {}
        self.segments: dict[str, Any] = {}  # name -> InsertBlock (ids + column envelopes)
        self.order: list[str] = []
        self.tombstones: set[bytes] = set()
        self.next_seg = 0
        self.ctx: Any = None  # the ombra._native.Engine once the keys are installed


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
        self.lock = threading.RLock()  # one operation at a time, except between search steps
        self._cols: OrderedDict[tuple[str, str], Any] = OrderedDict()  # -> native Columns
        self._cols_bytes = 0
        self._colls: dict[str, _Coll] = {}
        for d in sorted((self.root / "c").glob("*")) if (self.root / "c").exists() else []:
            spec = pb.CollectionSpec.FromString((d / "spec.pb").read_bytes())
            self._colls[spec.name] = self._load(spec, d)

    def close(self) -> None:
        self._cols.clear()
        self._cols_bytes = 0

    def __repr__(self) -> str:
        return "LocalServer()"

    # ---- storage ----------------------------------------------------------------------------
    def _uncache(self, key: tuple[str, str]) -> None:
        old = self._cols.pop(key, None)
        if old is not None:
            self._cols_bytes -= old.nbytes

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
            c.ctx = self._engine(c, c.keys)
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
            if "OpenFHE version mismatch" in msg:
                raise Refuse(
                    self.pb.ERROR_INVALID_ARGUMENT,
                    f"envelope was made with another OpenFHE version than the server's "
                    f"({OPENFHE_VERSION})",
                ) from None
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
            self._uncache(k)
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
            engine = self._engine(c, staged)
            for k, v in staged.items():
                _atomic(c.root / "keys" / k, v)
            c.keys, c.ctx = staged, engine
        except Refuse as e:
            return pb.UploadKeysResponse(error=self._err(e))
        return pb.UploadKeysResponse()

    def _engine(self, c: _Coll, files: dict[str, bytes]) -> Any:
        """Open the three envelopes and deserialize the keys in C++; the key tag is checked from
        the keys that were really deserialized."""
        native = _native()
        ctx = self._envelope(c, files["context"], Kind.CONTEXT)
        mult = self._envelope(c, files["evalmult"], Kind.PUBLIC_EVAL_KEY)
        auto = self._envelope(c, files["automorphism"], Kind.PUBLIC_EVAL_KEY)
        try:
            return native.Engine(ctx, mult, auto, c.spec.key_tag)
        except native.KeyTagError as ex:
            raise Refuse(self.pb.ERROR_KEY_TAG_MISMATCH, str(ex)) from None
        except ValueError:
            raise Refuse(
                self.pb.ERROR_INVALID_ARGUMENT, "cannot read the context or keys"
            ) from None

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
                if len(b.ids) > c.ctx.batch_size:
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
        self._uncache((c.spec.name, seg))

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
            if len(header.ids) > c.ctx.batch_size if c.ctx else False:
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
    def _columns(self, c: _Coll, seg: str) -> Any:
        key = (c.spec.name, seg)
        cols = self._cols.get(key)
        if cols is not None:
            self._cols.move_to_end(key)
            return cols
        b = c.segments[seg]
        payloads = [self._envelope(c, col, Kind.CIPHERTEXT) for col in b.columns]
        try:
            cols = c.ctx.load_columns(payloads)
        except _native().KeyTagError as ex:
            raise Refuse(self.pb.ERROR_KEY_TAG_MISMATCH, str(ex)) from None
        self._cols[key] = cols
        self._cols_bytes += cols.nbytes
        while self._cols_bytes > CACHE_BYTES and len(self._cols) > 1:
            _, old = self._cols.popitem(last=False)
            self._cols_bytes -= old.nbytes
        return cols

    def _score(self, c: _Coll, seg: str, leaves: Any, tomb: bool) -> bytes:
        b = c.segments[seg]
        penalty: list[float] = []
        if tomb:
            n_fields = len(c.spec.filter_fields)
            pen = [0.0] * c.ctx.batch_size
            for i, ident in enumerate(b.ids):
                if ident in c.tombstones:
                    pen[i] = -LAMBDA * (n_fields + 1)
                    penalty = pen
        res = c.ctx.score(leaves, self._columns(c, seg), penalty)
        return write_envelope(Kind.CIPHERTEXT, c.spec.preset, self.version, res)

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
            with self.lock:
                c = self._coll(st.collection)
                if st.key_tag != c.spec.key_tag:
                    raise Refuse(pb.ERROR_KEY_TAG_MISMATCH, "request key tag differs")
                if c.ctx is None:
                    raise Refuse(pb.ERROR_KEYS_MISSING, "evaluation keys are not uploaded")
                if st.filter_leaves != sum(f.cardinality for f in c.spec.filter_fields):
                    raise Refuse(pb.ERROR_INVALID_FILTER_LAYOUT, "filter_leaves differs")
                if not st.query:
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, "query is empty")
                payload = self._envelope(c, st.query, Kind.CIPHERTEXT)
                d = c.spec.dimension
                try:
                    leaves = c.ctx.expand(payload, d + st.filter_leaves)
                except _native().KeyTagError as ex:
                    raise Refuse(pb.ERROR_KEY_TAG_MISMATCH, str(ex)) from None
                except ValueError as ex:
                    raise Refuse(pb.ERROR_INVALID_ARGUMENT, str(ex)) from None
                center = c.spec.cluster_count
                out = pb.SearchCenterScores()
                for seg in c.order:
                    if c.segments[seg].cluster_id == center:
                        out.score_blocks.append(self._score(c, seg, leaves, False))
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
                    with self.lock:
                        for seg in c.order:
                            b = c.segments[seg]
                            if b.cluster_id == cl:
                                msg.score_blocks.append(self._score(c, seg, leaves, True))
                                msg.ids.extend(b.ids)
                    yield pb.SearchResponse(scores=msg)
        except Refuse as e:
            yield error(e)


def _locked(fn: Any) -> Any:
    """Run a server method under the server lock; streams are produced fully inside the lock."""
    import functools
    import inspect

    if inspect.isgeneratorfunction(fn):

        @functools.wraps(fn)
        def gen(self: LocalServer, *a: Any) -> Iterator[Any]:
            with self.lock:
                items = list(fn(self, *a))
            return iter(items)

        return gen

    @functools.wraps(fn)
    def run(self: LocalServer, *a: Any) -> Any:
        with self.lock:
            return fn(self, *a)

    return run


for _name in (
    "CreateCollection GetCollection ListCollections DropCollection UploadKeys Insert Delete "
    "ListBlocks GetBlock ReplaceBlock PutBlob GetBlob DeleteBlobs"
).split():
    setattr(LocalServer, _name, _locked(getattr(LocalServer, _name)))
