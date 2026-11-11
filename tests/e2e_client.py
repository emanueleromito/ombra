"""Small test client for the Ombra gRPC server (T2.7b): the client side of the protocol.

Holds the secret key (it never leaves this process; no message of the API has a field for it),
creates keys, encrypts vectors and queries with OpenFHE, talks to `ombra_server` over gRPC, and
decrypts and decodes the answers: the band decode of decision 0019 (slot = score + 4 * c, c =
number of matching filter fields, a deleted slot decodes to c < 0) and the top-k.

The gRPC stubs are generated from proto/ombra/v1/ombra.proto with grpc_tools into a temp directory
at test time. The copy lives under the Python package name `ombra_api` so it cannot collide with
the client package `ombra`; the wire names (package ombra.v1) are unchanged.

Synthetic data only. Not part of the shipped client.
"""

from __future__ import annotations

import importlib
import queue
import select
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ombra import params as ombra_params
from ombra import query as oq
from ombra.file_format import Kind, read_envelope, write_envelope

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTO = REPO_ROOT / "proto" / "ombra" / "v1" / "ombra.proto"
LAMBDA = 4.0
MAX_MESSAGE = 1 << 30
KEY_CHUNK = 4_000_000  # below LIMIT_MAX_CHUNK_BYTES (4 MiB)


class ApiError(Exception):
    """The server answered with an ombra.v1.Error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"error code {code}: {message}")
        self.code = code
        self.message = message


def load_stubs(out_dir: Path) -> tuple[Any, Any]:
    """Generate and import the Python stubs; returns (messages module, grpc module)."""
    from grpc_tools import protoc

    api = out_dir / "ombra_api" / "v1"
    api.mkdir(parents=True)
    shutil.copy(PROTO, api / "ombra.proto")
    code = protoc.main(
        [
            "protoc",
            f"-I{out_dir}",
            f"--python_out={out_dir}",
            f"--grpc_python_out={out_dir}",
            str(api / "ombra.proto"),
        ]
    )
    assert code == 0, "protoc failed"
    sys.path.insert(0, str(out_dir))
    pb = importlib.import_module("ombra_api.v1.ombra_pb2")
    pbg = importlib.import_module("ombra_api.v1.ombra_pb2_grpc")
    return pb, pbg


def find_server_binary() -> Path | None:
    import os

    for c in (
        os.environ.get("OMBRA_SERVER_BIN", ""),
        "/root/ombra-build/server/grpc/ombra_server",
        str(REPO_ROOT / "build" / "server" / "grpc" / "ombra_server"),
    ):
        if c and Path(c).is_file():
            return Path(c)
    return None


class ServerProcess:
    """`ombra_server` on 127.0.0.1 with a free port (the server prints the bound port)."""

    def __init__(
        self,
        binary: Path,
        data_dir: Path,
        version: str,
        log_path: Path,
        gpu_budget_mb: int = 1024,
        host_budget_mb: int = 512,
        extra_args: tuple[str, ...] = (),
        secure: bool = False,
    ) -> None:
        # Tests that do not exercise T2.9 run without TLS and keys; `secure=True` passes none of
        # the two flags, so extra_args must bring --tls-cert/--tls-key (and keys must exist).
        insecure_args = () if secure else ("--insecure", "--no-auth")
        self._log = open(log_path, "w")  # noqa: SIM115
        self.proc = subprocess.Popen(  # noqa: S603
            [
                str(binary),
                "--data-dir",
                str(data_dir),
                "--listen",
                "127.0.0.1:0",
                "--openfhe-version",
                version,
                "--gpu-budget-mb",
                str(gpu_budget_mb),
                "--host-budget-mb",
                str(host_budget_mb),
                *insecure_args,
                *extra_args,
            ],
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
        )
        self.log_path = log_path
        deadline = time.monotonic() + 180
        line = ""
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                break
            ready, _, _ = select.select([self.proc.stdout], [], [], 1.0)
            if ready:
                line = self.proc.stdout.readline()
                break
        if not line.startswith("ombra_server listening on "):
            tail = self.log_tail()  # before stop(), which closes the log
            self.stop()
            raise RuntimeError(f"server did not start; log tail:\n{tail}")
        self.address = line.strip().split()[-1]
        self.rest_address = ""
        if "--rest-port" in extra_args:
            ready, _, _ = select.select([self.proc.stdout], [], [], 30.0)
            rest_line = self.proc.stdout.readline() if ready else ""
            if not rest_line.startswith("ombra_server rest listening on "):
                self.stop()
                raise RuntimeError(f"REST port did not start; log tail:\n{self.log_tail()}")
            self.rest_address = rest_line.strip().split()[-1]

    def log_tail(self, n: int = 30) -> str:
        self._log.flush()
        return "\n".join(self.log_path.read_text().splitlines()[-n:])

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._log.close()


@dataclass
class SearchResult:
    ids: list[bytes]
    scores: list[float]
    rounds: int
    incomplete: bool
    scanned: list[int]
    first_round_matches: int
    candidates: dict[bytes, tuple[float, int]] = field(default_factory=dict)  # id -> (score, c)


class SearchStream:
    """One Search stream: the requests come from a queue so rounds can follow each other."""

    def __init__(self, stub: Any, first: Any) -> None:
        self._q: queue.Queue = queue.Queue()
        self._q.put(first)
        self._call = stub.Search(self._requests())

    def _requests(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item

    def send(self, request: Any) -> None:
        self._q.put(request)

    def next(self) -> Any:
        try:
            resp = next(self._call)
        except StopIteration:
            raise ApiError(0, "stream ended") from None
        if resp.WhichOneof("step") == "error":
            raise ApiError(resp.error.code, resp.error.message)
        return resp

    def close(self) -> None:
        self._q.put(None)  # half-close
        for _ in self._call:
            pass

    def abort(self) -> None:
        """After an error message: stop the request generator (the server ended the stream)."""
        self._q.put(None)


class OmbraClient:
    def __init__(
        self,
        pb: Any,
        pbg: Any,
        address: str,
        preset: str,
        version: str,
        root_cert: bytes | None = None,
        api_token: str | None = None,
    ) -> None:
        """`root_cert` switches to TLS; `api_token` is sent as `authorization: Bearer <token>`.

        The API token is an access token for the server, unrelated to the FHE keys.
        """
        import grpc

        self.pb, self.version, self.preset = pb, version, preset
        options = [
            ("grpc.max_send_message_length", MAX_MESSAGE),
            ("grpc.max_receive_message_length", MAX_MESSAGE),
        ]
        if root_cert is None:
            self.channel = grpc.insecure_channel(address, options=options)
        else:
            creds = grpc.ssl_channel_credentials(root_certificates=root_cert)
            if api_token is not None:
                creds = grpc.composite_channel_credentials(
                    creds, grpc.access_token_call_credentials(api_token)
                )
            self.channel = grpc.secure_channel(address, creds, options=options)
        self.stub = pbg.OmbraStub(self.channel)
        self.slots = ombra_params.get_preset(preset)["batch_size"]
        self.cc = None
        self.keys = None
        self.other = None
        self.key_tag = ""

    # ---- keys ---------------------------------------------------------------------------------
    def make_keys(self, n_leaves: int) -> None:
        """Key pair, EvalMult key and the automorphism keys for D = n_leaves (secret stays here)."""
        _, cc = ombra_params.make_context(self.preset)
        self.cc = cc
        self.keys = cc.KeyGen()
        cc.EvalMultKeyGen(self.keys.secretKey)
        cc.ClearEvalAutomorphismKeys()  # the key store is static: hold exactly this D's indices
        cc.EvalAutomorphismKeyGen(
            self.keys.secretKey, oq.automorphism_indices(cc.GetRingDimension(), n_leaves)
        )
        self.other = cc.KeyGen()  # a second key pair, for the wrong-key refusals
        self.key_tag = self.keys.publicKey.GetKeyTag()
        assert self.other.publicKey.GetKeyTag() != self.key_tag

    def _via_file(self, write) -> bytes:
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.bin"
            assert write(str(p))
            return p.read_bytes()

    def key_files(self) -> dict[int, bytes]:
        import openfhe as ofhe

        pb = self.pb
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
        return {
            k: write_envelope(kind, self.preset, self.version, b) for k, (kind, b) in raw.items()
        }

    def upload_keys(self, collection: str, files: dict[int, bytes], api_version: int = 1) -> Any:
        pb = self.pb

        def chunks():
            first = True
            for kind, data in files.items():
                for off in range(0, len(data), KEY_CHUNK):
                    c = pb.UploadKeysChunk(
                        api_version=api_version,
                        kind=kind,
                        offset=off,
                        data=data[off : off + KEY_CHUNK],
                        last=off + KEY_CHUNK >= len(data),
                    )
                    if first:
                        c.collection = collection
                        first = False
                    yield c

        return self.stub.UploadKeys(chunks())

    # ---- crypto helpers -----------------------------------------------------------------------
    def encrypt(self, values: np.ndarray, public_key: Any = None) -> Any:
        pt = self.cc.MakeCKKSPackedPlaintext([float(v) for v in values])
        return self.cc.Encrypt(public_key or self.keys.publicKey, pt)

    def ct_envelope(self, ct: Any) -> bytes:
        import openfhe as ofhe

        raw = self._via_file(lambda p: ofhe.SerializeToFile(p, ct, ofhe.BINARY))
        return write_envelope(Kind.CIPHERTEXT, self.preset, self.version, raw)

    def encrypt_query(self, ext: np.ndarray, public_key: Any = None) -> bytes:
        ring = self.cc.GetRingDimension()
        slots_v = oq.symmetric_slots(oq.prescale_query(ext), ring)
        pt = self.cc.MakeCKKSPackedPlaintext(slots_v.tolist())
        ct = self.cc.Encrypt(public_key or self.keys.publicKey, pt)
        return self.ct_envelope(ct)

    def decrypt_ct(self, ct: Any) -> np.ndarray:
        dec = self.cc.Decrypt(ct, self.keys.secretKey)
        dec.SetLength(self.slots)
        return np.array(dec.GetRealPackedValue(), dtype=np.float64)

    def decrypt_envelope(self, raw: bytes) -> np.ndarray:
        import openfhe as ofhe

        env = read_envelope(raw, Kind.CIPHERTEXT, self.preset, self.version)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "ct.bin"
            p.write_bytes(env.payload)
            ct, ok = ofhe.DeserializeCiphertext(str(p), ofhe.BINARY)
        assert ok
        return self.decrypt_ct(ct)

    def oracle_scores(self, cols: list[Any], leaves: np.ndarray) -> np.ndarray:
        """CPU OpenFHE oracle: sum_j leaf_j * column_j on the client's own ciphertext copies."""
        acc = None
        for ct, w in zip(cols, leaves, strict=True):
            term = self.cc.EvalMult(ct, self.cc.MakeCKKSPackedPlaintext([float(w)] * self.slots))
            acc = term if acc is None else self.cc.EvalAdd(acc, term)
        return self.decrypt_ct(self.cc.Rescale(acc))

    # ---- protocol -----------------------------------------------------------------------------
    @staticmethod
    def check(err: Any) -> None:
        if err.code != 0:
            raise ApiError(err.code, err.message)

    def make_spec(
        self,
        name: str,
        d: int,
        cards: list[int],
        clusters: int,
        key_tag: str | None = None,
        api_version: int = 1,
    ) -> Any:
        pb = self.pb
        spec = pb.CollectionSpec(
            api_version=api_version,
            name=name,
            preset=self.preset,
            dimension=d,
            security_level=pb.SECURITY_LEVEL_FAST,
            key_tag=self.key_tag if key_tag is None else key_tag,
            cluster_count=clusters,
        )
        for i, c in enumerate(cards):
            spec.filter_fields.add(field_id=i, cardinality=c)
        return spec

    def create_collection(self, spec: Any) -> Any:
        r = self.stub.CreateCollection(self.pb.CreateCollectionRequest(spec=spec))
        if r.WhichOneof("result") == "error":
            raise ApiError(r.error.code, r.error.message)
        return r.spec

    def get_collection(self, name: str) -> Any:
        r = self.stub.GetCollection(self.pb.GetCollectionRequest(name=name))
        if r.WhichOneof("result") == "error":
            raise ApiError(r.error.code, r.error.message)
        return r.info

    def drop_collection(self, name: str) -> None:
        self.check(self.stub.DropCollection(self.pb.DropCollectionRequest(name=name)).error)

    def insert(self, blocks: list[Any]) -> int:
        r = self.stub.Insert(iter(blocks))
        self.check(r.error)
        return r.inserted

    def delete(self, collection: str, ids: list[bytes], api_version: int = 1) -> int:
        r = self.stub.Delete(
            self.pb.DeleteRequest(api_version=api_version, collection=collection, ids=ids)
        )
        self.check(r.error)
        return r.deleted

    # ---- blocks (T2.8) ------------------------------------------------------------------------
    def list_blocks(self, collection: str) -> list[Any]:
        r = self.stub.ListBlocks(self.pb.ListBlocksRequest(api_version=1, collection=collection))
        self.check(r.error)
        return list(r.blocks)

    def get_block(self, collection: str, segment: str) -> tuple[Any, list[bytes]]:
        """(header, column envelopes) of one block."""
        stream = self.stub.GetBlock(
            self.pb.GetBlockRequest(api_version=1, collection=collection, segment=segment)
        )
        header, cols = None, []
        for m in stream:
            part = m.WhichOneof("part")
            if part == "error":
                raise ApiError(m.error.code, m.error.message)
            if part == "header":
                header = m.header
            else:
                cols.append(bytes(m.column))
        assert header is not None and len(cols) == header.n_columns
        return header, cols

    def replace_block(
        self, collection: str, segments: list[str] | str, ids: list[bytes], columns: list[bytes]
    ) -> str:
        """Replace 1..N blocks of one cluster by one block (the client has merged them)."""
        segs = [segments] if isinstance(segments, str) else list(segments)
        pb = self.pb

        def msgs():
            yield pb.ReplaceBlockRequest(
                api_version=1,
                header=pb.ReplaceBlockHeader(collection=collection, segments=segs, ids=ids),
            )
            for c in columns:
                yield pb.ReplaceBlockRequest(api_version=1, column=c)

        r = self.stub.ReplaceBlock(msgs())
        self.check(r.error)
        return r.segment

    def open_search(
        self, collection: str, query: bytes, filter_leaves: int, key_tag: str | None = None
    ) -> tuple[SearchStream, Any]:
        pb = self.pb
        start = pb.SearchStart(
            collection=collection,
            key_tag=self.key_tag if key_tag is None else key_tag,
            query=query,
            filter_leaves=filter_leaves,
        )
        stream = SearchStream(self.stub, pb.SearchRequest(start=start))
        try:
            return stream, stream.next()
        except ApiError:
            stream.abort()
            raise

    @staticmethod
    def ext_query(q: np.ndarray, cards: list[int], wants: list[int | None]) -> np.ndarray:
        """Query leaves: q, then per field lambda * one-hot (all lambda if the field is free)."""
        parts = [q]
        for card, want in zip(cards, wants, strict=True):
            oh = np.full(card, LAMBDA) if want is None else np.zeros(card)
            if want is not None:
                oh[want] = LAMBDA
            parts.append(oh)
        return np.concatenate(parts)

    @staticmethod
    def decode_band(slots_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(c, score) per slot: c = floor((slot + 2) / lambda), score = slot - lambda * c."""
        c = np.floor((slots_v + 2.0) / LAMBDA).astype(int)
        return c, slots_v - LAMBDA * c

    def search(
        self,
        collection: str,
        q: np.ndarray,
        cards: list[int],
        wants: list[int | None],
        k: int,
        order: list[int] | None = None,
        batch: int | None = None,
    ) -> SearchResult:
        """Full client flow: stage 1, then SearchChosen rounds until k matches.

        A vector matches the filter iff c == number of fields (a free field counts as matching).
        `order` is the cluster order to scan (default: by decrypted center score, best first).
        """
        n_fields = len(cards)
        ext = self.ext_query(q, cards, wants)
        stream, centers = self.open_search(collection, self.encrypt_query(ext), sum(cards))
        try:
            cs = centers.centers
            assert len(cs.score_blocks) == 1  # the test collections have fewer centers than slots
            center_scores = self.decrypt_envelope(cs.score_blocks[0])[: cs.center_count]
            if order is None:
                order = [int(i) for i in np.argsort(-center_scores, kind="stable")]
            step = batch or len(order)
            cand: dict[bytes, tuple[float, int]] = {}
            scanned: list[int] = []
            rounds = 0
            first_round_matches = -1
            pos = 0
            while pos < len(order):
                chosen = order[pos : pos + step]
                pos += step
                stream.send(self.pb.SearchRequest(chosen=self.pb.SearchChosen(cluster_ids=chosen)))
                got = 0
                while True:
                    msg = stream.next().scores
                    assert msg.cluster_id in chosen
                    scanned.append(msg.cluster_id)
                    for i, bl in enumerate(msg.score_blocks):
                        c, sc = self.decode_band(self.decrypt_envelope(bl))
                        ids = msg.ids[i * self.slots : (i + 1) * self.slots]
                        for j, ident in enumerate(ids):
                            if c[j] == n_fields:  # live (c >= 0) and every field matches
                                cand[bytes(ident)] = (float(sc[j]), int(c[j]))
                    got += 1
                    if msg.last:
                        break
                assert got == len(chosen)
                rounds += 1
                if first_round_matches < 0:
                    first_round_matches = len(cand)
                if len(cand) >= k:
                    break
        finally:
            stream.close()
        best = sorted(cand.items(), key=lambda kv: (-kv[1][0], kv[0]))[:k]
        return SearchResult(
            ids=[i for i, _ in best],
            scores=[v[0] for _, v in best],
            rounds=rounds,
            incomplete=len(best) < k,
            scanned=scanned,
            first_round_matches=first_round_matches,
            candidates=cand,
        )

    def close(self) -> None:
        self.channel.close()
