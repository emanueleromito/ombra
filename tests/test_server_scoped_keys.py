# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Collection-scoped API keys, the api_keys.v1 -> v2 migration, request ids and structured logs of
the CPU-only `ombra_server`.

One collection is built once (client-side encryption is slow); every test copies its data
directory and starts its own server with API keys. Calls use the raw gRPC stub with the key in the
`authorization` metadata (the Python client sends a key only over TLS). Synthetic data only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time

import pytest

np = pytest.importorskip("numpy")
grpc = pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")
pytest.importorskip("openfhe")

from server_harness import ServerProcess, find_server_binary  # noqa: E402

pytestmark = pytest.mark.openfhe

if find_server_binary() is None:
    pytest.skip("ombra_server (CPU build) not found: run `make build-cpu`", allow_module_level=True)

from ombra._proto import insert_requests, load_stubs  # noqa: E402

D = 384
A = "collsecret"  # collection names must never appear in logs
B = "betasecret"
ID_PREFIX = "idsecret-"  # vector ids must never appear in logs
N_BLOCK = 3
pb, pbg = load_stubs()
PERMISSION_DENIED = grpc.StatusCode.PERMISSION_DENIED


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """(data root with collections A (keys uploaded, 8 vectors) and B (empty), columns of a block)."""
    from ombra import Client

    root = tmp_path_factory.mktemp("base")
    rng = np.random.default_rng(5)
    x = rng.normal(size=(8, D))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    server = ServerProcess(root)
    with server:
        client = Client(server.address, insecure=True)
        client.create_collection(
            A,
            x,
            [f"{ID_PREFIX}{i}" for i in range(8)],
            key_file=root / "k.key",
            passphrase="pw",  # noqa: S106
            preset="fast",
            n_clusters=1,
            scrypt_params=(2**4, 8, 1),
        )
        col = client._core._get(A)[0]
        cols = col._columns(x[:N_BLOCK], np.zeros((N_BLOCK, 0)))
        client.close()
        ch = grpc.insecure_channel(server.address)
        stub = pbg.OmbraStub(ch)
        spec = pb.CollectionSpec()
        spec.CopyFrom(stub.GetCollection(pb.GetCollectionRequest(name=A)).info.spec)
        spec.name = B
        assert not stub.CreateCollection(pb.CreateCollectionRequest(spec=spec)).error.code
        ch.close()
    return root, cols


class Env:
    def __init__(self, tmp_path, root, cols, extra=()):
        shutil.copytree(root, tmp_path / "r")
        self.cols = cols
        self.server = ServerProcess(tmp_path / "r", auth=True, extra_args=list(extra))
        self.channel = None
        self.stub = None

    def start(self):
        self.server.start()
        self.channel = grpc.insecure_channel(self.server.address)
        self.stub = pbg.OmbraStub(self.channel)

    def close(self):
        if self.channel is not None:
            self.channel.close()
        self.server.kill()

    def key(self, scope, collection=None):
        return self.server.create_key(scope, collection)

    @property
    def data(self):
        return self.server.data_dir


@pytest.fixture
def env(base, tmp_path):
    root, cols = base
    made = []

    def make(*extra):
        e = Env(tmp_path, root, cols, extra)
        made.append(e)
        return e

    yield make
    for e in made:
        e.close()


def md(token, rid=None):
    m = [("authorization", f"Bearer {token}")]
    if rid:
        m.append(("x-request-id", rid))
    return tuple(m)


def denied(fn, *args, **kw):
    """The call must end with PERMISSION_DENIED."""
    with pytest.raises(grpc.RpcError) as ei:
        fn(*args, **kw)
    assert ei.value.code() == PERMISSION_DENIED, ei.value.code()


def n_blocks(env, token, name) -> int:
    r = env.stub.ListBlocks(
        pb.ListBlocksRequest(api_version=1, collection=name), metadata=md(token), timeout=10
    )
    assert not r.error.code
    return len(r.blocks)


def insert_stream(cols, name, ids):
    block = pb.InsertBlock(
        api_version=1, collection=name, cluster_id=0, ids=ids, columns=cols, is_center=False
    )
    return insert_requests(pb, [block])


# ---- a v1 file keeps working ------------------------------------------------------------------
def test_v1_file_works_unchanged(env):
    e = env()
    admin, write, read = e.key("admin"), e.key("write"), e.key("read")
    e.start()
    assert (e.data / "api_keys.v1").is_file() and not (e.data / "api_keys.v2").exists()
    assert (e.data / "api_keys.v1").read_text().startswith("OMBRA-API-KEYS v1\n")
    names = lambda r: sorted(c.spec.name for c in r.collections)  # noqa: E731
    for tok in (admin, write, read):  # unscoped: every collection is visible
        assert names(e.stub.ListCollections(pb.ListCollectionsRequest(), metadata=md(tok))) == [
            B,
            A,
        ]
    denied(  # read scope cannot write
        e.stub.Insert, insert_stream(e.cols, A, [b"x-1"]), metadata=md(read), timeout=20
    )
    r = e.stub.Insert(insert_stream(e.cols, A, [b"x-1"]), metadata=md(write), timeout=60)
    assert not r.error.code and r.inserted == 1
    r = e.stub.Insert(insert_stream(e.cols, B, [b"x-2"]), metadata=md(write), timeout=60)
    assert r.error.code == 9  # B has no keys: it was reachable, not refused by scope
    assert not (e.data / "api_keys.v2").exists()  # unscoped keys never trigger the migration


def test_admin_key_cannot_be_scoped(env):
    e = env()
    e.key("admin")
    before = (e.data / "api_keys.v1").read_text()
    out = subprocess.run(  # noqa: S603
        [str(e.server.binary), "--create-api-key", "--scope", "admin", "--collection", A]
        + ["--data-dir", str(e.data)],
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0 and out.stdout == ""
    assert (e.data / "api_keys.v1").read_text() == before and not (e.data / "api_keys.v2").exists()


# ---- scoped keys ------------------------------------------------------------------------------
def test_scoped_keys(env):
    e = env()
    admin = e.key("admin")  # v1 entry, migrated by the first scoped key
    read_a = e.key("read", A)
    write_a = e.key("write", A)
    assert (e.data / "api_keys.v2").read_text().startswith("OMBRA-API-KEYS v2\n")
    assert not (e.data / "api_keys.v1").exists()
    assert (e.data / "api_keys.v1.migrated").exists()
    assert oct((e.data / "api_keys.v2").stat().st_mode & 0o777) == "0o600"
    e.start()
    s = e.stub

    # The migrated v1 admin key is still unscoped.
    assert n_blocks(e, admin, B) == 0 and n_blocks(e, admin, A) >= 1

    # Read key scoped to A: reads A, nothing else.
    assert s.GetCollection(pb.GetCollectionRequest(name=A), metadata=md(read_a)).info.spec.name == A
    for other in (B, "no_such_collection"):  # same answer whether it exists or not
        denied(s.GetCollection, pb.GetCollectionRequest(name=other), metadata=md(read_a))
        denied(
            s.ListBlocks,
            pb.ListBlocksRequest(api_version=1, collection=other),
            metadata=md(read_a),
        )
    lst = s.ListCollections(pb.ListCollectionsRequest(), metadata=md(read_a))
    assert [c.spec.name for c in lst.collections] == [A]
    lst = s.ListCollections(pb.ListCollectionsRequest(), metadata=md(admin))
    assert sorted(c.spec.name for c in lst.collections) == [B, A]
    denied(s.CreateCollection, pb.CreateCollectionRequest(spec=pb.CollectionSpec(name=B)), metadata=md(read_a))
    denied(  # the scope is read: no insert even into A
        s.Insert, insert_stream(e.cols, A, [b"r-1"]), metadata=md(read_a), timeout=20
    )

    # Write key scoped to A.
    spec = s.GetCollection(pb.GetCollectionRequest(name=A), metadata=md(write_a)).info.spec
    again = s.CreateCollection(pb.CreateCollectionRequest(spec=spec), metadata=md(write_a))
    assert again.error.code == 3  # the name is allowed (and exists), not refused
    newspec = pb.CollectionSpec()
    newspec.CopyFrom(spec)
    newspec.name = "other_collection"
    denied(s.CreateCollection, pb.CreateCollectionRequest(spec=newspec), metadata=md(write_a))
    assert "other_collection" not in [
        c.spec.name for c in s.ListCollections(pb.ListCollectionsRequest(), metadata=md(admin)).collections
    ]

    n_a, n_b = n_blocks(e, admin, A), n_blocks(e, admin, B)
    r = s.Insert(insert_stream(e.cols, A, [b"w-1"]), metadata=md(write_a), timeout=60)
    assert not r.error.code and r.inserted == 1
    assert n_blocks(e, admin, A) == n_a + 1

    # Streams naming B in their first message are refused before anything is stored or parsed.
    denied(s.Insert, insert_stream(e.cols, B, [b"w-2"]), metadata=md(write_a), timeout=20)
    assert n_blocks(e, admin, B) == n_b

    def chunks():
        yield pb.UploadKeysChunk(
            api_version=1, collection=B, kind=1, offset=0, data=b"not a key", last=True
        )

    denied(s.UploadKeys, chunks(), metadata=md(write_a), timeout=20)

    def blob_msgs(name):
        yield pb.PutBlobRequest(
            api_version=1,
            header=pb.PutBlobHeader(collection=name, kind=1, id=b"blob-1", size=16384),
        )
        yield pb.PutBlobRequest(api_version=1, data=bytes(16384))

    denied(s.PutBlob, blob_msgs(B), metadata=md(write_a), timeout=20)
    assert not s.PutBlob(blob_msgs(A), metadata=md(write_a), timeout=20).error.code
    gb = list(s.GetBlob(pb.GetBlobRequest(api_version=1, collection=B, kind=1, id=b"blob-1"), metadata=md(admin)))
    assert gb[0].error.code == 14  # not stored in B

    def replace(name):
        yield pb.ReplaceBlockRequest(
            api_version=1,
            header=pb.ReplaceBlockHeader(collection=name, segments=["x"], ids=[b"y"]),
        )

    denied(s.ReplaceBlock, replace(B), metadata=md(write_a), timeout=20)

    def search(name):
        yield pb.SearchRequest(
            start=pb.SearchStart(collection=name, key_tag="t", query=b"q", filter_leaves=0)
        )

    denied(lambda: list(s.Search(search(B), metadata=md(read_a), timeout=20)))
    denied(s.Delete, pb.DeleteRequest(api_version=1, collection=B, ids=[b"i"]), metadata=md(write_a))
    denied(
        s.DeleteBlobs,
        pb.DeleteBlobsRequest(api_version=1, collection=B, kind=1, ids=[b"i"]),
        metadata=md(write_a),
    )
    denied(
        lambda: list(
            s.GetBlock(
                pb.GetBlockRequest(api_version=1, collection=B, segment="s"), metadata=md(read_a)
            )
        )
    )
    denied(
        lambda: list(
            s.GetBlob(
                pb.GetBlobRequest(api_version=1, collection=B, kind=1, id=b"i"), metadata=md(read_a)
            )
        )
    )
    # Search on its own collection is not refused by scope (the CPU build then reports no GPU).
    try:
        list(s.Search(search(A), metadata=md(read_a), timeout=20))
    except grpc.RpcError as exc:
        assert exc.code() != PERMISSION_DENIED

    # Drop needs admin (scoped keys are never admin); an unscoped admin may drop.
    denied(s.DropCollection, pb.DropCollectionRequest(name=A), metadata=md(write_a))
    denied(s.DropCollection, pb.DropCollectionRequest(name=B), metadata=md(write_a))
    assert [c.spec.name for c in s.ListCollections(pb.ListCollectionsRequest(), metadata=md(admin)).collections].count(A) == 1


def test_sighup_reload_with_a_new_scoped_key(env):
    e = env()
    admin = e.key("admin")
    e.start()
    assert (e.data / "api_keys.v1").exists()
    scoped = e.key("read", B)  # migrates the file while the server runs
    denied_auth = lambda: e.stub.GetCollection(  # noqa: E731
        pb.GetCollectionRequest(name=B), metadata=md(scoped), timeout=10
    )
    with pytest.raises(grpc.RpcError) as ei:  # not loaded yet
        denied_auth()
    assert ei.value.code() == grpc.StatusCode.UNAUTHENTICATED
    e.server._proc.send_signal(signal.SIGHUP)
    deadline = time.monotonic() + 10
    while True:
        try:
            assert denied_auth().info.spec.name == B
            break
        except grpc.RpcError:
            assert time.monotonic() < deadline
            time.sleep(0.1)
    denied(e.stub.GetCollection, pb.GetCollectionRequest(name=A), metadata=md(scoped), timeout=10)
    assert n_blocks(e, admin, A) >= 1  # the old key still works after the migration


# ---- request ids and logs ---------------------------------------------------------------------
def _lines(server):
    text = server.log_path.read_text(errors="replace")
    return text, text.splitlines()


def test_request_ids_and_json_logs_carry_no_data(env):
    e = env("--log-format", "json")
    admin, write_a, read_a = e.key("admin"), e.key("write", A), e.key("read", A)
    e.start()
    s = e.stub

    # The server returns the client's id (when valid) in the initial metadata.
    def rid_of(call):
        return dict(call.initial_metadata())["x-request-id"]

    _, c1 = s.GetCollection.with_call(
        pb.GetCollectionRequest(name=A), metadata=md(admin, "client-id.1_A"), timeout=10
    )
    assert rid_of(c1) == "client-id.1_A"
    # An invalid id (space, too long) is replaced by a random one.
    for bad in ("bad id", "x" * 65, "semi;colon"):
        try:
            _, c = s.GetCollection.with_call(
                pb.GetCollectionRequest(name=A), metadata=md(admin, bad), timeout=10
            )
        except grpc.RpcError:  # gRPC itself may refuse some values
            continue
        assert rid_of(c) != bad and re.fullmatch(r"[0-9a-f]{16}", rid_of(c))
    _, c2 = s.GetCollection.with_call(pb.GetCollectionRequest(name=A), metadata=md(admin), timeout=10)
    assert re.fullmatch(r"[0-9a-f]{16}", rid_of(c2))

    # A mix of calls with their own ids: success, refusals (scope, collection), bad key.
    r = s.Insert(
        insert_stream(e.cols, A, [(ID_PREFIX + "new").encode()]),
        metadata=md(write_a, "req-insert-1"),
        timeout=60,
    )
    assert r.inserted == 1
    denied(s.Insert, insert_stream(e.cols, A, [b"z"]), metadata=md(read_a, "req-insert-2"))
    denied(s.GetCollection, pb.GetCollectionRequest(name=B), metadata=md(read_a, "req-coll-1"))
    with pytest.raises(grpc.RpcError):
        s.GetCollection(
            pb.GetCollectionRequest(name=A), metadata=md("wrongtoken", "req-bad-1"), timeout=10
        )
    s.ListBlocks(
        pb.ListBlocksRequest(api_version=1, collection=A), metadata=md(admin, "req-list-1")
    )
    e.server.stop()

    text, lines = _lines(e.server)
    # No data: no token, no collection name, no id.
    for forbidden in (admin, write_a, read_a, "wrongtoken", A, B, ID_PREFIX, "Bearer"):
        assert forbidden not in text, forbidden
    by_id: dict[str, list[dict]] = {}
    for line in lines:
        if not line.startswith("{"):
            continue  # lines of the storage and search libraries are plain text
        rec = json.loads(line)
        assert {"ts", "level", "msg"} <= set(rec)
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", rec["ts"])
        assert rec["level"] in {"info", "warn", "error"}
        if "request_id" in rec:
            by_id.setdefault(rec["request_id"], []).append(rec)
    # Each call's lines share its id, its rpc name and its key id (8 hex).
    for rid, rpc in [
        ("req-insert-1", "Insert"),
        ("req-insert-2", "Insert"),
        ("req-coll-1", "GetCollection"),
        ("req-list-1", "ListBlocks"),
    ]:
        recs = by_id[rid]
        assert {r["rpc"] for r in recs} == {rpc}
        assert all(re.fullmatch(r"[0-9a-f]{8}", r["key_id"]) for r in recs)
        assert len({r["key_id"] for r in recs}) == 1
    assert {r["msg"] for r in by_id["req-insert-1"]} >= {"insert"}
    assert [r["reason"] for r in by_id["req-insert-2"] if r["msg"] == "refused"] == ["scope"]
    assert [r["reason"] for r in by_id["req-coll-1"] if r["msg"] == "refused"] == ["collection"]
    bad = by_id["req-bad-1"]  # unauthenticated: the id is there, the key id is not
    assert all("key_id" not in r for r in bad)
    # Distinct calls have distinct ids; no call's lines mix rpc names.
    for rid, recs in by_id.items():
        assert len({r.get("rpc") for r in recs if "rpc" in r}) <= 1, rid


def test_text_logs_are_the_default_and_carry_the_request_id(env):
    e = env()
    admin = e.key("admin")
    e.start()
    e.stub.ListBlocks(
        pb.ListBlocksRequest(api_version=1, collection=A), metadata=md(admin, "req-text-1")
    )
    with pytest.raises(grpc.RpcError):
        e.stub.GetCollection(pb.GetCollectionRequest(name=A), metadata=md("nope"), timeout=10)
    e.server.stop()
    text, lines = _lines(e.server)
    assert not any(line.startswith("{") for line in lines)
    assert any(line.startswith("[ombra.server] ") for line in lines)
    assert admin not in text and A not in text and ID_PREFIX not in text


def test_invalid_log_format_is_refused(env):
    e = env()
    e.key("admin")
    out = subprocess.run(  # noqa: S603
        [str(e.server.binary), "--data-dir", str(e.data), "--listen", "127.0.0.1:0"]
        + ["--insecure", "--log-format", "xml"],
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ},
    )
    assert out.returncode == 2 and "--log-format" in out.stderr
