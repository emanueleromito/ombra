# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Shutdown, idempotent Insert and message limits of the CPU-only `ombra_server`.

One collection is built once (client-side encryption is slow); every test copies its data
directory, starts its own server and talks to it with the raw gRPC stub. Synthetic data only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

import pytest

np = pytest.importorskip("numpy")
grpc = pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")
pytest.importorskip("openfhe")

from server_harness import REPO_ROOT, ServerProcess, find_server_binary  # noqa: E402

pytestmark = pytest.mark.openfhe

if find_server_binary() is None:
    pytest.skip("ombra_server (CPU build) not found: run `make build-cpu`", allow_module_level=True)

from ombra._proto import insert_requests, load_stubs  # noqa: E402

D = 384
NAME = "shut"
N_BLOCK = 3
VERIFY = REPO_ROOT / "build-cpu" / "server" / "storage" / "ombra_verify"
pb, pbg = load_stubs()


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """(data root with a collection and keys, the columns of one encrypted block)."""
    from ombra import Client

    root = tmp_path_factory.mktemp("base")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(8, D))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    server = ServerProcess(root)
    with server:
        client = Client(server.address, insecure=True)
        client.create_collection(
            NAME,
            x,
            [f"b{i}" for i in range(8)],
            key_file=root / "k.key",
            passphrase="pw",  # noqa: S106
            preset="fast",
            n_clusters=1,
            scrypt_params=(2**4, 8, 1),
        )
        col = client._core._get(NAME)[0]
        cols = col._columns(x[:N_BLOCK], np.zeros((N_BLOCK, 0)))
        client.close()
    return root, cols


@pytest.fixture
def srv(base, tmp_path):
    root, cols = base
    shutil.copytree(root, tmp_path / "r")
    holder = {}

    def make(*extra):
        s = ServerProcess(tmp_path / "r", rest=True, extra_args=list(extra))
        s.start()
        holder["s"] = s
        ch = grpc.insecure_channel(s.address)
        holder["ch"] = ch
        return s, pbg.OmbraStub(ch), cols

    yield make
    if "s" in holder:
        holder["s"].kill()
    if "ch" in holder:
        holder["ch"].close()


def _block(ids, cols, *, center=False):
    return pb.InsertBlock(
        api_version=1, collection=NAME, cluster_id=0, ids=ids, columns=cols, is_center=center
    )


def _n_blocks(stub) -> int:
    r = stub.ListBlocks(pb.ListBlocksRequest(api_version=1, collection=NAME))
    assert not r.error.code
    return len(r.blocks)


def _held_stream(ids, cols, reached: threading.Event, release: threading.Event):
    """Header and the first column, then wait for `release`, then the rest."""
    yield pb.InsertRequest(
        api_version=1, header=pb.InsertHeader(collection=NAME, cluster_id=0, ids=ids)
    )
    for i, c in enumerate(cols):
        yield pb.InsertRequest(api_version=1, column=c)
        if i == 0:
            reached.set()
            release.wait(60)


def _http(addr, path):
    try:
        with urllib.request.urlopen(f"http://{addr}{path}", timeout=5) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _verify_clean(server) -> None:
    out = subprocess.run(  # noqa: S603
        [str(VERIFY), "--data-dir", str(server.data_dir)], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stdout + out.stderr


def _wait_exit(server, timeout):
    return server._proc.wait(timeout)


def test_sigterm_lets_a_running_insert_finish(srv):
    server, stub, cols = srv()
    before = _n_blocks(stub)
    reached, release = threading.Event(), threading.Event()
    fut = stub.Insert.future(_held_stream([b"sd-1", b"sd-2"], cols, reached, release))
    assert reached.wait(30)
    server._proc.send_signal(15)
    deadline = time.monotonic() + 5
    while _http(server.rest_address, "/readyz")[0] != 503:  # not ready at once
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert _http(server.rest_address, "/healthz")[0] == 200  # still alive
    with pytest.raises(grpc.RpcError) as ei:  # a new call is refused
        stub.ListCollections(pb.ListCollectionsRequest(), timeout=5)
    assert ei.value.code() == grpc.StatusCode.UNAVAILABLE
    release.set()
    r = fut.result(timeout=30)
    assert not r.error.code and r.inserted == 2
    assert _wait_exit(server, 30) == 0
    server.start()
    ch = grpc.insecure_channel(server.address)
    try:
        assert _n_blocks(pbg.OmbraStub(ch)) == before + 1  # durable
    finally:
        ch.close()
    server.stop()
    _verify_clean(server)


def test_a_call_longer_than_the_grace_is_cancelled_and_leaves_nothing(srv):
    server, stub, cols = srv("--shutdown-grace-s", "1")
    before = _n_blocks(stub)
    reached, release = threading.Event(), threading.Event()
    fut = stub.Insert.future(_held_stream([b"sg-1"], cols, reached, release))
    assert reached.wait(30)
    t0 = time.monotonic()
    server._proc.send_signal(15)
    assert _wait_exit(server, 30) == 0
    assert time.monotonic() - t0 < 15
    release.set()
    with pytest.raises(grpc.RpcError):  # never acknowledged
        fut.result(timeout=30)
    server.start()
    ch = grpc.insecure_channel(server.address)
    try:
        assert _n_blocks(pbg.OmbraStub(ch)) == before  # the partial block is not there
    finally:
        ch.close()
    server.stop()
    _verify_clean(server)


def test_second_signal_exits_at_once(srv):
    server, stub, cols = srv("--shutdown-grace-s", "300")
    reached, release = threading.Event(), threading.Event()
    stub.Insert.future(_held_stream([b"s2-1"], cols, reached, release))
    assert reached.wait(30)
    server._proc.send_signal(15)
    time.sleep(0.5)
    assert server.running  # draining: waits for the call
    t0 = time.monotonic()
    server._proc.send_signal(15)
    assert _wait_exit(server, 15) == 1
    assert time.monotonic() - t0 < 5
    release.set()
    server.start()
    server.stop()
    _verify_clean(server)


def test_insert_is_idempotent(srv):
    server, stub, cols = srv()
    before = _n_blocks(stub)
    blk = _block([b"id-1", b"id-2", b"id-3"], cols)
    first = stub.Insert(insert_requests(pb, [blk]))
    again = stub.Insert(insert_requests(pb, [blk]))  # as after a lost answer
    assert not first.error.code and not again.error.code
    assert first.inserted == again.inserted == 3
    assert _n_blocks(stub) == before + 1
    # also after a crash and restart (the acknowledgement was lost with the connection)
    server.kill()
    server.start()
    ch = grpc.insecure_channel(server.address)
    try:
        stub = pbg.OmbraStub(ch)
        assert stub.Insert(insert_requests(pb, [blk])).inserted == 3
        assert _n_blocks(stub) == before + 1
        # refused: same ids with other bytes, and a partial overlap
        swapped = _block([b"id-1", b"id-2", b"id-3"], [cols[1], cols[0], *cols[2:]])
        assert stub.Insert(insert_requests(pb, [swapped])).error.code == pb.ERROR_INVALID_ARGUMENT
        part = _block([b"id-1", b"id-9", b"id-8"], cols)
        assert stub.Insert(insert_requests(pb, [part])).error.code == pb.ERROR_INVALID_ARGUMENT
        assert _n_blocks(stub) == before + 1
    finally:
        ch.close()
    server.stop()
    _verify_clean(server)


def test_insert_parts_out_of_order_and_message_limit(srv):
    server, stub, cols = srv()
    before = _n_blocks(stub)
    head = pb.InsertRequest(
        api_version=1, header=pb.InsertHeader(collection=NAME, cluster_id=0, ids=[b"q-1"])
    )
    col = lambda i: pb.InsertRequest(api_version=1, column=cols[i])  # noqa: E731
    assert stub.Insert(iter([col(0)])).error.code == pb.ERROR_BAD_SEQUENCE  # column first
    short = stub.Insert(iter([head, col(0)]))  # fewer columns than the layout
    assert short.error.code == pb.ERROR_BAD_SEQUENCE
    assert stub.Insert(iter([])).error.code == pb.ERROR_BAD_SEQUENCE  # nothing at all
    assert _n_blocks(stub) == before
    big = pb.PutBlobRequest(api_version=1, data=b"x" * (17 << 20))  # above the 16 MiB limit
    with pytest.raises(grpc.RpcError) as ei:
        stub.PutBlob(iter([big]))
    assert ei.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
    # the largest legitimate chunk still goes through (and is refused by the handler, not gRPC)
    chunk = pb.PutBlobRequest(api_version=1, data=b"x" * 4_194_304)
    r = stub.PutBlob(iter([chunk]))
    assert r.error.code == pb.ERROR_BAD_SEQUENCE  # data before the header
