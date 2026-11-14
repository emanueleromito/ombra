# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Deadlines, error types, retries and search-stream reconnects (CPU only, no OpenFHE).

An in-process gRPC server injects failures; the wrapper under test is `ResilientStub`.
"""

import random
import time
from concurrent import futures
from types import SimpleNamespace

import numpy as np
import pytest
from ombra import errors, search
from ombra._proto import load_stubs
from ombra._transport import ALL_RPCS, ResilientStub, default_deadlines, may_retry
from ombra.blobstore import BlobStoreError, GrpcBlobStore
from ombra.collection import CollectionError, _check, _create_on_server
from ombra.errors import from_proto_error, from_rpc_error
from ombra.presets import FILTER_LAMBDA
from ombra.retry import Replayable, RetryPolicy, call_with_retry, retrying

grpc = pytest.importorskip("grpc")
pb, pbg = load_stubs()


def _ns(**kw):
    return SimpleNamespace(**kw)


def _rpc_error(status: str):
    class E(grpc.RpcError):
        def code(self):
            return _ns(name=status)

        def details(self):
            return "secret-detail"

    return E()


# ---- error types ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("status", "cls", "retryable"),
    [
        ("UNAVAILABLE", errors.ServerUnavailableError, True),
        ("DEADLINE_EXCEEDED", errors.DeadlineExceededError, True),
        ("RESOURCE_EXHAUSTED", errors.ResourceExhaustedError, True),
        ("NOT_FOUND", errors.NotFoundError, False),
        ("INVALID_ARGUMENT", errors.InvalidArgumentError, False),
        ("PERMISSION_DENIED", errors.PermissionDeniedError, False),
        ("UNAUTHENTICATED", errors.UnauthenticatedError, False),
        ("INTERNAL", errors.ServerInternalError, False),
        ("DATA_LOSS", errors.OmbraError, False),
    ],
)
def test_grpc_status_mapping(status, cls, retryable):
    err = from_rpc_error(_rpc_error(status), "GetCollection")
    assert type(err) is cls and err.retryable is retryable and err.grpc_status == status
    assert str(err) == f"GetCollection failed with status {status}"  # no server detail text


def test_non_rpc_exception_is_not_mapped():
    assert from_rpc_error(ValueError("x"), "Delete") is None


def test_every_proto_error_code_has_a_class_and_is_not_retryable():
    for name, code in pb.ErrorCode.items():
        if code == 0:
            continue
        err = from_proto_error(code, "m")
        assert err.error_code == code, name
        assert err.retryable is (name == "ERROR_STORAGE_FULL"), name  # the one retryable code
        assert type(err) is not CollectionError  # a specific class
        assert isinstance(err, CollectionError) and err.code == code and err.message == "m"
        blob = from_proto_error(code, "m", BlobStoreError)
        assert isinstance(blob, BlobStoreError) and isinstance(blob, RuntimeError)


def test_specific_proto_classes():
    assert isinstance(from_proto_error(pb.ERROR_KEY_TAG_MISMATCH, ""), errors.KeyTagMismatchError)
    assert isinstance(from_proto_error(pb.ERROR_PRESET_MISMATCH, ""), errors.PresetMismatchError)
    assert isinstance(from_proto_error(pb.ERROR_COLLECTION_NOT_FOUND, ""), errors.NotFoundError)
    assert isinstance(from_proto_error(pb.ERROR_BLOB_NOT_FOUND, ""), errors.BlobNotFoundError)
    with pytest.raises(CollectionError):
        _check(pb.Error(code=pb.ERROR_INVALID_ARGUMENT, message="bad"))
    with pytest.raises(errors.InvalidArgumentError):
        _check(pb.Error(code=pb.ERROR_INVALID_ARGUMENT, message="bad"))
    _check(pb.Error())  # code 0 is success


def test_blob_store_error_still_catchable_as_before():
    store = GrpcBlobStore(_ns(), pb)
    with pytest.raises(BlobStoreError) as info:
        store._check(pb.Error(code=pb.ERROR_BLOB_NOT_FOUND, message="x"))
    assert isinstance(info.value, errors.BlobNotFoundError) and isinstance(info.value, RuntimeError)
    assert str(info.value) == f"server error code {pb.ERROR_BLOB_NOT_FOUND}"


# ---- retry loop -----------------------------------------------------------------------------
class _Clock:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _unavailable():
    return errors.ServerUnavailableError("x", grpc_status="UNAVAILABLE")


def test_backoff_is_exponential_with_full_jitter_and_capped():
    clock, calls = _Clock(), []
    policy = RetryPolicy(max_attempts=6, base_delay=1.0, max_delay=5.0, total_budget=1e9)

    def fn():
        calls.append(1)
        raise _unavailable()

    kw = {"rpc": "X", "policy": policy, "clock": clock, "sleep": clock.sleep}
    with pytest.raises(errors.ServerUnavailableError):
        call_with_retry(fn, rand=lambda: 1.0, **kw)
    assert len(calls) == 6
    assert clock.sleeps == [1.0, 2.0, 4.0, 5.0, 5.0]  # rand=1 gives the upper bounds
    clock.sleeps.clear()
    with pytest.raises(errors.ServerUnavailableError):
        call_with_retry(fn, rand=random.Random(1).random, **kw)  # noqa: S311
    assert all(0 <= s <= b for s, b in zip(clock.sleeps, [1, 2, 4, 5, 5], strict=True))
    assert len(clock.sleeps) == 5


def test_total_budget_stops_retries():
    clock, calls = _Clock(), []
    policy = RetryPolicy(max_attempts=50, base_delay=1.0, max_delay=1.0, total_budget=3.5)

    def fn():
        calls.append(1)
        clock.now += 1.0  # each attempt itself takes a second
        raise _unavailable()

    with pytest.raises(errors.ServerUnavailableError):
        call_with_retry(
            fn, rpc="X", policy=policy, clock=clock, sleep=clock.sleep, rand=lambda: 1.0
        )
    assert len(calls) == 2 and clock.now == 3.0  # a third try would pass the 3.5 s budget


def test_non_retryable_error_is_raised_at_once():
    calls = []

    def fn():
        calls.append(1)
        raise errors.NotFoundError("x")

    with pytest.raises(errors.NotFoundError):
        call_with_retry(fn, rpc="X", policy=RetryPolicy(), sleep=lambda s: None)
    assert calls == [1]


def test_never_policy_and_validation():
    calls = []

    def fn():
        calls.append(1)
        raise _unavailable()

    with pytest.raises(errors.ServerUnavailableError):
        call_with_retry(fn, rpc="X", policy=RetryPolicy.never(), sleep=lambda s: None)
    assert calls == [1]
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)


def test_retries_are_logged_at_debug_without_payload(caplog):
    caplog.set_level("DEBUG", logger="ombra.retry")
    n = []

    def fn():
        n.append(1)
        if len(n) < 2:
            raise _unavailable()
        return "ok"

    assert call_with_retry(fn, rpc="GetBlock", policy=RetryPolicy(), sleep=lambda s: None) == "ok"
    (rec,) = caplog.records
    assert rec.levelname == "DEBUG"
    assert rec.getMessage() == "retry GetBlock attempt=1 status=UNAVAILABLE"


# ---- in-process server with failure injection -----------------------------------------------
class _Servicer(pbg.OmbraServicer):
    """Fails the first `fail[rpc]` calls of a method with `status`; records what it saw."""

    def __init__(self):
        self.fail: dict[str, int] = {}
        self.status = grpc.StatusCode.UNAVAILABLE
        self.calls: dict[str, int] = {}
        self.remaining: dict[str, float] = {}
        self.requests: dict[str, list[int]] = {}
        self.spec = None
        self.create_error = None
        self.blob_size = 4

    def _enter(self, rpc, ctx):
        self.calls[rpc] = self.calls.get(rpc, 0) + 1
        self.remaining[rpc] = ctx.time_remaining()
        if self.fail.get(rpc, 0) >= self.calls[rpc]:
            ctx.abort(self.status, "secret-detail")

    def GetCollection(self, request, context):  # noqa: N802
        self._enter("GetCollection", context)
        if self.spec is None:
            return pb.GetCollectionResponse(error=pb.Error(code=pb.ERROR_COLLECTION_NOT_FOUND))
        return pb.GetCollectionResponse(info=pb.CollectionInfo(spec=self.spec))

    def CreateCollection(self, request, context):  # noqa: N802
        self._enter("CreateCollection", context)
        if self.create_error is not None:
            return pb.CreateCollectionResponse(error=pb.Error(code=self.create_error))
        return pb.CreateCollectionResponse(spec=request.spec)

    def DropCollection(self, request, context):  # noqa: N802
        self._enter("DropCollection", context)
        return pb.DropCollectionResponse()

    def Delete(self, request, context):  # noqa: N802
        self._enter("Delete", context)
        return pb.DeleteResponse(deleted=len(request.ids))

    def Insert(self, request_iterator, context):  # noqa: N802
        self._enter("Insert", context)
        return pb.InsertResponse(inserted=sum(1 for _ in request_iterator))

    def PutBlob(self, request_iterator, context):  # noqa: N802
        self._enter("PutBlob", context)
        self.requests.setdefault("PutBlob", []).append(sum(1 for _ in request_iterator))
        return pb.PutBlobResponse()

    def GetBlob(self, request, context):  # noqa: N802
        self.calls["GetBlob"] = self.calls.get("GetBlob", 0) + 1
        self.remaining["GetBlob"] = context.time_remaining()
        yield pb.GetBlobResponse(size=self.blob_size)
        if self.fail.get("GetBlob", 0) >= self.calls["GetBlob"]:
            context.abort(self.status, "secret-detail")  # the stream breaks after one message
        yield pb.GetBlobResponse(data=b"\0" * self.blob_size)

    def Search(self, request_iterator, context):  # noqa: N802
        self._enter("Search", context)
        yield pb.SearchResponse()


@pytest.fixture
def server():
    servicer = _Servicer()
    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pbg.add_OmbraServicer_to_server(servicer, srv)
    port = srv.add_insecure_port("127.0.0.1:0")
    srv.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    clock = _Clock()

    def stub(**kw):
        policy = kw.pop("policy", RetryPolicy(base_delay=0.5, total_budget=1e9))
        deadlines = kw.pop("deadlines", default_deadlines(120.0, 1800.0))
        s = ResilientStub(
            pbg.OmbraStub(channel), deadlines, policy, clock=clock, sleep=clock.sleep, **kw
        )
        s.clock = clock
        return s

    yield servicer, stub
    channel.close()
    srv.stop(None)


def test_every_service_method_has_a_deadline():
    methods = set(pb.DESCRIPTOR.services_by_name["Ombra"].methods_by_name)
    assert methods == set(ALL_RPCS)
    assert set(default_deadlines(120.0, 1800.0)) == methods
    assert all(v > 0 for v in default_deadlines(120.0, 1800.0).values())


def test_deadline_is_sent_with_each_call_kind(server):
    servicer, stub = server
    s = stub()
    s.GetCollection(pb.GetCollectionRequest(name="c"))  # short metadata call
    s.Delete(pb.DeleteRequest(collection="c"))  # single-answer call
    s.Insert(Replayable(lambda: iter([pb.InsertRequest()])))  # client stream
    assert list(s.GetBlob(pb.GetBlobRequest()))
    assert 20 < servicer.remaining["GetCollection"] <= 30 * 1.05
    assert 100 < servicer.remaining["Delete"] <= 120 * 1.05
    assert 1700 < servicer.remaining["Insert"] <= 1800 * 1.05
    assert 250 < servicer.remaining["GetBlob"] <= 300 * 1.05


def test_deadline_override_and_per_call_keyword(server):
    servicer, stub = server
    s = stub(deadlines={**default_deadlines(120.0, 1800.0), "Search": 7.0})
    assert list(s.Search(iter([])))
    assert 5 < servicer.remaining["Search"] <= 7 * 1.05
    s.GetCollection(pb.GetCollectionRequest(name="c"), timeout=3)  # per-call keyword wins
    assert 1 < servicer.remaining["GetCollection"] <= 3 * 1.05


def test_stream_deadline_bounds_a_stalled_stream():
    class Slow(pbg.OmbraServicer):
        def Search(self, request_iterator, context):  # noqa: N802
            yield pb.SearchResponse()
            time.sleep(2)
            yield pb.SearchResponse()

    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pbg.add_OmbraServicer_to_server(Slow(), srv)
    port = srv.add_insecure_port("127.0.0.1:0")
    srv.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        s = ResilientStub(
            pbg.OmbraStub(channel),
            {**default_deadlines(120.0, 1800.0), "Search": 0.5},
            RetryPolicy.never(),
        )
        call = s.Search(iter([]))
        next(call)
        with pytest.raises(grpc.RpcError) as info:
            next(call)
        assert info.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        mapped = from_rpc_error(info.value, "Search")
        assert isinstance(mapped, errors.DeadlineExceededError) and mapped.retryable
    finally:
        channel.close()
        srv.stop(None)


def test_idempotent_call_is_retried_with_backoff(server):
    servicer, stub = server
    servicer.fail["GetCollection"] = 2
    servicer.spec = pb.CollectionSpec(name="c")
    s = stub()
    r = s.GetCollection(pb.GetCollectionRequest(name="c"))
    assert r.info.spec.name == "c"
    assert servicer.calls["GetCollection"] == 3
    assert len(s.clock.sleeps) == 2
    assert s.clock.sleeps[0] <= 0.5 and s.clock.sleeps[1] <= 1.0


@pytest.mark.parametrize("status", [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED])
def test_retries_give_up_after_max_attempts(server, status):
    servicer, stub = server
    servicer.fail["Delete"] = 99
    servicer.status = status
    s = stub(policy=RetryPolicy(max_attempts=3, base_delay=0.1))
    with pytest.raises(errors.OmbraError) as info:
        s.Delete(pb.DeleteRequest(collection="c", ids=[b"x"]))
    assert info.value.retryable and info.value.grpc_status == status.name
    assert servicer.calls["Delete"] == 3
    assert "secret-detail" not in str(info.value)


def test_non_retryable_status_is_not_retried(server):
    servicer, stub = server
    servicer.fail["GetCollection"] = 99
    servicer.status = grpc.StatusCode.PERMISSION_DENIED
    with pytest.raises(errors.PermissionDeniedError) as info:
        stub().GetCollection(pb.GetCollectionRequest(name="c"))
    assert not info.value.retryable and servicer.calls["GetCollection"] == 1


def test_resource_exhausted_is_retried(server):
    servicer, stub = server
    servicer.fail["Delete"] = 1
    servicer.status = grpc.StatusCode.RESOURCE_EXHAUSTED
    assert stub().Delete(pb.DeleteRequest(collection="c", ids=[b"a"])).deleted == 1
    assert servicer.calls["Delete"] == 2


def test_insert_is_retried_but_drop_and_replace_are_not(server):
    servicer, stub = server
    servicer.fail["Insert"] = 1
    servicer.fail["DropCollection"] = 1
    s = stub()
    s.Insert(Replayable(lambda: iter([pb.InsertRequest()])))  # the server is idempotent: retried
    assert servicer.calls["Insert"] == 2
    with pytest.raises(errors.ServerUnavailableError):
        s.DropCollection(pb.DropCollectionRequest(name="c"))
    assert servicer.calls["DropCollection"] == 1
    assert may_retry("Insert") and not may_retry("ReplaceBlock")
    assert not may_retry("DropCollection") and not may_retry("CreateCollection")


def test_put_blob_retry_resends_the_whole_request_stream(server):
    servicer, stub = server
    servicer.fail["PutBlob"] = 1
    msgs = [pb.PutBlobRequest(), pb.PutBlobRequest(data=b"x"), pb.PutBlobRequest(data=b"y")]
    stub().PutBlob(Replayable(lambda: iter(msgs)))
    assert servicer.calls["PutBlob"] == 2
    assert servicer.requests["PutBlob"] == [3]  # the second try sent all three messages


def test_one_shot_request_stream_is_not_retried(server):
    servicer, stub = server
    servicer.fail["PutBlob"] = 1
    with pytest.raises(errors.ServerUnavailableError):
        stub().PutBlob(iter([pb.PutBlobRequest()]))  # a plain iterator cannot be replayed
    assert servicer.calls["PutBlob"] == 1


def test_broken_response_stream_is_read_again_in_full(server):
    servicer, stub = server
    servicer.fail["GetBlob"] = 1  # first stream breaks after its first message
    out = list(stub().GetBlob(pb.GetBlobRequest()))
    assert [m.WhichOneof("part") for m in out] == ["size", "data"]
    assert servicer.calls["GetBlob"] == 2


def test_blob_store_get_survives_a_break(server):
    servicer, stub = server
    servicer.fail["GetBlob"] = 1
    servicer.blob_size = 16384  # a bucket size
    store = GrpcBlobStore(stub(), pb)
    assert store.get("c", 1, b"id") == b"\0" * 16384
    assert servicer.calls["GetBlob"] == 2


def test_create_collection_retry_accepts_identical_existing_spec(server):
    servicer, stub = server
    spec = pb.CollectionSpec(name="c", preset="p", dimension=4, key_tag="t", cluster_count=2)
    servicer.fail["CreateCollection"] = 1  # first try fails; the repeat finds the collection there
    servicer.create_error = pb.ERROR_COLLECTION_EXISTS
    servicer.spec = spec
    _create_on_server(pb, stub(), spec)  # same spec: success
    assert servicer.calls["CreateCollection"] == 2

    servicer.calls.clear()
    servicer.spec = pb.CollectionSpec(name="c", preset="p", dimension=4, key_tag="OTHER")
    with pytest.raises(errors.CollectionExistsError):
        _create_on_server(pb, stub(), spec)


def test_create_collection_exists_on_first_try_is_an_error(server):
    servicer, stub = server
    spec = pb.CollectionSpec(name="c", key_tag="t")
    servicer.create_error = pb.ERROR_COLLECTION_EXISTS
    servicer.spec = spec  # even an identical spec: nothing was retried, so it is not ours
    with pytest.raises(errors.CollectionExistsError):
        _create_on_server(pb, stub(), spec)
    assert servicer.calls["CreateCollection"] == 1


def test_stubs_without_a_retry_runner_run_once():
    n = []
    assert retrying(_ns(), "X", lambda: n.append(1) or "v") == "v" and n == [1]


# ---- search: reconnect keeps the query and the cluster choice -------------------------------
C, N = 36, 20
GOOD = FILTER_LAMBDA + 0.1


class _Msg(SimpleNamespace):
    def WhichOneof(self, _):  # noqa: N802
        return self.kind


def _fake_search(break_after):
    """A collection whose first Search stream dies (UNAVAILABLE) at `break_after`: "start" (while
    the server reads the query), or "round1" / "round2" (after it read that round's request).
    Records every stream's start query and chosen lists."""
    streams: list[dict] = []
    encrypted: list[int] = []
    cs = np.random.default_rng(0).uniform(-1, 1, C)

    def search_stream(it):
        rec = {"query": None, "chosen": []}
        streams.append(rec)
        first = len(streams) == 1
        rec["query"] = next(it).start.query
        if first and break_after == "start":
            raise _rpc_error("UNAVAILABLE")
        yield _Msg(kind="centers", centers=_ns(score_blocks=[cs.tobytes()], center_count=C))
        for req in it:
            rec["chosen"].append(list(req.chosen.cluster_ids))
            if first and break_after == f"round{len(rec['chosen'])}":
                raise _rpc_error("UNAVAILABLE")
            for c in req.chosen.cluster_ids:
                ids = [bytes([c, j]) for j in range(N)]
                blk = np.full(N, GOOD).tobytes()
                yield _Msg(
                    kind="scores",
                    scores=_ns(cluster_id=c, score_blocks=[blk], ids=ids, block_sizes=[N]),
                )

    def enc(ext):
        encrypted.append(1)
        return random.randbytes(16)  # noqa: S311  # a different value on every call

    pb_fake = _ns(
        ListBlocksRequest=lambda **kw: None,
        SearchStart=lambda collection, key_tag, query, filter_leaves: _ns(query=query),
        SearchRequest=lambda start=None, chosen=None: _ns(start=start, chosen=chosen),
        SearchChosen=lambda cluster_ids: _ns(cluster_ids=cluster_ids),
    )
    clock = _Clock()
    stub = _ns(
        ListBlocks=lambda r: _ns(blocks=[_ns(cluster_id=c, n_vectors=N) for c in range(C)]),
        Search=search_stream,
        retry_runner=lambda rpc, fn: call_with_retry(
            fn, rpc=rpc, policy=RetryPolicy(), clock=clock, sleep=clock.sleep
        ),
    )
    kit = _ns(
        key_tag="t",
        encrypt_query=enc,
        decrypt_column=lambda env: np.frombuffer(env, dtype=np.float64),
    )
    col = _ns(
        pb=pb_fake,
        stub=stub,
        kit=kit,
        name="c",
        security_level="decoy",
        derived=_ns(dimension=4),
        cards=(),
        cluster_count=C,
        stats=None,
    )
    return col, streams, encrypted


@pytest.fixture(autouse=True)
def _no_rerank(monkeypatch):
    monkeypatch.setattr(search, "_rerank", lambda col, q, top, k: [])


def _search(col):
    # no _rng: the real SystemRandom, so a second draw would almost surely differ
    return search.run_search(col, np.array([1.0, 0, 0, 0]), 3, level="decoy")


@pytest.mark.parametrize("break_after", ["start", "round1", "round2"])
def test_search_reconnect_resends_same_query_and_same_cluster_choice(break_after):
    col, streams, encrypted = _fake_search(break_after)
    res = _search(col)
    assert res.rounds == 2 and len(streams) == 2
    first, second = streams
    assert first["query"] == second["query"]  # the same query ciphertext, not re-encrypted
    assert len(encrypted) == 1
    sent = len(first["chosen"])
    assert sent == {"start": 0, "round1": 1, "round2": 2}[break_after]
    assert second["chosen"][:sent] == first["chosen"]  # same real + decoy clusters, same order
    assert len(second["chosen"]) == 2
    flat = [c for r in second["chosen"] for c in r]
    assert len(flat) == len(set(flat))  # no cluster twice on one stream, even across the reconnect


def test_a_fresh_search_draws_other_decoys():
    # guards the test above: a fresh draw would differ, so equal lists mean the choice was reused
    col_a, streams_a, _ = _fake_search("none")
    col_b, streams_b, _ = _fake_search("none")
    _search(col_a)
    _search(col_b)
    assert streams_a[0]["chosen"] != streams_b[0]["chosen"]


def test_search_does_not_retry_a_non_retryable_stream_error():
    col, streams, _ = _fake_search("none")

    def failing(it):
        next(it)
        raise _rpc_error("PERMISSION_DENIED")
        yield

    col.stub.Search = failing
    with pytest.raises(errors.PermissionDeniedError):
        _search(col)


def test_search_reconnect_survives_a_block_layout_change():
    """The second stream answers with other block sizes (a repack happened); each answer carries
    its own sizes, so the search still works. A different answer kind or cluster is refused."""
    col, streams, _ = _fake_search("round1")
    res = _search(col)
    assert res.rounds == 2
    s = search._Stream
    assert s._identity(_Msg(kind="centers")) == ("centers",)
    assert s._identity(_Msg(kind="scores", scores=_ns(cluster_id=4))) == ("scores", 4)


def test_id_batches_stay_under_the_byte_cap():
    from ombra._transport import id_batches

    ids = [bytes([i % 256]) * 256 for i in range(100_000)]  # about 26 MB at the longest id
    batches = list(id_batches(ids, max_bytes=8 << 20))
    assert len(batches) > 1
    assert [i for b in batches for i in b] == ids
    assert all(sum(len(i) + 4 for i in b) <= 8 << 20 for b in batches)
    assert list(id_batches([])) == []
