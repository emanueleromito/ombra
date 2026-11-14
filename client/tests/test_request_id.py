# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""The client sends a random `x-request-id` with every call, reuses it across retries (with
`x-attempt`) and puts it on the OmbraError (CPU only, no OpenFHE). An in-process gRPC server
records the metadata it receives."""

import re
from concurrent import futures

import pytest
from ombra import errors
from ombra._proto import load_stubs
from ombra._transport import ResilientStub, default_deadlines
from ombra.retry import RetryPolicy

grpc = pytest.importorskip("grpc")
pb, pbg = load_stubs()


class _Servicer(pbg.OmbraServicer):
    def __init__(self) -> None:
        self.seen: list[dict[str, str]] = []
        self.fail_unavailable = 0
        self.deny = False

    def GetCollection(self, request, context):
        self.seen.append(dict(context.invocation_metadata()))
        if self.fail_unavailable > 0:
            self.fail_unavailable -= 1
            context.abort(grpc.StatusCode.UNAVAILABLE, "try again")
        if self.deny:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "no")
        return pb.GetCollectionResponse(error=pb.Error(code=2, message="collection not found"))


@pytest.fixture
def stub():
    servicer = _Servicer()
    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pbg.add_OmbraServicer_to_server(servicer, srv)
    port = srv.add_insecure_port("127.0.0.1:0")
    srv.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    s = ResilientStub(
        pbg.OmbraStub(channel),
        default_deadlines(30.0, 30.0),
        RetryPolicy(base_delay=0.0, max_delay=0.0),
    )
    yield servicer, s
    channel.close()
    srv.stop(None)


def test_each_call_sends_a_random_id(stub):
    servicer, s = stub
    s.GetCollection(pb.GetCollectionRequest(name="c"))
    s.GetCollection(pb.GetCollectionRequest(name="c"))
    ids = [m["x-request-id"] for m in servicer.seen]
    assert all(re.fullmatch(r"[0-9a-f]{16}", i) for i in ids)
    assert ids[0] != ids[1]
    assert [m["x-attempt"] for m in servicer.seen] == ["1", "1"]


def test_retries_reuse_the_id_with_an_attempt_number(stub):
    servicer, s = stub
    servicer.fail_unavailable = 2
    s.GetCollection(pb.GetCollectionRequest(name="c"))
    assert [m["x-attempt"] for m in servicer.seen] == ["1", "2", "3"]
    assert len({m["x-request-id"] for m in servicer.seen}) == 1


def test_the_error_carries_the_request_id(stub):
    servicer, s = stub
    servicer.deny = True
    with pytest.raises(errors.PermissionDeniedError) as ei:
        s.GetCollection(pb.GetCollectionRequest(name="c"))
    assert ei.value.request_id == servicer.seen[-1]["x-request-id"]
    # still a plain message: no server text, and the id is an attribute, not part of the text
    assert str(ei.value) == "GetCollection failed with status PERMISSION_DENIED"

    # an ombra.v1 error inside a normal response gets the id of the call that returned it
    servicer.deny = False
    resp = s.GetCollection(pb.GetCollectionRequest(name="c"))
    err = errors.from_proto_error(resp.error.code, resp.error.message)
    assert err.request_id == servicer.seen[-1]["x-request-id"]
    assert errors.OmbraError("x").request_id is None
