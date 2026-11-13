# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Sizes and counts a server reports are bounded; calls have deadlines (audit A-23)."""

from types import SimpleNamespace

import pytest
from ombra import _transport, presets
from ombra.blobstore import BUCKET_SIZES, BlobStoreError, GrpcBlobStore
from ombra.collection import CollectionError, check_server_spec
from ombra.presets import ParamsError


def _ns(**kw):
    return SimpleNamespace(**kw)


def test_server_spec_is_bounded():
    d = presets.derive("fast", 384)
    check_server_spec(d, (4, 8), 10)  # a legitimate spec
    with pytest.raises(CollectionError):
        check_server_spec(d, (4,), 0)
    with pytest.raises(CollectionError):
        check_server_spec(d, (4,), d.batch_size + 1)
    with pytest.raises(CollectionError):
        check_server_spec(d, (4,), 2**32 - 1)
    with pytest.raises(ParamsError):
        check_server_spec(d, (2**31,), 10)  # cardinality
    with pytest.raises(ParamsError):
        check_server_spec(d, (2,) * 100_000, 10)  # field count


class _Msg(SimpleNamespace):
    def WhichOneof(self, _):  # noqa: N802
        return self.part


def _store(msgs):
    stub = _ns(GetBlob=lambda req: iter(msgs))
    pb = _ns(GetBlobRequest=lambda **kw: None)
    return GrpcBlobStore(stub, pb)


def test_blob_stream_totals_are_bounded():
    size = BUCKET_SIZES[0]
    ok = [_Msg(part="size", size=size), _Msg(part="data", data=b"x" * size)]
    assert len(_store(ok).get("c", 1, b"i")) == size
    huge = [_Msg(part="size", size=2**40)]
    with pytest.raises(BlobStoreError):
        _store(huge).get("c", 1, b"i")
    more = [_Msg(part="size", size=size), _Msg(part="data", data=b"x" * (size + 1))]
    with pytest.raises(BlobStoreError):
        _store(more).get("c", 1, b"i")
    nosize = [_Msg(part="data", data=b"x")]
    with pytest.raises(BlobStoreError):
        _store(nosize).get("c", 1, b"i")


def test_deadlines_are_set_and_can_be_overridden():
    seen = {}
    inner = _ns(
        Search=lambda it, **kw: seen.update(search=kw),
        GetCollection=lambda r, **kw: seen.update(get=kw),
    )
    stub = _transport._DeadlineStub(inner, 5.0, 50.0)
    stub.Search(iter(()))
    stub.GetCollection(None)
    assert seen == {"search": {"timeout": 50.0}, "get": {"timeout": 5.0}}
    stub.GetCollection(None, timeout=1.0)
    assert seen["get"] == {"timeout": 1.0}


def test_receive_limit_is_far_below_a_gibibyte():
    assert _transport.MAX_RECEIVE_MESSAGE <= 64 << 20
    assert _transport.DEFAULT_TIMEOUT > 0 and _transport.DEFAULT_STREAM_TIMEOUT > 0


@pytest.mark.parametrize(
    ("address", "host"),
    [("localhost:50051", "localhost"), ("[::1]:5", "::1"), ("10.0.0.2:5", "10.0.0.2"), ("h", "h")],
)
def test_host_parsing(address, host):
    assert _transport._host(address) == host


def test_insecure_warns_for_non_loopback_hosts():
    import warnings

    with pytest.warns(UserWarning, match="not encrypted"):
        _transport._warn_if_not_loopback("10.255.255.1:1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for ok in ("localhost:1", "127.0.0.1:1", "[::1]:1"):
            _transport._warn_if_not_loopback(ok)
