# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""gRPC transport to a real Ombra server. Carries ciphertexts and public keys only."""

from __future__ import annotations

import random
import time
import warnings
from collections.abc import Callable, Mapping
from typing import Any

from ombra._proto import load_stubs
from ombra.errors import OmbraError, from_rpc_error
from ombra.retry import Replayable, RetryPolicy, call_with_retry

# Largest message the client sends. Every stream is sent in parts (Insert, ReplaceBlock: one column
# of about 0.8 MB at the biggest preset; PutBlob and UploadKeys: chunks of at most 4 MiB; Insert
# header: at most one block of ids, about 2 MB), so this equals the server's receive limit.
MAX_SEND_MESSAGE = 16 << 20
# Largest message the client accepts from a server. Server answers are one ciphertext or blob chunk
# (<= 4 MiB), one SearchScores (a few blocks and <= 100,000 ids of 256 bytes) or a ListBlocks
# catalog (~50 bytes a block); 64 MiB covers all of them (audit A-23).
MAX_RECEIVE_MESSAGE = 64 << 20
# Default deadlines in seconds. `timeout` / `stream_timeout` of the constructor replace the two
# general ones; `deadlines={"Search": 600}` replaces one call; a `timeout=` keyword on a single
# stub call beats both. A deadline bounds the whole call, so for a stream it bounds the stream.
DEFAULT_TIMEOUT = 120.0  # calls with one request and one answer that change data
DEFAULT_STREAM_TIMEOUT = 1800.0  # calls that stream big data or search
DEFAULT_BLOB_TIMEOUT = 300.0  # one blob is at most 16 MiB
DEFAULT_METADATA_TIMEOUT = 30.0  # small reads
_METADATA = frozenset({"GetCollection", "ListCollections", "ListBlocks"})
_BLOB = frozenset({"PutBlob", "GetBlob"})
_STREAMING = frozenset({"UploadKeys", "Insert", "Search", "GetBlock", "ReplaceBlock"})
_UNARY = frozenset({"CreateCollection", "DropCollection", "Delete", "DeleteBlobs"})
ALL_RPCS = _METADATA | _BLOB | _STREAMING | _UNARY
# Streams of answers that are read to the end inside the retry loop, so a break in the middle is
# repeated as a whole (the answers are bounded: one block or one blob).
_COLLECT = frozenset({"GetBlock", "GetBlob"})
# Calls whose retry is driven by the caller, not by the stub wrapper: Search by the stream
# (it must replay the same requests) and CreateCollection by `Collection.create`.
_CALLER_RETRIES = frozenset({"Search", "CreateCollection"})


def may_retry(rpc: str) -> bool:
    """The one place that decides which calls are repeated after a retryable error.

    A call is listed only if doing it twice leaves the same state: reads, Search, Delete (an id
    that is already gone is ignored), PutBlob (atomic replace), DeleteBlobs and UploadKeys (atomic
    replace with the same keys) and Insert: the server treats a block whose ids are all stored in
    one block with byte-identical columns as a success that stores nothing, so a repeat after a
    lost answer neither duplicates nor fails (the call site sends the same encrypted bytes, as a
    `Replayable`). ReplaceBlock is NOT listed: after a lost answer the old blocks are gone, so a
    repeat is refused as ERROR_BLOCK_NOT_FOUND although the first try worked.
    """
    return rpc in {
        "GetCollection",
        "ListCollections",
        "ListBlocks",
        "GetBlock",
        "GetBlob",
        "Search",
        "Insert",
        "Delete",
        "PutBlob",
        "DeleteBlobs",
        "UploadKeys",
    }


def default_deadlines(timeout: float, stream_timeout: float) -> dict[str, float]:
    """The deadline of every call: `timeout` for data-changing single calls (but short for small
    reads), `stream_timeout` for big streams and searches, five minutes for one blob."""
    out: dict[str, float] = {}
    for rpc in ALL_RPCS:
        if rpc in _METADATA:
            out[rpc] = min(DEFAULT_METADATA_TIMEOUT, timeout)
        elif rpc in _BLOB:
            out[rpc] = min(DEFAULT_BLOB_TIMEOUT, stream_timeout)
        elif rpc in _STREAMING:
            out[rpc] = stream_timeout
        else:
            out[rpc] = timeout
    return out


_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


def _host(address: str) -> str:
    """The host part of `host:port`, `[v6]:port` or a bare name."""
    if address.startswith("["):
        return address[1:].split("]", 1)[0]
    if address.count(":") == 1:
        return address.split(":", 1)[0]
    return address


def _warn_if_not_loopback(address: str) -> None:
    """insecure=True off the loopback interface: no encryption or server authentication (A-49)."""
    if _host(address) not in _LOOPBACK:
        warnings.warn(
            "insecure=True on a non-loopback address: the connection is not encrypted "
            "or authenticated; use TLS",
            stacklevel=4,
        )


class ResilientStub:
    """Wraps the generated stub: every call gets a deadline, gRPC errors become `OmbraError`s, and
    calls that are safe to repeat are retried under a `RetryPolicy` (see `may_retry`).

    A call that sends a request stream is repeated only if the stream is a `Replayable`.
    """

    def __init__(
        self,
        stub: Any,
        deadlines: Mapping[str, float],
        policy: RetryPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rand: Callable[[], float] = random.random,
    ) -> None:
        self._stub, self._deadlines, self.retry_policy = stub, dict(deadlines), policy
        self._clock, self._sleep, self._rand = clock, sleep, rand

    def retry_runner(self, rpc: str, fn: Callable[[], Any]) -> Any:
        """Run `fn` under this stub's policy (used by callers that drive their own retries)."""
        return call_with_retry(
            fn,
            rpc=rpc,
            policy=self.retry_policy,
            clock=self._clock,
            sleep=self._sleep,
            rand=self._rand,
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        method = getattr(self._stub, name)
        limit = self._deadlines.get(name, DEFAULT_TIMEOUT)

        def once(args: tuple, kwargs: dict) -> Any:
            real = [iter(a) if isinstance(a, Replayable) else a for a in args]
            try:
                res = method(*real, **kwargs)
                return list(res) if name in _COLLECT else res
            except OmbraError:
                raise
            except Exception as exc:
                mapped = from_rpc_error(exc, name)
                if mapped is None:
                    raise
                raise mapped from exc

        def call(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("timeout", limit)
            if name in _CALLER_RETRIES or not may_retry(name):
                return once(args, kwargs) if name != "Search" else method(*args, **kwargs)
            streams_in = any(not isinstance(a, Replayable) and _is_iterator(a) for a in args)
            if streams_in:  # a one-shot request stream cannot be sent again
                return once(args, kwargs)
            return self.retry_runner(name, lambda: once(args, kwargs))

        return call


def _is_iterator(value: Any) -> bool:
    return hasattr(value, "__next__")


class GrpcTransport:
    def __init__(
        self,
        address: str,
        api_key: str | None,
        root_certificates: bytes | None,
        insecure: bool,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        stream_timeout: float = DEFAULT_STREAM_TIMEOUT,
        deadlines: Mapping[str, float] | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        import grpc

        if timeout <= 0 or stream_timeout <= 0:
            raise ValueError("timeouts must be positive")
        per_call = default_deadlines(timeout, stream_timeout)
        for rpc, seconds in (deadlines or {}).items():
            if rpc not in ALL_RPCS:
                raise ValueError(f"unknown call name in deadlines: {rpc!r}")
            if seconds <= 0:
                raise ValueError("timeouts must be positive")
            per_call[rpc] = seconds
        options = [
            ("grpc.max_send_message_length", MAX_SEND_MESSAGE),
            ("grpc.max_receive_message_length", MAX_RECEIVE_MESSAGE),
        ]
        if insecure:
            if api_key is not None:
                raise ValueError(
                    "an API key is only sent over TLS; do not combine it with insecure"
                )
            _warn_if_not_loopback(address)
            self.channel = grpc.insecure_channel(address, options=options)
        else:
            creds: Any = grpc.ssl_channel_credentials(root_certificates=root_certificates)
            if api_key is not None:
                creds = grpc.composite_channel_credentials(
                    creds, grpc.access_token_call_credentials(api_key)
                )
            self.channel = grpc.secure_channel(address, creds, options=options)
        self.pb, pbg = load_stubs()
        self.stub = ResilientStub(
            pbg.OmbraStub(self.channel), per_call, RetryPolicy() if retry is None else retry
        )
        import openfhe

        self.openfhe_version = openfhe.__version__

    def __repr__(self) -> str:
        return "GrpcTransport()"

    def close(self) -> None:
        self.channel.close()
