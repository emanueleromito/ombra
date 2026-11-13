# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""gRPC transport to a real Ombra server. Carries ciphertexts and public keys only."""

from __future__ import annotations

import warnings
from typing import Any

from ombra._proto import load_stubs

# Largest message the client sends: one InsertBlock carries d (+ filter) column ciphertexts of
# about 0.8 MB each at the biggest preset (d = 768: ~600 MB), so the send limit stays high.
MAX_SEND_MESSAGE = 1 << 30
# Largest message the client accepts from a server. Server answers are one ciphertext or blob chunk
# (<= 4 MiB), one SearchScores (a few blocks and <= 100,000 ids of 256 bytes) or a ListBlocks
# catalog (~50 bytes a block); 64 MiB covers all of them (audit A-23).
MAX_RECEIVE_MESSAGE = 64 << 20
# Per-call deadlines in seconds: single-answer calls, and calls that stream data or search.
DEFAULT_TIMEOUT = 120.0
DEFAULT_STREAM_TIMEOUT = 1800.0
_STREAMING = frozenset(
    {"UploadKeys", "Insert", "Search", "GetBlock", "ReplaceBlock", "PutBlob", "GetBlob"}
)
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


class _DeadlineStub:
    """Gives every call of the generated stub a deadline unless the caller sets one."""

    def __init__(self, stub: Any, timeout: float, stream_timeout: float) -> None:
        self._stub, self._timeout, self._stream = stub, timeout, stream_timeout

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._stub, name)
        limit = self._stream if name in _STREAMING else self._timeout

        def call(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("timeout", limit)
            return method(*args, **kwargs)

        return call


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
    ) -> None:
        import grpc

        if timeout <= 0 or stream_timeout <= 0:
            raise ValueError("timeouts must be positive")
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
        self.stub = _DeadlineStub(pbg.OmbraStub(self.channel), timeout, stream_timeout)
        import openfhe

        self.openfhe_version = openfhe.__version__

    def __repr__(self) -> str:
        return "GrpcTransport()"

    def close(self) -> None:
        self.channel.close()
