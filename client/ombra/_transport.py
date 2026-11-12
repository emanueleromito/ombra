# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""gRPC transport to a real Ombra server. Carries ciphertexts and public keys only."""

from __future__ import annotations

from typing import Any

from ombra._proto import load_stubs

MAX_MESSAGE = 1 << 30


class GrpcTransport:
    def __init__(
        self, address: str, api_key: str | None, root_certificates: bytes | None, insecure: bool
    ) -> None:
        import grpc

        options = [
            ("grpc.max_send_message_length", MAX_MESSAGE),
            ("grpc.max_receive_message_length", MAX_MESSAGE),
        ]
        if insecure:
            if api_key is not None:
                raise ValueError(
                    "an API key is only sent over TLS; do not combine it with insecure"
                )
            self.channel = grpc.insecure_channel(address, options=options)
        else:
            creds: Any = grpc.ssl_channel_credentials(root_certificates=root_certificates)
            if api_key is not None:
                creds = grpc.composite_channel_credentials(
                    creds, grpc.access_token_call_credentials(api_key)
                )
            self.channel = grpc.secure_channel(address, creds, options=options)
        self.pb, pbg = load_stubs()
        self.stub = pbg.OmbraStub(self.channel)
        import openfhe

        self.openfhe_version = openfhe.__version__

    def __repr__(self) -> str:
        return "GrpcTransport()"

    def close(self) -> None:
        self.channel.close()
