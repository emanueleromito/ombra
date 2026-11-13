# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Resource limits and hardening of `ombra_server` (pre-release audit A-02, A-03, A-18, A-22, A-54).

Needs the server binary (it is built with CUDA), so the tests are marked gpu. They use synthetic
data only: a stream that sends nothing, a large junk message, bad flag values. The logic behind the caps
(session limiter, stream watchdog, flag parsing) also has CPU unit tests in server/grpc/tests.
"""

from __future__ import annotations

import os
import stat
import subprocess
import threading
import time

import pytest

pytest.importorskip("numpy")
grpc = pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")

import e2e_client as ec  # noqa: E402

pytestmark = pytest.mark.gpu

VERSION = "test-openfhe"


def _binary():
    b = ec.find_server_binary()
    if b is None:
        pytest.fail("ombra_server not found; build with `make build` or `make test-gpu`")
    return b


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [str(_binary()), *args], capture_output=True, text=True, timeout=60
    )


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--gpu-budget-mb", "abc"),
        ("--gpu-budget-mb", "99999999999999999999"),
        ("--gpu-budget-mb", "-1"),
        ("--max-message-mb", "0"),
        ("--max-sessions", "0"),
        ("--rest-port", "70000"),
    ],
)
def test_bad_numeric_flag_is_an_error_not_a_crash(tmp_path, flag, value):
    r = _run(
        [
            "--data-dir",
            str(tmp_path / "d"),
            "--listen",
            "127.0.0.1:0",
            "--insecure",
            "--no-auth",
            "--openfhe-version",
            VERSION,
            flag,
            value,
        ]
    )
    assert r.returncode == 2, r.stderr  # a clean exit code, not an abort from an exception
    assert "invalid value" in r.stderr


def test_rest_host_must_be_loopback_without_tls_or_auth(tmp_path):
    r = _run(
        [
            "--data-dir",
            str(tmp_path / "d"),
            "--listen",
            "127.0.0.1:0",
            "--insecure",
            "--no-auth",
            "--rest-port",
            "0",
            "--rest-host",
            "0.0.0.0",  # noqa: S104 - the test checks this public host is refused
            "--openfhe-version",
            VERSION,
        ]
    )
    assert r.returncode == 2
    assert "non-loopback" in r.stderr


def test_data_directory_is_private(tmp_path):
    data = tmp_path / "data"
    server = ec.ServerProcess(_binary(), data, VERSION, tmp_path / "server.log")
    try:
        assert stat.S_IMODE(data.stat().st_mode) == 0o700
        for sub in ("segments", "keys", "blobs"):
            assert stat.S_IMODE((data / sub).stat().st_mode) == 0o700
        for f in data.iterdir():
            if f.is_file():
                assert stat.S_IMODE(f.stat().st_mode) & 0o077 == 0, f.name
    finally:
        server.stop()


def test_idle_search_stream_is_cancelled(tmp_path):
    server = ec.ServerProcess(
        _binary(),
        tmp_path / "data",
        VERSION,
        tmp_path / "server.log",
        extra_args=("--search-idle-s", "1"),
    )
    pb, pbg = ec.load_stubs(tmp_path / "stubs")
    channel = grpc.insecure_channel(server.address)
    stub = pbg.OmbraStub(channel)
    stop = threading.Event()

    def requests():
        stop.wait(30)  # never sends a message
        yield from ()

    t0 = time.monotonic()
    try:
        with pytest.raises(grpc.RpcError):
            for _ in stub.Search(requests()):
                pass
        assert time.monotonic() - t0 < 20  # ended by the server's idle deadline, not our event
    finally:
        stop.set()
        channel.close()
        server.stop()


def test_large_unauthenticated_message_is_refused_with_unauthenticated(tmp_path):
    """With TLS and keys, a call without a key is refused before its body is processed."""
    binary = _binary()
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(  # noqa: S603, S607
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "2",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    data = tmp_path / "data"
    subprocess.run(  # noqa: S603
        [str(binary), "--create-api-key", "--scope", "read", "--data-dir", str(data)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    server = ec.ServerProcess(
        binary,
        data,
        VERSION,
        tmp_path / "server.log",
        extra_args=("--tls-cert", str(cert), "--tls-key", str(key)),
        secure=True,
    )
    pb, pbg = ec.load_stubs(tmp_path / "stubs")
    creds = grpc.ssl_channel_credentials(root_certificates=cert.read_bytes())
    channel = grpc.secure_channel(server.address, creds)
    stub = pbg.OmbraStub(channel)
    try:
        req = pb.GetCollectionRequest(name="x" * (32 << 20))  # 32 MiB of junk, no API key
        with pytest.raises(grpc.RpcError) as ei:
            stub.GetCollection(req, timeout=30)
        assert ei.value.code() == grpc.StatusCode.UNAUTHENTICATED
        assert os.path.exists(server.log_path)
    finally:
        channel.close()
        server.stop()
