"""T2.9: TLS, API keys with scopes and the REST admin port of `ombra_server`.

The server runs with TLS (a self-signed certificate made with the `openssl` CLI in tmp_path) and
three API keys made with `ombra_server --create-api-key`. Cases: no key -> UNAUTHENTICATED; a read
key on Insert and a write key on DropCollection -> PERMISSION_DENIED; an admin key works; REST
/healthz needs no key, /metrics needs an admin key (401 without, 403 with a lower scope) and its
JSON uses only an allow-list of keys and holds none of the synthetic names or ids used here; a
plain-text gRPC channel and a plain `http://` request fail when TLS is on; the server refuses to
start without TLS or without keys unless --insecure / --no-auth are given. Data is synthetic.
"""

from __future__ import annotations

import http.client
import json
import os
import ssl
import stat
import subprocess
import urllib.error
import urllib.request

import pytest

pytest.importorskip("numpy")
grpc = pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")
pytest.importorskip("openfhe")

import e2e_client as ec  # noqa: E402

pytestmark = pytest.mark.gpu

VERSION = "test-openfhe"
PRESET = "m0_compact_n8192"
COLLECTION = "synthetic-collection-name"
SECRET_ID = b"synthetic-id-7"

# Documented keys of GET /metrics (server/grpc/include/ombra/server/metrics.hpp).
TOP_KEYS = {
    "collections",
    "vectors_live",
    "vectors_deleted",
    "blocks",
    "rpc",
    "auth",
    "gpu_memory",
    "cache",
}
RPC_NAMES = {
    "CreateCollection",
    "GetCollection",
    "ListCollections",
    "DropCollection",
    "UploadKeys",
    "Insert",
    "Delete",
    "Search",
    "ListBlocks",
    "GetBlock",
    "ReplaceBlock",
}
RPC_KEYS = {"requests", "errors", "latency_ms"}
LATENCY_KEYS = {"sum", "count", "max"}
AUTH_KEYS = {"unauthenticated", "permission_denied"}
GPU_KEYS = {"used_bytes", "total_bytes"}
CACHE_KEYS = {"hits", "misses", "evictions", "hit_rate"}


def _binary():
    b = ec.find_server_binary()
    if b is None:
        pytest.fail("ombra_server not found; build with `make build` or `make test-gpu`")
    return b


def _make_key(binary, data_dir, scope: str) -> str:
    out = subprocess.run(  # noqa: S603
        [str(binary), "--create-api-key", "--scope", scope, "--data-dir", str(data_dir)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    token = out.stdout.strip()
    assert len(token) == 64
    assert token not in out.stderr  # shown once, on stdout only
    return token


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("auth")
    binary = _binary()
    cert, key = tmp / "cert.pem", tmp / "key.pem"
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
    data = tmp / "data"
    tokens = {s: _make_key(binary, data, s) for s in ("read", "write", "admin")}
    keyfile = data / "api_keys.v1"
    assert stat.S_IMODE(keyfile.stat().st_mode) == 0o600
    text = keyfile.read_text()
    assert text.startswith("OMBRA-API-KEYS v1\n")
    assert not any(t in text for t in tokens.values())  # only salted hashes on disk

    server = ec.ServerProcess(
        binary,
        data,
        VERSION,
        tmp / "server.log",
        extra_args=(
            "--tls-cert",
            str(cert),
            "--tls-key",
            str(key),
            "--rest-port",
            "0",
        ),
        secure=True,
    )
    pb, pbg = ec.load_stubs(tmp / "stubs")
    root = cert.read_bytes()
    clients = {
        name: ec.OmbraClient(pb, pbg, server.address, PRESET, VERSION, root, tok)
        for name, tok in tokens.items()
    }
    clients["none"] = ec.OmbraClient(pb, pbg, server.address, PRESET, VERSION, root, None)
    try:
        yield {
            "server": server,
            "clients": clients,
            "tokens": tokens,
            "pb": pb,
            "pbg": pbg,
            "cert": cert,
            "tmp": tmp,
            "binary": binary,
        }
    finally:
        server.stop()


def _code(exc_info) -> grpc.StatusCode:
    return exc_info.value.code()


def _rest(dep, path: str, token: str | None):
    ctx = ssl.create_default_context(cafile=str(dep["cert"]))
    req = urllib.request.Request(f"https://{dep['server'].rest_address}{path}")  # noqa: S310
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as r:  # noqa: S310
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_grpc_scopes(deployment):
    c, pb = deployment["clients"], deployment["pb"]
    list_req = pb.ListCollectionsRequest()

    with pytest.raises(grpc.RpcError) as ei:
        c["none"].stub.ListCollections(list_req)
    assert _code(ei) == grpc.StatusCode.UNAUTHENTICATED

    bad = ec.OmbraClient(
        pb,
        deployment["pbg"],
        deployment["server"].address,
        PRESET,
        VERSION,
        deployment["cert"].read_bytes(),
        "0" * 64,
    )
    with pytest.raises(grpc.RpcError) as ei:
        bad.stub.ListCollections(list_req)
    assert _code(ei) == grpc.StatusCode.UNAUTHENTICATED

    c["read"].stub.ListCollections(list_req)  # read works
    block = pb.InsertBlock(api_version=1, collection=COLLECTION, cluster_id=0, ids=[SECRET_ID])
    with pytest.raises(grpc.RpcError) as ei:
        c["read"].stub.Insert(iter([block]))
    assert _code(ei) == grpc.StatusCode.PERMISSION_DENIED
    with pytest.raises(grpc.RpcError) as ei:
        c["read"].create_collection(c["read"].make_spec(COLLECTION, 16, [3, 4], 6, "tag"))
    assert _code(ei) == grpc.StatusCode.PERMISSION_DENIED

    # write: create works, drop (admin) is refused
    c["write"].create_collection(c["write"].make_spec(COLLECTION, 16, [3, 4], 6, "tag"))
    c["write"].delete(COLLECTION, [SECRET_ID])
    with pytest.raises(grpc.RpcError) as ei:
        c["write"].stub.DropCollection(pb.DropCollectionRequest(name=COLLECTION))
    assert _code(ei) == grpc.StatusCode.PERMISSION_DENIED
    c["read"].get_collection(COLLECTION)

    # admin may do everything; the throwaway collection is dropped
    c["admin"].create_collection(c["admin"].make_spec("other", 16, [3, 4], 6, "tag"))
    c["admin"].drop_collection("other")


def test_rest_healthz_and_metrics(deployment):
    c = deployment["clients"]
    tok = deployment["tokens"]
    assert _rest(deployment, "/healthz", None) == (200, b'{"status":"ok"}')

    assert _rest(deployment, "/metrics", None)[0] == 401
    assert _rest(deployment, "/metrics", "0" * 64)[0] == 401
    assert _rest(deployment, "/metrics", tok["read"])[0] == 403
    assert _rest(deployment, "/metrics", tok["write"])[0] == 403
    assert _rest(deployment, "/nope", tok["admin"])[0] == 404

    # make sure the collection (with its name and an id in a request) exists before scraping
    try:
        c["admin"].create_collection(c["admin"].make_spec(COLLECTION, 16, [3, 4], 6, "tag"))
    except ec.ApiError:
        pass  # exists from test_grpc_scopes
    c["admin"].delete(COLLECTION, [SECRET_ID])

    status, body = _rest(deployment, "/metrics", tok["admin"])
    assert status == 200
    text = body.decode()
    assert COLLECTION not in text and SECRET_ID.decode() not in text
    m = json.loads(text)
    assert set(m) == TOP_KEYS
    assert set(m["rpc"]) == RPC_NAMES
    for v in m["rpc"].values():
        assert set(v) == RPC_KEYS and set(v["latency_ms"]) == LATENCY_KEYS
    assert set(m["auth"]) == AUTH_KEYS and set(m["gpu_memory"]) == GPU_KEYS
    assert set(m["cache"]) == CACHE_KEYS
    assert m["collections"] == 1
    assert m["auth"]["unauthenticated"] >= 2 and m["auth"]["permission_denied"] >= 1
    assert m["rpc"]["Delete"]["requests"] >= 2
    assert m["gpu_memory"]["total_bytes"] > 0


def test_rest_limits(deployment):
    host, port = deployment["server"].rest_address.rsplit(":", 1)
    ctx = ssl.create_default_context(cafile=str(deployment["cert"]))
    conn = http.client.HTTPSConnection(host, int(port), context=ctx, timeout=10)
    # The server answers 431 and closes with unread bytes pending, so the client may instead see
    # the connection reset before it reads the answer; either way the request is refused.
    try:
        conn.request("GET", "/healthz", headers={"X-Pad": "a" * 20000})
        assert conn.getresponse().status == 431
    except (OSError, http.client.HTTPException):
        pass
    finally:
        conn.close()


def test_plain_text_refused_when_tls_is_on(deployment):
    pb, server = deployment["pb"], deployment["server"]
    plain = grpc.insecure_channel(server.address)
    stub = deployment["pbg"].OmbraStub(plain)
    with pytest.raises(grpc.RpcError) as ei:
        stub.ListCollections(pb.ListCollectionsRequest(), timeout=10)
    assert _code(ei) == grpc.StatusCode.UNAVAILABLE
    plain.close()

    with pytest.raises((OSError, http.client.HTTPException)):
        urllib.request.urlopen(f"http://{server.rest_address}/healthz", timeout=10)  # noqa: S310


def test_refuses_unsafe_start(deployment, tmp_path):
    binary, tmp = deployment["binary"], deployment["tmp"]
    common = ["--listen", "127.0.0.1:0", "--openfhe-version", VERSION]

    def run(data_dir, *args):
        return subprocess.run(  # noqa: S603
            [str(binary), "--data-dir", str(data_dir), *common, *args],
            capture_output=True,
            text=True,
            timeout=60,
        )

    # no TLS and no --insecure
    r = run(tmp / "data2", "--no-auth")
    assert r.returncode == 2 and "TLS" in r.stderr
    # TLS but no key file and no --no-auth
    r = run(
        tmp_path / "empty",
        "--tls-cert",
        str(deployment["cert"]),
        "--tls-key",
        str(tmp / "key.pem"),
    )
    assert r.returncode == 2 and "API key" in r.stderr
    # a key file readable by others is refused
    os.chmod(tmp / "data" / "api_keys.v1", 0o644)
    try:
        r = run(
            tmp / "data",
            "--tls-cert",
            str(deployment["cert"]),
            "--tls-key",
            str(tmp / "key.pem"),
        )
        assert r.returncode == 2
    finally:
        os.chmod(tmp / "data" / "api_keys.v1", 0o600)
