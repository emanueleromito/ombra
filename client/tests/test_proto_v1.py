# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""The v1 proto compiles for Python and C++; messages round-trip; unknown versions refused.

Uses grpcio-tools' bundled protoc. The C++ check generates the C++ sources; building them needs
the protobuf C++ library, which arrives with the server. Synthetic data only.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

grpc_tools = pytest.importorskip("grpc_tools.protoc")
pytest.importorskip("google.protobuf")

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "proto" / "ombra" / "v1" / "ombra.proto"


def _protoc(out: Path, *flags: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    args = ["protoc", f"-I{ROOT / 'proto'}", *flags, str(PROTO)]
    code = grpc_tools.main([a.replace("OUT", str(out)) for a in args])
    assert code == 0


@pytest.fixture(scope="module")
def pb():
    """The messages module the client uses (built from the packaged copy of the proto, which a
    test keeps equal to proto/). One process can hold one copy of ombra.v1 in the protobuf pool."""
    from ombra._proto import load_stubs

    return load_stubs(grpc_module=False)[0]


def test_parses_with_descriptor_set(tmp_path: Path) -> None:
    """Syntax and imports check with the bundled protoc (it has no C++ generator)."""
    _protoc(tmp_path, f"--descriptor_set_out={tmp_path / 'ombra.desc'}")
    assert (tmp_path / "ombra.desc").stat().st_size > 0


@pytest.mark.skipif(shutil.which("protoc") is None, reason="system protoc not installed")
def test_compiles_for_cpp(tmp_path: Path) -> None:
    proc = subprocess.run(  # noqa: S603
        ["protoc", f"-I{ROOT / 'proto'}", f"--cpp_out={tmp_path}", str(PROTO)],  # noqa: S607
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "ombra" / "v1" / "ombra.pb.h").is_file()
    assert (tmp_path / "ombra" / "v1" / "ombra.pb.cc").is_file()


def test_package_is_versioned(pb) -> None:
    assert pb.DESCRIPTOR.package == "ombra.v1"
    assert pb.Limit.Value("LIMIT_API_VERSION") == 1


def test_v1_message_round_trips(pb) -> None:
    spec = pb.CollectionSpec(
        api_version=1,
        name="docs",
        preset="m0_compact_n8192",
        dimension=768,
        security_level=pb.SECURITY_LEVEL_DECOY,
        filter_fields=[pb.FilterField(field_id=0, cardinality=16)],
        key_tag="tag",
        cluster_count=64,
    )
    again = pb.CollectionSpec.FromString(spec.SerializeToString())
    assert again == spec
    req = pb.SearchRequest(start=pb.SearchStart(collection="docs", key_tag="tag", query=b"x"))
    assert pb.SearchRequest.FromString(req.SerializeToString()).WhichOneof("step") == "start"


def _validate(pb, msg) -> pb.ErrorCode:
    """What a server does first with a stored message: refuse an unknown version."""
    if msg.api_version != pb.LIMIT_API_VERSION:
        return pb.ERROR_UNSUPPORTED_VERSION
    return pb.ERROR_UNSPECIFIED


def test_unknown_version_is_refused(pb) -> None:
    for cls in (
        pb.CollectionSpec,
        pb.UploadKeysChunk,
        pb.InsertBlock,
        pb.DeleteRequest,
        pb.ListBlocksRequest,
        pb.GetBlockRequest,
        pb.ReplaceBlockRequest,
    ):
        assert _validate(pb, cls(api_version=2)) == pb.ERROR_UNSUPPORTED_VERSION, cls
        assert _validate(pb, cls(api_version=1)) == pb.ERROR_UNSPECIFIED, cls
        # An unset version (a v0 client) is refused too.
        assert _validate(pb, cls()) == pb.ERROR_UNSUPPORTED_VERSION, cls


def test_every_stored_message_has_api_version(pb) -> None:
    for name in (
        "CollectionSpec",
        "UploadKeysChunk",
        "InsertBlock",
        "DeleteRequest",
        "ListBlocksRequest",
        "ListBlocksResponse",
        "GetBlockRequest",
        "GetBlockResponse",
        "ReplaceBlockRequest",
        "ReplaceBlockResponse",
    ):
        assert "api_version" in getattr(pb, name).DESCRIPTOR.fields_by_name, name


def test_no_secret_key_field_anywhere(pb) -> None:
    """the secret key stays on the client: no field name suggests a secret key or plaintext."""
    bad = re.compile(r"secret|private|plaintext|password|passphrase", re.I)
    for msg in pb.DESCRIPTOR.message_types_by_name.values():
        for field in msg.fields:
            assert not bad.search(field.name), (msg.name, field.name)


def test_block_messages_carry_counts_only(pb) -> None:
    """ListBlocks exposes counts and the segment name, no ids (no plaintext in logs or errors)."""
    fields = set(pb.BlockInfo.DESCRIPTOR.fields_by_name)
    assert fields == {"segment", "cluster_id", "n_vectors", "n_deleted"}
    assert pb.ERROR_BLOCK_NOT_FOUND == 12 and pb.ERROR_BLOCK_MISMATCH == 13
