"""Protobuf (and gRPC) modules of ombra.v1, generated at first use from the packaged .proto.

The package ships `resources/ombra.proto`, a copy of proto/ombra/v1/ombra.proto (a test checks
that they are equal). The modules are generated with `grpc_tools` into a temporary directory, under
the Python package name `ombra_api` so they cannot collide with the client package `ombra`; the
wire names (package ombra.v1) are unchanged. Needs the `local` or `grpc` extra (grpcio-tools).
"""

from __future__ import annotations

import atexit
import importlib
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

_PROTO = Path(__file__).resolve().parent / "resources" / "ombra.proto"
_MODULES = ("ombra_api.v1.ombra_pb2", "ombra_api.v1.ombra_pb2_grpc")


def load_stubs(*, grpc_module: bool = True) -> tuple[Any, Any | None]:
    """Return (messages module, grpc module or None). Generated once per process."""
    if _MODULES[0] in sys.modules and (not grpc_module or _MODULES[1] in sys.modules):
        return sys.modules[_MODULES[0]], sys.modules.get(_MODULES[1])
    try:
        from grpc_tools import protoc
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "grpcio-tools is needed to build the API stubs: pip install 'ombra[local]' "
            "or 'ombra[grpc]'"
        ) from exc
    out = Path(tempfile.mkdtemp(prefix="ombra-proto-"))
    atexit.register(shutil.rmtree, out, ignore_errors=True)
    api = out / "ombra_api" / "v1"
    api.mkdir(parents=True)
    shutil.copy(_PROTO, api / "ombra.proto")
    args = ["protoc", f"-I{out}", f"--python_out={out}"]
    if grpc_module:
        args.append(f"--grpc_python_out={out}")
    code = protoc.main([*args, str(api / "ombra.proto")])
    if code != 0:
        raise RuntimeError("protoc failed on the packaged ombra.proto")
    sys.path.insert(0, str(out))
    pb = importlib.import_module(_MODULES[0])
    pbg = importlib.import_module(_MODULES[1]) if grpc_module else None
    return pb, pbg
