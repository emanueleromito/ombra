import inspect

import ombra
import pytest
from ombra import AsyncClient, Client


def _public(cls: type) -> dict[str, inspect.Signature]:
    return {
        n: inspect.signature(f)
        for n, f in inspect.getmembers(cls, inspect.isfunction)
        if not n.startswith("_") or n == "__init__"
    }


def test_same_methods_and_signatures() -> None:
    sync, asy = _public(Client), _public(AsyncClient)
    assert sync.keys() == asy.keys()
    assert {"create_collection", "search", "insert", "close"} <= sync.keys()
    for name in sync:
        assert sync[name] == asy[name], name


def test_async_methods_are_coroutines() -> None:
    for name in _public(AsyncClient):
        if name != "__init__":
            assert inspect.iscoroutinefunction(getattr(AsyncClient, name)), name


def test_versions_and_stubs() -> None:
    assert ombra.API_VERSION == "ombra.v1"
    assert ombra.__version__


def test_local_constructor_on_both() -> None:
    for cls in (Client, AsyncClient):
        sig = inspect.signature(cls.local)
        assert list(sig.parameters) == ["path"]


def test_secure_by_default() -> None:
    """A remote address needs TLS (or an explicit insecure=True); an API key never goes plain."""
    pytest.importorskip("grpc")
    pytest.importorskip("grpc_tools")
    pytest.importorskip("openfhe")
    with pytest.raises(ValueError):
        Client("localhost:1", api_key="x", insecure=True)
