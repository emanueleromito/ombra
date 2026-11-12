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


def test_fake_client_has_the_client_signatures() -> None:
    from ombra.testing import FakeClient

    real, fake = _public(Client), _public(FakeClient)
    assert real.keys() == fake.keys()
    for name in real:
        assert real[name].parameters == fake[name].parameters, name


def test_fake_client_round_trip() -> None:
    import numpy as np
    from ombra.testing import FakeClient

    x = np.eye(4)
    c = FakeClient("memory")
    c.create_collection(
        "t",
        x,
        ["a", "b", "c", "d"],
        key_file="unused",
        passphrase="x",  # noqa: S106
        metadata=[{"g": 0}, {"g": 1}, {"g": 0}, {"g": 1}],
        documents=["A", "B", "C", "D"],
    )
    assert [h.id for h in c.search("t", x[2], k=2)][0] == "c"
    assert [h.id for h in c.search("t", x[2], k=4, filter={"g": 1})] == ["b", "d"]
    c.delete("t", ["c"])
    assert c.get_collection("t").vector_count == 3


def test_documents_in_the_api() -> None:
    for cls in (Client, AsyncClient):
        assert "documents" in inspect.signature(cls.create_collection).parameters
        assert "documents" in inspect.signature(cls.insert).parameters
        assert inspect.signature(cls.create_collection).parameters["preset"].default is None
