"""T3.7: local mode runs the server's C++ expansion in a child process.

Checks that the child never imports openfhe-python and has the compiled extension loaded, that
the default preset is "fast", that envelopes of another OpenFHE version are refused, that keys of
another key tag are refused (the tag comes from the deserialized keys), and that the search result
sits at the level the server's search leaves it at (the expansion tree uses no level). Synthetic
data only. Needs `ombra._native` (built by `make test-openfhe` on Modal).
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("openfhe")
pytest.importorskip("grpc_tools")
pytest.importorskip("faiss")
pytest.importorskip("ombra._native", reason="the compiled extension is not built")

import ombra  # noqa: E402
from ombra import presets  # noqa: E402
from ombra._local import OPENFHE_VERSION, LocalTransport  # noqa: E402
from ombra.collection import Collection, CollectionError, KeyKit  # noqa: E402

pytestmark = pytest.mark.openfhe

D = 384
SERVER_RESULT_LEVEL = 1  # a fresh ciphertext at level 0, one rescale after the dot product


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _data(n=12, seed=3):
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal((n, D)))


def test_child_has_no_openfhe_wheel_and_default_preset_is_fast(tmp_path) -> None:
    x = _data()
    client = ombra.Client.local(tmp_path / "data")
    try:
        client.create_collection(
            "t",
            x,
            [f"v{i}" for i in range(len(x))],
            key_file=tmp_path / "t.key",
            passphrase="pw",  # noqa: S106
            scrypt_params=(2**4, 8, 1),
        )
        assert client.get_collection("t").preset == presets.derive("fast", D).preset
        mods = client._core._t.modules_loaded()
        assert mods == {"openfhe_loaded": False, "native_loaded": True}
        hits = client.search("t", x[4], k=3)
        assert hits[0].id == "v4" and abs(hits[0].score - 1.0) < 1e-5
    finally:
        client.close()


def test_other_openfhe_version_is_refused(tmp_path) -> None:
    transport = LocalTransport(tmp_path / "local")
    try:
        assert transport.openfhe_version == OPENFHE_VERSION
        x = _data()
        with pytest.raises(CollectionError) as err:
            Collection.create(
                transport.pb,
                transport.stub,
                "old",
                x,
                [b"v%d" % i for i in range(len(x))],
                preset="fast",
                openfhe_version="1.6.0",
            )
        assert "OpenFHE version" in err.value.message
        assert transport.stub.GetCollection(
            transport.pb.GetCollectionRequest(name="old")
        ).error.code  # the failed create was rolled back
    finally:
        transport.close()


def test_keys_of_another_key_tag_are_refused(tmp_path) -> None:
    transport = LocalTransport(tmp_path / "local")
    pb, stub = transport.pb, transport.stub
    try:
        derived = presets.derive("fast", D)
        mine = KeyKit(derived, transport.openfhe_version, D)
        other = KeyKit(derived, transport.openfhe_version, D)
        assert mine.key_tag != other.key_tag
        spec = pb.CollectionSpec(
            api_version=1,
            name="c",
            preset=derived.preset,
            dimension=D,
            key_tag=mine.key_tag,
            cluster_count=1,
        )
        assert stub.CreateCollection(pb.CreateCollectionRequest(spec=spec)).WhichOneof(
            "result"
        ) == ("spec")
        chunks = []
        for kind, data in other.key_files(pb).items():
            for off in range(0, len(data), 4_000_000):
                chunks.append(
                    pb.UploadKeysChunk(
                        api_version=1,
                        collection="c" if not chunks else "",
                        kind=kind,
                        offset=off,
                        data=data[off : off + 4_000_000],
                        last=off + 4_000_000 >= len(data),
                    )
                )
        resp = stub.UploadKeys(iter(chunks))
        assert resp.error.code == pb.ERROR_KEY_TAG_MISMATCH
    finally:
        transport.close()


def test_search_result_level_and_no_plaintext_in_the_child(tmp_path) -> None:
    """The score blocks come back at the server's level: the expansion used no level."""
    import openfhe as ofhe
    from ombra.file_format import Kind, read_envelope

    transport = LocalTransport(tmp_path / "local")
    pb, stub = transport.pb, transport.stub
    try:
        x = _data()
        col = Collection.create(
            pb,
            stub,
            "lv",
            x,
            [b"v%d" % i for i in range(len(x))],
            preset="fast",
            openfhe_version=transport.openfhe_version,
        )
        start = pb.SearchStart(
            api_version=1,
            collection="lv",
            key_tag=col.kit.key_tag,
            query=col.kit.encrypt_query(x[2]),
            filter_leaves=0,
        )
        stream = stub.Search(iter([pb.SearchRequest(start=start)]))
        centers = next(stream).centers
        stream.close()
        env = read_envelope(
            bytes(centers.score_blocks[0]), Kind.CIPHERTEXT, "m0_compact_n8192", "1.6.1"
        )
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "c.bin"
            p.write_bytes(env.payload)
            ct, ok = ofhe.DeserializeCiphertext(str(p), ofhe.BINARY)
        assert ok and ct.GetLevel() == SERVER_RESULT_LEVEL
        col.close()
    finally:
        transport.close()
