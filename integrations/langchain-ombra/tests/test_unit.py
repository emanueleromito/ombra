"""Unit tests of OmbraVectorStore on ombra.testing.FakeClient (no OpenFHE, nothing encrypted)."""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_ombra import OmbraVectorStore  # noqa: E402
from ombra.testing import FakeClient  # noqa: E402

EMB = DeterministicFakeEmbedding(size=384)
SCRYPT = (2**4, 8, 1)  # tiny: the fake client ignores it anyway


def make(tmp_path, **kw):
    return OmbraVectorStore(
        FakeClient("memory"),
        "docs",
        EMB,
        key_file=tmp_path / "k.key",
        passphrase="correct horse",
        scrypt_params=SCRYPT,
        **kw,
    )


def test_empty_store_returns_nothing(tmp_path):
    assert make(tmp_path).similarity_search("x") == []


def test_add_and_search_with_scores(tmp_path):
    s = make(tmp_path)
    ids = s.add_texts(["foo", "bar", "baz"], [{"n": 1}, {"n": 2}, {"n": 3}])
    assert len(ids) == 3
    res = s.similarity_search_with_score("bar", k=3)
    assert res[0][0] == Document(id=ids[1], page_content="bar", metadata={"n": 2})
    assert res[0][1] == pytest.approx(1.0)
    scores = [sc for _, sc in res]
    assert scores == sorted(scores, reverse=True)  # cosine, higher is better


def test_by_vector_matches_by_text(tmp_path):
    s = make(tmp_path)
    s.add_texts(["foo", "bar"])
    assert s.similarity_search_by_vector(EMB.embed_query("foo"), k=1)[0].page_content == "foo"


def test_existing_id_refused_and_kept_then_delete(tmp_path):
    s = make(tmp_path)
    s.add_texts(["foo", "bar"], ids=["1", "2"])
    with pytest.raises(ValueError):
        s.add_texts(["new foo"], ids=["1"])  # no replace in place; the old version stays
    assert sorted(d.page_content for d in s.similarity_search("foo", k=5)) == ["bar", "foo"]
    s.delete(["2", "missing"])
    assert [d.id for d in s.similarity_search("foo", k=5)] == ["1"]
    with pytest.raises(ValueError):
        s.delete()


def test_from_texts(tmp_path):
    s = OmbraVectorStore.from_texts(
        ["foo", "bar"],
        EMB,
        client=FakeClient("memory"),
        collection_name="c",
        key_file=tmp_path / "k.key",
        passphrase="pw",
        scrypt_params=SCRYPT,
    )
    assert s.similarity_search("foo", k=1)[0].page_content == "foo"


def test_filters_only_on_declared_fields(tmp_path):
    s = make(tmp_path, filter_fields={"topic": 3})
    s.add_texts(["a", "b", "c"], [{"topic": 0}, {"topic": 1}, {"topic": 1}])
    assert {d.page_content for d in s.similarity_search("a", k=5, filter={"topic": 1})} == {
        "b",
        "c",
    }
    with pytest.raises(ValueError, match="declared"):
        s.similarity_search("a", filter={"other": 1})
    with pytest.raises(ValueError, match="integer"):
        s.similarity_search("a", filter={"topic": "1"})


def test_reserved_metadata_key_and_repr(tmp_path):
    s = make(tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        s.add_texts(["a"], [{"_ombra_page_content": "x"}])
    assert "correct horse" not in repr(s)


def test_existing_key_file_is_never_overwritten(tmp_path):
    (tmp_path / "k.key").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        make(tmp_path).add_texts(["a"])
