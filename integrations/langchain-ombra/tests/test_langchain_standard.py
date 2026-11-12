# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""LangChain's own vector store integration tests, against Ombra's local mode (needs OpenFHE).

Run: pytest integrations -m openfhe   (slow: every test makes keys and each search takes seconds)

Skipped on purpose (see the properties below):
- every get_by_ids test, sync and async (6 cases, `has_get_by_ids = False`): the Ombra client has
  no call that fetches a stored vector or document by id, so `get_by_ids` is not implemented
  (it raises NotImplementedError). The tests that only need it to read documents back are
  skipped with it:
    test_get_by_ids, test_get_by_ids_missing, test_add_documents_documents,
    test_add_documents_with_existing_ids and their `_async` twins.
- test_add_documents_by_id_with_mutation, test_add_documents_with_ids_is_idempotent and their
  `_async` twins (4 cases, marked xfail strict: the suite allows nothing else): they add an id
  that is already stored. The store does not replace ids in place (a delete before the insert
  would lose the old version, because the server keeps a deleted id reserved until compaction
  and refuses the re-insert with "duplicate id"), so the server refuses the insert and the old
  version stays.
Adapted, not skipped: the standard embedding has 6 dimensions, which Ombra refuses (it supports
384, 768, 1536), so `get_embeddings` returns the same deterministic fake embedding at 384.
"""

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain_tests")
pytest.importorskip("openfhe")

import shutil  # noqa: E402
from collections.abc import Generator  # noqa: E402

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings  # noqa: E402
from langchain_core.vectorstores import VectorStore  # noqa: E402
from langchain_ombra import OmbraVectorStore  # noqa: E402
from langchain_tests.integration_tests import VectorStoreIntegrationTests  # noqa: E402
from ombra import Client  # noqa: E402

pytestmark = pytest.mark.openfhe

REUSE_REASON = "an id already stored is refused (no replace in place; deleted ids stay reserved)"


class TestOmbraVectorStoreLocal(VectorStoreIntegrationTests):
    @pytest.fixture
    def vectorstore(self, tmp_path) -> Generator[VectorStore, None, None]:
        store = OmbraVectorStore(
            Client.local(tmp_path / "server"),
            "standard",
            self.get_embeddings(),
            key_file=tmp_path / "standard.key",
            passphrase="test passphrase",
            scrypt_params=(2**4, 8, 1),  # tests only; the default is OWASP-level
        )
        try:
            yield store
        finally:
            store.close()
            shutil.rmtree(tmp_path, ignore_errors=True)  # local mode data is large

    @property
    def has_get_by_ids(self) -> bool:
        return False  # the client cannot fetch by id

    # The suite forbids overriding its tests except with xfail (test_no_overrides_DO_NOT_OVERRIDE).
    @pytest.mark.xfail(reason=REUSE_REASON, strict=True)
    def test_add_documents_by_id_with_mutation(self, vectorstore: VectorStore) -> None:
        super().test_add_documents_by_id_with_mutation(vectorstore)

    @pytest.mark.xfail(reason=REUSE_REASON, strict=True)
    async def test_add_documents_by_id_with_mutation_async(self, vectorstore: VectorStore) -> None:
        await super().test_add_documents_by_id_with_mutation_async(vectorstore)

    @pytest.mark.xfail(reason=REUSE_REASON, strict=True)
    def test_add_documents_with_ids_is_idempotent(self, vectorstore: VectorStore) -> None:
        super().test_add_documents_with_ids_is_idempotent(vectorstore)

    @pytest.mark.xfail(reason=REUSE_REASON, strict=True)
    async def test_add_documents_with_ids_is_idempotent_async(
        self, vectorstore: VectorStore
    ) -> None:
        await super().test_add_documents_with_ids_is_idempotent_async(vectorstore)

    @staticmethod
    def get_embeddings() -> Embeddings:
        return DeterministicFakeEmbedding(size=384)
