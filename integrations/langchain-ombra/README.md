# langchain-ombra

A LangChain `VectorStore` for Ombra. Texts, vectors and metadata are encrypted on your machine;
the server only computes on ciphertexts.

```python
store = OmbraVectorStore.from_texts(
    texts, embedding, client=Client.local("./ombra-data"),
    collection_name="docs", key_file="docs.key", passphrase="...",
)
store.similarity_search_with_score("query", k=4)
```

Run `examples/quickstart.py` for a complete example (local mode, no model download).

- **Score** is the exact cosine similarity, in [-1, 1], higher is better (not a distance).
- **Dimension** of the embedding must be 384, 768 or 1536.
- **Filters** are equality on integer fields declared with `filter_fields={"topic": 5}`; anything
  else raises `ValueError` (nothing is filtered on the client).
- **Key file:** back it up. Without it the data is lost. The passphrase is never shown in `repr`.
- **Not supported:** `get_by_ids`, MMR search, `delete()` without ids. Replacing an existing id can
  be refused by the server ("duplicate id") because deleted ids stay reserved until compaction.
- Async methods run the sync ones in a worker thread.
