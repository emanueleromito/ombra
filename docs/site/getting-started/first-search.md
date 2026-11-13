# Your first search

The [quickstart](quickstart.md) ran one script. This page goes through it in order and adds filters, inserts and deletes, using `examples/filters_and_updates.py`. Everything here runs in local mode; it works the same against a server, as [Local mode or a server](local-or-server.md) shows.

## 1. Embed your texts

Ombra stores vectors, so you bring the embeddings. Any model works if it gives 384, 768 or 1536 dimensions, and the vectors must have unit length.

```python
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384 dimensions
vectors = model.encode(texts, normalize_embeddings=True).tolist()  # Ombra needs unit length
```

## 2. Create a collection

A collection is a set of vectors that you search together. Creating one does four things on your machine: it generates the keys, groups the vectors into clusters, encrypts them, and uploads the ciphertexts and the public evaluation keys. The secret key is written to the key file and never leaves it.

```python
client = Client.local(workdir / "server")  # a CPU server inside this process
client.create_collection(
    "notes",
    vectors,
    ids=[str(i) for i in range(len(texts))],
    key_file=workdir / "notes.key",  # holds the secret key: back it up, never share it
    passphrase="change this passphrase",
)
```

The passphrase protects the key file. Use a real one, and do not put it in code you share. The collection uses the "fast" preset in local mode and "balanced" against a server; [presets](../reference/presets.md) lists what each one sets. You cannot choose ring sizes or moduli, and none of the presets is below 128-bit security.

## 3. Search

```python
query = model.encode(["What did the central bank do?"], normalize_embeddings=True)[0].tolist()
for hit in client.search("notes", query, k=3):
    print(f"{hit.score:.3f}  {texts[int(hit.id)]}")  # the score is the exact cosine similarity
```

The client encrypts the query and sends it. The server scores it against every encrypted vector without decrypting anything, and returns encrypted scores. The client decrypts them, picks the top `k`, and re-ranks those candidates exactly. A hit has an `id` and a `score`. The text is not stored on the server in this example, so you look it up yourself by id.

## 4. Filter

Filters work on integer fields that you declare when you create the collection. Here `topic` takes the values 0, 1 and 2, and each vector carries one. The filter values are encrypted too, so the server does not learn them.

```python
client.create_collection(
    "notes",
    embed(texts),
    ids=[str(i) for i in range(len(texts))],
    key_file=key_file,
    passphrase="change this passphrase",
    metadata=[{"topic": t} for t in topics],
    filter_fields={"topic": 3},  # the field "topic" takes the values 0, 1 and 2
)
```

```python
show("only topic 2 (tech)", client.search("notes", query, k=2, filter={"topic": 2}))
```

Only vectors with `topic` equal to 2 can come back. [Filters](../how-it-works/filters.md) explains how this works on encrypted data and what it costs.

## 5. Insert and delete

```python
client.insert("notes", embed(texts[-1:]), ids=["6"], metadata=[{"topic": 1}])
```

```python
client.delete("notes", ["3"])
```

An insert encrypts the new vector on your machine and adds it to the collection. A delete marks the vector as removed, and it stops appearing in results at once. The space comes back later, when the collection is compacted. See [deletes](../how-it-works/deletes.md).

## 6. Come back later

The server folder holds the encrypted collection. To search it in a new process, you need that folder, the key file and the passphrase. Name the filter fields in the order you created them.

```python
client = Client.local(workdir / "server")
client.open_collection(
    "notes", key_file=key_file, passphrase="change this passphrase", filter_fields=["topic"]
)
```

## Your key file

The key file holds your secret key, protected by the passphrase. It also holds the key that protects stored documents.

Back it up, and remember the passphrase. If you lose either one, nobody can recover the data, you included. Never send the file to a server or commit it to a repository. The server receives only ciphertexts and public evaluation keys. Anyone who has both the file and the passphrase can read your data.

[Keys](../concepts/keys.md) explains who holds what, and [what the server learns](../security/what-the-server-learns.md) lists what a server can still see: the collection size, the number of queries and their timing, and, depending on the security level, which clusters a query touched.
