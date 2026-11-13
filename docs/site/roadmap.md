# Roadmap

This page lists what is not built yet. Nothing here works today, and nothing here has a date. The order can change, and an item can be dropped. If you need one of these now, use what the "Today" line says.

Items are in two groups. "Next" is work that has started or is small and agreed. "Later" is work we want but have not started.

## Next

**Limits on uploads before the server reads them.** The server accepts messages of up to 1 GiB. For a single request, gRPC reads the whole message before the server checks the API key, so a caller without a key can make the server hold a lot of memory. We plan a small receive limit for ordinary calls, kept apart from the limit for streaming calls, and a key check before the body is read. Today: run the server on a network where only your own clients can connect.

**Checks on uploaded contexts and columns.** The server rebuilds the encryption context from bytes the client sends. It checks the file header first and the security floor afterwards, and it does not compare the context with the collection's preset. A holder of a write key can also store a column that makes later searches of its cluster fail. We plan to check the size, the security floor and the preset before the server builds the context, and to check each column when it is inserted. Today: give write keys only to clients you trust.

**Shorter global locks in the storage layer.** The server holds some store locks while it writes to disk, so one slow writer can delay every other request. We plan to release them before the disk work. Today: this costs speed under many concurrent writers. It does not affect results.

**Bucketing of filter value counts in the client.** The server sees how many fields a filter uses and how many distinct values each field has. The client does not yet round those counts up to a few sizes, so the exact counts leak. [What the server learns](security/what-the-server-learns.md) lists them as they are. Today: keep the number of values per field in mind when you design your fields.

## Later

### Privacy

**Hiding the record fetch after a search.** After a search, the client fetches the best candidates by id, so the server sees which records matched. Strict mode hides which clusters a query asks for, but not this step. Hiding it needs private retrieval, which we have not designed for Ombra. It would cost more time and bandwidth. Today: use random ids, so the ids reveal nothing by themselves.

**Hiding which cluster each record is in.** On insert, the client sends the cluster of each vector in the clear, next to its id. The server therefore holds the full split of ids into clusters. No setting hides this today. We want an option that does.

**Key rotation.** A key cannot be replaced in place. We plan to let the server switch stored vectors to a new key without seeing them, and to re-wrap the document keys. Today: create a new collection with a new key and insert the vectors again from your source data.

**An external review of the cryptography.** No outside cryptographer has reviewed Ombra yet. One part needs it most: stored ciphertexts are seeded, and the noise used there depends on the secret key, unlike the textbook form. No attack is known. We plan a review of that step and of the rest of the design. Until then, treat the cryptography as unreviewed.

### Scale and operations

**Several nodes and a router.** One server runs on one machine, so a collection must fit there. We plan a router that sends each query to the nodes holding the clusters it needs and merges the encrypted results. Today: split your data across separate collections or servers in your own code.

**A REST admin API.** The REST port serves only a health check and metrics. Collections are managed through gRPC and the Python client. We plan REST calls for collection management.

**Other operations features.** Mutual TLS as an alternative to API keys, API keys limited to one collection, a Helm chart for Kubernetes, backup to object storage such as S3, and a scheduler that runs several queries in one pass over the stored data. Today: see [Backups](operations/backups.md) for how to back up the data directory.

**More platforms.** Windows client wheels and support for Blackwell GPUs. Today the client has no Windows build, and the server needs an Ampere GPU.

### Features

**Range filters.** Filters work on integer fields. Ranges over dates or numbers need encrypted comparisons, which we have not built. Today: turn a range into buckets, for example one field per month, and filter on the bucket.

**A "precise" preset.** A preset with higher precision. The name exists, but the client and the server refuse it. Today: use the presets listed in [Presets](reference/presets.md).

**Plaintext-database mode.** For a corpus that any client may read, the server could keep vectors in the clear and hide only the query and the results. It would be faster. It does not exist: every vector is always encrypted.

**A LlamaIndex adapter.** Only the LangChain adapter exists. Today: use the Python client directly.

**Automatic normalization.** The client would normalize vectors for you. Today: normalize them before you insert.

**Migration tools.** One tool would build a new collection next to the old one when you change embedding model, and switch over with no downtime. Another would convert files from older format versions. Today: create the new collection yourself and switch your application when it is ready.

### Housekeeping

**Our own k-means.** The client uses faiss for clustering, in a child process, because faiss and OpenFHE cannot share a process on macOS. We may replace it with our own k-means. That removes a dependency and the child process, but it changes the clusters, so we would measure recall again first.
