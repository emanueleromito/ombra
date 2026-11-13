# server/ rules

C++20. gRPC data API, REST health and metrics endpoints, storage, clustering metadata and GPU scheduling. Work here when a task asks for it.

## Design to follow

See `docs/architecture.md`: one server per machine (a router and several nodes are planned), clusters as the unit of distribution, segment files plus write-ahead log plus catalog database, a GPU memory cache, and one two-way gRPC stream per search.

## Rules

- **The server never sees plaintext.** It receives ciphertexts and public evaluation keys only. Never accept, store or log a secret key, plaintext vector, query, score or document.
- **Logs and metrics carry no data.** Log sizes, timings, IDs and errors; never ciphertext contents or anything derived from them.
- **The router never decrypts.** It only forwards and merges encrypted results.
- **Segment files are immutable.** Writes go to the write-ahead log, then new segments; deletes are marks plus compaction.
- **Versioned formats and API.** Every on-disk format has a version header; the API is versioned (`v1`, ...).
- **Memory safety:** RAII only, no raw `new`/`delete`; tests run under AddressSanitizer and UndefinedBehaviorSanitizer in CI.
