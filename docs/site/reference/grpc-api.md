# gRPC API

The server speaks gRPC, package `ombra.v1`. The definition is `client/ombra/resources/ombra.proto`, and the server builds from the same file (`proto/ombra/v1/ombra.proto` is an identical copy). The Python client wraps all of this, so you only need this page to write your own client or to debug one.

What the API never carries: the secret key, plaintext vectors, queries, documents or metadata values. Ids are opaque bytes chosen by the client. The server receives ciphertexts inside the [versioned envelope](file-formats.md) and public evaluation keys only.

## Is there a REST API?

There is no REST data API. Everything that reads or writes vectors, blobs or collections goes over gRPC. The server has a small HTTP port for operators, turned on with `--rest-port` (the Docker image turns it on at 8080). It answers `GET` only and has two paths. `/healthz` needs no key and returns `{"status":"ok"}`. `/metrics` needs an admin API key and returns counters as JSON. See [Monitoring](../operations/monitoring.md). Any other path returns 404 and any other method 405.

## Connecting and authenticating

Use TLS. The server refuses to start without it unless it is started with `--insecure`, and then it only accepts a loopback address. Send the API key as call metadata: `authorization: Bearer <key>`. A missing or wrong key fails with `UNAUTHENTICATED`. A key whose scope is too low fails with `PERMISSION_DENIED`. A server started with `--no-auth` skips the check and is also limited to loopback.

An API key has one scope. `read` may call the read methods, `write` may also call the write methods, and `admin` may call everything. The table below gives the scope each method needs. Messages may be up to 1 GiB in both directions, since one block can be tens of megabytes.

## Methods

The service is `Ombra`. A method marked "stream" sends or receives a sequence of messages.

| Method | Scope | Request and response | What it does |
| --- | --- | --- | --- |
| `CreateCollection` | write | `CreateCollectionRequest` to `CreateCollectionResponse` | Creates a collection from a `CollectionSpec` and returns the stored spec. |
| `GetCollection` | read | `GetCollectionRequest` to `GetCollectionResponse` | Returns the `CollectionInfo`: spec, vector counts, and whether evaluation keys are uploaded. |
| `ListCollections` | read | `ListCollectionsRequest` to `ListCollectionsResponse` | Returns the `CollectionInfo` of every collection. |
| `DropCollection` | admin | `DropCollectionRequest` to `DropCollectionResponse` | Deletes a collection and everything stored for it. The error field is unset on success. |
| `UploadKeys` | write | stream `UploadKeysChunk` to `UploadKeysResponse` | Uploads the public evaluation keys in chunks. `kind` says which file the bytes belong to. |
| `Insert` | write | stream `InsertBlock` to `InsertResponse` | Inserts blocks of vectors, one cluster per block. |
| `Delete` | write | `DeleteRequest` to `DeleteResponse` | Marks the given ids deleted. |
| `Search` | read | stream `SearchRequest` to stream `SearchResponse` | One bidirectional stream per query, in two rounds. See below. |
| `ListBlocks` | read | `ListBlocksRequest` to `ListBlocksResponse` | Lists the blocks (storage segments) of a collection with their vector and deleted counts. No ids. |
| `GetBlock` | read | `GetBlockRequest` to stream `GetBlockResponse` | Streams one block: a header, then its column ciphertexts in order. The client uses it to repack. |
| `ReplaceBlock` | write | stream `ReplaceBlockRequest` to `ReplaceBlockResponse` | Replaces one or more blocks of one cluster with one new block. |
| `PutBlob` | write | stream `PutBlobRequest` to `PutBlobResponse` | Stores an encrypted blob under (collection, kind, id), replacing any blob with that key. |
| `GetBlob` | read | `GetBlobRequest` to stream `GetBlobResponse` | Streams a blob: its size, then data chunks. |
| `DeleteBlobs` | write | `DeleteBlobsRequest` to `DeleteBlobsResponse` | Deletes blobs by id. A missing id is not an error. |

Responses report most failures in an `Error` field (code and message) instead of a gRPC status. The message never contains ids, ciphertext bytes or values.

### Search

A search is one bidirectional stream, so the query ciphertext is sent and expanded once.

1. The client sends a `SearchStart` with the collection, the key tag, the query as one envelope and `filter_leaves`, the sum of the filter fields' cardinalities. The server checks `filter_leaves` against the collection's spec.
2. The server answers with `SearchCenterScores`: one score ciphertext per block of cluster centers. The client decrypts these and picks clusters.
3. The client sends `SearchChosen` with the cluster ids, real and decoy ones mixed so the server cannot tell them apart. The server answers with `SearchScores` messages, the last one with `last` set. Each score slot holds `score + lambda * c`, where `c` counts the filter fields a vector fails. A selective filter may need more clusters. The client can then send further `SearchChosen` messages on the same stream, and each reuses the expansion.
4. The client half-closes the stream to end it.

`SearchScores.ids` lists the ids in slot order, tombstoned slots included, so the client drops the deleted ones.

## Limits

These are the values of the `Limit` enum.

| Limit | Value |
| --- | --- |
| `LIMIT_API_VERSION` | 1 |
| `LIMIT_MAX_CHUNK_BYTES` | 4194304 (one `bytes` payload of a chunk) |
| `LIMIT_MAX_NAME_BYTES` | 128 (collection name) |
| `LIMIT_MAX_ID_BYTES` | 256 (one vector id) |
| `LIMIT_MAX_IDS_PER_REQUEST` | 100000 (delete ids) |
| `LIMIT_MAX_FILTER_FIELDS` | 16 (a preset's headroom may lower it, see [Presets](presets.md)) |
| `LIMIT_MAX_FILTER_CARDINALITY` | 64 values per field |
| `LIMIT_MAX_CLUSTERS_PER_ROUND` | 4096 |
| `LIMIT_BLOB_BUCKET_0` to `LIMIT_BLOB_BUCKET_3` | 16384, 262144, 2097152 and 16777216 bytes |

A stored blob has exactly one bucket size, so the server learns the bucket of a document and nothing finer.

## Messages

Every message that the server stores carries `api_version`, which must be 1. Any other value fails with `ERROR_UNSUPPORTED_VERSION`. An incompatible API will be a new package, `ombra.v2`, never a changed field.

### Collections

| Message | Fields |
| --- | --- |
| `FilterField` | `field_id` (0, 1, ... in order, not a name), `cardinality` (1 to 64). The server sees the count of fields and values, never names. |
| `CollectionSpec` | `api_version`, `name`, `preset`, `dimension`, `security_level`, `filter_fields`, `key_tag` (a public tag of the client's key), `cluster_count`. |
| `CollectionInfo` | `spec`, `vector_count`, `deleted_count`, `keys_uploaded`. |
| `CreateCollectionRequest`, `CreateCollectionResponse` | A `spec` in. The stored `spec` or an `Error` out. |
| `GetCollectionRequest`, `GetCollectionResponse` | A `name` in. An `info` or an `Error` out. |
| `ListCollectionsRequest`, `ListCollectionsResponse` | Empty in. Repeated `collections` out. |
| `DropCollectionRequest`, `DropCollectionResponse` | A `name` in. An optional `Error` out. |

The query ciphertext of a collection carries `dimension + sum(cardinality)` entries.

### Keys

| Message | Fields |
| --- | --- |
| `UploadKeysChunk` | `api_version`, `collection` (required on the first chunk), `kind`, `offset` (byte offset in this key file), `data` (at most `LIMIT_MAX_CHUNK_BYTES`), `last`. |
| `UploadKeysResponse` | An optional `Error`. |

`KeyKind` values: `KEY_KIND_CONTEXT`, `KEY_KIND_EVAL_MULT` and `KEY_KIND_AUTOMORPHISM`. There is no kind for a secret key, and none will be added.

### Insert and delete

| Message | Fields |
| --- | --- |
| `InsertBlock` | `api_version`, `collection`, `cluster_id`, `ids` (one per occupied slot, in slot order), `columns` (`dimension + sum(cardinality)` envelopes in column order), `is_center` (a cluster-center block for stage 1 instead of data). |
| `InsertResponse` | `inserted` count, optional `Error`. |
| `DeleteRequest` | `api_version`, `collection`, `ids`. |
| `DeleteResponse` | `deleted` count, optional `Error`. |

### Search messages

| Message | Fields |
| --- | --- |
| `SearchRequest` | One of `start` (`SearchStart`) or `chosen` (`SearchChosen`). |
| `SearchStart` | `collection`, `key_tag`, `query`, `filter_leaves`. |
| `SearchChosen` | `cluster_ids`, at most `LIMIT_MAX_CLUSTERS_PER_ROUND`. |
| `SearchResponse` | One of `centers` (`SearchCenterScores`), `scores` (`SearchScores`) or `error`. |
| `SearchCenterScores` | `score_blocks` (envelopes, each decrypts to one score per slot), `center_count`. |
| `SearchScores` | `cluster_id`, `score_blocks`, `ids`, `last`. |

`SecurityLevel` values: `SECURITY_LEVEL_FAST`, `SECURITY_LEVEL_DECOY` and `SECURITY_LEVEL_STRICT`. See [Clusters](../how-it-works/clusters.md) for what each one does.

### Blocks

A deleted vector is one slot in every column of its block, so the server cannot drop single slots. It drops blocks whose vectors are all deleted. For blocks with some deleted slots the client repacks: `GetBlock`, decrypt, drop the deleted slots, re-encrypt, `ReplaceBlock`. The server never sees keys or plaintext in this flow.

| Message | Fields |
| --- | --- |
| `BlockInfo` | `segment` (opaque name, stable until the block is replaced or dropped), `cluster_id` (equal to `cluster_count` for center blocks), `n_vectors` (deleted ones included), `n_deleted`. |
| `ListBlocksRequest` | `api_version`, `collection`. |
| `ListBlocksResponse` | `api_version`, `blocks`, optional `Error`. |
| `GetBlockRequest` | `api_version`, `collection`, `segment`. |
| `BlockHeader` | `cluster_id`, `ids` in slot order, `deleted` (parallel to `ids`), `n_columns`. |
| `GetBlockResponse` | `api_version`, then one of `header` (first), `column` (an envelope, `n_columns` of them) or `error`. |
| `ReplaceBlockHeader` | `collection`, `segments` (one or more blocks of one data cluster), `ids` (exactly the union of their live ids, in the new slot order, at most one block of slots). |
| `ReplaceBlockRequest` | `api_version`, then one of `header` (first) or `column` (in column order). |
| `ReplaceBlockResponse` | `api_version`, `segment` (the new block), optional `Error`. |

N old blocks become one new block, so storage falls by N minus 1 blocks.

### Blobs

Blobs are opaque client ciphertexts. The server never parses them. A blob becomes visible only after it is fully written, synced and committed in the catalog, so a crash leaves the old blob or the new one, never a part.

| Message | Fields |
| --- | --- |
| `BlobKind` | `BLOB_KIND_RECORD` (per-vector record) and `BLOB_KIND_DOCUMENT` (document body or encrypted reference). |
| `PutBlobHeader` | `collection`, `kind`, `id` (at most 256 bytes), `size` (must be a bucket size). |
| `PutBlobRequest` | `api_version`, then one of `header` (first) or `data` (chunks of at most 4 MiB summing to `size`). |
| `PutBlobResponse` | `api_version`, optional `Error`. |
| `GetBlobRequest` | `api_version`, `collection`, `kind`, `id`. |
| `GetBlobResponse` | `api_version`, then one of `size` (first), `data` or `error`. |
| `DeleteBlobsRequest` | `api_version`, `collection`, `kind`, `ids` (at most 100000). |
| `DeleteBlobsResponse` | `deleted` count, optional `Error`. |

## Errors

| Code | Meaning |
| --- | --- |
| `ERROR_UNSUPPORTED_VERSION` | `api_version` or the envelope version is not understood. |
| `ERROR_COLLECTION_NOT_FOUND` | No collection with that name. |
| `ERROR_COLLECTION_EXISTS` | A collection with that name exists. |
| `ERROR_INVALID_ARGUMENT` | Malformed message or an empty required field. |
| `ERROR_SIZE_LIMIT` | A limit above was exceeded, or a blob is not a bucket size. |
| `ERROR_KEY_TAG_MISMATCH` | A ciphertext or key was made under another key than the collection's. |
| `ERROR_PRESET_MISMATCH` | The envelope's preset differs from the collection's. |
| `ERROR_INVALID_FILTER_LAYOUT` | More filter fields than the limits or the preset's headroom allow. |
| `ERROR_KEYS_MISSING` | A search or insert before the evaluation keys were uploaded. |
| `ERROR_BAD_SEQUENCE` | Stream messages arrived out of order. |
| `ERROR_INTERNAL` | A server fault. |
| `ERROR_BLOCK_NOT_FOUND` | No such block in the collection. |
| `ERROR_BLOCK_MISMATCH` | The ids of a `ReplaceBlock` are not exactly the live ids, or the cluster or columns differ. |
| `ERROR_BLOB_NOT_FOUND` | No blob with this kind and id in the collection. |
