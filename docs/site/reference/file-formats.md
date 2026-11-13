# File formats

Every file Ombra writes starts with a magic string and a version number, so a program can tell what it is reading and refuse a version it does not know. All integers are little-endian. A reader never guesses: it refuses bad magic, an unknown version, a wrong kind, truncation and trailing bytes.

| Format | Written by | Magic | Version |
| --- | --- | --- | --- |
| [Envelope](#envelope) | client, server | `{{ const("ombra.file_format.MAGIC").decode() }}` | {{ const("ombra.file_format.FORMAT_VERSION") }}, and {{ const("ombra.file_format.FORMAT_VERSION_SEEDED") }} for seeded ciphertexts |
| [Key file](#key-file) | client | `{{ const("ombra.keyfile.MAGIC").decode() }}` | {{ const("ombra.keyfile.KEYFILE_VERSION") }} |
| [Blob](#blobs) | client | `{{ const("ombra.documents.MAGIC").decode() }}` | {{ const("ombra.documents.BLOB_VERSION") }} |
| [Segment](#server-files) | server | `OSEG` | {{ cpp("server/storage/include/ombra/storage/segment.hpp", "kSegmentVersion") }} |
| [Stored blob](#server-files) | server | `OBLB` | {{ cpp("server/storage/include/ombra/storage/blob.hpp", "kBlobVersion") }} |
| [Write-ahead log](#server-files) | server | `OWAL` | {{ cpp("server/storage/include/ombra/storage/wal.hpp", "kWalVersion") }} |
| [Catalog](#server-files) | server | SQLite file, `schema_version` in its `meta` table | {{ cpp("server/storage/include/ombra/storage/catalog.hpp", "kCatalogSchemaVersion") }} |
| [API key file](#server-files) | server | first line `OMBRA-API-KEYS v1` | 1 |

A format changes only with a version bump. The version of a file never depends on the version of the package that wrote it.

## Envelope

Every ciphertext, crypto context and public evaluation key crosses the network and is stored on the server inside an envelope. It is a small header around OpenFHE's binary serialization. The version is {{ const("ombra.file_format.FORMAT_VERSION") }} for kinds 1 to 3 and {{ const("ombra.file_format.FORMAT_VERSION_SEEDED") }} for kind 4, the seeded ciphertext. A reader refuses a kind that does not match its version.

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | magic `OMBR` |
| 4 | 2 | format version |
| 6 | 2 | kind: 1 context, 2 ciphertext, 3 public evaluation key, 4 seeded ciphertext |
| 8 | 2 | preset name length $P$, 1 to {{ const("ombra.file_format.MAX_NAME") }} |
| 10 | $P$ | preset name |
| . | 2 | OpenFHE version length $V$, 1 to {{ const("ombra.file_format.MAX_NAME") }} |
| . | $V$ | OpenFHE version string |
| . | 8 | payload length $L$ |
| . | $L$ | payload: OpenFHE binary serialization (kinds 1 to 3) or the seeded layout below (kind 4). The file ends here. |

There is no kind for a secret key, so the secret key cannot travel in an envelope. A reader checks the kind, the preset name and the OpenFHE version against what it expects, and checks the payload length against the bytes present before it allocates anything. The Python side is `client/ombra/file_format.py` and the C++ side is `core/include/ombra/core/file_envelope.hpp`. The C++ constants `kEnvelopeVersion` ({{ cpp("core/include/ombra/core/file_envelope.hpp", "kEnvelopeVersion") }}) and `kEnvelopeVersionSeeded` ({{ cpp("core/include/ombra/core/file_envelope.hpp", "kEnvelopeVersionSeeded") }}) match the Python ones.

### Seeded ciphertext

A seeded ciphertext stores a 32-byte seed in place of the uniformly random half of the ciphertext, so it is about half the size. The server expands the seed back into the full ciphertext when it loads a block, and every kernel sees the same words as for an ordinary one. Payload of kind 4:

| Size | Field |
| --- | --- |
| 2 | payload version, 1 |
| 2 | generator, 1 is BLAKE2xb |
| 32 | seed, a fresh one from the operating system for each ciphertext |
| 4 | ring dimension $N$ |
| 2 | towers $T$ |
| 4 | level |
| 4 | noise scale degree |
| 4 | slots |
| 8 | scaling factor, IEEE double |
| 2 + $K$ | key tag length $K$ and the key tag |
| $8T$ | moduli, 8 bytes each |
| $8TN$ | the first polynomial, tower by tower, in evaluation form |

A reader refuses an unknown payload version or generator, a size that does not match $N$ and $T$, a word that is not reduced, and trailing bytes. A collection may hold both kinds of column. The client writes kind 4 when its compiled module is installed. A client without it writes kind 2 and refuses to read kind 4.

## Key file

The key file holds the client's OpenFHE secret key and the master key for documents, protected by a passphrase. It has its own magic because the envelope has no secret-key kind. The version is {{ const("ombra.keyfile.KEYFILE_VERSION") }}. Source: `client/ombra/keyfile.py`.

| Size | Field |
| --- | --- |
| 4 | magic `OMBK` |
| 2 | version |
| 1 | key derivation function: 1 is scrypt |
| 4, 4, 4 | scrypt $n$, $r$, $p$ |
| 16 | salt |
| 12 | AES-GCM nonce |
| rest | AES-256-GCM ciphertext and tag |

The plaintext is the 32-byte master key followed by the serialized secret key. Everything before the ciphertext, including the scrypt parameters and the salt, is the GCM associated data, so changing any header byte makes decryption fail. The default scrypt cost is $n = {{ const("ombra.keyfile.DEFAULT_SCRYPT")[0] }}$, $r = {{ const("ombra.keyfile.DEFAULT_SCRYPT")[1] }}$, $p = {{ const("ombra.keyfile.DEFAULT_SCRYPT")[2] }}$, which takes about 128 MiB of memory to derive. When reading, the client caps $n$, $r$ and $p$ so that a hostile file cannot make it allocate gigabytes.

If you lose this file or its passphrase, the data is lost. There is no recovery path, by design.

## Blobs

Records and documents are stored as blobs that the client encrypts. The version is {{ const("ombra.documents.BLOB_VERSION") }}. Source: `client/ombra/documents.py`. The stored blob is exactly one bucket size (16384, 262144, 2097152 or 16777216 bytes), the smallest that fits.

| Size | Field |
| --- | --- |
| 4 | magic `OMBD` |
| 2 | version |
| 12 | nonce of the key wrap |
| 48 | wrapped data key: a fresh 32-byte AES key, wrapped with the master key, plus a 16-byte tag |
| 12 | nonce of the data |
| rest | AES-256-GCM ciphertext and tag of the padded plaintext |

The padded plaintext is a mode byte (1 text, 2 bytes, 3 reference, 4 record), the true length as 8 bytes, the payload and zero padding up to the bucket. The true length and the padding are inside the ciphertext, so the server sees the bucket and nothing finer. A record payload is the vector's byte length, the float32 vector and JSON metadata. A reference payload is JSON with `uri`, `sha256` and `size`, for a document that is larger than the inline threshold of {{ const("ombra.documents.DEFAULT_INLINE_THRESHOLD") }} bytes.

The associated data binds a blob to its place: the magic, version, kind, collection name and id, and for the body also the whole header. A blob copied to another id, collection or kind fails authentication, as does any changed byte. See [Documents and blobs](../how-it-works/documents.md).

## Server files

The server keeps everything in its data directory, laid out as in [Backups](../operations/backups.md). Clients never read these files, but the formats are versioned the same way.

**Segment** (`segments/<name>.oseg`): one block of one cluster, immutable once written. Magic `OSEG`, a 2-byte version, a CRC32 of the body, then the cluster number, the ids and the column envelopes. Each column is a complete [envelope](#envelope) of kind ciphertext. Source: `server/storage/include/ombra/storage/segment.hpp`.

**Stored blob** (`blobs/<name>.oblob`): magic `OBLB`, a 2-byte version, a CRC32 of the body, then the blob kind and the payload, which is the client's blob above and is never parsed. Source: `server/storage/include/ombra/storage/blob.hpp`.

**Write-ahead log** (`wal.log`): magic `OWAL`, a 2-byte version, then records of a length, a CRC32 and a body. Records are insert intent, insert commit and delete. On start the server replays the log: an insert that was acknowledged is rolled forward, one that was not is discarded. Source: `server/storage/include/ombra/storage/wal.hpp`.

**Catalog** (`catalog.db`): a SQLite database with the collections, segments, blobs and tombstones. Its `meta` table holds `schema_version` and `api_version`. The server upgrades an older schema when it opens it, and refuses a newer one. The current schema version is {{ cpp("server/storage/include/ombra/storage/catalog.hpp", "kCatalogSchemaVersion") }}.

**API key file** (`api_keys.v1`): text, mode 0600. The first line is `OMBRA-API-KEYS v1`, then one line per key with a salt, a salted SHA-256 hash and a scope. It holds no token.

**Collection format**: a collection is the set of the files above plus its catalog rows. The wire version is the `api_version` field, 1, stored with every collection spec. See the [gRPC API](grpc-api.md).
