# Command line tools

Two tools work on a server's data directory: `ombra_verify` checks it and `ombra_backup` copies it. Both run on the CPU, need no GPU and no OpenFHE, and never read a secret key (the server has none). They ship in the server image next to `ombra_server`, in `/usr/local/bin`. The server itself is described on the [Configuration](configuration.md) page.

Both print counts and file names only. They never print vector ids, collection names or any content. File names are counters such as `s0000000000000007.oseg`.

## ombra_verify

```text
ombra_verify --data-dir DIR [--json] [--openfhe-version V]
```

| Flag | Meaning |
| --- | --- |
| `--data-dir DIR` | The data directory to check. Required. |
| `--json` | Prints the result as one JSON object instead of text. |
| `--openfhe-version V` | Also requires every stored column and key file to carry this OpenFHE version, as the server does. Without it each file only has to agree with itself. |

| Exit code | Meaning |
| --- | --- |
| 0 | Clean. Warnings may be printed. |
| 1 | Something is corrupt. |
| 2 | Usage error, or `DIR` is not a readable directory. |

### What it checks

| Part | Check |
| --- | --- |
| Segment files | Header, version, CRC, that the file parses, and that its vector count, column count, CRC and cluster equal the catalog row. |
| Blob files | Header, version, CRC, kind and size against the catalog row. |
| Key files | Magic, format version, kind, preset, OpenFHE version and lengths. See the limit below. |
| Catalog | SQLite `integrity_check`, `quick_check` and `foreign_key_check`, the schema and API versions. |
| Cross-references | Every catalog row has its file. The vector rows of each block add up to the block's vector count. |
| Orphans | A file that no catalog row names is a warning. The server deletes such files at its next start. |
| Write-ahead log | Header, version and the CRC of every record. A torn tail is a warning. A damaged record that valid ones follow is corrupt, because the server refuses to start on it. |

A key file has no checksum in the [key file envelope](file-formats.md), so a flipped byte in its payload is not found here. A truncated or extended file is. The same holds for the ciphertext payload inside a segment, but the segment's CRC covers it.

### Read-only

`ombra_verify` opens everything read-only. It does not run recovery, write, truncate or repair a file, and it takes no lock. It is safe to run while a server uses the directory, and it does not change what the server does.

The server takes no lock on the data directory, so `ombra_verify` cannot tell whether a server is running. On a live directory it can report a warning for a write in progress (a new file that the catalog does not list yet, a half-written log tail). It drops a "file missing" finding again when the catalog row is gone by the time it looks a second time, which happens when the server replaces or removes a block during the check. For a result you can rely on, check a backup or a stopped directory.

### Example

```text
$ ombra_verify --data-dir /data
checked: 2 collections, 4 segments, 2 blobs, 3 key files, 9 vectors, 3 log records
result: clean
```

```text
$ ombra_verify --data-dir /data
CORRUPT segment s0000000000000004.oseg: segment: CRC mismatch
checked: 2 collections, 4 segments, 2 blobs, 3 key files, 9 vectors, 3 log records
result: CORRUPT (1 problem(s))
```

In Docker:

```bash
docker compose exec ombra ombra_verify --data-dir /data
```

## ombra_backup

```text
ombra_backup --data-dir DIR --out DIR2
```

| Flag | Meaning |
| --- | --- |
| `--data-dir DIR` | The data directory to copy. Required. |
| `--out DIR2` | Where to put the copy. It must not exist or be empty, and must not lie inside `DIR`. Required. |

| Exit code | Meaning |
| --- | --- |
| 0 | The copy is done and was verified. |
| 1 | The backup failed. `DIR2` is empty again. |
| 2 | Usage error. |

The copy is the state of one instant, and it is safe to take while the server runs. The server does not need to cooperate and `DIR` is not changed. The [Backups](../operations/backups.md) page says how to use it and why the copy is consistent. After copying, `ombra_backup` runs the same checks as `ombra_verify` on the copy and fails if any file is corrupt.

The copy holds `catalog.db`, `segments/`, `blobs/`, `keys/` and `api_keys.v1` when present. It has no `wal.log`, no `tls/` and no `admin_api_key`. The server creates a new log on its first start from the copy.

In Docker, run the tool in a second container on the same volume and write to a folder on the host. The folder must be writable by user 10001, the user of the image.

```bash
mkdir -p backup && sudo chown 10001 backup
docker run --rm -v ombra_ombra-data:/data -v "$PWD/backup":/backup \
    --entrypoint ombra_backup ghcr.io/erailabs/ombra-server:0.1.0 \
    --data-dir /data --out /backup/ombra-2026-01-15
```

See [Backups](../operations/backups.md#online-backup) for the volume name and for restoring.
