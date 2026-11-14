# Backups

Two things need a backup, and they live in different places. The server's data directory holds the encrypted collections. The client's key file holds the only key that can read them.

## The key file

Back up each collection's key file, and its passphrase, on the client side. Without the key file the data on the server cannot be decrypted by anyone, and there is no recovery. Keep the copy somewhere other than the server. The file is passphrase-protected, see the [key file format](../reference/file-formats.md#key-file), so it is safe to store like any other sensitive file. A backup of the server data without the key file is useless, and a key file without the server data has nothing to open.

### Copy and check a key file

`ombra-keys` (also `python -m ombra.keys`) copies a key file and checks it. It never takes the passphrase as an argument, because arguments show in the process list. Give it a prompt, an environment variable, a file or a file descriptor.

```bash
# Copy atomically (temporary file, fsync, rename), mode 0600. It refuses to replace an existing
# file unless you pass --force, and then checks the copy.
ombra-keys backup my.key /mnt/offsite/my.key.2026-01-15 --check-passphrase

# Open a key file or a backup and compare its key tag with the one on the server
OMBRA_KEY_PASSPHRASE=... OMBRA_API_KEY=... \
  ombra-keys check /mnt/offsite/my.key.2026-01-15 --server ombra.example.com:443 --collection docs

# Offline: compare with a key tag you wrote down
ombra-keys check /mnt/offsite/my.key.2026-01-15 --tag "$KEY_TAG"
```

`check` opens the file, checks that the secret key and the public key in it work and have the same key tag, and prints `OK` with that tag. The key tag is not secret: the server stores it. The tool never prints a key or the passphrase. Run `check` on every backup after you make it, and after each restore test. The exit code is 0 for OK and 1 for a failed check. See the [Python API](../reference/python-api.md#ombra-keys) for the options.

## The server data directory

In Docker this is the `ombra-data` volume, mounted at `/data`. Outside Docker it is whatever you pass as `--data-dir`.

| Path | Contents |
| --- | --- |
| `catalog.db` | The catalog: collections, blocks, ids, tombstones and where every file is. |
| `wal.log` | The write-ahead log. |
| `segments/` | One file per block: the encrypted vectors of a cluster. |
| `blobs/` | One file per encrypted blob: records and documents. |
| `keys/` | The public evaluation keys of each collection. |
| `api_keys.v1` or `api_keys.v2` | The salted hashes of the API keys. Version 2 is used once a key is limited to a collection, see [API keys](../security/api-keys.md). |
| `tls/` | The self-signed certificate and its key (Docker image). |
| `admin_api_key` | The admin token made on the first start (Docker image). |

Nothing here is plaintext vectors or documents, and nothing here can decrypt anything: the server never has the secret key. A stolen backup gives the same access as a stolen disk of the server, and no more. It does contain the TLS private key and the admin token, so treat the backup as a secret anyway.

### Online backup

`ombra_backup` copies the data directory as of one instant while the server keeps running. Use it in preference to the stop-and-copy below. It reads the directory without changing it. The server does not need to know.

```bash
mkdir -p backup && sudo chown 10001 backup
docker run --rm -v ombra_ombra-data:/data -v "$PWD/backup":/backup \
    --entrypoint ombra_backup ghcr.io/erailabs/ombra-server:0.1.0 \
    --data-dir /data --out /backup/ombra-2026-01-15
```

Outside Docker: `ombra_backup --data-dir /var/lib/ombra --out /backups/ombra-2026-01-15`. The output directory must not exist or be empty, and must not lie inside the data directory. When it finishes it has checked every copied file, the same way [`ombra_verify`](../reference/cli.md#ombra_verify) does, and it fails if one is corrupt. A failed run leaves the output directory empty. The volume name has the compose project name as a prefix, `ombra_ombra-data` for the shipped file. Check yours with `docker volume ls`.

The copy has `catalog.db`, `segments/`, `blobs/`, `keys/` and the API key file (`api_keys.v1` or `api_keys.v2`). It has no write-ahead log, no `tls/` and no `admin_api_key`. Copy those two from the volume if you want them, for example with the tar command below. Without `tls/` the image makes a new certificate on its first start, and clients need the new one. Without `admin_api_key` you cannot read the admin token again, see [Docker and compose](docker.md).

!!! deep-dive "Why the copy is consistent while the server writes"
    The catalog is the one record of what exists. SQLite copies it in one read snapshot, so it is the catalog of a single instant. The server writes a segment, blob or key file completely to disk before the catalog row that names it, and never reuses a file name, so every file the catalog copy names existed at that instant and cannot change afterwards. `ombra_backup` opens all of them right after the snapshot. An open file stays readable even if the server removes it a moment later, for example when a client replaces a block.

    If a file is already gone when `ombra_backup` opens it, the catalog copy is out of date, and it starts again with a new snapshot, up to 5 times. After that it fails with a message and you run it again. The write-ahead log is not copied. The server acknowledges a write only after the catalog has it, so every acknowledged write is in the copy. A write in progress at that instant was not acknowledged yet, and may or may not be in the copy, as after a crash. Files that no catalog row names are not copied, so the copy holds no leftovers.

    One limit: a store of many thousands of files that changes all the time (blocks replaced every few milliseconds) can make the run restart. The check at the end then still protects you: a copy is never reported as done unless it verifies.

### Taking a backup with the server stopped

Stop the server, copy the directory, start it again. This also copies `tls/` and `admin_api_key`.

```bash
docker compose stop ombra
docker run --rm -v ombra_ombra-data:/data -v "$PWD":/backup busybox \
    tar czf /backup/ombra-data.tar.gz -C /data .
docker compose start ombra
```

A plain copy of a running server's directory is not safe, because `catalog.db`, its journal and the segment files are copied at different moments. Use `ombra_backup`, stop the server, or take a snapshot of the whole filesystem (LVM, ZFS, a cloud disk snapshot), which gives you the state of one instant.

### Checking a data directory

`ombra_verify` re-checks every file and the catalog without changing anything. Run it on a backup after you take it, on a restored copy before you start the server on it, and now and then on the live directory to find disk damage early.

```bash
docker run --rm -v ombra_ombra-data:/data --entrypoint ombra_verify \
    ghcr.io/erailabs/ombra-server:0.1.0 --data-dir /data
```

It exits 0 when clean, 1 when something is corrupt and 2 on a usage error, and prints counts and file names only. On a live directory a write in progress can show up as a warning. What it checks, and its limits, are on the [Command line tools](../reference/cli.md#ombra_verify) page. If it reports a corrupt file, restore from the last backup. A file that does not verify is not repaired by the server either: it refuses to read it.

### Disk full

When the disk or quota fills, a write fails with an error (the gRPC API reports it as an internal error for now) and the server stays up and consistent. The call that hit the limit either did not happen or happened completely, never half, and files it had started are removed. Writes work again as soon as there is space, and the client can send the same call again. After a restart the server finishes or drops any write that was in progress and deletes leftover files. Searches are not affected. Watch the free space of the data volume, a full disk blocks every write.

### Restoring

Stop the server, put the files back into an empty volume or directory, start the server.

```bash
docker compose down
docker volume rm ombra_ombra-data
docker run --rm -v ombra_ombra-data:/data -v "$PWD":/backup busybox \
    tar xzf /backup/ombra-data.tar.gz -C /data
docker compose up -d
```

On start the server checks every file it reads against its version and CRC, and replays the write-ahead log: an insert that was acknowledged is kept, one that was not is dropped. Run `ombra_verify` on the restored directory first if you want to know before you start. A restored copy must come from the same or an older version of Ombra than the one that reads it, see [Upgrades](upgrades.md). The files belong to the user `ombra` (uid 10001). The `tar` command above keeps the owners stored in the archive.

### What a backup does not cover

A backup is a copy of one moment. Vectors inserted after it are not in it, and deleted vectors come back. Ombra does not replicate the server or take backups on its own.
