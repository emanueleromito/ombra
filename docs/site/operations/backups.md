# Backups

Two things need a backup, and they live in different places. The server's data directory holds the encrypted collections. The client's key file holds the only key that can read them.

## The key file

Back up each collection's key file, and its passphrase, on the client side. Without the key file the data on the server cannot be decrypted by anyone, and there is no recovery. Keep the copy somewhere other than the server. The file is passphrase-protected, see the [key file format](../reference/file-formats.md#key-file), so it is safe to store like any other sensitive file. A backup of the server data without the key file is useless, and a key file without the server data has nothing to open.

## The server data directory

In Docker this is the `ombra-data` volume, mounted at `/data`. Outside Docker it is whatever you pass as `--data-dir`.

| Path | Contents |
| --- | --- |
| `catalog.db` | The catalog: collections, blocks, ids, tombstones and where every file is. |
| `wal.log` | The write-ahead log. |
| `segments/` | One file per block: the encrypted vectors of a cluster. |
| `blobs/` | One file per encrypted blob: records and documents. |
| `keys/` | The public evaluation keys of each collection. |
| `api_keys.v1` | The salted hashes of the API keys. |
| `tls/` | The self-signed certificate and its key (Docker image). |
| `admin_api_key` | The admin token made on the first start (Docker image). |

Nothing here is plaintext vectors or documents, and nothing here can decrypt anything: the server never has the secret key. A stolen backup gives the same access as a stolen disk of the server, and no more. It does contain the TLS private key and the admin token, so treat the backup as a secret anyway.

### Taking a backup

Stop the server, copy the directory, start it again.

```bash
docker compose stop ombra
docker run --rm -v ombra_ombra-data:/data -v "$PWD":/backup busybox \
    tar czf /backup/ombra-data.tar.gz -C /data .
docker compose start ombra
```

The volume name has the compose project name as a prefix, `ombra_ombra-data` for the shipped file. Check yours with `docker volume ls`.

A copy of a running server is not documented as safe. Files are written atomically and the server recovers from a crash by replaying the log, so a copy taken while it runs may still open, but you could copy `catalog.db` and the segment files at different moments. Stop the server, or use a snapshot of the whole filesystem (LVM, ZFS, a cloud disk snapshot), which gives you the state of one instant.

### Restoring

Stop the server, put the files back into an empty volume or directory, start the server.

```bash
docker compose down
docker volume rm ombra_ombra-data
docker run --rm -v ombra_ombra-data:/data -v "$PWD":/backup busybox \
    tar xzf /backup/ombra-data.tar.gz -C /data
docker compose up -d
```

On start the server checks every file it reads against its version and CRC, and replays the write-ahead log: an insert that was acknowledged is kept, one that was not is dropped. A restored copy must come from the same or an older version of Ombra than the one that reads it, see [Upgrades](upgrades.md). The files belong to the user `ombra` (uid 10001). The `tar` command above keeps the owners stored in the archive.

### What a backup does not cover

A backup is a copy of one moment. Vectors inserted after it are not in it, and deleted vectors come back. Ombra does not replicate the server or take backups on its own.
