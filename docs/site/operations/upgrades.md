# Upgrades

This page says what the code does today, for version 0.1.0 and the versions after it. Every stored file and every message carries a version, and a reader refuses a version it does not know. So the failure you can expect from a mismatch is a clear error at start or on the first call, not silent misreading.

## Before you upgrade

Take a [backup](backups.md) of the data volume, with the server stopped. Read the entry for the new version in `CHANGELOG.md`, which lists any format change.

## Upgrading the server

With compose, set the new tag and recreate the container. The data volume stays.

```bash
OMBRA_VERSION=0.2.0 docker compose pull
OMBRA_VERSION=0.2.0 docker compose up -d
```

The version number here is an example. Without `OMBRA_VERSION` the compose file uses the version it shipped with.

On start the server opens the catalog. If the catalog has an older schema version, the server upgrades it in place. If it has a newer one, the server refuses to start with `unsupported schema_version`. Segments, blobs, the write-ahead log and the API key file are read as they are, and a file with a version the server does not know is refused. The certificate and the API keys in the volume are reused.

You cannot go back by starting an older image on a volume that a newer one has upgraded. Restore the backup instead.

## Upgrading the client

Install the new wheel with `pip install -U ombra`. The client and the server talk through the gRPC package `ombra.v1`. A change that is not compatible will be a new package, `ombra.v2`, so an old client keeps working against a server that still serves `ombra.v1`, and a new client reports a version error against a server that does not.

The OpenFHE version matters more than the Ombra version. Every envelope carries the OpenFHE version that wrote it (1.6.1 for the 0.1 series), and a reader refuses an envelope from another version, so a collection created with one OpenFHE version can only be used by a client and a server built with the same one. The `ombra[grpc]` and `ombra[local]` extras pin the matching `openfhe` package.

## What the tests check

The repository has upgrade tests (`make test-upgrade`, marker `upgrade`). They run on the CPU build of the server, so Search is checked only as far as it starts and fails with the "without GPU support" error, not with a protocol error. The previous release is the highest `v*` tag below the commit under test. The tests build that release's CPU server and run its client, and they skip while no release tag exists.

- **Old client, new server.** The previous client creates a collection, uploads keys, inserts, deletes, replaces a block, writes and deletes blobs, and reads everything back, against the new server.
- **New client, old server.** The same session with the new client against the previous server.
- **Old data directory, new server.** The previous server writes a data directory and is stopped. `ombra_verify` of the new version reports it clean, the new server opens it, the new client reads all of it and can write, and the recovery counts in the log are plausible (no torn bytes after a clean stop).
- **Every frozen data directory.** At each release a small data directory is frozen under `tests/fixtures/datadir-<version>/`. The new server must open every one of them, and the tests also write one with the current code. If a frozen directory would be too large for git, it is kept as a release asset instead.

For a release with a tag, any failure in these tests is a failure of the build. For an untagged commit given by `OMBRA_PREV_REF`, a failure in a pairing is reported as an expected incompatibility, so a deliberate protocol change before the first release does not fail the build.

## What is promised

Only what the tests above check: a client and a server that are one release apart work together for the calls listed, and a data directory written by a release can be opened by the next one. Two releases apart is not tested. The OpenFHE version rule below still applies. This page does not promise anything about GPU search across versions, because the upgrade tests have no GPU.

## Existing collections

A collection keeps its preset, its dimension and its key tag for life. A new Ombra version reads it as long as the formats are at versions the new version still reads. The formats, with their versions, are listed on [File formats](../reference/file-formats.md).

The client's key file has a version too, currently 1. A key file with a version the client does not know is refused with an error.

If a release changes a ciphertext layout or a preset, collections made before it cannot be read in place. You would create a new collection with the new version and insert your vectors again. Ombra has no tool to convert encrypted data between layouts, because the server cannot decrypt it. No release so far has needed this.
