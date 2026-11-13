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

## Existing collections

A collection keeps its preset, its dimension and its key tag for life. A new Ombra version reads it as long as the formats are at versions the new version still reads. The formats, with their versions, are listed on [File formats](../reference/file-formats.md).

The client's key file has a version too, currently 1. A key file with a version the client does not know is refused with an error.

If a release changes a ciphertext layout or a preset, collections made before it cannot be read in place. You would create a new collection with the new version and insert your vectors again. Ombra has no tool to convert encrypted data between layouts, because the server cannot decrypt it. No release so far has needed this.
