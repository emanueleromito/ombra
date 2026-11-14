# API keys

An API key is an access token for the server. It is not an FHE key and has nothing to do with your secret key, which never leaves the client. The server stores only a salted hash of each key. You see the token once, when you create it.

## Scopes

A key has one scope. `read` may call the read methods and `search`. `write` may also change data. `admin` may call everything, including `DropCollection` and the `/metrics` port. The table of methods is on the [gRPC API](../reference/grpc-api.md) page.

## Limiting a key to one collection

A `read` or `write` key can be limited to one collection:

```bash
ombra_server --create-api-key --scope write --collection docs --data-dir /data
```

A limited key can only call methods on that collection. Any call that names another collection fails with `PERMISSION_DENIED`, and the answer is the same whether that collection exists or not, so the key cannot be used to find out which names exist. `ListCollections` returns only the key's own collection. `CreateCollection` works only for that name. For calls that send a stream, the check happens on the first message, before the server stores any data or reads any key bytes.

An `admin` key cannot be limited: the server refuses to create one, and refuses a key file with such a line. An admin key can drop every collection and read the metrics, so a limit on it would mean nothing. `DropCollection` needs an admin key, which is never limited.

Limited keys do not make the server multi-tenant. Collections still share the GPU, the disk and the limits, and a client with a write key can fill the disk. For hard separation run one server per trust domain, see the [threat model](threat-model.md). The HTTP port still takes only the admin key.

## The key file

The keys are in `api_keys.v1`, or `api_keys.v2` once a key is limited to a collection. See [File formats](../reference/file-formats.md#server-files). The file must be mode 0600. Create keys while the server runs and then send it SIGHUP to load them. To revoke a key, remove its line and send SIGHUP. A file that does not parse keeps the loaded keys and logs an error.

The first limited key moves the file to version 2 and keeps the old file as `api_keys.v1.migrated`. Old keys work as before. Do not edit `api_keys.v1.migrated` expecting an effect, and delete it when you no longer need it. Going back to a server without version 2 support means keeping a copy of the old file.
