# Configuration

The server is the `ombra_server` binary. The tools `ombra_verify` and `ombra_backup` check and copy its data directory, see [Command line tools](cli.md). In the Docker image a small entrypoint script builds its command line from environment variables; see [Docker and compose](../operations/docker.md). This page lists both.

## Flags

```text
ombra_server --data-dir D --listen HOST:PORT --tls-cert F --tls-key F [options]
ombra_server --create-api-key --scope admin|read|write --data-dir D
ombra_server --version
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data-dir D` | none, required | Directory for all server state. See [Backups](../operations/backups.md). |
| `--listen HOST:PORT` | none, required | gRPC address. Port 0 picks a free port. |
| `--tls-cert F`, `--tls-key F` | none | PEM certificate and key. Give both or neither. |
| `--insecure` | off | Allows starting without TLS. Prints a warning. Only accepted on a loopback address. |
| `--no-auth` | off | Skips the API key check. Prints a warning. Only accepted on a loopback address. |
| `--create-api-key` | off | Creates an API key, prints it once on stdout and exits. Needs `--scope` and `--data-dir`. |
| `--scope S` | none | Scope of the key to create: `admin`, `read` or `write`. |
| `--rest-port N` | off | Starts the HTTP port for `/healthz` and `/metrics`. 0 picks a free port. |
| `--rest-host H` | the host of `--listen` | IPv4 address of the HTTP port. It uses the same TLS certificate as gRPC. |
| `--gpu-budget-mb N` | 4096 | GPU memory the block cache may use, in MiB. |
| `--host-budget-mb N` | 2048 | Host memory the block cache may use, in MiB. |
| `--compact-interval-ms N` | 60000 | How often the server drops blocks whose vectors are all deleted. 0 turns it off. |
| `--openfhe-version V` | the build's | OpenFHE version written into stored files and checked when reading them. |
| `--version` | | Prints the version and exits. |

Without `--tls-cert` and `--tls-key` the server refuses to start unless you pass `--insecure`. Without an API key file in the data directory it refuses to start unless you pass `--no-auth`. The key file is named `api_keys.v1`, and the server also refuses a file that group or others can read.

On start the server prints `ombra_server listening on HOST:PORT` to stdout, and with `--rest-port` also `ombra_server rest listening on HOST:PORT`. It shuts down on SIGINT or SIGTERM.

### API keys

An API key is an access token for the gRPC and HTTP ports. It is not an FHE key, and the server never sees the client's secret key. `--create-api-key` prints the token once. The file keeps only a salted SHA-256 hash, so a lost token cannot be recovered and you create a new one. The three scopes are described on the [gRPC API](grpc-api.md) page. A key has a scope but no collection: the server is single-tenant, so any read key reads every collection and any write key can change any of them.

```bash
ombra_server --create-api-key --scope read --data-dir /data
```

## Environment variables of the Docker image

All are optional. Arguments after the image name are passed on to `ombra_server`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OMBRA_LISTEN` | `0.0.0.0:50051` | gRPC address, as `--listen`. |
| `OMBRA_REST_PORT` | `8080` | HTTP port. Set it to `0` or empty to turn the HTTP port off. |
| `OMBRA_GPU_MB` | server default | As `--gpu-budget-mb`. |
| `OMBRA_HOST_MB` | server default | As `--host-budget-mb`. |
| `OMBRA_OPENFHE_VERSION` | the build's | As `--openfhe-version`. |
| `OMBRA_TLS_HOSTS` | `DNS:localhost,IP:127.0.0.1` | Subject alternative names of the self-signed certificate made on the first start. Read only while no certificate exists yet. |

The entrypoint always passes `--data-dir /data` and the TLS files in `/data/tls/`. On the first start it makes a certificate and an admin API key if they are missing.

## Client side

The Python client takes its connection settings as arguments: `Client(address, api_key=..., root_certificates=..., insecure=...)`. It reads no environment variables for a server connection. In local mode, `OMBRA_LOCAL_THREADS` sets the number of threads of the local server's OpenFHE work. See the [Python API](python-api.md).
