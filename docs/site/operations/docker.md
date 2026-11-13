# Docker and compose

The server runs from a Docker image. You need Docker, an NVIDIA GPU from the [supported list](gpus.md) and the NVIDIA container toolkit.

## Images

| Image | What it is |
| --- | --- |
| `ghcr.io/erailabs/ombra-server` | The GPU server. Built for the five architectures of [Supported GPUs](gpus.md), on the CUDA 12.6 runtime. Runs as the unprivileged user `ombra`. |
| `ghcr.io/erailabs/ombra-cpu` | The Python client with local mode, for tests and development. It has no `ombra_server`: serving search over gRPC needs the GPU image. |

## Compose

This is the `docker-compose.yml` at the root of the repository.

```yaml
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# Starts the Ombra GPU server. Needs Docker, an NVIDIA GPU (Ampere: A100, A10, RTX 30xx) and the
# NVIDIA container toolkit.
#
#   docker compose up -d
#   docker compose exec ombra cat /data/admin_api_key    # the admin API key, made on the first start
#   docker compose cp ombra:/data/tls/server.crt ./server.crt    # the certificate, for clients
#
# Collections, the API key file and the certificate live in the named volume "ombra-data", so they
# survive `docker compose down`. `docker compose down -v` deletes them. Back the volume up.
# The server has no secrets in this file: the API key is made on the first start and kept in the
# volume. The image tag is set with OMBRA_VERSION (default 0.1.0).
name: ombra

services:
  ombra:
    image: ghcr.io/erailabs/ombra-server:${OMBRA_VERSION:-0.1.0}
    build:
      context: .
      dockerfile: docker/Dockerfile.server
    ports:
      - "50051:50051"  # gRPC
      - "127.0.0.1:8080:8080"  # REST admin and metrics, on this machine only
    volumes:
      - ombra-data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    stop_grace_period: 10s

volumes:
  ombra-data:
```

## First start

```bash
docker compose up -d
docker compose exec ombra cat /data/admin_api_key
docker compose cp ombra:/data/tls/server.crt ./server.crt
```

On the first start the entrypoint makes two things in the `/data` volume and reuses them on every later start. The first is a self-signed TLS certificate in `/data/tls/`, valid for 825 days, for `localhost` and `127.0.0.1`. The second is one admin API key in `/data/admin_api_key`, readable only by the server's user. The key is never printed to the log, so read it with the `cat` command above. The volume stores only its salted hash in `api_keys.v1`, so if you delete `admin_api_key` you cannot get the token back.

The certificate only names `localhost` and `127.0.0.1`. To reach the server by another name, set `OMBRA_TLS_HOSTS` before the first start, for example `OMBRA_TLS_HOSTS=DNS:ombra.example.com`. The variable is ignored once a certificate exists, so to change it later, delete `/data/tls/` in the volume and restart.

Connect with the key and the certificate as the trusted root:

```python
from pathlib import Path
from ombra import Client

client = Client(
    "localhost:50051",
    api_key=Path("admin_api_key").read_text().strip(),
    root_certificates=Path("server.crt").read_bytes(),
)
```

An admin key can do everything, including dropping collections. For an application, make a read or write key as described in [Configuration](../reference/configuration.md), and keep the admin key for operations.

## Ports

Port 50051 is gRPC. Port 8080 is the HTTP admin port, with `/healthz` and `/metrics`. The compose file publishes it on `127.0.0.1` only, so it is reachable from the machine and not from the network. Change the mapping only if you put something in front of it that restricts access. To turn it off, set `OMBRA_REST_PORT=0` in the service's `environment`.

## Settings

Environment variables for the cache budgets, the listen address and the certificate names are listed in [Configuration](../reference/configuration.md). Anything you put in `command:` is passed to `ombra_server`.

## Data and stopping

Everything the server stores is in the `ombra-data` volume. `docker compose down` keeps it, `docker compose down -v` deletes it with all collections. See [Backups](backups.md). On a stop, Docker sends SIGTERM and the server shuts down cleanly, with 10 seconds before Docker kills it.

## Building the image yourself

```bash
docker build -f docker/Dockerfile.server -t ombra-server .
```

The build compiles OpenFHE 1.6.1 and the CUDA code for the five architectures, and takes about an hour. `docker compose build` does the same through the `build:` entry of the compose file.
