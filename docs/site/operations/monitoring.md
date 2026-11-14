# Monitoring

The server has two sources of operational data: a JSON metrics endpoint on the HTTP admin port and logs on stderr. There is no Prometheus endpoint and no OpenTelemetry export. If you use Prometheus, poll the JSON with a small exporter or a script.

Neither source carries plaintext, ids, collection names, scores or keys. They hold counts, byte sizes and timings only.

## Health check

```bash
curl --cacert server.crt https://localhost:8080/healthz
```

returns `{"status":"ok"}` and needs no API key. The HTTP port uses the same TLS certificate as gRPC, so drop `--cacert` only if the server runs with `--insecure`. It only tells you that the process is up. It does not check the GPU or the data.

## Readiness check

```bash
curl --cacert server.crt https://localhost:8080/readyz
```

returns `{"status":"ready"}` while the server takes calls, and status 503 with `{"status":"draining"}` from the moment it gets SIGTERM until it exits. Use `/readyz` for a load balancer or a Kubernetes readiness probe, so traffic stops before the server does. Use `/healthz` for the liveness probe: it stays 200 while the server drains, so the server is not restarted in the middle of a shutdown.

## Metrics

```bash
curl --cacert server.crt -H "Authorization: Bearer $(cat admin_api_key)" \
    https://localhost:8080/metrics
```

The key must have the admin scope. A missing key returns 401 and a key with a lower scope 403. Both are counted. The body is JSON:

| Key | Meaning |
| --- | --- |
| `collections` | Number of collections. |
| `vectors_live`, `vectors_deleted` | Vectors in all collections, live and deleted but not yet compacted. |
| `blocks` | Number of stored blocks. |
| `rpc.<Method>.requests` | Calls since the server started, for each gRPC method. |
| `rpc.<Method>.errors` | Calls that ended in an error. |
| `rpc.<Method>.latency_ms` | `sum`, `count` and `max`, in milliseconds. Divide `sum` by `count` for the mean. |
| `auth.unauthenticated`, `auth.permission_denied` | Calls refused for a missing or wrong key, or a key with too low a scope. |
| `gpu_memory.used_bytes`, `gpu_memory.total_bytes` | GPU memory in use and in total. |
| `cache.hits`, `cache.misses`, `cache.evictions`, `cache.hit_rate` | Block cache counters since the start, and hits over hits plus misses. |

Counters reset when the server restarts. A latency histogram or percentiles are not available, only the sum, the count and the maximum. A rising `cache.misses` or `cache.evictions` with a low `hit_rate` means the cache budgets are too small for your collections, see [Supported GPUs](gpus.md#memory).

## Logs

Logs go to stderr. In Docker, read them with `docker compose logs ombra`. Each line starts with `[ombra.server]`, followed by what happened and numbers. Examples of the shape, with made-up numbers:

```text
[ombra.server] insert count=8192 bytes=104857600 ms=412.30
[ombra.server] search_round count=3 bytes=52428800 ms=95.10
[ombra.server] compaction dropped=2
[ombra.server] refused rpc=Search reason=unauthenticated
```

`count` is a number of items, `bytes` a size and `ms` the time of the call. The server logs a line for each write and each search round, for refused calls (with the first characters of the key's hash, not the token), for REST requests denied by scope, for compaction, and for internal errors. A line never contains an id, a collection name, a value or a token.

On start, the server prints the gRPC address (and the REST address) to stdout, and warns on stderr when it runs with `--insecure` or `--no-auth`. The Docker entrypoint also prints one line when it makes a certificate or an API key on the first start, and never the key itself.

## What to watch

Whether `/healthz` answers; the error counts of `Search` and `Insert`; `auth.unauthenticated`, which climbs when a client has a wrong key or when someone is probing; GPU memory against its total; and free disk space in the data volume, which the server does not report.
