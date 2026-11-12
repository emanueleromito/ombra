#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# Starts ombra_server in the container. On the first start it makes what the server needs in /data:
# a self-signed TLS certificate (data/tls/) and one admin API key, printed once to the log. Both
# stay in the volume, so a restart reuses them. The API key is an access token; the client's secret
# key never reaches the server.
#
# Environment (all optional):
#   OMBRA_TLS_HOSTS   subject alternative names of the certificate (default: DNS:localhost,IP:127.0.0.1)
#   OMBRA_LISTEN      gRPC address (default 0.0.0.0:50051)
#   OMBRA_REST_PORT   REST admin port, 0 or empty to turn it off (default 8080)
#   OMBRA_GPU_MB, OMBRA_HOST_MB   cache budgets in MB
#   OMBRA_OPENFHE_VERSION   OpenFHE version written into stored files (default: the build's)
# Arguments to the container are passed on to ombra_server.
set -euo pipefail

DATA=/data
TLS="$DATA/tls"
LISTEN="${OMBRA_LISTEN:-0.0.0.0:50051}"
REST="${OMBRA_REST_PORT-8080}"

if [ ! -f "$TLS/server.crt" ] || [ ! -f "$TLS/server.key" ]; then
  mkdir -p "$TLS"
  openssl req -x509 -newkey rsa:3072 -nodes -days 825 -subj "/CN=ombra" \
    -addext "subjectAltName=${OMBRA_TLS_HOSTS:-DNS:localhost,IP:127.0.0.1}" \
    -keyout "$TLS/server.key" -out "$TLS/server.crt" 2>/dev/null
  chmod 600 "$TLS/server.key"
  echo "ombra: made a self-signed certificate in $TLS/server.crt (give it to clients as the CA)" >&2
fi

if [ ! -f "$DATA/api_keys.v1" ]; then
  echo "ombra: first start, creating one admin API key (shown once):" >&2
  ombra_server --create-api-key --scope admin --data-dir "$DATA"
fi

args=(--data-dir "$DATA" --listen "$LISTEN" --tls-cert "$TLS/server.crt" --tls-key "$TLS/server.key")
[ -n "${OMBRA_OPENFHE_VERSION:-}" ] && args+=(--openfhe-version "$OMBRA_OPENFHE_VERSION")
[ -n "$REST" ] && [ "$REST" != "0" ] && args+=(--rest-port "$REST" --rest-host 0.0.0.0)
[ -n "${OMBRA_GPU_MB:-}" ] && args+=(--gpu-budget-mb "$OMBRA_GPU_MB")
[ -n "${OMBRA_HOST_MB:-}" ] && args+=(--host-budget-mb "$OMBRA_HOST_MB")

exec ombra_server "${args[@]}" "$@"
