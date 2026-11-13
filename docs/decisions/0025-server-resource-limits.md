# 0025. Server resource limits and pre-body authentication

- **Status:** proposed
- **Date:** 2026-11-13

## Context

The pre-release audit found that the gRPC server buffers up to 1 GiB per call before its handler
authenticates, that Search streams are not capped or timed, and that most quotas are missing. The
gRPC C++ API has one receive-size limit for the whole server, not one per method.

## Options considered

- **A. A second listener for the large streaming methods.** Exact limits, but two ports and a
  breaking change for clients.
- **B. One receive limit (flag, default 1 GiB, enough for one block insert), authentication before the body, a ResourceQuota
  for all buffered memory and threads, per-field caps in the handlers.** Weaker per method, no API
  change.

## Decision

B. API keys are checked by an auth metadata processor when the headers arrive (TLS only; the
handlers check again for plain-text loopback use). Search streams are capped globally and per key,
count their device bytes against a session budget (default a quarter of the GPU budget), and have
idle and total deadlines. All caps are flags of `ombra_server` with defaults in
`server/grpc/include/ombra/server/limits.hpp`. The test hook `OMBRA_TEST_LOG2QP_LIMIT` is compiled
only with CMake option `OMBRA_ENABLE_TEST_HOOKS` (default ON for development; a release or container
build passes OFF), and the server always unsets it at start.

## Consequences

An authenticated key can still send one message up to the receive limit to a unary method. Release
builds must pass `-DOMBRA_ENABLE_TEST_HOOKS=OFF`. Global locks of the store are still held across
file I/O; removing that needs a rework of the store's locking and is not part of this change.
