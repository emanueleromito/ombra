# Plan

Milestones, in order. Each milestone has its own file with small tasks; every task has a **Done when** line that works as its acceptance test.

| Milestone | Goal | Status |
| --- | --- | --- |
| [M0: Technical spike](m0-spike.md) | Prove the GPU core works on Ampere and measure it | done ([report](results/spike-report.md)) |
| [M1: Core library](m1-core.md) | Reviewed `core/` library running a full encrypted query on the GPU; end-to-end latency, speedups and product presets | done ([report](results/m1-report.md)) |
| [M2: Minimal server](m2-server.md) | Single-node gRPC server with storage and clustered search | done (T2.1 to T2.10, see [m2-server.md](m2-server.md)) |
| [M3: Client and LangChain](m3-client.md) | Python client (sync + async), key file, top-k, re-ranking, LangChain adapter | done ([report](results/m3-report.md)) |
| [M4: First release](m4-release.md) | Docker images, compose file, docs, licensing page, client wheels | not started |

Benchmark results go in [results/](results/), one file per run, so numbers are never only in chat.

## Task format

```markdown
### T<milestone>.<n> Title

**Why:** one line on what this proves or unlocks.
**Depends on:** task IDs that must be done first (or "none").
**Steps:** short list.
**Done when:** a concrete, checkable condition.
**Status:** todo | in progress | done
```
