# CLAUDE.md

Instructions for AI coding agents working in this repository. Read this first, every session.

## What this project is

Ombra is a vector database with fully homomorphic encryption (FHE): vectors, queries, filters and scores stay encrypted, and the server only ever computes on ciphertexts. The Python client holds the secret key; the server runs encrypted search on GPUs.

Background, in this order:
1. `docs/product-requirements.md`: what we build and why, design decisions for users
2. `docs/architecture.md`: how it is built
3. `docs/decisions/`: one file per decision, with the reasoning
4. `docs/plan/`: milestones and the current tasks

**Current stage:** milestone M2, the minimal server (`docs/plan/m2-server.md`), with M3 (`docs/plan/m3-client.md`) next. M1 is done (`docs/plan/results/m1-report.md`); M0 findings are in `docs/plan/results/spike-report.md`. Build only what the current M2 task asks for.

## Layout

| Folder | Language | Purpose |
| --- | --- | --- |
| `core/` | C++20 / CUDA | CUDA kernels; reading and writing OpenFHE's ciphertext format |
| `server/` | C++20 | gRPC API, storage, clustering, GPU scheduling (not started) |
| `client/` | Python 3.11+ | keys, encryption (openfhe-python), top-k, re-ranking |
| `integrations/` | Python | LangChain and LlamaIndex adapters (not started) |
| `tests/` | Python + C++ | cross-component and end-to-end tests |
| `modal_app.py` | Python | GPU environment on Modal |

Each folder may have its own `CLAUDE.md` with rules for that part. Read it before working there.

## Commands

Always use the Makefile; run `make help` for the full list.

| Command | What it does |
| --- | --- |
| `make build` | Configure and build C++/CUDA locally (needs CUDA) |
| `make venv` | Create the gitignored `.venv` with pytest and ruff, which `make` then uses |
| `make test` | CPU tests (Python + C++) that need neither OpenFHE nor a GPU, runs locally |
| `make test-openfhe` | Tests marked `openfhe`, on Modal in the pinned image, CPU only (no GPU) |
| `make test-gpu` | Tests marked `gpu` and C++ tests, on Modal (A10) |
| `make bench` | Benchmarks on Modal (A100) |
| `make fmt` | Format all code (clang-format, ruff) |
| `make lint` | Lint all code |

There is no local GPU. Anything that needs CUDA to *run* goes through Modal (`make test-gpu`, `make bench`). Modal needs `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the environment; never write them to a file.

## Hard rules

These are not style preferences. Breaking one is a bug.

1. **Security floor.** Never use FHE parameters below 128-bit security, in any code path, test or benchmark. Parameters come from presets; never hand-pick ring sizes or moduli outside `core/params` (when it exists).
2. **The secret key stays on the client.** No server-side code may accept, store, log or derive the secret key. The server only receives ciphertexts and public evaluation keys.
3. **OpenFHE is the oracle.** Key generation, encryption and decryption use OpenFHE only. Every CUDA kernel has a test that compares its output with OpenFHE's CPU result on the same inputs. No kernel is merged without one.
4. **No plaintext leaks.** Never log, print or write plaintext vectors, queries, documents, scores or keys outside tests that use synthetic data.
5. **Crypto changes need tests first.** Do not change key handling, parameter selection or ciphertext layout without adding or updating tests in the same change.
6. **Versioned formats.** Every ciphertext, key file and collection format carries a version header. Do not change a format without bumping its version.
7. **Ampere only, for now.** Compile for `sm_80` and `sm_86`. Do not add other architectures without a task asking for it.
8. **Wording.** Never call this project "open source". It is "source-available" (see `LICENSE`).

## Conventions

- **C++:** C++20, `clang-format` (repo `.clang-format`), RAII, no raw `new`/`delete`, errors as exceptions at boundaries and status values inside kernels' host code. Files `snake_case`, types `PascalCase`, functions `snake_case`.
- **CUDA:** one kernel family per file in `core/kernels/`, host wrappers in `core/`. Check every CUDA call with the project's error macro. Document the expected launch shape above each kernel.
- **Python:** 3.11+, `ruff` for format and lint, type hints on public functions, `pytest` for tests.
- **Tests:** small sizes by default so tests are fast; large sizes only in `make bench`.
- **Commits:** small and focused, imperative subject line ("Add query expansion kernel").

## Agents and models

The main session runs on **Opus 5.5** and acts as planner and reviewer. Hands-on work goes to cheaper subagents in `.claude/agents/`:

| Agent | Model | Does | Can edit? |
| --- | --- | --- | --- |
| main session | Opus 5.5 | plans, delegates, reviews every diff, commits, opens PRs | yes |
| `implementer` | Sonnet | writes code and tests for one scoped task, in its own git worktree | yes, commits on its worktree branch only |
| `test-runner` | Haiku | runs `make` commands, summarizes results | no |
| `Explore` | Haiku | read-only search of code and docs | no |

Rules for the main session:
1. **Delegate implementation** to `implementer`, one task per implementer, passing the task ID and its "Done when" line.
2. **Run up to 3 implementers in parallel**, but only on tasks whose "Depends on" tasks are all done and that touch different files. If two tasks would edit the same files, run them one after the other.
3. **Delegate test and benchmark runs** to `test-runner`. Keep at most 2 Modal runs going at once, to limit GPU cost.
4. **Review each implementer branch before merging.** Read the full diff yourself and check it against the Hard rules: parameters, key handling, oracle tests for kernels, no plaintext logging, format versions. Anything in `core/` crypto code, key handling or parameter selection gets your line-by-line review, never only the subagent's summary.
5. **Merge one branch at a time** into your working branch, then run `make test` (and `make test-gpu` if `core/` changed) before merging the next.
6. **Only the main session pushes and opens PRs.** Implementers commit only on their own worktree branch; `test-runner` and `Explore` never write.
7. If a subagent reports "blocked", decide yourself; do not send the same task back unchanged.

## How to work on a task

1. Pick the task in `docs/plan/`. Its "Done when" line is the acceptance test.
2. Write or update the test first when possible.
3. Implement, then run `make test` (and `make test-gpu` for anything in `core/`).
4. Iterate until the "Done when" condition holds. Do not mark a task done on a partial result.
5. If you make a design choice not already in `docs/`, add a short file in `docs/decisions/` (copy `docs/decisions/template.md`). Agents mark a decision "proposed" only; the user accepts or rejects it, and an agent never changes a status to "accepted" or "rejected".
6. Record benchmark numbers in `docs/plan/results/`, never only in chat.

## Do not

- Commit secrets, keys, key files, Modal tokens or datasets (see `.gitignore`).
- Push to `main` or `dev` directly. Open pull requests against `dev`; `main` only takes release merges from `dev`.
- Add dependencies without saying why in the PR description.
- Add `Co-Authored-By` or session lines (e.g. `Claude-Session:` links) to commit messages or PR descriptions.
- Silence a failing test to make CI green.
