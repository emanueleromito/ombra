# Contributing to Ombra

Ombra is source-available under the PolyForm Noncommercial License 1.0.0 or a commercial license (see [LICENSE](LICENSE) and the License section of the [README](README.md)). Outside contributors sign a Contributor License Agreement (CLA) before their first pull request is merged. The CLA Assistant bot asks for it on the pull request.

## Workflow

Open pull requests against `dev`. `main` only takes release merges from `dev`. Keep each pull request small and focused, with an imperative subject line ("Add query expansion kernel"). Say in the description why you add any new dependency.

Run `make help` for the commands. Before you open a pull request, run `make fmt`, `make lint` and `make test`. CI runs them again, plus the OpenFHE tests. Changes to `core/` also need the GPU tests (`make test-gpu`), which a maintainer runs.

## CPU-only build

`make build-cpu` builds without CUDA. If OpenFHE 1.6.1 and gRPC for C++ are installed, it also builds `ombra_server` without a GPU. That server is for tests and development, not for production: every call works except Search, which fails with the error "this server was built without GPU support". Key upload and collection load run the same checks as in the GPU build. `make test-openfhe-local` builds it and runs `tests/test_server_cpu.py` against it. `tests/server_harness.py` starts, stops and kills the binary for tests; set `OMBRA_SERVER_BIN` to use another binary.

Report security problems privately, as [SECURITY.md](SECURITY.md) explains, not in a public issue.

## Security invariants

These are project rules, not style preferences. A change that breaks one is a bug, and review will ask for it to be fixed before merge.

1. **128-bit minimum.** No code path, test or benchmark uses FHE parameters below 128-bit security. Parameters come from the presets in `core/params/`. Never pick a ring size or a modulus by hand anywhere else.
2. **The secret key stays on the client.** No server code accepts, stores, logs or derives the secret key. The server only receives ciphertexts and public evaluation keys.
3. **Every GPU kernel agrees with OpenFHE.** Key generation, encryption and decryption use OpenFHE only. Each CUDA kernel has a test that compares its output with OpenFHE's CPU result on the same inputs and requires exact agreement. A kernel without that test is not merged.
4. **No plaintext in logs.** Never log, print or write plaintext vectors, queries, documents, scores or keys, except in tests that use synthetic data.
5. **Versioned file formats.** Every ciphertext, key file and collection format starts with a version header. Changing a format means bumping its version, with tests for the old and the new one.

Changes to key handling, parameter selection or ciphertext layout come with new or updated tests in the same pull request.

## Docs

The public docs are in `docs/site/` and are built with MkDocs (`pip install -r docs/requirements.txt`, then `mkdocs build --strict`). [Writing these docs](docs/site/contributing-docs.md) explains the style rules, how a page reads values from the code, and the deep-dive marker.
