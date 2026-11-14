# Contributing to Ombra

Ombra is source-available under the PolyForm Noncommercial License 1.0.0 or a commercial license (see [LICENSE](LICENSE) and the License section of the [README](README.md)). Outside contributors sign a Contributor License Agreement (CLA) before their first pull request is merged. The CLA Assistant bot asks for it on the pull request.

## Workflow

Open pull requests against `dev`. `main` only takes release merges from `dev`. Keep each pull request small and focused, with an imperative subject line ("Add query expansion kernel"). Say in the description why you add any new dependency.

Run `make help` for the commands. Before you open a pull request, run `make fmt`, `make lint` and `make test`. CI runs them again, plus the OpenFHE tests. Changes to `core/` also need the GPU tests (`make test-gpu`), which a maintainer runs.

Report security problems privately, as [SECURITY.md](SECURITY.md) explains, not in a public issue.

## Sanitizers and fuzzing

`make test-asan` builds all CPU C++ tests in `build-asan` with AddressSanitizer and UBSan and runs them. A UBSan finding stops the test, so it fails. Run it when you change a parser, a file format or memory handling.

The parsers of untrusted bytes (file envelope, seeded payload, segment, blob, write-ahead log, recovery, API key file, HTTP request) have libFuzzer targets in `fuzz/`. `make fuzz FUZZ_SECONDS=60` needs clang, builds `build-fuzz` and runs each target for that long on a copy of its seed corpus. If you change one of these parsers, run it for a few minutes. A parser must reject bad input with its normal error, never crash, read out of bounds or allocate memory from a length field without checking it against the input size. Every fuzzing finding gets a regression test in the normal test suite. [fuzz/README.md](fuzz/README.md) has the details. CI runs both targets in the "Sanitizers and fuzzing" job.

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
