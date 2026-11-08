---
name: run-on-gpu
description: Run GPU tests or benchmarks for Ombra on Modal (A10 for tests, A100 for benchmarks) and bring the results back. Use whenever code in core/ must be run or measured on a GPU.
---

# Run on GPU (Modal)

There is no local GPU. All CUDA execution goes through Modal via the Makefile.

## Commands

| Goal | Command | GPU |
| --- | --- | --- |
| Check the environment | `make gpu-info` | A10 |
| Build and run GPU tests | `make test-gpu` | A10 |
| Run one test file or filter | `modal run modal_app.py::test_gpu --pytest-args "-k expansion"` | A10 |
| Run benchmarks | `make bench TASK=t0.4` | A100 |
| Copy results locally | `make results-pull` | none |

## Steps

1. Run `make test` locally first; fix CPU failures before spending GPU time.
2. Run `make test-gpu`. Read the full output: CMake errors, ctest output, then pytest output.
3. On failure, fix and rerun. The image is cached, so only your code is rebuilt; if the first run rebuilds OpenFHE, that is expected and takes several minutes.
4. For benchmarks, run `make bench TASK=<task id>`, then `make results-pull`.
5. Write a result summary in `docs/plan/results/YYYY-MM-DD-<task>-<gpu>.md` with: the command, git commit (`git rev-parse --short HEAD`), GPU, CUDA and OpenFHE versions (from `make gpu-info`), the parameter set, and the numbers in a table.

## If something goes wrong

- **Authentication error:** Modal tokens are missing. Stop and tell the user; never create or write tokens yourself.
- **Image build fails in OpenFHE:** do not change `OPENFHE_TAG` or `OPENFHE_PYTHON_TAG` independently; they must match. Report the error to the user.
- **Out of memory on the GPU:** reduce the test size; large sizes belong in benchmarks, not tests.
