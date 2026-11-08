# tests/

Cross-component and end-to-end tests (Python client talking to C++/CUDA code). Unit tests live next to their code (`core/tests/`, `client/tests/`).

- `make test` runs the CPU tests locally.
- `make test-gpu` runs tests marked `@pytest.mark.gpu` on Modal (A10).
- `make test-openfhe` runs tests marked `@pytest.mark.openfhe` on Modal, CPU only, in the pinned image with OpenFHE and openfhe-python. Zero collected tests counts as a failure. CPU timings are written to the Modal volume under `results/`.

`make test` excludes both `gpu` and `openfhe` tests. `make venv` creates `.venv` with pytest and ruff; the Makefile uses it automatically.
