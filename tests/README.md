# tests/

Cross-component and end-to-end tests (Python client talking to C++/CUDA code). Unit tests live next to their code (`core/tests/`, `client/tests/`).

- `make test` runs the CPU tests locally.
- `make test-gpu` runs tests marked `@pytest.mark.gpu` on Modal (A10).
