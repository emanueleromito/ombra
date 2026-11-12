# M4: First release

**Goal:** Docker images, compose file, docs, licensing page. Tasks are added as M3 closes; this file holds the ones already decided.

---

### T4.1 Client wheels with the exact local expansion

**Why:** local mode (T3.7) expands queries in Python with one extra level and about 5x the server's time (option A of decision 0022, measured too slow); M3 compiles the exact expansion for Linux x86_64 and Python 3.12 only.
**Depends on:** T3.7
**Steps:** extend the M3 build (T3.7) to our own `ombra` client wheels for Linux x86_64 and aarch64 and macOS (arm64, x86_64), Python 3.11 to 3.13, with OpenFHE 1.6.1 and the C++ OpenFHE CPU expansion (the server's exact tree method, `core/`) compiled in; local mode uses it instead of the Python expansion; widen `requires-python` again; CI builds and tests every wheel.
**Done when:** each wheel installs on its platform and Python version without a compiler, the local-server parity test (T3.7) passes with the compiled expansion, local mode uses the same number of levels as the server, and query time at d = 768 is measured and recorded in `results/`.
**Status:** todo
