# Low-severity findings, triage for M4 (T4.7)

Sources searched: `docs/plan/*.md` and `docs/plan/results/*.md`. The only review with severity labels is `m1-crypto-review.md` (15 low findings, F13 to F27; `m1-report.md` lists them as open). `m3-report.md` and the M2 notes carry no open low or minor findings. Status was checked against the tree at `origin/dev`.

Outcomes: 2 fixed, 13 deferred, 0 won't fix. Four of the deferred ones need main-session review (crypto, parameters or rule text).

| ID | Source | Summary | Outcome |
| --- | --- | --- | --- |
| F13 | m1-crypto-review, `core/src/gpu_search.cpp` | `have_stored` is true after one block, so `run()` can read device memory of blocks never uploaded. | Deferred. Small, but it is `core/` search code and needs a GPU test (`make test-gpu`) that this task cannot run. Suggested fix: per-block flags like `have_mask`. |
| F14 | m1-crypto-review, `client/ombra/query.py` | A NaN entry passes the norm check. | Fixed. `encode_query` now rejects non-finite input first; test `test_encode_rejects_non_finite` covers NaN and both infinities. |
| F15 | m1-crypto-review, GPU roundtrip and search tests | Add, ct x pt and column search only tested at level 0, degree 1. | Deferred. New GPU test cases, only needed if these paths run at other levels. |
| F16 | m1-crypto-review, `expansion_device.cu`, `ciphertext_ops.cpp` | Key, monomial and `unflatten` words not checked as reduced. | Deferred, needs main-session review. Touches the ciphertext and key upload path in `core/`; `expansion_device.cu:638` now checks one path, the rest is unverified here. |
| F17 | m1-crypto-review, `client/ombra/params.py`, `ckks_presets.json` | Secret-key distribution and digit count come from OpenFHE defaults; unknown preset keys are ignored. | Deferred, needs main-session review (parameter selection and preset schema). |
| F18 | m1-crypto-review, GPU oracle tests | Oracles run on `m0_depth2` only; small-ring path in `ntt.cu` unexercised. | Deferred. Needs GPU runs. |
| F19 | m1-crypto-review, expansion and relin tests | d = 384, 768, 1536 oracles run only with `--expansion-large`, by hand. | Deferred. Belongs in the scheduled GPU workflow of T4.2. |
| F20 | m1-crypto-review, `modal_app.py` | Oracles build in Debug, production in Release. | Deferred. Needs a Modal run; fits the scheduled GPU workflow of T4.2. `modal_app.py` already builds the expansion drivers and benchmarks in Release. |
| F21 | m1-crypto-review, `openfhe_expand_ops.hpp` | CPU reference for multiplying by X^-k is Ombra's own code. | Deferred. Test-only cross-check; needs OpenFHE and a rebuild. |
| F22 | m1-crypto-review, preset file | `num_large_digits` added without a version bump. | Deferred, needs main-session review (format version, hard rule 6). Bumping the version also changes what every loader accepts. |
| F23 | m1-crypto-review, relin and mask test | Compact preset uses a 1e-5 bound where others use 1e-6. | Deferred. Tied to the open tolerance decision 0012 and the known compact expansion flake; loosening or tightening a tolerance is not a low-risk edit. |
| F24 | m1-crypto-review, `core/bench/*.py` vs `core/CLAUDE.md` | Bench Python under `core/` holds the secret key while the rule says core code never does. | Deferred, needs main-session review. The fix is a wording change in `core/CLAUDE.md`, which implementers may not edit. |
| F25 | m1-crypto-review, tests and bench | Ring sizes hard-coded against `core/CLAUDE.md`. | Deferred. Touches many tests and parameters; some are deliberate pins. |
| F26 | m1-crypto-review, `modal_app.py` | Base image and OpenFHE pinned by movable tags; pip packages unpinned. | Deferred. T4.2 builds the pinned CI image, and Dependabot (added in T4.7) covers pip, Actions and Docker. Pin digests there. |
| F27 | m1-crypto-review, `Makefile`, `expansion_device.cu` | `make venv` lacks numpy; one `cudaEventDestroy` unchecked. | Both parts closed. `make venv` already installs numpy on `dev`. The destructor-path call in `clear()` now casts to `(void)`, as elsewhere (not compiled here, no CUDA locally; run `make test-gpu`). |

Counts: 15 findings. Fixed 2 (F14, F27). Deferred 13 (F13, F15 to F26), of which 4 need main-session review (F16, F17, F22, F24). Won't fix 0.
