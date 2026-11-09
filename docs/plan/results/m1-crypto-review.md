# M1 crypto review (T1.1)

- **Reviewed:** all of `core/`, `client/` and `tests/` at commit `7a8126b` (`main`, including T1.2), plus `modal_app.py`, the `Makefile` and the threat model in `docs/product-requirements.md`.
- **Checked against:** OpenFHE v1.6.1 and openfhe-python v1.6.1.0 source at the pinned tags. Every claim below about OpenFHE's behaviour was read there.
- **Reviewer:** a fresh agent session with no access to the M0 conversation.
- **Result:** 1 high, 11 medium, 15 low. No secret-key leak, no plaintext leak and no parameter below 128 bits was found. Reading the arithmetic found no bug in any kernel. The findings are mostly about checks that are missing or weaker than they claim.

## Limits of this review

- It is a static review. No GPU test, oracle or benchmark was run for it. "Passes" below means the code compares what it says it compares, and M0/T1.2 results files record that it passed. The only code executed was one call to `encode_query` with a stub context, for F14.
- The reviewer is the same model family that wrote M0, so it may share the authors' blind spots. This is not a substitute for review by a human cryptographer before a security claim is made to customers.
- Hard rule 5 (tests in the same change as crypto changes) is about history and was not audited PR by PR.
- Client key handling and key files could not be reviewed, because that code does not exist yet (`client/ombra/` has only `params.py` and `query.py`).

**Severity:** high = a hard rule is broken on merged code and nothing exact compensates. Medium = a guarantee is missing or weaker than claimed, and could yield a wrong or unsafe result on a path M1 or M2 builds on. Low = hardening, coverage or tidiness.

## T1.1 checklist

| Point | Verdict | Detail |
| --- | --- | --- |
| Parameters only from `core/params/`, at least 128-bit | Holds on every path that exists; enforced in Python only | Every context comes from `make_context`, and no other `CCParams` use exists. OpenFHE also rejects a non-compliant ring dimension at context creation. C++ never checks: F7. |
| No secret key outside the client | Holds | No C++ file references a secret key. No test or bench serializes one. See F12 and F24 for adjacent gaps. |
| Every kernel has an exact oracle comparing RNS words | Partly | See the table below: F1, F2, F3. |
| No plaintext logged or written outside synthetic tests | Holds | Drivers print verdicts, indices and timings only. Tests write plaintext only to temp dirs, with synthetic data. `core/bench/recall.py` uses a public embedding dataset and stores aggregates only. |
| Format version headers | Not met | F8. |
| Level, noise-scale degree, scaling factor | Correct where derived; several gaps | F3, F4, F15. The `GpuCtOps` and `gpu_eval_*` rules match OpenFHE's source. |
| Rules 7 and 8 (Ampere only, wording) | Hold | `CUDA_ARCHS` is 80;86. Every "open source" in the repo is a statement that the project is not. |
| Secrets in git history | None found | No key, `.env`, or `.bin` file was ever added. |

## Oracle audit

"Metadata" is **real** only when the code under test derives it independently of the OpenFHE result it is compared with.

| Kernel | Driver | RNS words exact | Metadata | Verdict |
| --- | --- | --- | --- | --- |
| `rns_add_kernel`, `rns_mul_kernel` | `ombra_roundtrip` | yes | real (`gpu_eval_*`) | sound |
| `column_search_kernel` | `ombra_column_search` | yes, 2 presets, ragged tile | real | sound |
| `column_search_batched_kernel` | `ombra_column_search_batch` | yes, `m0_depth2` only | real | sound (F18) |
| NTT forward and inverse | `ombra_expand_gpu` stage 1 | yes, Q and P towers | n/a | sound; one variant unexercised (F18) |
| Key-switch digits, inner product, ModDown | `ombra_expand_gpu` stages 2 and 3 | yes, against `EvalAutomorphism` too | n/a | sound |
| Expansion tree (batched, chunked, per node) | `ombra_expand_gpu`, `_batch` | yes | carried from the root, compared with OpenFHE's leaf | sound for matching inputs (F4, F21) |
| `ks_inner_product` for relinearization | `ombra_search_gpu` | yes, levels 0 and 1 | real, including `GetScalingFactorInt` | sound for one node (F2) |
| `rescale_switch`, `rescale_combine` | `ombra_search_gpu` | yes, levels 0 and 1 | real | sound for one node (F2) |
| `ct_tensor_kernel` | `ombra_search_gpu` | yes, level 1 | real | sound for one node (F2) |
| `plain_search_kernel` | `ombra_plain_search` check 2 | yes, with OpenFHE plaintexts | **tautological** | F3 |
| `decode`, `fft_stage`, `finish` (`encode.cu`) | `ombra_plain_search` check 1 | **no: tolerance only** | none | F1 |

## High

### F1. The GPU encoder has no exact oracle, and GPU-encoded search is never compared with OpenFHE

- **File:** `core/src/ombra_plain_search.cpp:193`, `:240`, `:244-246`; `tests/test_gpu_plain_search.py:43`; kernels in `core/kernels/encode.cu`
- **Severity:** high
- **What is wrong:** hard rule 3 needs a comparison with OpenFHE, and `core/CLAUDE.md` needs exact matches for integer RNS arithmetic. Three weaknesses combine here.
  1. Coefficients are compared within 2 units, for the integer reduction steps as well as the floating-point FFT.
  2. `enc_ok` is `max_diff <= tol && (max_diff != 0 || exact_eval == gpu_eval.size())`. As soon as one coefficient differs by one unit, the evaluation-form words are not checked at all. T0.11 recorded exactly that case at `m0_depth2` (1 unit of difference), so the NTT of the encoder output was not compared on that preset.
  3. Check 3 compares the fused path with the two-pass path: GPU against GPU. The comparison with OpenFHE on `:244` only prints an `info:` line and cannot fail.

  What is left is value-level: decoded slots within 1e-9 and decrypted scores within 1e-6. A defect within those bounds, for example in tower reduction, would pass. Floating-point encoding cannot promise identical words, which is a fair reason for a tolerance on the FFT, but not on the steps after it.
- **Suggested fix:** split the oracle. (a) Run reduction and NTT on OpenFHE's own rounded integer coefficients and compare words exactly with OpenFHE's plaintext. (b) Keep the tolerance only for FFT and rounding. (c) Always compare the GPU evaluation form with OpenFHE's NTT of the GPU's own coefficients (`DCRTPoly::SetFormat`), so no case is skipped. (d) Record the tolerance as a decision, since it is an exception to the rule.
- **Follow-up:** T1.9

## Medium

### F2. Every T1.2 oracle runs one node and one block

- **File:** `core/src/ombra_search_gpu.cpp:135`; `core/src/gpu_search.cpp:252-261`; `core/src/expansion_device.cu:371`, `:395`, `:422`
- **Severity:** medium
- **What is wrong:** `GpuSearch` is built with `blocks = 1`, and each `GpuCtOps` call passes one pointer. So `blockIdx.z > 0`, `node = z >> 1`, the per-block strides and chunking by `max_batch` are never exercised for relinearize, rescale and multiply. The strides read correct, but no oracle covers them. T1.3 will use many blocks and measures only latency, so a stride error there would produce a plausible timing on wrong ciphertexts.
- **Suggested fix:** run the driver with several blocks of different data and a ragged count, with `set_max_batch(3)` as the expansion test does. Compare every block at every stage.
- **Follow-up:** T1.10 (do before T1.3 reports numbers)

### F3. The plain-search metadata comparison is tautological

- **File:** `core/src/ombra_plain_search.cpp:224-225`, `:235`, `:239`; `core/include/ombra/core/plain_search.hpp:124-125`
- **Severity:** medium
- **What is wrong:** `gpu_ofhe_pt` is built with `unflatten_ciphertext(cpu, words)`, so its level, degree and scale are copied from the OpenFHE result and then compared with that same result by `identical()`. The check cannot fail on metadata. The cause is that `PlainSearchDevice` returns raw words and no library code derives the metadata, unlike `gpu_eval_column_search`. The driver's header claims a metadata comparison.
- **Suggested fix:** add a `gpu_eval_plain_search` wrapper in `ciphertext_ops` that derives metadata from the query and plaintext (level unchanged, degrees added, scales multiplied). Compare its output with OpenFHE.
- **Follow-up:** T1.11

### F4. `GpuExpansion::upload` discards metadata and `download` stamps the sample's

- **File:** `core/src/expansion_gpu.cpp:224-231`, `:233-235`; `core/src/ombra_recall_search.cpp:157`; `core/src/ombra_expand.cpp:76`
- **Severity:** medium
- **What is wrong:** `upload` checks only the moduli. `download` always returns the metadata of the constructor's `sample_`. A root at the same tower count but a different noise-scale degree or scale is accepted and its leaves are labelled with the sample's values. Under FIXEDMANUAL, `Decode` divides by `2^(p * (noiseScaleDeg - 1))`, so a wrong degree silently gives wrong values. The refusal described in the decision 0009 addendum exists only in the CPU driver, and `ombra_recall_search` checks only its first query.
- **Suggested fix:** in `upload`, reject a ciphertext whose level, degree or scale differs from `sample_`. Add a negative test.
- **Follow-up:** T1.12

### F5. GPU operations skip the key and context checks OpenFHE enforces

- **File:** `core/src/ciphertext_ops.cpp:84-89`, `:107-111`, `:187-217`, `:237`; `core/src/expansion_gpu.cpp:200`, `:224`; `core/src/gpu_search.cpp:38`, `:51`; oracle comparators `ombra_roundtrip.cpp:58`, `ombra_column_search.cpp:46`, `ombra_column_search_batch.cpp:44`, `ombra_plain_search.cpp:95`
- **Severity:** medium
- **What is wrong:** OpenFHE's `TypeCheck` (`cryptocontext.h:266`) rejects operands from another context, key tag or encoding type. No GPU path compares any of them. `gpu_eval_add` also checks FIXEDMANUAL only on `a`. Mixing ciphertexts under two keys, which a shared multi-tenant batch could do through a bug, gives garbage labelled with one tenant's key tag, and `Decrypt` does not check the tag either. Oracles use a single key and their comparators ignore key tag, encoding type and slots, so they cannot notice.
- **Suggested fix:** record key tag and encoding type in `flatten_ciphertexts` and `flatten_one`, and reject mixes. Check the evaluation key's tag against the ciphertext. Replace the four copies of `identical()` with one shared comparator that includes these fields. Add negative tests with two key pairs.
- **Follow-up:** T1.13

### F6. Batched search can read past the end of the query buffer

- **File:** `core/src/ciphertext_ops.cpp:284-285`, `:307`
- **Severity:** medium
- **What is wrong:** when comparing later queries with the first, the code checks level, scale, degree and moduli but not `ring_dim`, unlike the single-query path at `:237-239`. A prime valid for ring dimension N is also valid for N/2, so a smaller query can pass. `qwords` is then shorter than `q * query_words()` assumes, and `upload_query` reads past it. No network path reaches this today, but M2 batches queries from different clients, and this is a host heap over-read.
- **Suggested fix:** compare `ring_dim` and `num_towers`, and check `qwords.size()` before uploading.
- **Follow-up:** T1.14

### F7. The 128-bit floor is not enforced on the C++ side

- **File:** `core/src/ciphertext_ops.cpp:75`; `core/src/expansion_gpu.cpp:42`; `core/src/gpu_search.cpp:38`; each driver's `Serial::DeserializeFromFile(argv[1], cc, ...)`
- **Severity:** medium
- **What is wrong:** hard rule 1 applies to "any code path", and `docs/product-requirements.md` says the library refuses anything below 128 bits. Only `client/ombra/params.py` checks. A context created with `HEStd_NotSet` passes through every kernel and driver. No such context is created today, and M2 will receive contexts from clients.
- **Suggested fix:** add one check, called from `require_fixedmanual` and `rns_params`, that rejects `GetStdLevel() == HEStd_NotSet` (the level is serialized, `rlwe-cryptoparameters.h:481`) and confirms log2(QP) against the standard's limit.
- **Follow-up:** T1.15

### F8. No Ombra format has a version header

- **File:** every `SerializeToFile` in `core/src/*.cpp` and `tests/`; `core/src/ombra_encoding_check.cpp:29`; `tests/test_query_encoding.py:184-185`; `tests/test_gpu_plain_search.py:103-104`; `core/src/ombra_plain_search.cpp:207-208`
- **Severity:** medium
- **What is wrong:** hard rule 6 requires a version header on every ciphertext and key file. Ciphertext and evaluation-key files are raw OpenFHE serialization. Its per-class version (`ciphertext.h:563`) tracks OpenFHE's layout, not Ombra's format, preset or scale conventions. The only versioned files are the preset JSON and one test input. All of this is in temp dirs today, which is why this is not high, but M2 storage will build on it.
- **Suggested fix:** define a small envelope (magic, format version, preset name, OpenFHE version) before M2 storage, or record a decision saying what counts as compliant for spike-only files.
- **Follow-up:** T1.16

### F9. The decryption-leak mitigation is heuristic, unpinned and never tested

- **File:** `client/ombra/params.py:71-89`; `core/params/ckks_presets.json`; `docs/decisions/0010-query-expansion-via-automorphisms.md:16`
- **Severity:** medium
- **What is wrong:** decision 0010 relies on "noise flooding on decryption" as "the mitigation against key recovery". The preset does not set it. OpenFHE's default is `FIXED_NOISE_DECRYPT`, and the `NOISE_FLOODING_DECRYPT` path in `ckksrns-pke.cpp:52` is off. What actually runs is `Decode`'s added Gaussian noise, sized from an estimate, which OpenFHE's own comment says needs "at least 128 decryption queries (in practice the values are typically closer to 10,000 or so)" to average out, before a further attack. The threat model already lets the server see which clusters and document IDs the client picks, and those choices are functions of decrypted scores. This leak is ordinal only and this review knows of no practical attack from it, which is why it is not high, but nobody has analysed it. It also depends on `ckksDataType` staying at its default of REAL, which nothing pins.
- **Suggested fix:** write a decision that states the IND-CPA-D position and why it is acceptable, or adopts flooding (which runs computations twice). Pin `ckks_data_type` and the decryption mode in the preset. Add a test that two decryptions of one ciphertext differ, which proves the noise is active. Add a rule that decrypted values never leave the client.
- **Follow-up:** T1.17

### F10. Plaintext-DB mode lets a client read the database

- **File:** `core/kernels/plain_search.cu:6-7`; `core/include/ombra/core/plain_search.hpp:3`; `client/ombra/query.py:114`; `docs/product-requirements.md:31`, `:57-63`
- **Severity:** medium
- **What is wrong:** the requirements offer this mode for a "corpus the server owner already holds (public or licensed data)". The result holds one score per vector in each searched block. A client encrypting the unit vector e_j, which passes the norm check, receives coordinate j of every vector, so d queries recover each searched block. The norm check runs on the client and the server cannot verify anything about an encrypted query. The threat model covers only a curious server, so this loss for the data owner is undocumented. CKKS noise in the result is also not flooded in this mode.
- **Suggested fix:** record a decision before offering the mode. State plainly that the query side is protected and the corpus is not, or restrict the mode to public corpora.
- **Follow-up:** T1.18

### F11. Nothing enforces that the oracles run

- **File:** no `.github/` directory; `Makefile:30-34`
- **Severity:** medium
- **What is wrong:** hard rule 3 and the merge rules depend on `make test-gpu`, which someone must run by hand. There is no CI, so a merge cannot be blocked by a failing oracle, and the record of a pass lives only in results files and PR text.
- **Suggested fix:** add a required check that runs `make test` on every PR and `make test-gpu` on PRs touching `core/`, with Modal credentials as repository secrets.
- **Follow-up:** T1.19

### F12. The Modal image upload is much looser than `.gitignore`

- **File:** `modal_app.py:62-66`
- **Severity:** medium
- **What is wrong:** the image copies the working tree, ignoring only `.git`, `build`, `__pycache__`, `.venv`, `data`, `*.key` and `.env`. Paths that `.gitignore` treats as secrets are not excluded: `.env.*`, `*.pem`, `*.keyfile`, `*.ombrakey`, `keys/`, `secrets/`, `.modal.toml`. A client key file kept in the repo folder, the place `.gitignore` anticipates, would be copied into a cloud image, which breaks the intent of hard rule 2 without any code touching the key.
- **Suggested fix:** switch to an allowlist of the folders the image needs (`client`, `core`, `tests`, build files). Add a test asserting that the secret patterns in `.gitignore` are excluded.
- **Follow-up:** T1.20

## Low

No follow-up task is required for these.

| ID | File and line | What is wrong | Suggested fix |
| --- | --- | --- | --- |
| F13 | `core/src/gpu_search.cpp:192`, `:237` | `have_stored` becomes true after one block, so `run()` with more blocks reads uninitialized device memory and returns a result. Masks are tracked per block. | Track uploads per block, like `have_mask`. |
| F14 | `client/ombra/query.py:114` | A NaN entry passes the norm check, because `abs(nan - 1) > tol` is false. Confirmed with a stub context. | Reject non-finite input first with `np.isfinite`. |
| F15 | `tests/test_gpu_ciphertext_roundtrip.py:66-68`; `tests/test_gpu_column_search.py:73-78` | Add, ct x pt and column search are only tested on fresh level 0, degree 1 inputs. Their metadata rules are correct by reading, and M1 uses them only there. | Add level 1 and degree 2 cases if these paths are ever used at other levels. |
| F16 | `core/src/expansion_device.cu:611`, `:626`, `:522`; `core/src/ciphertext_ops.cpp:131` | Key, monomial and `unflatten` words are not checked as reduced, although ciphertext uploads are. Barrett needs x < q^2 (`modarith.cuh:27`). Wrong output only. | Apply the reduced-word check on all inputs from outside. |
| F17 | `client/ombra/params.py:53-61`; `core/params/ckks_presets.json:4-13`; `client/tests/test_ckks_baseline.py:23` | Secret-key distribution and, for `m0_depth2`, the digit count come from OpenFHE defaults (uniform ternary, 2 digits). Unknown keys are ignored, so a typo changes parameters silently. The log2(QP) test covers named presets only. | Pin these in the preset, reject unknown keys, and parametrize the test over `load_presets()`. |
| F18 | `tests/test_gpu_column_search_batch.py:23`; `tests/test_gpu_query_expansion_batch.py:26`; `tests/test_gpu_ciphertext_roundtrip.py:25`; `tests/test_query_encoding.py:28`; `core/kernels/ntt.cu:317-319` | These oracles run on `m0_depth2` only, and the conversion-load path for N <= 2048 is unexercised. | Add the compact preset. Remove or test the small-ring path. |
| F19 | `tests/test_gpu_query_expansion.py:36-37`; `tests/test_gpu_relin_rescale_mask.py:42-43` | Oracles at d = 384, 768, 1536 run only with `--expansion-large`, by hand. Results files record that they were run. | Run them in the gate from T1.19 on a schedule. |
| F20 | `modal_app.py:136` | Oracles build in Debug and production paths in Release. The floating-point encoder is the most sensitive to this. | Run the oracle set once in Release. |
| F21 | `core/src/openfhe_expand_ops.hpp:21-47` | The CPU reference for multiplying by X^-k is Ombra's own code, not OpenFHE. The decrypted-value check backs it up. | Cross-check it against OpenFHE's polynomial multiplication by the monomial. |
| F22 | `core/params/ckks_presets.json:23`; `client/ombra/params.py:14` | `num_large_digits` was added (`1251d55`) without a version bump. An older loader would silently build a different context. | Bump the preset version on schema changes. |
| F23 | `tests/test_gpu_relin_rescale_mask.py:39`, `:179` | The compact preset gets a 1e-5 bound where other tests say the 1e-6 bound must not be loosened. It is documented and T1.6/T1.7 own the decision. | Remove it once T1.7 sets the rule for this preset. |
| F24 | `core/bench/recall.py:231`, `:308`; `core/bench/t011.py:159`, `:195`; `core/bench/run.py:106` | Python under `core/` holds the secret key, while `core/CLAUDE.md:21` says core code never touches it. These act as the client and never pass it to C++. | Amend the rule to cover benches, or move them. |
| F25 | `client/tests/test_compact_preset.py:23-32`, `:55`; `tests/test_openfhe_sizes.py:59`; `tests/test_gpu_bench_batch.py:67`; `core/bench/recall.py:418` | Ring sizes are hard-coded, against `core/CLAUDE.md:22`. Some are deliberate pins. | Read them from the preset, and keep only intentional pins. |
| F26 | `modal_app.py:24`, `:26-27`, `:39` | The base image and OpenFHE are pinned by movable tags, and pip packages are unpinned. | Pin digests, commit hashes and versions. |
| F27 | `Makefile:16`; `client/ombra/query.py:27`; `tests/test_query_encoding.py:18-20`; `core/src/expansion_device.cu:84` | `make venv` installs no numpy, and `ombra.query` imports it before `importorskip`, so a fresh venv fails at collection. Separately, one `cudaEventDestroy` is unchecked. | Add numpy to `make venv`. Check the call or mark it `(void)`, as elsewhere. |

## What the hard rules do not cover but should

1. **GPU operations enforce everything OpenFHE's checks enforce** (context, key tag, encoding, metadata). F4, F5.
2. **An oracle may not build the value under test from the OpenFHE result it is compared with**, and a check that cannot fail does not count. F3, F1.
3. **Batched paths need their own oracle.** A kernel with a batch dimension is tested at more than one node. F2.
4. **Inputs reaching the server are untrusted**, including context, keys and shapes, with bounds checks and fuzzing before M2. F6, F7, F16.
5. **Decrypted values never leave the client**, and the decryption-noise position is a documented decision. F9.
6. **Each collection mode states who is protected from whom**, including the data owner against the querier. F10.
7. **The working tree is not uploaded wholesale** to any remote environment. F12.
8. **Oracle runs are enforced by a required check**, not by convention. F11.
