# M3 report

M3 (Python client and LangChain, `docs/plan/m3-client.md`) is done: all eleven tasks, T3.1 to T3.11. Sources are in `docs/plan/results/`.

## What works

| Area | Tasks | State |
| --- | --- | --- |
| Package, sync and async clients with the same API | T3.1 | Signature test; wheel builds |
| Key file (passphrase, versioned) | T3.2 | Round trip; refuses a wrong passphrase, a changed byte and an unknown version; no key material in errors |
| Presets "fast" and "balanced", d = 384, 768, 1536 | T3.3 | Match the C++ presets; refuses unsupported dimensions and anything below 128-bit security |
| Encrypted documents and metadata, blob store | T3.4 | Associated data binds a record to its id; bucket sizes only; crash-safe blob writes |
| Create, re-cluster, repack | T3.5 | Search stays correct during and after re-clustering; repack frees space |
| Insert, delete, search with exact re-ranking | T3.6 | Matches plaintext top-k at every security level and with a filter |
| Local mode (CPU server in a child process, the server's C++ expansion) | T3.7 | Same top-k as the GPU server (A10 parity test); e2e test passes on both backends |
| `langchain-ombra` | T3.8 | Example runs in local mode; LangChain's standard vector store tests pass, except the cases below |
| Recall tool `ombra-recall` | T3.9 | Prints plaintext and encrypted recall@10 and the 0012 verdict |
| Quickstart | T3.10 | `docs/quickstart.md`; samples tested in local mode; CI keeps the page and samples identical |
| Filtered recall on real data, 0019 amendment 1 | T3.11 | Filter-aware cluster choice plus one capped widening round |

Decisions accepted during M3: 0020 (key file, "balanced" preset), 0021 (blob store, re-clustering by generation), 0022 (local mode uses the server's C++ expansion) and 0019 amendment 1 (filter-aware cluster choice), all on 2026-11-12.

## Numbers

| What | Result | Source |
| --- | --- | --- |
| Local mode search, 10k vectors, fast preset | 2.28 s (d = 384) and 2.71 s (d = 768) on 4 hardware threads; 1.75 s and 2.18 s on 8 (target under 5 s at d = 768 on 4) | `2026-11-11-t3.7-local-mode.md` (local WSL, Ryzen 5 5600H) |
| Filtered recall@10 at ~1% selectivity, old 0019 rule | 0.76 to 0.81 | `2026-11-11-t3.11-filtered-recall.md` |
| Same, with amendment 1, Fast, plaintext simulation | min 0.991 on arXiv (40k, 20 clusters) and banking77 (10k, 6 clusters) | `2026-11-11-t3.11-probe-policies.md` |
| Encrypted vs plaintext recall under the same clusters | gap +0.0000 in all 21 cells | `2026-11-11-t3.11-filtered-recall.md` |
| Encrypted check of amendment 1, banking77, A10 | every filtered cell 1.000, same clusters as the simulation; unfiltered Fast 0.999 (one round, 3 clusters); 70 to 148 ms mean latency | `2026-11-11-t3.11-a10-confirmation.md` |
| Recall tool, 1,000 queries, synthetic 3k x 384 | 1.0000 plaintext and encrypted, gap +0.0000, PASS; about 1 s per query in local mode | `2026-11-11-t3.9-recall-tool.md` |
| Quickstart `local_search.py`, six texts | about 22 s wall clock including the model load; writes about 300 MB | T3.10, local WSL |
| LangChain standard suite in local mode | 13 passed, 8 skipped, 4 xfailed (strict) | `integrations/langchain-ombra/tests/` |

Modal spend in M3 batch D was about $0.70 (two A10 runs). Batch E used no Modal.

## Open items

**Before release (M4):**
- **Wheels (T4.1).** Until then the client is built from source against OpenFHE 1.6.1, on CPython 3.12 and x86-64 Linux only. Building OpenFHE takes longer than 10 minutes, so T3.10's "first search in 10 minutes" holds only once Ombra is installed. Nobody new to the project has followed the page yet.
- **Local-mode tests are not in CI.** CI's CPU job has no OpenFHE, and the Modal job runs only `gpu` tests. The quickstart run, the LangChain example and LangChain's standard suite (all `openfhe`-marked) were run locally only. CI checks that the quickstart page and its samples are identical and that the samples compile.
- **Encrypted arXiv check of amendment 1 not run.** banking77 has 6 clusters, so most filtered queries there search all of them. The arXiv numbers, where the cluster choice matters, are from the plaintext simulation, which matched the encrypted run exactly on banking77. About $0.60 on one A10.
- **The quickstart's server section is not run.** Its commands come from `ombra_server`'s help header.

**Client and adapter limits:**
- **Ids cannot be replaced in place.** A deleted id stays reserved until compaction, so upsert is not possible. `langchain-ombra` refuses an id that is already stored and keeps the stored version. Before review it deleted the stored version first and then failed the insert, which lost data. 4 standard LangChain tests are strict xfails because of this.
- **No fetch by id in the client API,** so `langchain-ombra` has no `get_by_ids` (8 standard tests skipped), and the document text is stored in the encrypted record metadata. There is no MMR search either, and filters are equality on declared integer fields only.
- **Possible partial insert (not verified).** `insert_vectors` uploads one block per cluster. A duplicate id in a later cluster would leave earlier blocks inserted, without their records.

**Small fixes:**
- `Client.local` fails with "the local server process did not start" when `TMPDIR` is long: the Unix socket path exceeds the AF_UNIX limit. The error should name the cause, or the socket should go in a short path.
- The root `pytest` `pythonpath = ["client"]` shadows an installed `ombra` that carries the compiled `_native` module. Local `openfhe` runs need `-o pythonpath=integrations/langchain-ombra`.
- The speed table in `docs/local-mode.md` still shows the old Python-expansion numbers (19.7 s per search); the current ones are above.
- A local collection of six texts writes about 300 MB of keys and ciphertexts.
