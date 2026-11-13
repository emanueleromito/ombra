# Reproducing the numbers

Every figure on these pages comes from a script in the repository, run on a known machine. Figures marked TBD are waiting for a new run of these scripts on the current code. This page lists the scripts and the commands. You need a Modal account for the GPU runs, because there is no local GPU in the project's setup. Set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in your environment, and never write them to a file.

Expect small differences. The same kernel took up to 26% longer in one run than another on the same card, and the host CPU of a Modal container is hidden, so CPU-bound steps (the client's decrypt and ranking) move between runs. Compare ratios and orders of magnitude before you compare single milliseconds.

## The make targets

| Command | What it runs |
| --- | --- |
| `make bench TASK=<name>` | `core/bench/run.py` on an A100 on Modal, results to a Modal volume |
| `make results-pull` | copies the results from the volume into a local folder (see the `results-pull` target in the `Makefile`) |
| `make test-gpu` | the GPU tests on an A10 on Modal (they check kernels against OpenFHE) |
| `make test-openfhe` | the tests that need OpenFHE, CPU only, in the pinned image |
| `make test` | the CPU tests, locally, no OpenFHE or GPU needed |

`make bench` takes a task name. `column-search`, `expansion`, `batching`, `e2e-latency`, `gpu-cache`, `filter-measurements` and `grpc-e2e` are the tasks in `core/bench/run.py`. Pass extra options to a task through Modal: `modal run modal_app.py::bench --task <name> --extra "<options>"`.

## Which script made which number

| Figures | Script and command | Hardware |
| --- | --- | --- |
| Latency per query, by dimension, size and share searched | `core/bench/e2e.py`, `modal run modal_app.py::bench --task e2e-latency --extra "--presets-e2e-latency m0_depth2,m0_compact_n8192"` | A100 40 GB |
| Over gRPC, with a filter, 100k and 1M vectors, ingest speed, bytes on disk, recall | `core/bench/grpc_e2e.py`, `modal run modal_app.py::bench_e2e --extra "--presets-grpc <preset> --sizes-grpc <N>"`, one run per preset and size | A100 80 GB, 16 CPUs |
| Search kernel, batching | `core/bench/run.py --task column-search`, `--task batching` | A100 40 GB |
| Cache paging and host to GPU bandwidth | `core/bench/run.py --task gpu-cache` | A100 40 GB |
| Object sizes, keys, vectors per GPU | `modal run modal_app.py::test_openfhe --pytest-args "-s -k test_openfhe_sizes"` | CPU only |
| Local mode search and create time | `python tests/bench_local_mode.py --d <d>`, pinned with `taskset -c <cpus>` | a Ryzen 5 5600H laptop |
| Filtered recall and latency on real text | `core/bench/filtered_recall.py`, `modal run modal_app.py::filtered_recall` | A100 |
| Recall of a preset against plaintext | `modal run modal_app.py::recall --preset <preset>` | A100 |

`m0_depth2` is the Balanced preset and `m0_compact_n8192` is Fast; the benchmark scripts use these names, which come from `core/params/ckks_presets.json`. The Python client has its own recall check, `ombra-recall`, which compares plaintext and encrypted recall on your own data.

Each run writes its raw JSON. Keep it together with a short note that gives the command, the commit, the GPU, the CUDA and OpenFHE versions and the parameters used.

## Where the data in the charts comes from

The charts are drawn by `docs/charts/make_charts.py` from three small CSV files next to it: `latency.csv`, `storage.csv` and `e2e.csv`. Each row names where its numbers came from, and a number that waits for a new run says TBD. A chart with TBD numbers is not drawn. To redraw them:

```bash
pip install -r docs/charts/requirements.txt
python docs/charts/make_charts.py
```

This writes the SVG files in `docs/site/performance/img/` and removes the SVG of a chart whose numbers are TBD. The script produces the same bytes every time, so a `git diff` after running it tells you whether a chart is current. If you rerun a benchmark, copy the new numbers into the CSV file, run the script, and commit both.

## Checking the sizes yourself

The storage figures need no GPU. The sizes test (`tests/test_openfhe_sizes.py`) prints the serialized size of every object for a preset and dimension on a CPU. The per-vector size follows from it: a ciphertext holds one dimension for as many vectors as there are slots, so divide its size by the slots. A stored column is half of the printed size when it is seeded, because the random half is replaced by a 32-byte seed.
