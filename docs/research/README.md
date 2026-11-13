# Research log

One folder per experiment, kept whether it worked or not, so that anyone can rerun it and nobody
repeats a dead end. Benchmark results of plan tasks stay in `docs/plan/results/`; this folder is for
experiments that compare candidates.

## Rules

- **One folder per experiment:** `NNNN-short-name/` with
  - `README.md`: what was tried and why, the exact setup, the single command that reruns it, the
    results and their reading (copy [template.md](template.md));
  - `run.py` or `run.sh`: that single command's script;
  - `raw/`: the raw output, unedited, one JSON file per run, each starting with the environment
    record from [tools/envinfo.py](tools/envinfo.py) (commit, GPU, CPU, versions);
  - `analyze.py`: turns the raw output into the tables and figures of the README.
- **Failed experiments stay,** with the outcome "rejected" and the reason.
- **Exact or rejected.** Every candidate is compared with the OpenFHE oracle. A candidate that
  changes an RNS word, or recall, is rejected whatever its speed.
- **Same-run comparisons.** Before and after are measured in one run on one machine. Numbers from
  different machines are never compared directly (the M1 refresh showed 26% run-to-run variation).
- **Synthetic data only** unless the README names a public dataset. No keys, plaintext or scores in
  `raw/`.
- **Hard rules still apply** (`CLAUDE.md`): 128-bit floor, parameters from presets only. A
  candidate that needs a preset, level-budget, security-level or format change is recorded as
  "needs approval" and is not built into the product without the user's decision.

## Index

| # | Experiment | Area | Outcome |
| --- | --- | --- | --- |
