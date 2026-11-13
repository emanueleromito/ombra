# NNNN. Title

- **Outcome:** adopted | rejected | inconclusive | baseline | needs approval | not run
- **Date:** YYYY-MM-DD
- **Area:** GPU kernel | GPU host | client CPU | server CPU | memory | GPU families

## What was tried and why

Two or three sentences: the idea, where it comes from (paper, library, own profile) and what it
was expected to change.

## Setup

| | |
| --- | --- |
| Commit | `abcdef0` (branch) |
| GPU | name, memory, driver |
| CPU | model, threads used |
| CUDA / OpenFHE | versions |
| Preset | `m0_depth2`, `m0_compact_n8192` |
| Dimensions | d = 384, 768, 1536 |
| Data | synthetic, how generated |
| Seeds | every seed used |

## Command

The single command that reruns it, from the repository root:

```
...
```

Raw output: `raw/` (JSON). Tables and figures: `python analyze.py raw/<file>.json`.

## Exactness

How the candidate was checked against the OpenFHE oracle (which test, which sizes) and the result.
A candidate that is not exact is rejected, whatever its speed.

## Results

Table from `analyze.py`. Before and after from the same run on the same machine.

## Reading

What the numbers mean, and what they do not show.

## Limits

One line each: what was not measured, what may differ on another machine.
