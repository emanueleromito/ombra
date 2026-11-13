# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Benchmark runner for core/ kernels (run via `make bench`, on an A100 in Modal).

    python core/bench/run.py --task column-search --out <dir>      # or --task all

Column search: column-packed search throughput for d = 384, 768, 1536. This script creates the CKKS
context from the `m0_depth2` preset and a key pair, writes ONLY the context and the public key to a
temporary directory (the secret key is never serialized or used), and runs the C++ benchmark binary
`ombra_bench_column_search` once per dimension. It merges the binary's per-run JSON into
`<out>/column-search-a100.json`. Nothing written contains plaintext or keys.

expansion: GPU query expansion time for d = 384, 768, 1536 (`ombra_bench_expand`, one run per d).
Per d this script generates the automorphism keys for N/2^k + 1, k < ceil(log2(2d)), in-process
(that needs the secret key, which stays in this process and is never written), and writes ONLY the
context, the public key and the automorphism keys (public evaluation keys) to a temporary directory.
Output: `<out>/expansion-a100.json` with d, L, nodes, automorphisms, keys, key_bytes, gpu_seconds
and iterations per d.

    python core/bench/run.py --task expansion --out <dir>

Query batching (`ombra_bench_batch`, one run per d): search and expansion throughput for
Q = 1, 8, 32, 128 queries per pass over the same stored blocks, the batched search kernel's
achieved memory bandwidth, and the per-(d, Q) memory plan (queries per pass, passes, peak device
memory). Output `<out>/batching-a100.json`; speedups are relative to the Q = 1 row of the same run.
Runs only when asked for (`--task batching`), not as part of `--task all`.

End-to-end query latency per step (client encrypt, expansion, stage 1, stage 2, client decrypt and
top-k), see core/bench/e2e.py. Output `<out>/e2e-latency-a100.json`. Runs only when asked for
(`--task e2e-latency`), not as part of `--task all`.

GPU cluster cache (`ombra_bench_cache`, built from server/cache): paging bandwidth pinned vs
pageable, serial vs overlapped search over a collection 3x the GPU budget, hit rate for uniform and
Zipf access. Output `<out>/gpu-cache-a100.json`. Runs only when asked for (`--task gpu-cache`).

Filter measurements: the filter measurements through the gRPC server
(core/bench/filter_measurements.py): score error with filter columns, the recall rule with the extra
columns, filtered recall with the selective-filter loop. Output
`<out>/filter-measurements-<preset>-a100.json` for each preset in
`--presets-filter`. Runs only when asked for (`--task filter-measurements`).
gRPC end-to-end (`--task grpc-e2e`): end-to-end over gRPC at 100k and 1M, see grpc_e2e.py.

    python core/bench/run.py --task batching --out <dir>
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Before openfhe loads: forked client workers can deadlock if the parent has OpenMP threads.
if "grpc-e2e" in _sys.argv:
    _os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "client"))

PRESET = "m0_depth2"
COLUMN_SEARCH_DIMS = (384, 768, 1536)
# Stored data per benchmark is kept near constant: blocks = BLOCK_DIM_BUDGET // d. At the preset
# (3 towers, ring 16384, 8192 slots) one ciphertext is 786,432 bytes, so the stored data is about
# 16 * 1536 * 786,432 B = 19.3 GB for every d, plus 1.2 GB (d = 1536) of query: under 40 GB.
BLOCK_DIM_BUDGET = 16 * 1536
DEFAULT_WARMUP = 2
DEFAULT_ITERS = 10
DEFAULT_OUT_NAME = "column-search-a100.json"
EXPANSION_OUT_NAME = "expansion-a100.json"
BATCHING_OUT_NAME = "batching-a100.json"
GPU_CACHE_OUT_NAME = "gpu-cache-a100.json"
BATCHING_QUERIES = (1, 8, 32, 128)
# batching stored data: STORED_CTS ciphertexts per d (blocks = STORED_CTS // d), 4 * 1536 * 786,432
# B =
# 4.8 GB. Smaller than column-search's 19 GB on purpose: expanded queries are 1.2 GB each at d =
# 1536, so a large stored set would leave room for only a handful of them in 40 GB.
BATCHING_STORED_CTS = 4 * 1536
# Device-memory budget for everything the process holds (CUDA context, keys, stored blocks,
# queries, scratch). Under the A100-40GB's 40 GiB so that planning error and the driver's own use
# do not cause an out-of-memory failure.
BATCHING_BUDGET_GIB = 34.0
# NVIDIA's published peak memory bandwidth of the A100 40 GB SXM (HBM2): 1,555 GB/s. A published
# figure, not measured here.
A100_40GB_PUBLISHED_PEAK_GBPS = 1555.0


def find_binary(name: str) -> Path:
    env = os.environ.get("OMBRA_BENCH_BIN_DIR")
    candidates = [
        Path(env) / name if env else None,
        Path("/root/ombra-build/core") / name,
        Path("/root/ombra-build/server/cache") / name,
        REPO_ROOT / "build" / "core" / name,
        REPO_ROOT / "build" / "server" / "cache" / name,
    ]
    for c in candidates:
        if c is not None and c.is_file():
            return c
    raise SystemExit(f"{name} not found; build with `make build` (or via modal_app.py)")


def openfhe_version() -> str:
    """Installed OpenFHE C++ version, read from the config file `cmake --install` wrote."""
    for path in sorted(Path("/usr/local").glob("lib*/OpenFHE/OpenFHEConfigVersion.cmake")):
        match = re.search(r'set\(PACKAGE_VERSION\s+"([^"]+)"\)', path.read_text())
        if match:
            return match.group(1)
    return "unknown"


def run_column_search(
    out_dir: Path,
    dims: tuple[int, ...],
    blocks: int | None,
    warmup: int,
    iters: int,
    out_name: str,
) -> Path:
    import openfhe as ofhe
    from ombra import params as ombra_params

    binary = find_binary("ombra_bench_column_search")
    _, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()

    results = []
    gpu_name = "unknown"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Public material only. The secret key stays in this process and is never written.
        if not ofhe.SerializeToFile(str(tmp_dir / "context.bin"), cc, ofhe.BINARY):
            raise SystemExit("cannot serialize context")
        if not ofhe.SerializeToFile(str(tmp_dir / "public_key.bin"), keys.publicKey, ofhe.BINARY):
            raise SystemExit("cannot serialize public key")
        for d in dims:
            n_blocks = blocks if blocks is not None else max(1, BLOCK_DIM_BUDGET // d)
            part = tmp_dir / f"part_{d}.json"
            cmd = [
                str(binary),
                str(tmp_dir / "context.bin"),
                str(tmp_dir / "public_key.bin"),
                str(d),
                str(n_blocks),
                str(warmup),
                str(iters),
                str(part),
            ]
            print("$", " ".join(cmd), flush=True)
            rows = {}
            for kernel in ("mont", "lazy"):  # old Montgomery kernel, then the lazy 128-bit one
                env = {**os.environ, "OMBRA_COLUMN_SEARCH_KERNEL": kernel}
                subprocess.run(cmd, check=True, env=env)  # noqa: S603
                entry = json.loads(part.read_text())
                gpu_name = entry.pop("gpu_name")
                rows[kernel] = entry
                results.append(entry)
                print(
                    f"d={d} blocks={n_blocks} kernel={kernel} "
                    f"vectors/s={entry['vectors_per_second']:.3e}",
                    flush=True,
                )
            speedup = rows["mont"]["kernel_seconds_median"] / rows["lazy"]["kernel_seconds_median"]
            rows["lazy"]["speedup_vs_mont"] = speedup
            print(f"d={d} lazy speedup over mont: {speedup:.3f}x", flush=True)

    record = {
        "task": "column-search",
        "format_version": 1,
        "preset": PRESET,
        "openfhe_version": openfhe_version(),
        "gpu_name": gpu_name,
        "note": (
            "Column-packed search: sum over d of stored_j * query_j (ciphertext x ciphertext, "
            "evaluation form, 3 components). Relinearization and rescale are excluded. Kernel "
            "time only (CUDA events); copies are reported separately."
        ),
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out_path}")
    return out_path


def run_expansion(
    out_dir: Path,
    dims: tuple[int, ...],
    warmup: int,
    iters: int,
    out_name: str,
) -> Path:
    import openfhe as ofhe
    from ombra import params as ombra_params
    from ombra import query as oq

    binary = find_binary("ombra_bench_expand")
    _, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()
    ring = cc.GetRingDimension()

    results = []
    gpu_name = "unknown"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Public material only. The secret key is used in-process to derive the automorphism
        # keys and is never serialized.
        if not ofhe.SerializeToFile(str(tmp_dir / "context.bin"), cc, ofhe.BINARY):
            raise SystemExit("cannot serialize context")
        if not ofhe.SerializeToFile(str(tmp_dir / "public_key.bin"), keys.publicKey, ofhe.BINARY):
            raise SystemExit("cannot serialize public key")
        for d in dims:
            indices = oq.automorphism_indices(ring, d)
            cc.ClearEvalAutomorphismKeys()  # the key store is static: exactly this d's indices
            cc.EvalAutomorphismKeyGen(keys.secretKey, indices)
            key_path = tmp_dir / f"autokeys_{d}.bin"
            if not cc.SerializeEvalAutomorphismKey(str(key_path), ofhe.BINARY):
                raise SystemExit("cannot serialize automorphism keys")
            part = tmp_dir / f"part_{d}.json"
            cmd = [
                str(binary),
                str(tmp_dir / "context.bin"),
                str(tmp_dir / "public_key.bin"),
                str(key_path),
                str(d),
                str(warmup),
                str(iters),
                str(part),
            ]
            print("$", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)  # noqa: S603
            entry = json.loads(part.read_text())
            gpu_name = entry.pop("gpu_name")
            results.append(entry)
            print(
                f"d={d} L={entry['L']} automorphisms={entry['automorphisms']} "
                f"keys={entry['keys']} key_bytes={entry['key_bytes']} "
                f"gpu_seconds={entry['gpu_seconds']:.4f}",
                flush=True,
            )

    record = {
        "task": "expansion",
        "format_version": 1,
        "preset": PRESET,
        "openfhe_version": openfhe_version(),
        "gpu_name": gpu_name,
        "note": (
            "GPU query expansion tree (one key-switched automorphism per split, d leaves). "
            "gpu_seconds is the median wall time of the whole tree with leaves resident on the "
            "device, excluding key and root upload and warm-up. key_bytes is the size of the "
            "serialized automorphism key file."
        ),
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out_path}")
    return out_path


def run_gpu_cache(
    out_dir: Path, dims: tuple[int, ...], budget_blocks: int, per_block: int, out_name: str
) -> Path:
    """GPU cluster cache (`ombra_bench_cache`): paging bandwidth, overlap, hit rate.

    Per d (block size) one run with a collection of 3 * budget_blocks blocks and a GPU budget of
    budget_blocks blocks. Only the context and the public key are written (no secret key).
    """
    import openfhe as ofhe
    from ombra import params as ombra_params

    binary = find_binary("ombra_bench_cache")
    _, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()
    results = []
    gpu_name = "unknown"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        if not ofhe.SerializeToFile(str(tmp_dir / "context.bin"), cc, ofhe.BINARY):
            raise SystemExit("cannot serialize context")
        if not ofhe.SerializeToFile(str(tmp_dir / "public_key.bin"), keys.publicKey, ofhe.BINARY):
            raise SystemExit("cannot serialize public key")
        for d in dims:
            part = tmp_dir / f"part_{d}.json"
            cmd = [
                str(binary),
                str(tmp_dir / "context.bin"),
                str(tmp_dir / "public_key.bin"),
                str(d),
                str(budget_blocks),
                str(per_block),
                str(part),
            ]
            print("$", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)  # noqa: S603
            entry = json.loads(part.read_text())
            gpu_name = entry.pop("gpu_name")
            results.append(entry)
    record = {
        "task": "gpu-cache",
        "format_version": 1,
        "preset": PRESET,
        "openfhe_version": openfhe_version(),
        "gpu_name": gpu_name,
        "gpu_memory_clocks": gpu_memory_clock_record(),
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out_path}")
    return out_path


def gpu_memory_clock_record() -> dict:
    """Output of `nvidia-smi --query-gpu=clocks.max.memory,memory.total --format=csv`."""
    try:
        proc = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=clocks.max.memory,memory.total", "--format=csv"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return {"command_output": proc.stdout.strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command_output": f"unavailable: {exc}"}


def add_speedups(rows: list[dict]) -> None:
    """Adds `speedup_vs_q1` to every row, relative to the Q = 1 row of the same list."""
    base = next((r for r in rows if r["queries"] == 1), None)
    if base is None:
        return
    for r in rows:
        r["speedup_vs_q1"] = r["queries_per_second"] / base["queries_per_second"]


def run_batching(
    out_dir: Path,
    dims: tuple[int, ...],
    queries: tuple[int, ...],
    stored_cts: int,
    budget_gib: float,
    max_per_pass: int,
    warmup: int,
    iters: int,
    out_name: str,
) -> Path:
    import openfhe as ofhe
    from ombra import params as ombra_params
    from ombra import query as oq

    binary = find_binary("ombra_bench_batch")
    _, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()
    ring = cc.GetRingDimension()
    budget = int(budget_gib * 1024**3)

    results = []
    gpu_name = "unknown"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Public material only. The secret key is used in-process to derive the automorphism
        # keys and is never serialized.
        if not ofhe.SerializeToFile(str(tmp_dir / "context.bin"), cc, ofhe.BINARY):
            raise SystemExit("cannot serialize context")
        if not ofhe.SerializeToFile(str(tmp_dir / "public_key.bin"), keys.publicKey, ofhe.BINARY):
            raise SystemExit("cannot serialize public key")
        for d in dims:
            indices = oq.automorphism_indices(ring, d)
            cc.ClearEvalAutomorphismKeys()  # the key store is static: exactly this d's indices
            cc.EvalAutomorphismKeyGen(keys.secretKey, indices)
            key_path = tmp_dir / f"autokeys_{d}.bin"
            if not cc.SerializeEvalAutomorphismKey(str(key_path), ofhe.BINARY):
                raise SystemExit("cannot serialize automorphism keys")
            part = tmp_dir / f"part_{d}.json"
            n_blocks = max(1, stored_cts // d)
            cmd = [
                str(binary),
                str(tmp_dir / "context.bin"),
                str(tmp_dir / "public_key.bin"),
                str(key_path),
                str(d),
                str(n_blocks),
                str(warmup),
                str(iters),
                str(budget),
                ",".join(str(q) for q in queries),
                str(part),
                *([str(max_per_pass)] if max_per_pass else []),
            ]
            print("$", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)  # noqa: S603
            entry = json.loads(part.read_text())
            gpu_name = entry.pop("gpu_name")
            add_speedups(entry["search"])
            add_speedups(entry["expansion"])
            results.append(entry)
            for kind in ("search", "expansion"):
                for r in entry[kind]:
                    print(
                        f"d={d} {kind} Q={r['queries']} per_pass={r['queries_per_pass']} "
                        f"passes={r['passes']} q/s={r['queries_per_second']:.3f} "
                        f"speedup={r['speedup_vs_q1']:.2f} "
                        f"peak_GiB={r['peak_device_bytes_used'] / 1024**3:.2f}",
                        flush=True,
                    )

    record = {
        "task": "batching",
        "format_version": 1,
        "preset": PRESET,
        "openfhe_version": openfhe_version(),
        "gpu_name": gpu_name,
        "gpu_memory_query": gpu_memory_clock_record(),
        "published_peak_memory_bandwidth_gb_per_s": A100_40GB_PUBLISHED_PEAK_GBPS,
        "published_peak_note": (
            "NVIDIA's published peak memory bandwidth of the A100 40 GB (1,555 GB/s); a "
            "published figure, not a measurement."
        ),
        "memory_budget_bytes": budget,
        "note": (
            "Query batching. search: Q queries scored against the same stored blocks by the "
            "batched kernel (3-component results, no relinearization or rescale); total time is "
            "the wall clock over all passes. expansion: trees of Q queries run together, one "
            "batched split per tree level; total time is the wall clock over all passes, leaves "
            "dropped after each pass. When Q does not fit the memory budget the work runs as "
            "consecutive passes (queries_per_pass, passes). Medians of timed iterations after "
            "warm-up. achieved_gb_per_s_access_pattern counts bytes the kernel's threads request "
            "and write (caches may serve repeats); achieved_gb_per_s_compulsory counts each "
            "stored, query and result word once. Search queries are copies of one synthetic "
            "query and the stored blocks copies of one block (speed does not depend on values)."
        ),
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="all",
        help="column-search, expansion, batching or all (all = column-search and expansion)",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dims", default=",".join(map(str, COLUMN_SEARCH_DIMS)))
    parser.add_argument("--blocks", type=int, default=None, help="override the block count")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument(
        "--name", default=DEFAULT_OUT_NAME, help="output file name for column-search"
    )
    parser.add_argument(
        "--name-expansion", default=EXPANSION_OUT_NAME, help="output file name for expansion"
    )
    parser.add_argument(
        "--name-batching", default=BATCHING_OUT_NAME, help="output file name for batching"
    )
    parser.add_argument(
        "--name-gpu-cache", default=GPU_CACHE_OUT_NAME, help="output file name for gpu-cache"
    )
    parser.add_argument(
        "--dims-gpu-cache", default="16,64,256", help="gpu-cache: dims = block sizes"
    )
    parser.add_argument(
        "--budget-blocks", type=int, default=8, help="gpu-cache: GPU budget in blocks"
    )
    parser.add_argument(
        "--passes-gpu-cache", type=int, default=4, help="gpu-cache: passes over the data"
    )
    parser.add_argument(
        "--queries",
        default=",".join(map(str, BATCHING_QUERIES)),
        help="batching: comma-separated batch sizes",
    )
    parser.add_argument(
        "--stored-cts",
        type=int,
        default=BATCHING_STORED_CTS,
        help="batching: stored ciphertexts per d (blocks = this // d)",
    )
    parser.add_argument(
        "--mem-budget-gib",
        type=float,
        default=BATCHING_BUDGET_GIB,
        help="batching: device-memory budget",
    )
    parser.add_argument(
        "--max-per-pass",
        type=int,
        default=0,
        help="batching: extra cap on queries per pass (0 = only the memory budget; tests use it)",
    )
    parser.add_argument(
        "--name-e2e-latency",
        default="e2e-latency-a100.json",
        help="output file name for e2e-latency",
    )
    parser.add_argument(
        "--presets-e2e-latency",
        default="",
        help="e2e-latency: comma-separated presets from core/params, run in one go; "
        "empty = m0_depth2 only, written to --name-e2e-latency",
    )
    parser.add_argument(
        "--e2e-queries", type=int, default=20, help="e2e-latency: timed queries per case"
    )
    parser.add_argument(
        "--e2e-warmup", type=int, default=3, help="e2e-latency: untimed queries per case"
    )
    parser.add_argument(
        "--presets-filter",
        default="m0_depth2,m0_compact_n8192",
        help="filter-measurements: presets, one file each",
    )
    parser.add_argument(
        "--queries-filter", type=int, default=1000, help="filter-measurements: queries for part 2"
    )
    parser.add_argument(
        "--presets-grpc", default="m0_depth2,m0_compact_n8192", help="grpc-e2e: presets"
    )
    parser.add_argument("--sizes-grpc", default="100000,1000000", help="grpc-e2e: vector counts")
    parser.add_argument(
        "--queries-grpc", type=int, default=30, help="grpc-e2e: queries per variant"
    )
    parser.add_argument(
        "--workers-grpc", type=int, default=0, help="grpc-e2e: client processes (0 = affinity)"
    )
    parser.add_argument("--rerun-grpc", action="store_true", help="grpc-e2e: redo existing files")
    args = parser.parse_args()

    task = args.task.lower()
    if task not in (
        "all",
        "column-search",
        "expansion",
        "batching",
        "e2e-latency",
        "gpu-cache",
        "filter-measurements",
        "grpc-e2e",
    ):
        raise SystemExit(f"unknown task {args.task!r}")
    dims = tuple(int(x) for x in args.dims.split(","))
    if task in ("all", "column-search"):
        run_column_search(args.out, dims, args.blocks, args.warmup, args.iters, args.name)
    if task in ("all", "expansion"):
        run_expansion(args.out, dims, args.warmup, args.iters, args.name_expansion)
    if task == "grpc-e2e":
        from grpc_e2e import run_grpc

        run_grpc(
            args.out,
            tuple(args.presets_grpc.split(",")),
            tuple(int(x) for x in args.sizes_grpc.split(",")),
            args.queries_grpc,
            args.workers_grpc,
            args.rerun_grpc,
        )
    if task == "filter-measurements":
        from filter_measurements import run_filter

        run_filter(args.out, tuple(args.presets_filter.split(",")), args.queries_filter)
    if task == "gpu-cache":
        run_gpu_cache(
            args.out,
            tuple(int(x) for x in args.dims_gpu_cache.split(",")),
            args.budget_blocks,
            args.passes_gpu_cache,
            args.name_gpu_cache,
        )
    if task == "batching":
        run_batching(
            args.out,
            dims,
            tuple(int(x) for x in args.queries.split(",")),
            args.stored_cts,
            args.mem_budget_gib,
            args.max_per_pass,
            args.warmup,
            args.iters,
            args.name_batching,
        )
    if task == "e2e-latency":
        from e2e import run_e2e_latency

        if args.presets_e2e_latency:
            for preset in args.presets_e2e_latency.split(","):
                run_e2e_latency(
                    args.out,
                    dims,
                    args.e2e_queries,
                    args.e2e_warmup,
                    find_binary,
                    openfhe_version(),
                    name=f"compact-latency-{preset}-a100.json",
                    preset=preset,
                )
        else:
            run_e2e_latency(
                args.out,
                dims,
                args.e2e_queries,
                args.e2e_warmup,
                find_binary,
                openfhe_version(),
                name=args.name_e2e_latency,
            )


if __name__ == "__main__":
    main()
