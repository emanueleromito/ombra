"""Benchmark runner for core/ kernels (run via `make bench`, on an A100 in Modal).

    python core/bench/run.py --task t0.4 --out <dir>      # or --task all

T0.4: column-packed search throughput for d = 384, 768, 1536. This script creates the CKKS context
from the `m0_depth2` preset and a key pair, writes ONLY the context and the public key to a
temporary directory (the secret key is never serialized or used), and runs the C++ benchmark binary
`ombra_bench_column_search` once per dimension. It merges the binary's per-run JSON into
`<out>/t0.4-a100.json`. Nothing written contains plaintext or keys.

T0.5: GPU query expansion time for d = 384, 768, 1536 (`ombra_bench_expand`, one run per d). Per d
this script generates the automorphism keys for N/2^k + 1, k < ceil(log2(2d)), in-process (that
needs the secret key, which stays in this process and is never written), and writes ONLY the
context, the public key and the automorphism keys (public evaluation keys) to a temporary
directory. Output: `<out>/t0.5-a100.json` with d, L, nodes, automorphisms, keys, key_bytes,
gpu_seconds and iterations per d.

    python core/bench/run.py --task t0.5 --out <dir>

T0.11: compact configurations (ring dimension 8192 preset, fp16/int8 plaintext storage), see
core/bench/t011.py. Only runs when asked for: `--task t0.11` (not part of `all`).

T0.7: query batching (`ombra_bench_batch`, one run per d): search and expansion throughput for
Q = 1, 8, 32, 128 queries per pass over the same stored blocks, the batched search kernel's
achieved memory bandwidth, and the per-(d, Q) memory plan (queries per pass, passes, peak device
memory). Output `<out>/t0.7-a100.json`; speedups are relative to the Q = 1 row of the same run.
Runs only when asked for (`--task t0.7`), not as part of `--task all`.

T1.3: end-to-end query latency per step (client encrypt, expansion, stage 1, stage 2, client
decrypt and top-k), see core/bench/e2e.py. Output `<out>/t1.3-a100.json`. Runs only when asked for
(`--task t1.3`), not as part of `--task all`.

T2.6: GPU cluster cache (`ombra_bench_cache`, built from server/cache): paging bandwidth pinned vs
pageable, serial vs overlapped search over a collection 3x the GPU budget, hit rate for uniform and
Zipf access. Output `<out>/t2.6-a100.json`. Runs only when asked for (`--task t2.6`).

T2.7: the measurements of decision 0019 through the gRPC server (core/bench/filter_0019.py): score
error with filter columns, the decision-0012 recall rule with the extra columns, filtered recall
with the selective-filter loop. Output `<out>/t2.7-<preset>-a100.json` for each preset in
`--presets-t27`. Runs only when asked for (`--task t2.7`).
T2.10 (`--task t2.10`): end-to-end over gRPC at 100k and 1M, see grpc_e2e.py.

    python core/bench/run.py --task t0.7 --out <dir>
"""

from __future__ import annotations

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
T04_DIMS = (384, 768, 1536)
# Stored data per benchmark is kept near constant: blocks = BLOCK_DIM_BUDGET // d. At the preset
# (3 towers, ring 16384, 8192 slots) one ciphertext is 786,432 bytes, so the stored data is about
# 16 * 1536 * 786,432 B = 19.3 GB for every d, plus 1.2 GB (d = 1536) of query: under 40 GB.
BLOCK_DIM_BUDGET = 16 * 1536
DEFAULT_WARMUP = 2
DEFAULT_ITERS = 10
DEFAULT_OUT_NAME = "t0.4-a100.json"
T05_OUT_NAME = "t0.5-a100.json"
T07_OUT_NAME = "t0.7-a100.json"
T26_OUT_NAME = "t2.6-a100.json"
T07_QUERIES = (1, 8, 32, 128)
# T0.7 stored data: STORED_CTS ciphertexts per d (blocks = STORED_CTS // d), 4 * 1536 * 786,432 B =
# 4.8 GB. Smaller than T0.4's 19 GB on purpose: expanded queries are 1.2 GB each at d = 1536, so a
# large stored set would leave room for only a handful of them in 40 GB.
T07_STORED_CTS = 4 * 1536
# Device-memory budget for everything the process holds (CUDA context, keys, stored blocks,
# queries, scratch). Under the A100-40GB's 40 GiB so that planning error and the driver's own use
# do not cause an out-of-memory failure.
T07_BUDGET_GIB = 34.0
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


def run_t04(
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
            subprocess.run(cmd, check=True)  # noqa: S603
            entry = json.loads(part.read_text())
            gpu_name = entry.pop("gpu_name")
            results.append(entry)
            print(
                f"d={d} blocks={n_blocks} vectors/s={entry['vectors_per_second']:.3e}",
                flush=True,
            )

    record = {
        "task": "t0.4",
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


def run_t05(
    out_dir: Path,
    dims: tuple[int, ...],
    warmup: int,
    iters: int,
    out_name: str,
    profile: bool = False,
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
                *(["profile"] if profile else []),
            ]
            print("$", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)  # noqa: S603
            entry = json.loads(part.read_text())
            gpu_name = entry.pop("gpu_name")
            results.append(entry)
            if profile:
                print("profile", json.dumps(entry.get("profile")), flush=True)
            print(
                f"d={d} L={entry['L']} automorphisms={entry['automorphisms']} "
                f"keys={entry['keys']} key_bytes={entry['key_bytes']} "
                f"gpu_seconds={entry['gpu_seconds']:.4f}",
                flush=True,
            )

    record = {
        "task": "t0.5",
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


def run_t26(
    out_dir: Path, dims: tuple[int, ...], budget_blocks: int, per_block: int, out_name: str
) -> Path:
    """T2.6: GPU cluster cache (`ombra_bench_cache`): paging bandwidth, overlap, hit rate.

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
        "task": "t2.6",
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


def run_t07(
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
        "task": "t0.7",
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
        "--task", default="all", help="t0.4, t0.5, t0.7 or all (all = t0.4 and t0.5)"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dims", default=",".join(map(str, T04_DIMS)))
    parser.add_argument("--blocks", type=int, default=None, help="override the block count")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--name", default=DEFAULT_OUT_NAME, help="output file name for t0.4")
    parser.add_argument("--name-t05", default=T05_OUT_NAME, help="output file name for t0.5")
    parser.add_argument("--name-t07", default=T07_OUT_NAME, help="output file name for t0.7")
    parser.add_argument("--name-t26", default=T26_OUT_NAME, help="output file name for t2.6")
    parser.add_argument("--dims-t26", default="16,64,256", help="t2.6: dims = block sizes")
    parser.add_argument("--budget-blocks", type=int, default=8, help="t2.6: GPU budget in blocks")
    parser.add_argument("--passes-t26", type=int, default=4, help="t2.6: passes over the data")
    parser.add_argument(
        "--queries",
        default=",".join(map(str, T07_QUERIES)),
        help="t0.7: comma-separated batch sizes",
    )
    parser.add_argument(
        "--stored-cts",
        type=int,
        default=T07_STORED_CTS,
        help="t0.7: stored ciphertexts per d (blocks = this // d)",
    )
    parser.add_argument(
        "--mem-budget-gib", type=float, default=T07_BUDGET_GIB, help="t0.7: device-memory budget"
    )
    parser.add_argument(
        "--max-per-pass",
        type=int,
        default=0,
        help="t0.7: extra cap on queries per pass (0 = only the memory budget; tests use it)",
    )
    parser.add_argument(
        "--profile", action="store_true", help="t0.5: also time the stages of one split"
    )
    parser.add_argument("--name-t13", default="t1.3-a100.json", help="output file name for t1.3")
    parser.add_argument(
        "--presets-t13",
        default="",
        help="t1.3: comma-separated presets from core/params, run in one go (T2.2); "
        "empty = m0_depth2 only, written to --name-t13",
    )
    parser.add_argument("--e2e-queries", type=int, default=20, help="t1.3: timed queries per case")
    parser.add_argument("--e2e-warmup", type=int, default=3, help="t1.3: untimed queries per case")
    parser.add_argument(
        "--presets-t27", default="m0_depth2,m0_compact_n8192", help="t2.7: presets, one file each"
    )
    parser.add_argument("--queries-t27", type=int, default=1000, help="t2.7: queries for part 2")
    parser.add_argument(
        "--presets-t210", default="m0_depth2,m0_compact_n8192", help="t2.10: presets"
    )
    parser.add_argument("--sizes-t210", default="100000,1000000", help="t2.10: vector counts")
    parser.add_argument("--queries-t210", type=int, default=30, help="t2.10: queries per variant")
    parser.add_argument(
        "--workers-t210", type=int, default=0, help="t2.10: client processes (0 = affinity)"
    )
    parser.add_argument("--rerun-t210", action="store_true", help="t2.10: redo existing files")
    parser.add_argument("--presets-t011", default="m0_depth2,m0_compact_n8192", help="t0.11")
    parser.add_argument(
        "--error-presets-t011", default="m0_compact_n8192", help="t0.11: presets with the e2e error"
    )
    args = parser.parse_args()

    task = args.task.lower()
    if task not in ("all", "t0.4", "t0.5", "t0.7", "t0.11", "t1.3", "t2.6", "t2.7", "t2.10"):
        raise SystemExit(f"unknown task {args.task!r}")
    dims = tuple(int(x) for x in args.dims.split(","))
    if task in ("all", "t0.4"):
        run_t04(args.out, dims, args.blocks, args.warmup, args.iters, args.name)
    if task in ("all", "t0.5"):
        run_t05(args.out, dims, args.warmup, args.iters, args.name_t05, args.profile)
    if task == "t2.10":
        from grpc_e2e import run_t210

        run_t210(
            args.out,
            tuple(args.presets_t210.split(",")),
            tuple(int(x) for x in args.sizes_t210.split(",")),
            args.queries_t210,
            args.workers_t210,
            args.rerun_t210,
        )
    if task == "t2.7":
        from filter_0019 import run_t27

        run_t27(args.out, tuple(args.presets_t27.split(",")), args.queries_t27)
    if task == "t2.6":
        run_t26(
            args.out,
            tuple(int(x) for x in args.dims_t26.split(",")),
            args.budget_blocks,
            args.passes_t26,
            args.name_t26,
        )
    if task == "t0.7":
        run_t07(
            args.out,
            dims,
            tuple(int(x) for x in args.queries.split(",")),
            args.stored_cts,
            args.mem_budget_gib,
            args.max_per_pass,
            args.warmup,
            args.iters,
            args.name_t07,
        )
    if task == "t1.3":
        from e2e import run_t13

        if args.presets_t13:
            for preset in args.presets_t13.split(","):
                run_t13(
                    args.out,
                    dims,
                    args.e2e_queries,
                    args.e2e_warmup,
                    find_binary,
                    openfhe_version(),
                    name=f"t2.2-{preset}-a100.json",
                    preset=preset,
                )
        else:
            run_t13(
                args.out,
                dims,
                args.e2e_queries,
                args.e2e_warmup,
                find_binary,
                openfhe_version(),
                name=args.name_t13,
            )
    if task == "t0.11":
        from t011 import run_t011

        run_t011(
            args.out,
            dims,
            tuple(args.presets_t011.split(",")),
            tuple(x for x in args.error_presets_t011.split(",") if x),
            args.warmup,
            args.iters,
            find_binary,
            openfhe_version(),
        )


if __name__ == "__main__":
    main()
