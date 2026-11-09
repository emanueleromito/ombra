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


def find_binary(name: str) -> Path:
    env = os.environ.get("OMBRA_BENCH_BIN_DIR")
    candidates = [
        Path(env) / name if env else None,
        Path("/root/ombra-build/core") / name,
        REPO_ROOT / "build" / "core" / name,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all", help="t0.4, t0.5 or all")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dims", default=",".join(map(str, T04_DIMS)))
    parser.add_argument("--blocks", type=int, default=None, help="override the block count")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--name", default=DEFAULT_OUT_NAME, help="output file name for t0.4")
    parser.add_argument("--name-t05", default=T05_OUT_NAME, help="output file name for t0.5")
    parser.add_argument(
        "--profile", action="store_true", help="t0.5: also time the stages of one split"
    )
    parser.add_argument("--presets-t011", default="m0_depth2,m0_compact_n8192", help="t0.11")
    parser.add_argument(
        "--error-presets-t011", default="m0_compact_n8192", help="t0.11: presets with the e2e error"
    )
    args = parser.parse_args()

    task = args.task.lower()
    if task not in ("all", "t0.4", "t0.5", "t0.11"):
        raise SystemExit(f"unknown task {args.task!r}")
    dims = tuple(int(x) for x in args.dims.split(","))
    if task in ("all", "t0.4"):
        run_t04(args.out, dims, args.blocks, args.warmup, args.iters, args.name)
    if task in ("all", "t0.5"):
        run_t05(args.out, dims, args.warmup, args.iters, args.name_t05, args.profile)
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
