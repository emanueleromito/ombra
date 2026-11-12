"""Compact configurations benchmark (called from core/bench/run.py, `--task compact-configs`).

Per preset (m0_depth2 = baseline, m0_compact_n8192 = ring dimension 8192) and per d in
384, 768, 1536 (default), one case per measurement:

  * "expansion": GPU query expansion time and automorphism-key size (binary `ombra_bench_expand`);
  * "search": per mode ct_x_ct (column-search kernel), pt_preencoded, fp16, int8 (binary
    `ombra_bench_plain_search`): vectors/s, storage bytes per vector, encode/search split;
  * "quantization": score error of fp16 and int8 storage against numpy on the unquantized
    vectors (seeded synthetic normalized vectors, max and mean absolute error, no bound);
  * "end_to_end_error" (only for the presets in `error_presets`): GPU expansion, GPU search,
    Relinearize, Rescale, encrypted 0/1 mask, Rescale, decrypted and compared with numpy, reported
    against the 1e-6 bound (a failure is a result, not an error). The secret key stays in this
    process; only the context, the public key, the automorphism keys and ciphertexts go to a
    temporary directory. Nothing printed or written contains vectors or scores.

Output: `<out>/compact-configs-a100.json`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

PRESETS = ("m0_depth2", "m0_compact_n8192")
DEFAULT_ERROR_PRESETS = ("m0_compact_n8192",)
BOUND = 1e-6
SEED = 20260111
# Stored data per search benchmark is kept near constant: blocks = BUDGET // d.
BLOCK_DIM_BUDGET = 8 * 1536
OUT_NAME = "compact-configs-a100.json"


def _run(cmd: list[str], timeout: int = 3000) -> str:
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)  # noqa: S603
    if proc.returncode != 0:
        raise SystemExit(
            f"{Path(cmd[0]).name} failed (exit {proc.returncode}): {proc.stderr[-2000:]}"
        )
    return proc.stdout


def quantization_case(preset: str, d: int, slots: int) -> list[dict[str, Any]]:
    """Score error of fp16 / int8 storage against numpy on the unquantized vectors."""
    rng = np.random.default_rng(SEED + d)
    x = rng.standard_normal((slots, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = rng.standard_normal(d)
    q /= np.linalg.norm(q)
    exact = x @ q
    out = []
    variants = {
        "fp16": x.astype(np.float16).astype(np.float64),
    }
    for name, scale in (
        ("int8_global_scale", 127.0 / float(np.max(np.abs(x)))),
        ("int8_fixed_127", 127.0),
    ):
        codes = np.clip(np.rint(x * scale), -127, 127).astype(np.int8)
        variants[name] = codes.astype(np.float64) / scale
    for name, decoded in variants.items():
        err = np.abs(decoded @ q - exact)
        out.append(
            {
                "kind": "quantization",
                "preset": preset,
                "mode": name,
                "d": d,
                "vectors": slots,
                "max_abs_score_error": float(err.max()),
                "mean_abs_score_error": float(err.mean()),
                "note": "decoded stored values @ q against unquantized stored @ q, float64",
            }
        )
    return out


def end_to_end_error(
    cc: Any,
    keys: Any,
    d: int,
    slots: int,
    tmp: Path,
    find_binary: Callable[[str], Path],
) -> dict[str, Any]:
    """Expansion, search, relinearize, rescale, mask, rescale on the GPU path; error vs numpy."""
    import openfhe as ofhe
    from ombra import query as oq

    rng = np.random.default_rng(SEED + 1000 + d)
    stored = rng.standard_normal((slots, d))
    stored /= np.linalg.norm(stored, axis=1, keepdims=True)
    q = rng.standard_normal(d)
    q /= np.linalg.norm(q)

    in_dir = tmp / "search_in"
    in_dir.mkdir()
    for j in range(d):
        col = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(stored[:, j].tolist()))
        if not ofhe.SerializeToFile(str(in_dir / f"stored_0_{j}.bin"), col, ofhe.BINARY):
            raise SystemExit("cannot serialize a stored column")
    qct = oq.encode_query(cc, keys.publicKey, q)
    if not ofhe.SerializeToFile(str(tmp / "query.bin"), qct, ofhe.BINARY):
        raise SystemExit("cannot serialize the query")

    exp_dir = tmp / "expand_out"
    exp_dir.mkdir()
    out = _run(
        [
            str(find_binary("ombra_expand_gpu")),
            str(tmp / "context.bin"),
            str(tmp / "query.bin"),
            str(tmp / "autokeys.bin"),
            str(d),
            str(exp_dir),
        ]
    )
    if "GPU expansion exact RNS match with OpenFHE: PASS" not in out:
        raise SystemExit("GPU expansion does not match OpenFHE")
    for j in range(d):
        shutil.copy(exp_dir / f"gpu_{j}.bin", in_dir / f"query_{j}.bin")
    search_dir = tmp / "search_out"
    search_dir.mkdir()
    out = _run(
        [
            str(find_binary("ombra_column_search")),
            str(tmp / "context.bin"),
            str(in_dir),
            str(search_dir),
            str(d),
            "1",
        ]
    )
    if "column search exact RNS match: PASS" not in out:
        raise SystemExit("GPU search does not match OpenFHE")

    expected = stored @ q
    random_mask = rng.integers(0, 2, size=slots).astype(np.float64)
    random_mask[0], random_mask[1] = 0.0, 1.0
    masks = {"random": random_mask, "ones": np.ones(slots), "zeros": np.zeros(slots)}
    ct, ok = ofhe.DeserializeCiphertext(str(search_dir / "gpu_0.bin"), ofhe.BINARY)
    if not ok:
        raise SystemExit("cannot read the search result")
    errors = {}
    for kind, mask in masks.items():
        mask_ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(mask.tolist(), 1, 1))
        score = cc.Rescale(cc.Relinearize(ct))
        masked = cc.Rescale(cc.EvalMult(score, mask_ct))
        if masked.GetLevel() != 2:
            raise SystemExit("unexpected level after the mask")
        dec = cc.Decrypt(masked, keys.secretKey)
        dec.SetLength(slots)
        got = np.array(dec.GetRealPackedValue())
        errors[kind] = float(np.max(np.abs(got - expected * mask)))
    worst = max(errors.values())
    return {
        "levels_used": 2,
        "max_abs_error_by_mask": errors,
        "max_abs_error": worst,
        "bound": BOUND,
        "within_bound": worst < BOUND,
    }


def run_compact_configs(
    out_dir: Path,
    dims: tuple[int, ...],
    presets: tuple[str, ...],
    error_presets: tuple[str, ...],
    warmup: int,
    iters: int,
    find_binary: Callable[[str], Path],
    openfhe_version: str,
) -> Path:
    import openfhe as ofhe
    from ombra import params as ombra_params
    from ombra import query as oq

    bench_expand = find_binary("ombra_bench_expand")
    bench_plain = find_binary("ombra_bench_plain_search")
    results: list[dict[str, Any]] = []
    gpu_name = "unknown"
    preset_info: dict[str, Any] = {}

    for preset in presets:
        _, cc = ombra_params.make_context(preset)
        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        ring = cc.GetRingDimension()
        pr = ombra_params.get_preset(preset)
        slots = pr["batch_size"]
        preset_info[preset] = {**pr, "towers_q": pr["multiplicative_depth"] + 1}
        for d in dims:
            with tempfile.TemporaryDirectory() as tmp_s:
                tmp = Path(tmp_s)
                cc.ClearEvalAutomorphismKeys()  # the key store is static: exactly this d's keys
                cc.EvalAutomorphismKeyGen(keys.secretKey, oq.automorphism_indices(ring, d))
                # Public material only; the secret key is never serialized.
                if not ofhe.SerializeToFile(str(tmp / "context.bin"), cc, ofhe.BINARY):
                    raise SystemExit("cannot serialize context")
                if not ofhe.SerializeToFile(
                    str(tmp / "public_key.bin"), keys.publicKey, ofhe.BINARY
                ):
                    raise SystemExit("cannot serialize public key")
                if not cc.SerializeEvalAutomorphismKey(str(tmp / "autokeys.bin"), ofhe.BINARY):
                    raise SystemExit("cannot serialize automorphism keys")

                part = tmp / "expand.json"
                _run(
                    [
                        str(bench_expand),
                        str(tmp / "context.bin"),
                        str(tmp / "public_key.bin"),
                        str(tmp / "autokeys.bin"),
                        str(d),
                        str(warmup),
                        str(iters),
                        str(part),
                    ]
                )
                entry = json.loads(part.read_text())
                gpu_name = entry.pop("gpu_name")
                results.append(
                    {"kind": "expansion", "preset": preset, "mode": "expansion", **entry}
                )
                print(
                    f"{preset} d={d} expansion gpu_seconds={entry['gpu_seconds']:.4f} "
                    f"key_bytes={entry['key_bytes']}",
                    flush=True,
                )

                blocks = max(1, BLOCK_DIM_BUDGET // d)
                part = tmp / "search.json"
                _run(
                    [
                        str(bench_plain),
                        str(tmp / "context.bin"),
                        str(tmp / "public_key.bin"),
                        str(d),
                        str(blocks),
                        str(warmup),
                        str(iters),
                        str(part),
                    ]
                )
                entry = json.loads(part.read_text())
                gpu_name = entry.pop("gpu_name")
                modes = entry.pop("modes")
                for m in modes:
                    results.append({"kind": "search", "preset": preset, "d": d, **entry, **m})
                    print(
                        f"{preset} d={d} {m['mode']} vectors/s={m['vectors_per_second']:.3e} "
                        f"bytes/vector={m['storage_bytes_per_vector']}",
                        flush=True,
                    )

                results.extend(quantization_case(preset, d, slots))

                if preset in error_presets:
                    err = end_to_end_error(cc, keys, d, slots, tmp, find_binary)
                    results.append(
                        {
                            "kind": "end_to_end_error",
                            "preset": preset,
                            "mode": "gpu_pipeline",
                            "d": d,
                            **err,
                        }
                    )
                    print(
                        f"{preset} d={d} end-to-end max_abs_error={err['max_abs_error']:.3e} "
                        f"within_1e-6={err['within_bound']}",
                        flush=True,
                    )

    record = {
        "task": "compact-configs",
        "format_version": 1,
        "openfhe_version": openfhe_version,
        "gpu_name": gpu_name,
        "presets": preset_info,
        "note": (
            "expansion: median wall time of the whole tree. search: kernel time with "
            "CUDA events; ct_x_ct is the column search kernel (3 components), pt_preencoded "
            "the ct x pt kernel on RNS plaintexts, fp16/int8 encode on the GPU in chunks of "
            "blocks then "
            "search (seconds_median = encode + search). Data is seeded synthetic; quantization "
            "errors are computed in float64 on the decoded stored values. end_to_end_error is "
            "reported against the 1e-6 bound; a failure is a result."
        ),
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_NAME
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out_path}")
    return out_path
