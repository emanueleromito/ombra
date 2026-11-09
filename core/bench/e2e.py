"""T1.3: end-to-end encrypted query latency, step by step (preset m0_depth2).

    python core/bench/run.py --task t1.3 --out <dir>      # writes <dir>/t1.3-a100.json

This process is the CLIENT: it holds the secret key (never written to disk, never sent to the C++
process), encodes and encrypts queries, decrypts, picks the clusters and ranks. The SERVER is the
C++ process `ombra_bench_e2e`, driven over stdin/stdout, which sees only the context, public
evaluation keys and ciphertexts and runs expansion -> stage 1 -> stage 2 on the GPU, with the
expansion leaves handed to both searches device to device.

Per query, the steps (network time excluded; bytes reported):
  client_encode_encrypt       oq.encode_query (encode + public-key encryption)
  server_deserialize_query    OpenFHE deserialization of the uploaded ciphertext
  expansion                   upload root, GPU expansion tree, wait
  leaf_handoff_d2d            leaves copied device to device into the two searches' query buffers
  stage1_gpu                  search kernel, relinearize, rescale, all-ones mask, rescale (centers)
  stage1_d2h                  result words to the host
  client_stage1               deserialize, decrypt, select the top nprobe clusters
  stage2_gpu                  same five GPU steps over the chosen blocks (with the mask)
  stage2_d2h                  result words to the host
  client_stage2               deserialize, decrypt, top-k over the candidates
Result serialization on the server is not timed (it stands in for the network encoding); the client
steps include deserialization (a real client must parse the bytes).

Data: the cluster-center block and stored block 0 are real encrypted random unit vectors, so stage 1
selection and the decrypted scores are real and checked against numpy (bound 1e-6, as T0.6). The
other chosen blocks are device-to-device copies of block 0, because the kernel timing does not
depend on the values (as in T0.4/T0.7); the client still deserializes and decrypts one result
ciphertext per chosen block. The chosen clusters are assumed packed contiguously into
ceil(frac * N / slots) blocks (the last one masked to the valid slots); with clusters straddling
block boundaries a real layout touches up to one extra block per chosen cluster, so per-block stage
2 numbers are reported to scale from. Host-to-device time of the chosen blocks is reported apart:
it is a stand-in for M2 cluster paging and not part of the search latency.

Only aggregates are printed or stored (times, sizes, max errors); data is synthetic.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

PRESET = "m0_depth2"
K = 10
SEED = 20260301
ERROR_BOUND = 1e-6  # same agreed score bound as T0.2 to T0.6
FRACTIONS = (0.05, 0.20)
SIZES = ((100_000, 100), (1_000_000, 1000))  # (N, nlist); a centers block holds up to 8192

STEPS = (
    "client_encode_encrypt",
    "server_deserialize_query",
    "expansion",
    "leaf_handoff_d2d",
    "stage1_gpu",
    "stage1_d2h",
    "client_stage1",
    "stage2_gpu",
    "stage2_d2h",
    "client_stage2",
)
SERVER_STEPS = tuple(s for s in STEPS if not s.startswith("client_"))
GPU_PARTS = ("search_s", "relin_s", "rescale1_s", "mask_s", "rescale2_s")


def _parse(line: str, tag: str) -> dict[str, float]:
    parts = line.split()
    if not parts or parts[0] != tag:
        raise RuntimeError(f"unexpected reply from the server process (wanted {tag})")
    return {k: float(v) for k, v in (p.split("=") for p in parts[1:])}


class Server:
    """The C++ process; line protocol documented in core/src/ombra_bench_e2e.cpp."""

    def __init__(self, binary: Path, args: list[str]) -> None:
        self.proc = subprocess.Popen(  # noqa: S603
            [str(binary), *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def call(self, line: str, tag: str) -> dict[str, float]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        if line:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        reply = self.proc.stdout.readline()
        if not reply:
            code = self.proc.wait()
            raise RuntimeError(f"ombra_bench_e2e exited (code {code})")
        if tag == "OK":
            if reply.strip() != "OK":
                raise RuntimeError("expected OK from the server process")
            return {}
        return _parse(reply, tag)

    def close(self) -> None:
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
            except OSError:
                pass
        self.proc.wait(timeout=120)


def _unit(rng: np.random.Generator, rows: int, d: int) -> np.ndarray:
    x = rng.standard_normal((rows, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _stats(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def _gpu_name() -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        return out.strip().splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        return "unknown"


def run_d(
    d: int,
    cases: list[tuple[int, float, int]],
    queries: int,
    warmup: int,
    binary: Path,
    work: Path,
    log: Callable[[str], None],
) -> list[dict[str, Any]]:
    """All cases (N, fraction, nlist) for one dimension d; returns one record per case."""
    import openfhe as ofhe
    from ombra import params as ombra_params
    from ombra import query as oq

    rng = np.random.default_rng(SEED + d)
    _, cc = ombra_params.make_context(PRESET)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    slots = ombra_params.get_preset(PRESET)["batch_size"]
    ring = cc.GetRingDimension()
    pk, sk = keys.publicKey, keys.secretKey
    cc.ClearEvalAutomorphismKeys()  # the key store is static: hold exactly this d's indices
    cc.EvalAutomorphismKeyGen(sk, oq.automorphism_indices(ring, d))
    assert ofhe.SerializeToFile(str(work / "context.bin"), cc, ofhe.BINARY)
    assert cc.SerializeEvalAutomorphismKey(str(work / "autokeys.bin"), ofhe.BINARY)
    assert ofhe.CryptoContext.SerializeEvalMultKey(
        str(work / "evalmult.bin"), ofhe.BINARY, pk.GetKeyTag()
    )

    nlist_max = max(nl for _, _, nl in cases)
    if nlist_max > slots:
        raise SystemExit("nlist must fit in one block")
    centers = np.zeros((slots, d))
    centers[:nlist_max] = _unit(rng, nlist_max, d)
    stored = _unit(rng, slots, d)  # stored block 0: slot s holds vector s
    t0 = time.time()
    for j in range(d):
        for name, mat in (("centers", centers), ("stored", stored)):
            col = cc.Encrypt(pk, cc.MakeCKKSPackedPlaintext(mat[:, j].tolist()))
            assert ofhe.SerializeToFile(str(work / f"{name}_{j}.bin"), col, ofhe.BINARY)
    log(f"d={d}: encrypted centers and stored block in {time.time() - t0:.1f} s")

    def mask_file(path: Path, valid: int) -> None:
        vals = np.zeros(slots)
        vals[:valid] = 1.0
        ct = cc.Encrypt(pk, cc.MakeCKKSPackedPlaintext(vals.tolist(), 1, 1))  # level 1
        assert ofhe.SerializeToFile(str(path), ct, ofhe.BINARY)

    mask_file(work / "mask_ones.bin", slots)

    def make_query(vec: np.ndarray) -> tuple[Path, float]:
        t = time.perf_counter()
        ct = oq.encode_query(cc, pk, vec)
        enc_s = time.perf_counter() - t
        path = work / "query.bin"
        assert ofhe.SerializeToFile(str(path), ct, ofhe.BINARY)
        return path, enc_s

    def decrypt_file(path: Path, count: int) -> np.ndarray:
        ct, ok = ofhe.DeserializeCiphertext(str(path), ofhe.BINARY)
        assert ok
        assert ct.GetLevel() == 2
        dec = cc.Decrypt(ct, sk)
        dec.SetLength(slots)
        return np.array(dec.GetRealPackedValue())[:count]

    server = Server(
        binary,
        [
            str(work / "context.bin"),
            str(work / "autokeys.bin"),
            str(work / "evalmult.bin"),
            str(work),
            str(d),
        ],
    )
    records = []
    try:
        ready = server.call("", "READY")
        block_bytes = int(ready["block_bytes"])
        # Allocate the expansion's pool before any memory plan is made.
        warm_path, _ = make_query(_unit(rng, 1, d)[0])
        server.call(f"WARM {warm_path}", "OK")

        for n, frac, nlist in cases:
            chosen = round(frac * n)
            blocks = -(-chosen // slots)
            valid_last = chosen - (blocks - 1) * slots
            nprobe = max(1, round(frac * nlist))
            mask_last = work / f"mask_last_{blocks}_{valid_last}.bin"
            mask_file(mask_last, valid_last)
            case = server.call(f"CASE {blocks} {mask_last}", "CASE")
            resident = int(case["resident_blocks"])
            log(
                f"d={d} N={n} frac={frac}: {blocks} chosen blocks ({chosen} vectors), "
                f"{resident} resident, passes={int(case['passes'])}"
            )

            rows: dict[str, list[float]] = {s: [] for s in STEPS}
            gpu1: dict[str, list[float]] = {p: [] for p in GPU_PARTS}
            gpu2: dict[str, list[float]] = {p: [] for p in GPU_PARTS}
            totals: list[float] = []
            up_bytes = s1_bytes = s2_file_bytes = 0
            err1 = err_first = err_last = 0.0
            for it in range(warmup + queries):
                q = _unit(rng, 1, d)[0]
                qpath, enc_s = make_query(q)
                up_bytes = qpath.stat().st_size
                s1 = server.call(f"QUERY {qpath}", "S1")

                t = time.perf_counter()
                res1 = work / "s1_result.bin"
                s1_bytes = res1.stat().st_size
                cscore = decrypt_file(res1, nlist)
                chosen_ids = np.argpartition(-cscore, nprobe - 1)[:nprobe]
                c1_s = time.perf_counter() - t
                assert chosen_ids.size == nprobe

                s2 = server.call("GO", "S2")
                t = time.perf_counter()
                parts = []
                for b in range(blocks):
                    f = work / f"s2_result_{b % resident}.bin"
                    got = decrypt_file(f, slots)
                    if b == blocks - 1:  # the mask zeroes the padding
                        got = got[:valid_last]
                    parts.append(got)
                    s2_file_bytes = f.stat().st_size
                allscores = np.concatenate(parts)
                kk = min(K, allscores.size)
                top = np.argpartition(-allscores, kk - 1)[:kk]
                top = top[np.argsort(-allscores[top])]
                c2_s = time.perf_counter() - t
                assert top.size == kk

                # Accuracy against numpy, outside the timers.
                err1 = max(err1, float(np.max(np.abs(cscore - (centers[:nlist] @ q)))))
                ref = stored @ q
                err_first = max(
                    err_first, float(np.max(np.abs(parts[0][:slots] - ref[: parts[0].size])))
                )
                if blocks > 1 and resident == blocks:
                    # block b copies block 0; the last block's mask keeps `valid_last` slots
                    err_last = max(err_last, float(np.max(np.abs(parts[-1] - ref[:valid_last]))))
                if it < warmup:
                    continue
                step = {
                    "client_encode_encrypt": enc_s,
                    "server_deserialize_query": s1["deserialize_s"],
                    "expansion": s1["expansion_s"],
                    "leaf_handoff_d2d": s1["handoff_s"],
                    "stage1_gpu": s1["gpu_total_s"],
                    "stage1_d2h": s1["d2h_s"],
                    "client_stage1": c1_s,
                    "stage2_gpu": s2["gpu_total_s"],
                    "stage2_d2h": s2["d2h_s"],
                    "client_stage2": c2_s,
                }
                for s in STEPS:
                    rows[s].append(step[s])
                for p in GPU_PARTS:
                    gpu1[p].append(s1[p])
                    gpu2[p].append(s2[p])
                totals.append(sum(step.values()))

            med = {s: statistics.median(rows[s]) for s in STEPS}
            total_med_steps = sum(med.values())
            server_med = sum(med[s] for s in SERVER_STEPS)
            records.append(
                {
                    "d": d,
                    "n_vectors": n,
                    "fraction_searched": frac,
                    "nlist": nlist,
                    "nprobe": nprobe,
                    "chosen_vectors": chosen,
                    "chosen_blocks": blocks,
                    "queries": queries,
                    "warmup_queries": warmup,
                    "steps_seconds": {s: _stats(rows[s]) for s in STEPS},
                    "step_share": {s: med[s] / total_med_steps for s in STEPS},
                    "stage1_gpu_parts_median_seconds": {
                        p: statistics.median(gpu1[p]) for p in GPU_PARTS
                    },
                    "stage2_gpu_parts_median_seconds": {
                        p: statistics.median(gpu2[p]) for p in GPU_PARTS
                    },
                    "stage2_gpu_per_block_median_seconds": med["stage2_gpu"] / blocks,
                    "total_seconds_sum_of_median_steps": total_med_steps,
                    "total_seconds_median_of_queries": statistics.median(totals),
                    "server_seconds_sum_of_median_steps": server_med,
                    "queries_per_second_single_stream": 1.0 / total_med_steps,
                    "queries_per_second_server_only": 1.0 / server_med,
                    "bytes": {
                        "upload_query": up_bytes,
                        "download_stage1": s1_bytes,
                        "download_stage2": s2_file_bytes * blocks,
                        "download_total": s1_bytes + s2_file_bytes * blocks,
                    },
                    "chosen_block_h2d_standin_for_m2_paging": {
                        "seconds_one_block_measured": case["h2d_one_block_seconds"],
                        "seconds_all_chosen_blocks_estimate": case["h2d_one_block_seconds"]
                        * blocks,
                        "note": "pageable host memory, cudaMemcpy of one block times the block "
                        "count; not part of the search latency",
                    },
                    "memory": {
                        "block_bytes": block_bytes,
                        "full_collection_blocks": -(-n // slots),
                        "full_collection_bytes": -(-n // slots) * block_bytes,
                        "chosen_blocks_bytes": blocks * block_bytes,
                        "resident_blocks": resident,
                        "passes": int(case["passes"]),
                        "stage2_device_bytes": int(case["stage2_device_bytes"]),
                        "device_total_bytes": int(case["total_bytes"]),
                        "device_free_bytes_before_stage2": int(case["free_bytes_before"]),
                        "device_free_bytes_after_stage2": int(case["free_bytes_after"]),
                        "device_used_since_start_bytes": int(case["device_used_since_start_bytes"]),
                        "plan": "resident: centers block + query buffers + chosen blocks + masks "
                        "+ stage buffers + expansion pool; the whole collection is not resident",
                        "fits_whole_collection": -(-n // slots) * block_bytes
                        < int(case["total_bytes"]),
                        "fits_chosen_blocks": resident == blocks,
                    },
                    "max_abs_score_error": {
                        "stage1_center_scores": err1,
                        "stage2_block0": err_first,
                        "stage2_last_masked_block": err_last,
                        "bound": ERROR_BOUND,
                    },
                }
            )
    finally:
        server.close()
    return records


def run_t13(
    out_dir: Path,
    dims: tuple[int, ...],
    queries: int,
    warmup: int,
    find_binary: Callable[[str], Path],
    openfhe_version: str,
    cases_spec: list[tuple[int, float, int]] | None = None,
    name: str = "t1.3-a100.json",
    log: Callable[[str], None] = lambda s: print(s, flush=True),
) -> Path:
    cases = cases_spec or [(n, f, nl) for n, nl in SIZES for f in FRACTIONS]
    binary = find_binary("ombra_bench_e2e")
    records: list[dict[str, Any]] = []
    for d in dims:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:  # noqa: S108
            records += run_d(d, cases, queries, warmup, binary, Path(tmp), log)
        for r in records[-len(cases) :]:
            log(
                f"d={r['d']} N={r['n_vectors']} frac={r['fraction_searched']}: "
                f"total {r['total_seconds_sum_of_median_steps'] * 1000:.1f} ms, "
                f"{r['queries_per_second_single_stream']:.2f} QPS"
            )
    result = {
        "task": "T1.3",
        "preset": PRESET,
        "gpu_name": _gpu_name(),
        "openfhe_version": openfhe_version,
        "queries_per_case": queries,
        "warmup_queries": warmup,
        "k": K,
        "steps": list(STEPS),
        "data_note": "centers block and stored block 0 are real encrypted random unit vectors; "
        "the other chosen blocks are device-to-device copies of block 0 (kernel timing does not "
        "depend on the values); network time excluded",
        "cases": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(result, indent=2) + "\n")
    log(f"wrote {path}")
    return path
