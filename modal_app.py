"""GPU environment for Ombra on Modal.

Development runs on an A10 (Ampere, sm_86); benchmarks on an A100 (Ampere, sm_80).
Use the Makefile instead of calling these directly:

    make gpu-info      # check GPU, CUDA and OpenFHE in the image
    make test-gpu      # build and run GPU tests on an A10
    make test-openfhe  # run tests marked `openfhe` on CPU in the pinned image
    make bench         # build and run benchmarks on an A100
    make results-pull  # copy benchmark results from the Modal volume to docs/plan/results/raw/

Needs MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in the environment (or `modal token new`).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import modal

# Pinned versions. OpenFHE and openfhe-python must be a matching pair; change both together.
CUDA_IMAGE = "nvidia/cuda:12.6.3-devel-ubuntu22.04"
PYTHON_VERSION = "3.11"
OPENFHE_TAG = "v1.6.1"
OPENFHE_PYTHON_TAG = "v1.6.1.0"

# Ampere only for now (see docs/decisions/0008-ampere-first.md).
CUDA_ARCHS = "80;86"

REPO_DIR = "/root/ombra"
BUILD_DIR = "/root/ombra-build"
DATA_DIR = "/data"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python=PYTHON_VERSION)
    .apt_install("git", "build-essential", "cmake", "ninja-build", "libomp-dev")
    .pip_install("pybind11[global]", "numpy", "pytest", "faiss-cpu")
    .run_commands(
        # OpenFHE (C++), installed to /usr/local
        f"git clone --depth 1 --branch {OPENFHE_TAG} "
        "https://github.com/openfheorg/openfhe-development /opt/openfhe-src",
        "cmake -S /opt/openfhe-src -B /opt/openfhe-build -G Ninja "
        "-DCMAKE_BUILD_TYPE=Release -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF "
        "-DBUILD_BENCHMARKS=OFF",
        "cmake --build /opt/openfhe-build -j",
        "cmake --install /opt/openfhe-build",
        # openfhe-python, built against the OpenFHE above
        f"git clone --depth 1 --branch {OPENFHE_PYTHON_TAG} "
        "https://github.com/openfheorg/openfhe-python /opt/openfhe-python-src",
        "cmake -S /opt/openfhe-python-src -B /opt/openfhe-python-build -G Ninja "
        "-DCMAKE_BUILD_TYPE=Release",
        "cmake --build /opt/openfhe-python-build -j",
        "cmake --install /opt/openfhe-python-build",
    )
    .env({"LD_LIBRARY_PATH": "/usr/local/lib"})
    # T0.8: reads the Hugging Face parquet embedding shards. A layer after the OpenFHE build, so
    # OpenFHE is not rebuilt. (ann-benchmarks.com does not host the dbpedia hdf5, so no h5py.)
    .pip_install("pyarrow==17.0.0")
    # Source code last, so code changes don't rebuild OpenFHE.
    .add_local_dir(
        ".",
        remote_path=REPO_DIR,
        ignore=[".git", "build", "**/__pycache__", ".venv", "data", "*.key", ".env"],
    )
)

app = modal.App("ombra", image=image)
volume = modal.Volume.from_name("ombra-data", create_if_missing=True)


def _run(cmd: list[str], cwd: str = REPO_DIR, ok_codes: tuple[int, ...] = (0,)) -> None:
    print("$", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=cwd).returncode
    if code not in ok_codes:
        raise SystemExit(f"command failed with exit code {code}: {' '.join(cmd)}")


def _build(build_type: str, targets: tuple[str, ...] = ()) -> None:
    """Configure and build; `targets` limits the build to those CMake targets (default: all)."""
    _run(
        [
            "cmake",
            "-S",
            REPO_DIR,
            "-B",
            BUILD_DIR,
            "-G",
            "Ninja",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCHS}",
            "-DOMBRA_ENABLE_CUDA=ON",
        ]
    )
    build_cmd = ["cmake", "--build", BUILD_DIR, "-j"]
    if targets:
        build_cmd += ["--target", *targets]
    _run(build_cmd)


@app.function(gpu="A10", timeout=600)
def gpu_info() -> None:
    """Print GPU, CUDA and OpenFHE versions, to check the image."""
    _run(["nvidia-smi"])
    _run(["nvcc", "--version"])
    _run(["python", "-c", "import openfhe; print('openfhe-python imported OK')"])

    # Installed versions, read from the image (not from the pins above). openfhe-python has no
    # binding for the C++ GetOPENFHEVersion(), so read PACKAGE_VERSION from the OpenFHEConfigVersion
    # file that `cmake --install` wrote (lib/ or lib64/ under /usr/local).
    version_files = sorted(Path("/usr/local").glob("lib*/OpenFHE/OpenFHEConfigVersion.cmake"))
    if not version_files:
        raise SystemExit("OpenFHEConfigVersion.cmake not found under /usr/local/lib*/OpenFHE")
    match = re.search(r'set\(PACKAGE_VERSION\s+"([^"]+)"\)', version_files[0].read_text())
    if match is None:
        raise SystemExit(f"no PACKAGE_VERSION in {version_files[0]}")
    installed_openfhe = match.group(1)
    installed_python = subprocess.run(
        ["python", "-c", "import openfhe; print(openfhe.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"OpenFHE installed: {installed_openfhe} (pinned: {OPENFHE_TAG})")
    print(f"openfhe-python installed: {installed_python} (pinned: {OPENFHE_PYTHON_TAG})")
    if installed_openfhe != OPENFHE_TAG.lstrip("v"):
        raise SystemExit("installed OpenFHE does not match the pinned tag")
    if installed_python != OPENFHE_PYTHON_TAG.lstrip("v"):
        raise SystemExit("installed openfhe-python does not match the pinned tag")


@app.function(gpu="A10", timeout=1800, volumes={DATA_DIR: volume})
def test_gpu(pytest_args: str = "") -> None:
    """Build in Debug and run C++ (ctest) and Python GPU tests on an A10."""
    _build("Debug")
    _run(["ctest", "--test-dir", BUILD_DIR, "--output-on-failure", "--no-tests=ignore"])
    # Zero collected tests (pytest exit code 5) is a failure: GPU tests exist now (T0.3). The
    # driver for tests/test_gpu_ciphertext_roundtrip.py is built into BUILD_DIR above.
    _run(
        ["python", "-m", "pytest", "-m", "gpu", "-v", "-s", *pytest_args.split()],
        ok_codes=(0,),
    )


# memory: with --expansion-large the C++ expansion driver holds all d = 1536 expanded ciphertexts
# (about 1.2 GB) plus its tree levels and automorphism keys, while the pytest process keeps its keys
# and one score accumulator; 16 GiB leaves headroom over that estimate of about 5 to 6 GB.
@app.function(cpu=4.0, memory=16384, timeout=3600, volumes={DATA_DIR: volume})
def test_openfhe(pytest_args: str = "") -> None:
    """Run tests marked `openfhe` on CPU. Zero collected tests (exit code 5) is a failure.

    First builds the CPU-side C++ drivers those tests call. Compiling needs the CUDA toolchain
    (the targets link the CUDA library) but no GPU, and the drivers make no CUDA calls at runtime.
    Release build, so the CPU timings are meaningful.
    """
    _build("Release", targets=("ombra_expand", "ombra_encoding_check"))
    cmd = ["python", "-m", "pytest", "-m", "openfhe", "-v", "-s", *pytest_args.split()]
    _run(cmd, ok_codes=(0,))
    volume.commit()


# memory: the T0.4 benchmark holds one block (about 1.2 GB at d = 1536) as OpenFHE ciphertexts plus
# its flattened copy on the host at a time, so about 3 to 4 GB peak; 16 GiB leaves headroom.
# T1.3 needs the 40 GB card (the plain "A100" sometimes lands on the 80 GB PCIe one) and more time:
# 12 cases, each with 23 queries, plus encrypting two blocks per d. Host memory: block 0 flattened
# (1.2 GB at d = 1536) plus its OpenFHE ciphertexts and the Python client; 16 GiB is enough.
@app.function(gpu="A100-40GB", memory=16384, timeout=7200, volumes={DATA_DIR: volume})
def bench(task: str = "all") -> None:
    """Build in Release and run benchmarks on an A100; results go to the volume."""
    _build("Release")
    out_dir = Path(DATA_DIR) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_script = Path(REPO_DIR) / "core" / "bench" / "run.py"
    if not bench_script.exists():
        print("No benchmarks yet (core/bench/run.py is missing). Nothing to run.")
        return
    _run(["python", str(bench_script), "--task", task, "--out", str(out_dir)])
    volume.commit()


@app.function(gpu="A10", memory=16384, timeout=3600, volumes={DATA_DIR: volume})
def profile_expand(dims: str = "1536") -> None:
    """Release build; time the stages of one expansion split on an A10 (T0.5b profiling)."""
    _build("Release")
    out_dir = Path(DATA_DIR) / "results-profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_script = Path(REPO_DIR) / "core" / "bench" / "run.py"
    _run(
        [
            "python",
            str(bench_script),
            "--task",
            "t0.5",
            "--profile",
            "--dims",
            dims,
            "--warmup",
            "2",
            "--iters",
            "5",
            "--name-t05",
            "t0.5-profile-a10.json",
            "--out",
            str(out_dir),
        ]
    )
    volume.commit()


# Recall acceptance check (T0.8, T1.6). At N = 100,000 and d = 1536 the run encrypts 13 database
# blocks plus 3 centers blocks (about 19 GB of ciphertexts on the container's local disk, kept on
# the GPU as well: about 20 GB of the A100's 40 GB). The host holds one block at a time (3 to 4 GB)
# plus the dataset subset (about 1 GB as float64) and faiss. Query results are decrypted while the
# search runs and deleted at once, so 1,000 queries do not add to the disk. Several datasets run
# one after the other (default: one per dimension), so the timeout is long.
@app.function(gpu="A100", cpu=8.0, memory=32768, timeout=43200, volumes={DATA_DIR: volume})
def recall(
    dry_run: bool = False,
    queries: int = 1000,
    preset: str = "m0_depth2",
    dataset: str = "",
    n: int = 100000,
) -> None:
    """Release build; recall@10 acceptance check of a preset on the encrypted pipeline.

    `--dry-run`: 10,000 vectors and 5 queries. `--dataset`: space-separated names or `all`
    (default: one dataset per dimension; names are listed by `recall.py --help`).
    Example: modal run modal_app.py::recall --preset m0_depth2
    """
    _build("Release", targets=("ombra_recall_search",))
    out_dir = Path(DATA_DIR) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(Path(REPO_DIR) / "core" / "bench" / "recall.py"),
        "--out",
        str(out_dir),
        "--datasets",
        str(Path(DATA_DIR) / "datasets"),
        "--scratch",
        "/tmp",  # noqa: S108 - container-local scratch for the encrypted data
        "--queries",
        str(queries),
        "--preset",
        preset,
        "--n",
        str(n),
    ]
    if dataset:
        cmd += ["--dataset", *dataset.split()]
    if dry_run:
        cmd.append("--small")
    try:
        _run(cmd)
    finally:
        volume.commit()


# "A100-40GB" pins the SXM4 40 GB part that T0.4 ran on (plain "A100" may be an 80 GB PCIe card).
@app.function(gpu="A100-40GB", memory=16384, timeout=3600)
def profile_search(ncu: bool = True, micro: bool = True) -> None:
    """T1.4 profiling: micro kernels, instruction mix and an Nsight Compute attempt on an A100."""
    import json
    import shutil

    _run(["nvidia-smi"])
    print(
        "ncu on PATH:", shutil.which("ncu"), "| cuobjdump:", shutil.which("cuobjdump"), flush=True
    )
    _build("Release", targets=("ombra_profile_search",))
    preset = json.loads((Path(REPO_DIR) / "core/params/ckks_presets.json").read_text())
    p = preset["presets"]["m0_depth2"]
    towers = p["multiplicative_depth"] + 1
    bits = ",".join([str(p["first_mod_size"])] + [str(p["scaling_mod_size"])] * (towers - 1))
    exe = f"{BUILD_DIR}/core/ombra_profile_search"
    common = ["--ring", str(p["ring_dim"]), "--bits", bits]
    if micro:
        _run([exe, *common])
    _run(
        [
            "python",
            str(Path(REPO_DIR) / "core/bench/sass_mix.py"),
            exe,
            "column_search_kernel",
            "column_search_batched_kernel",
            "plain_search_kernel",
            "prof_kernelILi0ELi1ELb1E",
            "prof_kernelILi1ELi1ELb1E",
            "prof_kernelILi2ELi1ELb1E",
            "prof_kernelILi4ELi1ELb0E",
            "prof_kernelILi5ELi1ELb0E",
            "prof_kernelILi6ELi1ELb0E",
        ]
    )
    if not ncu:
        return
    # Hardware counters may be refused inside the container; report whatever happens.
    cmd = [
        "ncu",
        "--clock-control",
        "none",  # the container may not lock clocks (first attempt failed on that)
        "--set",
        "full",
        "-k",
        "regex:column_search_kernel",
        "-c",
        "1",
        exe,
        *common,
        "--ref-only",
        "--d",
        "384",
        "--blocks",
        "16",
        "--warmup",
        "0",
        "--iters",
        "1",
    ]
    print("$", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=REPO_DIR).returncode
    print("ncu exit code:", code, flush=True)
