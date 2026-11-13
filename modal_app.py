# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
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

# Ampere only for now.
CUDA_ARCHS = "80;86"

REPO_DIR = "/root/ombra"
BUILD_DIR = "/root/ombra-build"
DATA_DIR = "/data"


def _upload_ignore(p: Path) -> bool:
    # Runs only locally at image build; imported lazily because Modal puts only modal_app.py
    # in the container, so a top-level import of modal_upload would fail there.
    from modal_upload import is_excluded

    return is_excluded(str(p))


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
    # Recall check: reads the Hugging Face parquet embedding shards. A layer after the OpenFHE
    # build, so OpenFHE is not rebuilt. (ann-benchmarks.com does not host the dbpedia hdf5, so no
    # h5py.)
    .pip_install("pyarrow==17.0.0")
    # gRPC for C++ (the server) and Python (tests/e2e_client.py). Layers after the OpenFHE
    # build, so OpenFHE is not rebuilt.
    .apt_install(
        "libgrpc++-dev",
        "libprotobuf-dev",
        "protobuf-compiler",
        "protobuf-compiler-grpc",
        "pkg-config",
        "libssl-dev",  # REST over TLS
        "openssl",  # tests make a self-signed certificate with the CLI
    )
    .pip_install("grpcio", "grpcio-tools")
    .pip_install("cryptography")  # client key file and document blobs (AES-GCM)
    # Source code last, so code changes don't rebuild OpenFHE.
    # Allowlist, not a blocklist: see modal_upload.py. The callable gets paths relative to ".".
    .add_local_dir(".", remote_path=REPO_DIR, ignore=_upload_ignore)
)

app = modal.App("ombra", image=image)
volume = modal.Volume.from_name("ombra-data", create_if_missing=True)


def _run(cmd: list[str], cwd: str = REPO_DIR, ok_codes: tuple[int, ...] = (0,)) -> None:
    print("$", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=cwd).returncode
    if code not in ok_codes:
        raise SystemExit(f"command failed with exit code {code}: {' '.join(cmd)}")


def _build(build_type: str, targets: tuple[str, ...] = (), archs: str = CUDA_ARCHS) -> None:
    """Configure and build; `targets` limits the build to those CMake targets (default: all).
    `archs` is the CUDA architecture list (default: Ampere only, decision 0008)."""
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
            f"-DCMAKE_CUDA_ARCHITECTURES={archs}",
            "-DOMBRA_ENABLE_CUDA=ON",
        ]
    )
    build_cmd = ["cmake", "--build", BUILD_DIR, "-j"]
    if targets:
        build_cmd += ["--target", *targets]
    _run(build_cmd)


NATIVE_BUILD_DIR = "/root/ombra-native-build"


def _build_native() -> None:
    """Build ombra._native (client/CMakeLists.txt) against OpenFHE in /usr/local and put it
    next to the client sources (tests import the client from source). CPU only, no CUDA."""
    import glob
    import os
    import shutil

    pybind = subprocess.run(
        ["python", "-m", "pybind11", "--cmakedir"], check=True, capture_output=True, text=True
    ).stdout.strip()
    env_backup = os.environ.get("OMBRA_REQUIRE_NATIVE")
    os.environ["OMBRA_REQUIRE_NATIVE"] = "1"
    try:
        _run(
            [
                "cmake",
                "-S",
                f"{REPO_DIR}/client",
                "-B",
                NATIVE_BUILD_DIR,
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-Dpybind11_DIR={pybind}",
                f"-DPython_EXECUTABLE={shutil.which('python')}",
            ]
        )
        _run(["cmake", "--build", NATIVE_BUILD_DIR, "-j"])
    finally:
        if env_backup is None:
            os.environ.pop("OMBRA_REQUIRE_NATIVE", None)
        else:
            os.environ["OMBRA_REQUIRE_NATIVE"] = env_backup
    for so in glob.glob(f"{NATIVE_BUILD_DIR}/_native*.so"):
        shutil.copy(so, f"{REPO_DIR}/client/ombra/")


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
    _build_native()  # the parity test also runs local mode
    _run(["ctest", "--test-dir", BUILD_DIR, "--output-on-failure", "--no-tests=ignore"])
    # Zero collected tests (pytest exit code 5) is a failure: GPU tests exist now. The
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
    _build_native()  # local mode runs the compiled expansion in a child process
    cmd = ["python", "-m", "pytest", "-m", "openfhe", "-v", "-s", *pytest_args.split()]
    _run(cmd, ok_codes=(0,))
    volume.commit()


# memory: the column search benchmark holds one block (about 1.2 GB at d = 1536) as OpenFHE
# ciphertexts plus its flattened copy on the host at a time, so about 3 to 4 GB peak; 16 GiB leaves
# headroom. The filter measurements (`--task filter-measurements`) runs here too: it starts
# ombra_server (built with everything else) and drives it over localhost; 16 GiB covers the client's
# cached ciphertexts (about 1 GB) and the server's pinned pool (2 GiB budget). The end-to-end
# latency benchmark needs the 40 GB card (the plain "A100" sometimes lands on the 80 GB PCIe one)
# and more time: 12 cases, each with 23 queries, plus encrypting two blocks per d. Host memory:
# block 0 flattened (1.2 GB at d = 1536) plus its OpenFHE ciphertexts and the Python client; 16 GiB
# is enough.
@app.function(gpu="A100-40GB", memory=16384, timeout=14400, volumes={DATA_DIR: volume})
def bench(task: str = "all", extra: str = "") -> None:
    """Build in Release and run benchmarks on an A100; results go to the volume."""
    _build("Release")
    out_dir = Path(DATA_DIR) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_script = Path(REPO_DIR) / "core" / "bench" / "run.py"
    if not bench_script.exists():
        print("No benchmarks yet (core/bench/run.py is missing). Nothing to run.")
        return
    # `extra` carries run.py options, e.g. --extra "--presets-e2e-latency
    # m0_depth2,m0_compact_n8192"
    _run(["python", str(bench_script), "--task", task, "--out", str(out_dir), *extra.split()])
    volume.commit()


# The gRPC end-to-end benchmark (`--task grpc-e2e`): the client encrypts about 50k to 100k
# ciphertexts per preset at 1M vectors in parallel processes, and the server keeps about 40 GB of
# ciphertexts per (preset, size) on the container's local disk (deleted after each run). The run
# refuses, and says so in its JSON, if the free disk is below the estimate. Run presets/sizes as
# separate invocations to stay within limits:
#   modal run modal_app.py::bench_e2e --extra "--presets-grpc m0_depth2 --sizes-grpc 1000000"
@app.function(gpu="A100-40GB", cpu=16.0, memory=65536, timeout=86400, volumes={DATA_DIR: volume})
def bench_e2e(extra: str = "") -> None:
    """Release build; grpc-e2e end-to-end benchmark over gRPC on an A100."""
    _build("Release")
    out_dir = Path(DATA_DIR) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(REPO_DIR) / "core" / "bench" / "run.py"
    cmd = ["python", str(script), "--task", "grpc-e2e", "--out", str(out_dir)]
    _run([*cmd, "--workers-grpc", "16", *extra.split()])
    volume.commit()


# ---- GPU families beyond Ampere (research: docs/research/, proposal in docs/decisions/) -------
# One build on CPUs, many runs on GPUs. `prebuild` compiles the tree once for every listed
# architecture on a CPU-only container (nvcc needs no GPU) and stores the build directory in the
# volume; `gpu_run` restores it on the GPU asked for and runs commands. A GPU container therefore
# never spends its billed time compiling, and every family runs the same binaries.
#   modal run modal_app.py::prebuild --build-type Release
#   modal run modal_app.py::on_gpu --gpu l4 --cmd "python -m pytest -m gpu -q"
GPU_FAMILIES = {
    # name: (Modal GPU type, generation, compute capability)
    "t4": ("T4", "Turing", 75),
    "a10": ("A10", "Ampere", 86),
    "a100": ("A100-40GB", "Ampere", 80),
    "l4": ("L4", "Ada", 89),
    "l40s": ("L40S", "Ada", 89),
    "h100": ("H100", "Hopper", 90),
}
ALL_CUDA_ARCHS = ";".join(str(a) for a in sorted({v[2] for v in GPU_FAMILIES.values()}))
BUILD_CACHE = f"{DATA_DIR}/build-cache"


def _cache_tar(build_type: str) -> str:
    return f"{BUILD_CACHE}/{build_type.lower()}.tar"


@app.function(cpu=8.0, memory=16384, timeout=3600, volumes={DATA_DIR: volume})
def prebuild(build_type: str = "Release", archs: str = ALL_CUDA_ARCHS, commit: str = "") -> None:
    """Build everything on CPUs for `archs` and store the build directories in the volume.
    `commit` (the caller's `git rev-parse --short HEAD`) is stored next to it: the image has no
    .git, and a result must name the commit it was measured on."""
    import os

    _build(build_type, archs=archs)
    _build_native()
    os.makedirs(BUILD_CACHE, exist_ok=True)
    natives = [str(p) for p in Path(f"{REPO_DIR}/client/ombra").glob("_native*.so")]
    _run(["tar", "-cf", _cache_tar(build_type), BUILD_DIR, *natives], cwd="/")
    Path(_cache_tar(build_type) + ".json").write_text(
        f'{{"commit": "{commit}", "archs": "{archs}", "build_type": "{build_type}"}}\n'
    )
    _run(["ls", "-la", BUILD_CACHE])
    volume.commit()


@app.function(gpu="A10", cpu=4.0, memory=16384, timeout=3600, volumes={DATA_DIR: volume})
def gpu_run(cmd: str, build_type: str = "Release", commit: str = "") -> None:
    """Restore the `prebuild` output and run `cmd` in the repository. Commands are separated by
    " ;; "; leading KEY=VALUE words of a command are its environment. The GPU type is chosen by
    the caller (`on_gpu`), not here. Refuses a cached build made from another commit."""
    import json
    import os

    meta = json.loads(Path(_cache_tar(build_type) + ".json").read_text())
    if commit and meta["commit"] != commit:
        raise SystemExit(f"cached build is from commit {meta['commit']!r}, asked for {commit!r}")
    _run(["tar", "-xf", _cache_tar(build_type)], cwd="/")
    _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    out_dir = Path(DATA_DIR) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for one in cmd.split(" ;; "):
            env = {
                **os.environ,
                "OMBRA_GIT_COMMIT": meta["commit"],
                "OMBRA_BENCH_BIN_DIR": f"{BUILD_DIR}/core",
                "OMBRA_RESULTS_DIR": str(out_dir),
            }
            words = one.split()
            while words and "=" in words[0] and not words[0].startswith("-"):
                key, _, value = words.pop(0).partition("=")
                env[key] = value
            print("$", " ".join(words), flush=True)
            code = subprocess.run(words, cwd=REPO_DIR, env=env).returncode
            if code != 0:
                raise SystemExit(f"command failed with exit code {code}: {one}")
    finally:
        volume.commit()


@app.local_entrypoint()
def on_gpu(gpu: str = "a10", cmd: str = "nvidia-smi", build_type: str = "Release") -> None:
    """Run `cmd` on one GPU family from the prebuilt binaries (see GPU_FAMILIES)."""
    if gpu not in GPU_FAMILIES:
        raise SystemExit(f"unknown gpu {gpu!r}; one of {sorted(GPU_FAMILIES)}")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    gpu_run.with_options(gpu=GPU_FAMILIES[gpu][0]).remote(cmd, build_type, commit)


def _cpu_run(cmd: str) -> None:
    import os

    _build_native()
    env = {
        **os.environ,
        "PYTHONPATH": f"{REPO_DIR}/client:{REPO_DIR}/tests",
        "OMBRA_LOCAL_DEBUG": "1",
    }
    words = cmd.split()
    while words and "=" in words[0] and not words[0].startswith("-"):
        key, _, value = words.pop(0).partition("=")  # leading KEY=VALUE words are environment
        env[key] = value
    print("$", " ".join(words), flush=True)
    code = subprocess.run(words, cwd=REPO_DIR, env=env).returncode
    if code != 0:
        raise SystemExit(f"command failed with exit code {code}: {cmd}")


@app.function(cpu=4.0, memory=32768, timeout=7200)
def cpu_run(cmd: str) -> None:
    """Run one command in the repo on 4 CPUs in the pinned image, after building ombra._native."""
    _cpu_run(cmd)


@app.function(cpu=8.0, memory=32768, timeout=7200)
def cpu_run8(cmd: str) -> None:
    """Same as cpu_run on 8 CPUs (local mode timings)."""
    _cpu_run(cmd)


# Local mode: a machine WITHOUT a GPU: a plain Python 3.12 image, OpenFHE 1.6.1 (C++) installed the
# way the docs describe, then `pip install ./client[local]`: openfhe-python comes from PyPI (a wheel
# for CPython 3.12), and ombra._native is compiled by pip against the OpenFHE above (wheels will
# ship wheels, then no compiler is needed). Only the client folder, the headers of core/ the
# extension shares with the server, and the check script are uploaded.
_CORE_FILES = ("src/security_floor.cpp", "src/openfhe_expand_ops.hpp")


def _skip_core(p: Path) -> bool:
    s = str(p)
    keep = s in ("include", "src") or s.startswith("include/") or s in _CORE_FILES
    return not keep


local_install_image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")  # libstdc++ of GCC 13+
    .apt_install("git", "build-essential", "cmake", "ninja-build", "libomp-dev")
    .run_commands(
        f"git clone --depth 1 --branch {OPENFHE_TAG} "
        "https://github.com/openfheorg/openfhe-development /opt/openfhe-src",
        "cmake -S /opt/openfhe-src -B /opt/openfhe-build -G Ninja "
        "-DCMAKE_BUILD_TYPE=Release -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF "
        "-DBUILD_BENCHMARKS=OFF -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++",
        "cmake --build /opt/openfhe-build -j",
        "cmake --install /opt/openfhe-build",
        "ldconfig",
    )
    .add_local_dir("core", remote_path="/work/core", copy=True, ignore=_skip_core)
    .add_local_dir(
        "client",
        remote_path="/work/client",
        copy=True,
        ignore=lambda p: any(
            s in str(p)
            for s in ("__pycache__", ".pytest_cache", "tests", ".ruff_cache", "_native.cpython")
        ),
    )
    .add_local_file("tests/local_install_check.py", "/work/local_install_check.py", copy=True)
    .run_commands(
        "CC=gcc CXX=g++ OMBRA_REQUIRE_NATIVE=1 pip install '/work/client[local]'",
    )
)


@app.function(image=local_install_image, cpu=4.0, memory=8192, timeout=1800)
def test_local_install() -> None:
    """`pip install ./client[local]` on a CPU-only machine, then a tiny encrypted search."""
    # Run from /tmp so the installed package is imported, not a source tree.
    _run(["python", "/work/local_install_check.py"], cwd="/tmp")  # noqa: S108


# Recall acceptance check. At N = 100,000 and d = 1536 the run encrypts 13 database
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


# Filtered recall on two real datasets through the client and the real server. Embeddings
# are computed here with sentence-transformers (installed at run time, so the image is unchanged);
# the data is downloaded to the volume's dataset cache and never committed. The server holds the
# encrypted collection (about 20 blocks of about 180 MB), so host memory is large.
#   modal run modal_app.py::filtered_recall --dry-run     (tiny sizes, 5 queries)
@app.function(gpu="A100-40GB", cpu=8.0, memory=65536, timeout=43200, volumes={DATA_DIR: volume})
def filtered_recall(dry_run: bool = False, queries: int = 1000, dataset: str = "") -> None:
    """Release build; filtered recall@10 on two real datasets, preset fast."""
    _build("Release")
    _run(["pip", "install", "--quiet", "sentence-transformers"])
    out_dir = Path(DATA_DIR) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(Path(REPO_DIR) / "core" / "bench" / "filtered_recall.py"),
        "--out",
        str(out_dir),
        "--datasets",
        str(Path(DATA_DIR) / "datasets" / "filtered-recall"),
        "--queries",
        str(queries),
    ]
    if dataset:
        cmd += ["--dataset", *dataset.split()]
    if dry_run:
        cmd.append("--small")
    try:
        _run(cmd)
    finally:
        volume.commit()
