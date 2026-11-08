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


def _build(build_type: str) -> None:
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
    _run(["cmake", "--build", BUILD_DIR, "-j"])


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
    # Exit code 5 means "no tests collected", which is fine while the repo is empty.
    _run(["python", "-m", "pytest", "-m", "gpu", *pytest_args.split()], ok_codes=(0, 5))


@app.function(cpu=4.0, memory=8192, timeout=1800, volumes={DATA_DIR: volume})
def test_openfhe(pytest_args: str = "") -> None:
    """Run tests marked `openfhe` on CPU. Zero collected tests (exit code 5) is a failure."""
    _run(["python", "-m", "pytest", "-m", "openfhe", *pytest_args.split()], ok_codes=(0,))
    volume.commit()


@app.function(gpu="A100", timeout=3600, volumes={DATA_DIR: volume})
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
