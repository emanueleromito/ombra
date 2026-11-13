# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Record the setup of an experiment run: commit, CPU, GPU, library versions.

    from envinfo import environment
    json.dump({"environment": environment(), ...}, f)

Every raw output file in docs/research/ starts from this record, so a result can be traced to the
machine and the commit that produced it. Nothing here reads keys or data.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=REPO_ROOT).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _openfhe_version() -> str:
    roots = [Path("/usr/local"), Path.home() / ".local" / "openfhe-1.6.1"]
    for root in roots:
        for path in sorted(root.glob("lib*/OpenFHE/OpenFHEConfigVersion.cmake")):
            m = re.search(r'set\(PACKAGE_VERSION\s+"([^"]+)"\)', path.read_text())
            if m:
                return m.group(1)
    return "unknown"


def environment() -> dict:
    """The setup record. `git_commit` is "unknown" where there is no .git (the Modal image);
    callers there pass the commit in through OMBRA_GIT_COMMIT."""
    commit = (
        os.environ.get("OMBRA_GIT_COMMIT") or _run(["git", "rev-parse", "--short", "HEAD"]).strip()
    )
    dirty = bool(_run(["git", "status", "--porcelain", "--untracked-files=no"]).strip())
    gpu = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    ).strip()
    nvcc = re.search(r"release ([\d.]+)", _run(["nvcc", "--version"]))
    return {
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit or "unknown",
        "git_dirty": dirty,
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "gpu": gpu or "none",
        "cuda_toolkit": nvcc.group(1) if nvcc else "unknown",
        "openfhe": _openfhe_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(environment(), indent=2))
