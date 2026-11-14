# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Helpers shared by the upgrade tests and `make_fixture.py`: run the driver with a chosen
client, run `ombra_verify`, read the server's recovery line. Synthetic data only."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))  # server_harness

DRIVER = HERE / "driver.py"
VERIFY = REPO_ROOT / "build-cpu" / "server" / "storage" / "ombra_verify"
FIXTURES = REPO_ROOT / "tests" / "fixtures"
RECOVERY = re.compile(r"\[ombra\.storage\] recovery count=(\d+) bytes=(\d+)")


class DriverFailure(Exception):
    def __init__(self, where: str, error: str) -> None:
        super().__init__(f"{where}: {error}")
        self.where, self.error = where, error


def run_driver(client_path: Path, phase: str, addr: str, work: Path, timeout: float = 1500) -> dict:
    """Run one driver phase with the `ombra` client found under `client_path`."""
    env = dict(os.environ, PYTHONPATH=str(client_path), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(  # noqa: S603 - our own script
        [sys.executable, str(DRIVER), phase, addr, str(work)],
        env=env,
        cwd=work,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:  # the client could not even start (import error, crash)
        raise DriverFailure(phase, f"no result (exit {proc.returncode}): {proc.stderr[-1500:]}")
    out = json.loads(lines[-1])
    if not out["ok"]:
        raise DriverFailure(out["where"], out["error"])
    return out


def verify(data_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [str(VERIFY), "--data-dir", str(data_dir)], capture_output=True, text=True, check=False
    )


def recovery_after(log_path: Path, offset: int) -> tuple[int, int] | None:
    """(records replayed, torn bytes dropped) from the last recovery line after `offset`."""
    text = log_path.read_bytes()[offset:].decode(errors="replace")
    found = RECOVERY.findall(text)
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def log_size(log_path: Path) -> int:
    return log_path.stat().st_size if log_path.exists() else 0
