# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
"""Repairs a macOS ombra wheel so ombra._native uses the libraries of the ombra-openfhe wheel.

Usage: python repair_macos.py WHEEL DEST_DIR OPENFHE_LIBDIR LIBOMP_LIBDIR

Both wheels need OpenFHE 1.6.1 and libomp. If each bundled its own copy, a process that imports
openfhe and ombra._native would load libomp twice, and libomp aborts when that happens
("OMP: Error #15"). So this wheel bundles nothing: _native loads OpenFHE and libomp from
site-packages/openfhe/.dylibs/, where delocate put them in the ombra-openfhe wheel (same build
script, same file names). _native is only loaded in local mode, whose extra pins
ombra-openfhe==1.6.1.0.

Each OpenFHE or libomp reference of _native becomes @rpath/<real file name>, and the only rpath
is @loader_path/../openfhe/.dylibs. The extension is then signed again (ad hoc), as arm64 needs.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

from delocate.wheeltools import InWheel

SHARED = re.compile(r"^(libOPENFHE\w+|libomp)\b")
RPATH = "@loader_path/../openfhe/.dylibs"


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def deps(lib: Path) -> list[str]:
    return [line.split(" (")[0].strip() for line in run("otool", "-L", str(lib)).splitlines()[1:]]


def rpaths(lib: Path) -> list[str]:
    lines = run("otool", "-l", str(lib)).splitlines()
    return [
        lines[i + 2].split()[1]
        for i, line in enumerate(lines)
        if line.strip() == "cmd LC_RPATH"
    ]


def real_name(dep: str, libdirs: list[Path]) -> str:
    name = os.path.basename(dep)
    for d in libdirs:
        if (d / name).exists():
            return os.path.basename(os.path.realpath(d / name))
    raise SystemExit(f"repair_macos: {name} is not in {', '.join(map(str, libdirs))}")


def main() -> None:
    wheel, dest, openfhe_libdir, omp_libdir = sys.argv[1:5]
    libdirs = [Path(openfhe_libdir), Path(omp_libdir)]
    out = Path(dest) / Path(wheel).name
    with InWheel(wheel, str(out)):
        natives = list(Path("ombra").glob("_native*.so"))
        if len(natives) != 1:
            raise SystemExit(f"repair_macos: expected one ombra/_native*.so, found {natives}")
        lib = natives[0]
        for dep in deps(lib):
            if SHARED.match(os.path.basename(dep)):
                run("install_name_tool", "-change", dep, f"@rpath/{real_name(dep, libdirs)}", str(lib))
        for rp in rpaths(lib):
            run("install_name_tool", "-delete_rpath", rp, str(lib))
        run("install_name_tool", "-add_rpath", RPATH, str(lib))
        run("codesign", "--force", "--sign", "-", str(lib))
        left = [
            d
            for d in deps(lib)
            if not d.startswith(("/usr/lib/", "/System/", "@rpath/"))
        ]
        if left:
            raise SystemExit(f"repair_macos: unresolved libraries in {lib}: {left}")
        print(f"repair_macos: {lib} -> {RPATH}")
        print("\n".join(deps(lib)))
    print(f"repair_macos: wrote {out}")


if __name__ == "__main__":
    main()
