"""Instruction mix of compiled sm_80 kernels (bench-only).

Usage: python sass_mix.py <binary-or-cubin> <regex on the mangled kernel name> [...]
Runs `cuobjdump -sass -arch sm_80`, finds each matching function and prints the opcode histogram of
the whole function and of its longest loop (the longest backward branch: the loop over dimensions).
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter

INSTR = re.compile(r"/\*([0-9a-f]{4})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?);")
TARGET = re.compile(r"(0x[0-9a-f]+)")


def functions(text: str) -> dict[str, list[tuple[int, str, str]]]:
    out: dict[str, list[tuple[int, str, str]]] = {}
    name = None
    for line in text.splitlines():
        if line.strip().startswith("Function :"):
            name = line.split(":", 1)[1].strip()
            out[name] = []
        elif name:
            m = INSTR.search(line)
            if m:
                out[name].append((int(m.group(1), 16), m.group(3), m.group(4)))
    return out


def hist(ins: list[tuple[int, str, str]]) -> str:
    c = Counter(op for _, op, _ in ins)
    return ", ".join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))


def main() -> None:
    text = subprocess.run(  # noqa: S603, S607
        ["cuobjdump", "-sass", "-arch", "sm_80", sys.argv[1]],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    funcs = functions(text)
    for pat in sys.argv[2:]:
        for name, ins in funcs.items():
            if not re.search(pat, name):
                continue
            print(f"\n=== {name}: {len(ins)} instructions")
            print("whole function:", hist(ins))
            best: list[tuple[int, str, str]] = []
            for addr, op, rest in ins:
                if op.startswith("BRA"):
                    t = TARGET.search(rest)
                    if t and int(t.group(1), 16) < addr:
                        body = [x for x in ins if int(t.group(1), 16) <= x[0] <= addr]
                        if len(body) > len(best):
                            best = body
            if best:
                print(f"longest loop: {len(best)} instructions")
                print("loop body:", hist(best))
            else:
                print("no loop found")


if __name__ == "__main__":
    main()
