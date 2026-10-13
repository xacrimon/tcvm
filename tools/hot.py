#!/usr/bin/env python3
"""Instruction counts of the loops in compiled regions: runs tcvm-cli with
TCVM_JIT_DUMP=asm and finds each backward branch in each region's code.

    tools/hot.py FILE [ARGS...] [--only CHUNK:LINE] [--body] [--bin PATH]

Each loop prints as `region  start-end  N instrs`, innermost (shortest)
first per region; --body prints the instructions of loops under 80.
"""
import argparse
import os
import re
import subprocess


def regions(dump):
    """(header, [(offset, op, args)]) per region in an asm dump."""
    out, cur = [], None
    for line in dump.splitlines():
        if line.startswith("-- jit asm"):
            cur = (line[len("-- jit asm "):].split(" at ")[0], [])
            out.append(cur)
            continue
        m = re.match(r"\s+([0-9a-f]+):\s+\t(\S+)\s*(.*)", line)
        if cur and m:
            cur[1].append((int(m.group(1), 16), m.group(2), m.group(3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--only", default="")
    ap.add_argument("--body", action="store_true")
    ap.add_argument("--bin", default=os.path.join(os.path.dirname(__file__), "..",
                                                  "target", "release", "tcvm-cli"))
    a = ap.parse_args()
    env = dict(os.environ, TCVM_JIT_DUMP="asm")
    if a.only:
        env["TCVM_JIT_ONLY"] = a.only
    p = subprocess.run([a.bin, "-f", a.file] + a.args, env=env, capture_output=True,
                       text=True, cwd=os.path.dirname(os.path.abspath(a.file)))
    for name, ins in regions(p.stderr):
        loops = set()
        for off, op, args in ins:
            m = re.search(r"0x([0-9a-f]+) <", args)
            if op.startswith("b") and m and int(m.group(1), 16) < off:
                loops.add((int(m.group(1), 16), off))
        for start, end in sorted(loops, key=lambda l: l[1] - l[0]):
            n = (end - start) // 4 + 1
            print(f"{name}  {start:x}-{end:x}  {n} instrs")
            if a.body and n < 80:
                for off, op, args in ins:
                    if start <= off <= end:
                        print(f"    {off:5x}  {op} {args}")


if __name__ == "__main__":
    main()
