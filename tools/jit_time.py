#!/usr/bin/env python3
"""Compile time per benchmark from TCVM_JIT_TIME: per stage and per pass,
summed over the run's compiles, and their share of the run's wall time.

    tools/jit_time.py [--suite DIR] [--only a,b] [--bin PATH] [--passes]
"""
import argparse
import collections
import os
import re
import subprocess
import tempfile
import time

STAGES = ["build", "passes", "lower", "regalloc", "emit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=os.path.expanduser("~/repos/luajit-remake/luabench"))
    ap.add_argument("--only", default="")
    ap.add_argument("--passes", action="store_true")
    ap.add_argument("--bin", default=os.path.join(os.path.dirname(__file__), "..",
                                                  "target", "release", "tcvm-cli"))
    a = ap.parse_args()
    names = sorted(f[:-4] for f in os.listdir(a.suite) if f.endswith(".lua") and f != "life.lua")
    if a.only:
        names = [n for n in names if n in set(a.only.split(","))]
    wrapdir = tempfile.mkdtemp(prefix="jit-time-")
    print(f"{'benchmark':24}{'compiles':>9}{'compile ms':>12}{'wall ms':>10}{'share':>8}  "
          + " ".join(f"{s:>8}" for s in STAGES))
    for n in names:
        w = os.path.join(wrapdir, n + ".lua")
        with open(w, "w") as fh:
            fh.write("unpack = unpack or table.unpack\nloadstring = loadstring or load\n"
                     f"return assert(loadfile({os.path.join(a.suite, n + '.lua')!r}))(...)\n")
        args = ["200000"] if n == "partialsums" else []
        t = time.time()
        p = subprocess.run([a.bin, "-f", w] + args, cwd=a.suite, capture_output=True, text=True,
                           env=dict(os.environ, TCVM_JIT_TIME="1"), stdin=subprocess.DEVNULL)
        wall = (time.time() - t) * 1000
        stages = collections.Counter()
        passes = collections.Counter()
        count = 0
        for line in p.stderr.splitlines():
            if not line.startswith("jit time"):
                continue
            count += 1
            for s in STAGES:
                m = re.search(rf"{s} (\d+)us", line)
                if m:
                    stages[s] += int(m.group(1))
            tail = line.split("; passes: ", 1)
            if len(tail) == 2:
                for part in tail[1].split(", "):
                    name, _, us = part.rpartition(" ")
                    passes[name] += int(us)
        total = sum(stages.values()) / 1000
        print(f"{n:24}{count:>9}{total:>12.1f}{wall:>10.0f}{total / wall * 100:>7.1f}%  "
              + " ".join(f"{stages[s] / 1000:>8.1f}" for s in STAGES))
        if a.passes and passes:
            print("    " + ", ".join(f"{k} {v / 1000:.1f}" for k, v in passes.most_common()))


if __name__ == "__main__":
    main()
