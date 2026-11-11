#!/usr/bin/env python3
"""mini.py [--bin PATH] [--mode MODE] [--grep REGEX] IN OUT: shrink a program that shows a finding.

Removes runs of lines, halving the run length, while the program still runs with the JIT off and
still fails in MODE (a run.py mode, default `hot`): its stdout or exit status differs from the
JIT-off run, or with --grep, its stderr matches REGEX (for a failed compile). Each candidate runs
twice in MODE, for findings that depend on timing.
"""
import argparse
import os
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import MODES, REPO, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=os.path.join(REPO, "target", "release", "tcvm-cli"))
    ap.add_argument("--mode", choices=MODES, default="hot")
    ap.add_argument("--grep", type=re.compile)
    ap.add_argument("--timeout", type=int, default=5)
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    tmp = a.dst + ".tmp.lua"

    def fails(lines):
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        rc, want, _ = run(a.bin, tmp, {"TCVM_JIT": "off"}, a.timeout)
        if rc != 0:
            return False
        for _ in range(2):
            rc, got, stderr = run(a.bin, tmp, MODES[a.mode], a.timeout)
            if a.grep.search(stderr) if a.grep else rc != 0 or got != want:
                return True
        return False

    lines = open(a.src).read().split("\n")
    try:
        if not fails(lines):
            sys.exit("the input does not show the finding")
        n = len(lines) // 2
        while n >= 1:
            i, shrunk = 0, False
            while i < len(lines):
                rest = lines[:i] + lines[i + n:]
                if fails(rest):
                    lines, shrunk = rest, True
                else:
                    i += n
            if not shrunk:
                n //= 2
            print(f"{len(lines)} lines, runs of {n}", file=sys.stderr, flush=True)
    finally:
        os.unlink(tmp)
    with open(a.dst, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
