#!/usr/bin/env python3
"""Run the LJR benchmark suite in interleaved rounds and report cycles and
instructions per benchmark and mode, from `/usr/bin/time -l`.

Modes: `jit` (tcvm-cli), `interp` (tcvm-cli with TCVM_JIT=off), `luajit`,
`luajit-joff`, and per `--tcvm NAME=PATH` the modes `NAME` and `NAME-off`. Each benchmark runs through a wrapper that defines
`unpack` and `loadstring` first and is loaded from its own directory.
life.lua is skipped: its source is Latin-1, which tcvm-cli rejects.

    tools/ljr.py [--suite DIR] [--bin PATH] [--modes jit,interp,luajit]
                 [--rounds N] [--only a,b] [--json OUT] [--timeout S]
                 [--tcvm NAME=PATH ...]
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile

SKIP = {"life"}
# The suite's own sizes (luajit-remake's run_bench.sh); k-nucleotide and
# revcomp read the output of `fasta.lua 5000000` on stdin.
ARGS = {
    "array3d": ["300", "packed"], "binary-trees-num": ["16"], "binary-trees-name": ["15"],
    "bounce": ["3000"], "chameneos": ["1e7"], "coroutine-ring": ["2e7"],
    "fannkuch": ["11"], "fasta": ["5e6"], "fixpoint-fact": ["1000"],
    "heapsort": ["1", "3000000"], "k-nucleotide": ["5e6"], "linear-sieve": ["3e7"],
    "mandelbrot": ["3000"], "mandel-metatable": ["256"], "nbody": ["5e6"], "nsieve": ["12"],
    "partialsums": ["3e7"], "pidigits-nogmp": ["5000"], "qt": ["14"], "quadtree-2": ["14"],
    "queen": ["12"], "ray": ["9"], "ray-prop": ["9"], "recursive-fib-uv": ["40"],
    "recursive-fib-gv": ["40"], "revcomp": ["5e6"], "scimark-fft": ["10"],
    "scimark-lu": ["5"], "scimark-sor": ["5"], "scimark-sparse": ["300"], "series": ["5000"],
    "spectral-norm": ["2000"], "table-sort": ["5e6"], "table-sort-cmp": ["1e6"],
}
STDIN = {"k-nucleotide", "revcomp"}


def measure(cmd, cwd, env, timeout, stdin=None):
    """Cycles, instructions, wall seconds and max RSS of one run, or None."""
    full = ["/usr/bin/time", "-l"] + cmd
    try:
        with open(stdin or os.devnull, "rb") as inp:
            p = subprocess.run(full, cwd=cwd, env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=timeout, stdin=inp)
    except subprocess.TimeoutExpired:
        return None
    err = p.stderr.decode("utf-8", "replace")
    out = {}
    for key, pat in [("cycles", r"(\d+)\s+cycles elapsed"),
                     ("instructions", r"(\d+)\s+instructions retired"),
                     ("rss", r"(\d+)\s+maximum resident set size"),
                     ("wall", r"([\d.]+)\s+real")]:
        m = re.search(pat, err)
        if m:
            out[key] = float(m.group(1)) if key == "wall" else int(m.group(1))
    out["status"] = p.returncode
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default=os.path.expanduser("~/repos/luajit-remake/luabench"))
    ap.add_argument("--bin", default=os.path.join(os.path.dirname(__file__), "..",
                                                  "target", "release", "tcvm-cli"))
    ap.add_argument("--luajit", default="luajit")
    ap.add_argument("--modes", default="jit,interp,luajit")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--tcvm", action="append", default=[])
    ap.add_argument("--fasta", default=os.path.join(tempfile.gettempdir(), "FASTA_5000000"),
                    help="stdin of k-nucleotide and revcomp, made with luajit when missing")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    bins = {"jit": os.path.abspath(a.bin)}
    for spec in a.tcvm:
        name, _, path = spec.partition("=")
        bins[name] = os.path.abspath(path)
    modes = [m for m in a.modes.split(",") if m]
    names = sorted(f[:-4] for f in os.listdir(a.suite) if f.endswith(".lua"))
    names = [n for n in names if n not in SKIP]
    if a.only:
        want = set(a.only.split(","))
        names = [n for n in names if n in want]
    wrapdir = tempfile.mkdtemp(prefix="ljr-")
    if STDIN & set(names) and not os.path.exists(a.fasta) and not a.dry_run:
        with open(a.fasta, "w") as fh:
            subprocess.run([a.luajit, os.path.join(a.suite, "fasta.lua"), "5000000"],
                           stdout=fh, check=True)
    results = {n: {m: [] for m in modes} for n in names}

    def command(name, mode):
        src = os.path.join(a.suite, name + ".lua")
        args = ARGS.get(name, [])
        base = "jit" if mode == "interp" else mode.removesuffix("-off")
        if base in bins:
            w = os.path.join(wrapdir, name + ".lua")
            with open(w, "w") as fh:
                fh.write("unpack = unpack or table.unpack\n"
                         "loadstring = loadstring or load\n"
                         f"return assert(loadfile({src!r}))(...)\n")
            env = dict(os.environ)
            if mode == "interp" or mode.endswith("-off"):
                env["TCVM_JIT"] = "off"
            return [bins[base], "-f", w] + args, env
        flags = ["-joff"] if mode == "luajit-joff" else []
        return [a.luajit] + flags + [src] + args, dict(os.environ)

    for r in range(a.rounds):
        for n in names:
            for m in modes:
                cmd, env = command(n, m)
                if a.dry_run:
                    print(" ".join(cmd))
                    continue
                res = measure(cmd, a.suite, env, a.timeout, a.fasta if n in STDIN else None)
                results[n][m].append(res)
                tag = "timeout" if res is None else f"{res.get('cycles', 0) / 1e6:.1f}M cycles"
                print(f"round {r + 1} {n} {m}: {tag}", file=sys.stderr)
    if a.dry_run:
        return

    def med(runs, key):
        xs = [x[key] for x in runs if x and x.get("status") == 0 and key in x]
        return statistics.median(xs) if xs else None

    # Ratios of each mode to the first, by median cycles, and their geomean.
    head = f"{'benchmark':24}" + "".join(f"{m + ' Mcyc':>16}" for m in modes)
    head += "".join(f"{m + '/' + modes[0]:>16}" for m in modes[1:])
    print(head)
    logs = {m: [] for m in modes[1:]}
    for n in names:
        cyc = {m: med(results[n][m], "cycles") for m in modes}
        row = f"{n:24}" + "".join(
            f"{(c / 1e6 if c else float('nan')):>16.1f}" for c in cyc.values())
        for m in modes[1:]:
            if cyc[m] and cyc[modes[0]]:
                q = cyc[m] / cyc[modes[0]]
                logs[m].append(q)
                row += f"{q:>16.2f}"
            else:
                row += f"{'-':>16}"
        print(row)
    geo = f"{'geomean':24}" + " " * 16 * len(modes)
    for m in modes[1:]:
        xs = logs[m]
        g = statistics.geometric_mean(xs) if xs else float("nan")
        geo += f"{g:>16.2f}"
    print(geo)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
