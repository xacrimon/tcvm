#!/usr/bin/env python3
"""Run the LJR benchmark suite in interleaved rounds and report cycles and
instructions per benchmark and mode, from `/usr/bin/time -l`.

Modes: `jit` (tcvm-cli), `interp` (tcvm-cli with TCVM_JIT=off), `lua`, `luajit`,
`luajit-joff`, and per `--tcvm NAME=PATH` the modes `NAME` and `NAME-off`. Every mode but
LuaJIT's has a `-lualua` variant (`jit-lualua`, `lua-lualua`, ...): that VM runs the program
under bench/lualua, a Lua 5.5 interpreter written in Lua, at the sizes in LUALUA_ARGS, and its
ratio is to the first `-lualua` mode instead of the first mode. The suite
defaults to bench/ljr. A benchmark NAME runs as NAME-5.5.lua under the Lua 5.5 modes (tcvm,
lua) when the suite has that file, and as NAME.lua otherwise and under LuaJIT. Every mode
but LuaJIT runs it through a wrapper that defines `unpack` and `loadstring` first; the cwd
is the suite. life.lua is skipped: its source is Latin-1, which tcvm-cli rejects.

    tools/ljr.py [--suite DIR] [--bin PATH] [--modes jit,interp,luajit]
                 [--rounds N] [--only a,b] [--json OUT] [--timeout S]
                 [--tcvm NAME=PATH ...] [--lua PATH] [--luajit PATH] [--lualua PATH]
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
# The suite's own sizes (luajit-remake's run_bench.sh, particles' own); k-nucleotide and
# revcomp read the output of `fasta.lua 5000000` on stdin.
ARGS = {
    "array3d": ["300", "packed"], "binary-trees-num": ["16"], "binary-trees-name": ["15"],
    "bounce": ["3000"], "chameneos": ["10000000"], "coroutine-ring": ["20000000"],
    "fannkuch": ["11"], "fasta": ["5000000"], "fixpoint-fact": ["1000"],
    "heapsort": ["1", "3000000"], "k-nucleotide": ["5000000"], "linear-sieve": ["30000000"],
    "mandelbrot": ["3000"], "mandel-metatable": ["256"], "nbody": ["5000000"], "nsieve": ["12"],
    "partialsums": ["30000000"], "particles": ["10000"], "pidigits-nogmp": ["5000"], "qt": ["14"],
    "quadtree-2": ["14"], "queen": ["12"], "ray": ["9"], "ray-prop": ["9"],
    "recursive-fib-uv": ["40"], "recursive-fib-gv": ["40"], "revcomp": ["5000000"],
    "scimark-fft": ["10"], "scimark-lu": ["5"], "scimark-sor": ["5"], "scimark-sparse": ["300"],
    "series": ["5000"], "spectral-norm": ["2000"], "table-sort": ["5000000"],
    "table-sort-cmp": ["1000000"],
}
# Sizes for the -lualua modes, ~5 s each under lua-lualua; havlak, cd and scimark-fft/lu/sor
# are at their smallest verified or nonzero size. k-nucleotide and revcomp read FASTA of that size.
LUALUA_ARGS = {
    "array3d": ["87", "packed"], "binary-trees-name": ["12"], "binary-trees-num": ["12"],
    "bounce": ["246"], "cd": ["10"], "chameneos": ["393220"], "coroutine-ring": ["2440097"],
    "deltablue": ["4663"], "fannkuch": ["9"], "fasta": ["287029"], "fixpoint-fact": ["161"],
    "havlak": ["1"], "heapsort": ["1", "178937"], "json": ["23"], "k-nucleotide": [],
    "linear-sieve": ["2003934"], "list": ["263"], "mandel-metatable": ["92"],
    "mandelbrot": ["442"], "nbody": ["77606"], "nsieve": ["8"], "partialsums": ["1045802"],
    "particles": ["500"], "permute": ["153"], "pidigits-nogmp": ["622"], "qt": ["9"],
    "quadtree-2": ["9"], "queen": ["11"], "ray": ["9", "23"], "ray-prop": ["9", "23"],
    "recursive-fib-gv": ["32"], "recursive-fib-uv": ["31"], "revcomp": [], "richard": ["5"],
    "scimark-fft": ["1"], "scimark-lu": ["1"], "scimark-sor": ["1"], "scimark-sparse": ["20"],
    "series": ["2954"], "spectral-norm": ["271"], "storage": ["229"], "table-sort": ["591874"],
    "table-sort-cmp": ["143717"], "towers": ["83"],
}
LUALUA_FASTA = {"k-nucleotide": 108770, "revcomp": 1648045}
STDIN = {"k-nucleotide", "revcomp"}
TOOLS = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.join(TOOLS, "..", "bench", "ljr")
LL = "-lualua"
VARIANT = "-5.5"


def programs(suite):
    """Benchmark names in `suite`: its NAME.lua files but SKIP and the NAME-5.5.lua variants."""
    return sorted(f[:-4] for f in os.listdir(suite)
                  if f.endswith(".lua") and not f[:-4].endswith(VARIANT) and f[:-4] not in SKIP)


def source(suite, name, lua55):
    """The file benchmark `name` runs from: NAME-5.5.lua for a Lua 5.5 VM when it exists."""
    v = os.path.join(suite, name + VARIANT + ".lua")
    return v if lua55 and os.path.exists(v) else os.path.join(suite, name + ".lua")


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
    ap.add_argument("--suite", default=SUITE)
    ap.add_argument("--bin", default=os.path.join(os.path.dirname(__file__), "..",
                                                  "target", "release", "tcvm-cli"))
    ap.add_argument("--luajit", default="luajit")
    ap.add_argument("--lua", default="lua", help="Lua 5.5, for the `lua` mode")
    ap.add_argument("--lualua", default=os.path.join(TOOLS, "..", "bench", "lualua", "lualua.lua"),
                    help="lualua.lua, for the -lualua modes (the bench/lualua submodule)")
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
    # Benchmarks run with the suite as cwd, so a relative path would not resolve.
    a.suite = os.path.abspath(a.suite)

    bins = {"jit": os.path.abspath(a.bin)}
    for spec in a.tcvm:
        name, _, path = spec.partition("=")
        bins[name] = os.path.abspath(path)
    modes = [m for m in a.modes.split(",") if m]
    tcvm = {m: m for m in bins} | {m + "-off": m for m in bins} | {"interp": "jit"}
    base = {m: m.removesuffix(LL) for m in modes}
    bad = [m for m in modes if base[m] not in tcvm and base[m] != "lua"
           and (m.endswith(LL) or m not in ("luajit", "luajit-joff"))]
    if bad:
        sys.exit(f"ljr.py: unknown modes {','.join(bad)}")
    a.lualua = os.path.abspath(a.lualua)
    shim = os.path.join(TOOLS, "lualua-shim.lua")
    names = programs(a.suite)
    if a.only:
        want = set(a.only.split(","))
        names = [n for n in names if n in want]
    lualua = any(m.endswith(LL) for m in modes)
    if lualua:
        if not os.path.exists(a.lualua):
            sys.exit(f"ljr.py: no {a.lualua} (git submodule update --init bench/lualua)")
        missing = [n for n in names if n not in LUALUA_ARGS]
        if missing:
            sys.exit(f"ljr.py: no LUALUA_ARGS for {','.join(missing)}")
    wrapdir = tempfile.mkdtemp(prefix="ljr-")
    # stdin of k-nucleotide and revcomp per size group, made with luajit when missing.
    fastas = {n: a.fasta for n in STDIN}
    fastas |= {n + LL: os.path.join(tempfile.gettempdir(), f"FASTA_{LUALUA_FASTA[n]}")
               for n in STDIN}
    sizes = {a.fasta: 5000000} | {fastas[n + LL]: LUALUA_FASTA[n] for n in STDIN}
    used = {fastas[n + LL if m.endswith(LL) else n] for n in STDIN & set(names) for m in modes}
    for path in sorted(used):
        if os.path.exists(path) or a.dry_run:
            continue
        with open(path + ".tmp", "w") as fh:
            subprocess.run([a.luajit, os.path.join(a.suite, "fasta.lua"), str(sizes[path])],
                           stdout=fh, check=True)
        os.rename(path + ".tmp", path)
    results = {n: {m: [] for m in modes} for n in names}

    def command(name, mode):
        nested = mode.endswith(LL)
        args = LUALUA_ARGS[name] if nested else ARGS.get(name, [])
        mode = base[mode]
        env = dict(os.environ)
        if mode.startswith("luajit"):
            flags = ["-joff"] if mode == "luajit-joff" else []
            return [a.luajit] + flags + [source(a.suite, name, False)] + args, env
        w = os.path.join(wrapdir, name + ".lua")
        with open(w, "w") as fh:
            fh.write("unpack = unpack or table.unpack\n"
                     "loadstring = loadstring or load\n"
                     f"return assert(loadfile({source(a.suite, name, True)!r}))(...)\n")
        if mode == "lua":
            return [a.lua] + ([a.lualua] if nested else []) + [w] + args, env
        if mode == "interp" or mode.endswith("-off"):
            env["TCVM_JIT"] = "off"
        return [bins[tcvm[mode]], "-f"] + ([shim, a.lualua] if nested else []) + [w] + args, env

    for r in range(a.rounds):
        for n in names:
            for m in modes:
                cmd, env = command(n, m)
                if a.dry_run:
                    print(" ".join(cmd))
                    continue
                stdin = fastas[n + LL if m.endswith(LL) else n] if n in STDIN else None
                res = measure(cmd, a.suite, env, a.timeout, stdin)
                results[n][m].append(res)
                if res is None:
                    tag = "timeout"
                elif res["status"] != 0:
                    tag = f"failed (status {res['status']})"
                else:
                    tag = f"{res.get('cycles', 0) / 1e6:.1f}M cycles"
                print(f"round {r + 1} {n} {m}: {tag}", file=sys.stderr)
    if a.dry_run:
        return

    def med(runs, key):
        xs = [x[key] for x in runs if x and x.get("status") == 0 and key in x]
        return statistics.median(xs) if xs else None

    # Ratios of each mode to the first of its size group (-lualua or not), by median cycles,
    # and their geomean.
    ref = {m: next(x for x in modes if x.endswith(LL) == m.endswith(LL)) for m in modes}
    rest = [m for m in modes if ref[m] != m]
    cw = {m: max(16, len(m) + 7) for m in modes}
    rw = {m: max(16, len(m + ref[m]) + 3) for m in rest}
    head = f"{'benchmark':24}" + "".join(f"{m + ' Mcyc':>{cw[m]}}" for m in modes)
    head += "".join(f"{m + '/' + ref[m]:>{rw[m]}}" for m in rest)
    print(head)
    logs = {m: [] for m in rest}
    for n in names:
        cyc = {m: med(results[n][m], "cycles") for m in modes}
        row = f"{n:24}" + "".join(
            f"{(c / 1e6 if c else float('nan')):>{cw[m]}.1f}" for m, c in cyc.items())
        for m in rest:
            if cyc[m] and cyc[ref[m]]:
                q = cyc[m] / cyc[ref[m]]
                logs[m].append(q)
                row += f"{q:>{rw[m]}.2f}"
            else:
                row += f"{'-':>{rw[m]}}"
        print(row)
    geo = f"{'geomean':24}" + " " * sum(cw.values())
    for m in rest:
        xs = logs[m]
        g = statistics.geometric_mean(xs) if xs else float("nan")
        geo += f"{g:>{rw[m]}.2f}"
    print(geo)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
