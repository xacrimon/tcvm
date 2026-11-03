#!/usr/bin/env python3
"""The operation cost table (jit-perf-fixes2.md G1): instructions and cycles
per operation, compiled and interpreted, with LuaJIT for reference. A row
whose compiled cost exceeds the interpreted one fails the run.

Each case in tools/opcost/*.lua is a loop around one operation. It runs at
two sizes, N/4 and N, and the difference cancels start-up and compilation.
Header lines set the case up:

    -- live: 0,4,8     variants with that many extra values live across it
    -- n: 2000000      operations at the larger size (default 2e6)
    -- lj: no          LuaJIT cannot run it
    -- env: K=V        environment for the tcvm runs; @FILE is the case's
                       file name (TCVM_JIT_ONLY=@FILE keeps chunks that
                       `load` makes interpreted)

The body reads `N` and may hold three markers, each on a line of its own:
`--@decl` (declares the live values), `--@live` (updates them, inside the
loop) and `--@use` (reads them after it). Each live value costs one integer
add per iteration on top of the operation.

    tools/opcost.py [--bin PATH] [--rounds N] [--only a,b] [--modes ...]
                    [--tol F] [--accept a,b] [--json OUT] [--compare OLD]
"""
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_N = 2_000_000


def measure(cmd, env, timeout):
    """Instructions and cycles of one run, or None if it failed."""
    try:
        p = subprocess.run(["/usr/bin/time", "-l"] + cmd, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    err = p.stderr.decode("utf-8", "replace")
    if p.returncode != 0:
        sys.stderr.write(f"failed ({p.returncode}): {' '.join(cmd)}\n{err[-2000:]}\n")
        return None
    i = re.search(r"(\d+)\s+instructions retired", err)
    c = re.search(r"(\d+)\s+cycles elapsed", err)
    if not (i and c):
        return None
    return int(i.group(1)), int(c.group(1))


def parse_case(path):
    head, body = {}, []
    with open(path) as fh:
        lines = fh.read().splitlines()
    k = 0
    while k < len(lines):
        m = re.match(r"--\s*(\w+):\s*(.*)$", lines[k])
        if not m:
            break
        head.setdefault(m.group(1), []).append(m.group(2).strip())
        k += 1
    body = lines[k:]
    name = os.path.basename(path)[:-4]
    lives = [int(x) for x in head.get("live", ["0"])[0].split(",")]
    n = int(float(head.get("n", [str(DEFAULT_N)])[0]))
    lj = head.get("lj", ["yes"])[0] != "no"
    env = dict(e.split("=", 1) for e in head.get("env", []))
    return {"name": name, "body": body, "lives": lives, "n": n, "lj": lj, "env": env}


def program(case, live):
    """The case's source with `live` extra live values."""
    vs = [f"v{k}" for k in range(1, live + 1)]
    out = ["local N = tonumber(arg[1])"]
    for line in case["body"]:
        s = line.strip()
        ind = line[: len(line) - len(line.lstrip())]
        if s == "--@decl":
            if vs:
                out.append(ind + "local " + ", ".join(vs) + " = " + ", ".join("0" for _ in vs))
        elif s == "--@live":
            out.extend(ind + f"{v} = {v} + 1" for v in vs)
        elif s == "--@use":
            if vs:
                out.append(ind + "if " + " + ".join(vs) + " < 0 then print('live') end")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=os.path.join(HERE, "..", "target", "release", "tcvm-cli"))
    ap.add_argument("--luajit", default="luajit")
    ap.add_argument("--modes", default="jit,interp,lj,ljoff")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated substrings of row names")
    ap.add_argument("--tol", type=float, default=0.03,
                    help="a row fails when compiled cycles exceed interpreted by this share")
    ap.add_argument("--accept", default="", help="rows whose failure is accepted")
    ap.add_argument("--json", default="")
    ap.add_argument("--compare", default="", help="an earlier --json run to show deltas against")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--cases", default=os.path.join(HERE, "opcost"))
    a = ap.parse_args()

    modes = [m for m in a.modes.split(",") if m]
    if not shutil.which(a.luajit):
        modes = [m for m in modes if not m.startswith("lj")]
    bin_ = os.path.abspath(a.bin)
    cases = [parse_case(os.path.join(a.cases, f))
             for f in sorted(os.listdir(a.cases)) if f.endswith(".lua")]
    tmp = tempfile.mkdtemp(prefix="opcost-")
    rows = []
    for c in cases:
        for live in c["lives"]:
            name = c["name"] if c["lives"] == [0] else f"{c['name']}/{live}"
            if a.only and not any(s in name for s in a.only.split(",")):
                continue
            fname = name.replace("/", "-L") + ".lua"
            path = os.path.join(tmp, fname)
            with open(path, "w") as fh:
                fh.write(program(c, live))
            rows.append({"name": name, "case": c, "path": path, "file": fname})

    def command(row, mode, n):
        c = row["case"]
        if mode.startswith("lj"):
            if not c["lj"]:
                return None
            flags = ["-joff"] if mode == "ljoff" else []
            return [a.luajit] + flags + [row["path"], str(n)], dict(os.environ)
        env = dict(os.environ)
        for k, v in c["env"].items():
            env[k] = v.replace("@FILE", row["file"])
        if mode == "interp":
            env["TCVM_JIT"] = "off"
        return [bin_, "-f", row["path"], str(n)], env

    # Per row and mode, the per-operation (instructions, cycles) of each round.
    per = {r["name"]: {m: [] for m in modes} for r in rows}
    for rnd in range(a.rounds):
        for r in rows:
            big = r["case"]["n"]
            small = big // 4
            for m in modes:
                cmds = [command(r, m, small), command(r, m, big)]
                if cmds[0] is None:
                    continue
                res = [measure(cmd, env, a.timeout) for cmd, env in cmds]
                if None in res:
                    per[r["name"]][m].append(None)
                    continue
                d = big - small
                per[r["name"]][m].append(((res[1][0] - res[0][0]) / d, (res[1][1] - res[0][1]) / d))
            print(f"round {rnd + 1} {r['name']}", file=sys.stderr)

    def med(xs, k):
        vals = [x[k] for x in xs if x is not None]
        return statistics.median(vals) if vals else None

    old = {}
    if a.compare:
        with open(a.compare) as fh:
            old = json.load(fh)["rows"]
    accept = set(x for x in a.accept.split(",") if x)
    out = {}
    cols = "".join(f"{m + ' ins':>11}{m + ' cyc':>10}" for m in modes)
    print(f"{'row':22}{cols}{'jit/int':>9}" + (f"{'Δcyc':>8}" if old else "") + "  status")
    failed = []
    for r in rows:
        n = r["name"]
        stats = {m: (med(per[n][m], 0), med(per[n][m], 1)) for m in modes}
        out[n] = {m: {"ins": s[0], "cyc": s[1]} for m, s in stats.items()}
        line = f"{n:22}"
        for m in modes:
            ins, cyc = stats[m]
            line += f"{ins:>11.1f}{cyc:>10.1f}" if ins is not None else f"{'-':>11}{'-':>10}"
        status = ""
        j, i = stats.get("jit", (None, None))[1], stats.get("interp", (None, None))[1]
        if j is not None and i is not None:
            line += f"{(j / i if i > 0 else float('inf')):>9.2f}"
            # A row fails past the tolerance, and past two cycles, where the
            # difference of two runs is noise.
            if j > i * (1 + a.tol) and j - i > 2:
                status = "ACCEPTED" if n in accept else "FAIL"
                if n not in accept:
                    failed.append(n)
        else:
            line += f"{'-':>9}"
            if "jit" in modes and "interp" in modes:
                status = "ERROR"
                failed.append(n)
        if old:
            o = old.get(n, {}).get("jit", {}).get("cyc")
            line += f"{((j - o) / o * 100 if (o and j) else float('nan')):>7.1f}%"
        print(line + "  " + status)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"bin": bin_, "rows": out}, fh, indent=1)
    if failed:
        print(f"\n{len(failed)} failing: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
