#!/usr/bin/env python3
"""run.py [--bin [NAME=]PATH]... [--kind KIND] [--out DIR] [--jobs N] [--timeout SECS] LO HI

Differential JIT fuzz: generates the programs of seeds LO..HI-1 (gen.py), runs each with the JIT
off and in every mode below, and reports
  - runs whose stdout or exit status differ from the same binary's JIT-off run,
  - compiles that failed on a verifier, checker or backend error or a panic (TCVM_JIT_LOG); those
    fall back to the interpreter, so their output is right,
  - JIT-off runs that fail.
Needs `tcvm-dev` builds. The default binary is target/release/tcvm-cli; give several to cover
several targets at once (`--bin a64=... --bin x86=...`). Each binary is compared with itself
only: libm results differ in the last bit between targets.

Findings go to DIR/results-LO-HI.jsonl (default DIR target/fuzz) and their programs stay in
DIR/prog; triage.py groups them, mini.py shrinks one.
"""
import argparse
import json
import multiprocessing
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import gen  # noqa: E402

HOT = {"TCVM_JIT_HOT_CALL": "1", "TCVM_JIT_HOT_LOOP": "1"}
MODES = {
    "default": {},
    "hot": HOT,
    "noinline": dict(HOT, TCVM_JIT_INLINE="0"),
    "check": dict(HOT, TCVM_JIT_CHECK="1"),
    "deopt": dict(HOT, TCVM_JIT_DEOPT_ALL="1"),
    "gcstress": dict(HOT, TCVM_GC_STRESS="1"),
}
# The slow modes run on every Nth seed, with a longer timeout.
EVERY = {"deopt": 4, "gcstress": 8}
SLOW = {"gcstress": 3}
INTERNAL = re.compile(r"failed: (verifier|backend)|panicked|allocation check|checker", re.I)


def run(binary, path, env, secs):
    """(exit status, stdout, stderr); a run past `secs` dies of SIGALRM."""
    env = dict(os.environ, TCVM_JIT_LOG="1", **env)
    cmd = ["perl", "-e", "alarm shift; exec @ARGV", str(secs), binary, "-f", path]
    p = subprocess.run(cmd, env=env, capture_output=True)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def first_diff(want, got):
    a, b = want.splitlines(), got.splitlines()
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else "<eof>"
        y = b[i] if i < len(b) else "<eof>"
        if x != y:
            return x, y
    return None


def fuzz(job):
    seed, kind, bins, out, secs = job
    src, insts = gen.G(seed).program(kind)
    path = os.path.join(out, "prog", f"p{seed}.lua")
    with open(path, "w") as f:
        f.write(src)
    found = []
    base = {}
    for name, binary in bins:
        rc, stdout, stderr = run(binary, path, {"TCVM_JIT": "off"}, secs)
        if rc != 0:
            found.append({"what": "interp", "bin": name, "rc": rc, "err": stderr[-400:]})
        else:
            base[name] = stdout
    for name, binary in bins:
        if name not in base:
            continue
        for mode, env in MODES.items():
            if seed % EVERY.get(mode, 1) != 0:
                continue
            rc, stdout, stderr = run(binary, path, env, secs * SLOW.get(mode, 1))
            for line in stderr.splitlines():
                if INTERNAL.search(line):
                    found.append({"what": "internal", "bin": name, "mode": mode, "msg": line[:300]})
            if rc == 0 and stdout == base[name]:
                continue
            d = {"what": "diverges", "bin": name, "mode": mode, "rc": rc}
            diff = first_diff(base[name], stdout)
            if diff:
                d["want"], d["got"] = diff[0][:300], diff[1][:300]
                # The first differing line names its instance.
                m = re.match(r"I(\d+)", diff[0] if diff[0] != "<eof>" else diff[1])
                if m:
                    d["kind"], d["body"] = insts[int(m.group(1))]
            if rc != 0:
                d["err"] = stderr[-600:]
            found.append(d)
    if not found:
        os.unlink(path)
    return {"seed": seed, "found": found}


def is_dev(binary, out):
    probe = os.path.join(out, "probe.lua")
    with open(probe, "w") as f:
        f.write("local s = 0 for i = 1, 100 do s = s + i end\n")
    rc, _, stderr = run(binary, probe, HOT, 10)
    os.unlink(probe)
    return rc == 0 and "jit:" in stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", action="append", metavar="[NAME=]PATH")
    ap.add_argument("--kind", choices=gen.KINDS)
    ap.add_argument("--out", default=os.path.join(REPO, "target", "fuzz"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("lo", type=int)
    ap.add_argument("hi", type=int)
    a = ap.parse_args()
    bins = []
    for b in a.bin or [os.path.join(REPO, "target", "release", "tcvm-cli")]:
        name, _, path = b.rpartition("=")
        bins.append((name or os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path)))), os.path.abspath(path)))
    if len({n for n, _ in bins}) != len(bins):
        sys.exit("give the binaries distinct names: --bin NAME=PATH")
    os.makedirs(os.path.join(a.out, "prog"), exist_ok=True)
    for name, path in bins:
        if not is_dev(path, a.out):
            sys.exit(f"{path}: does not run, or is not a tcvm-dev build (no TCVM_JIT_LOG output)")
    results = os.path.join(a.out, f"results-{a.lo}-{a.hi}.jsonl")
    hits = 0
    jobs = [(s, a.kind, bins, a.out, a.timeout) for s in range(a.lo, a.hi)]
    with multiprocessing.Pool(a.jobs) as pool, open(results, "w") as f:
        for r in pool.imap_unordered(fuzz, jobs):
            if not r["found"]:
                continue
            hits += 1
            f.write(json.dumps(r) + "\n")
            f.flush()
            what = sorted({":".join(str(x[k]) for k in ("what", "bin", "mode") if k in x) for x in r["found"]})
            print(r["seed"], *what, flush=True)
    print(f"{hits} of {a.hi - a.lo} seeds with findings; {results}")


if __name__ == "__main__":
    main()
