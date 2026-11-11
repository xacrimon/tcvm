#!/usr/bin/env python3
"""triage.py [DIR]: group the findings in DIR/results-*.jsonl (default target/fuzz) by cause.

Divergences group by the template and body of the function whose output differed first; failed
compiles by their message with names and numbers removed. Each group lists a few seeds and the
binary:mode pairs it showed in.
"""
import collections
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normal(msg):
    msg = re.sub(r"@\S+", "@F", msg)
    msg = re.sub(r"\b[vib]\d+\b", "N", msg)
    msg = re.sub(r"pc\d+", "pcN", msg)
    msg = re.sub(r"thread '[^']*' \(\d+\)", "thread", msg)
    return re.sub(r"\b\d+\b", "K", msg) if "panic" in msg and "panicked at" not in msg else msg


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "target", "fuzz")
    diverges = collections.defaultdict(list)
    internal = collections.defaultdict(list)
    other = []
    for fn in sorted(glob.glob(os.path.join(d, "results-*.jsonl"))):
        for line in open(fn):
            r = json.loads(line)
            for x in r["found"]:
                where = (r["seed"], f'{x["bin"]}:{x.get("mode", "off")}')
                if x["what"] == "internal":
                    internal[normal(x["msg"])].append(where)
                elif x["what"] == "diverges" and "body" in x:
                    diverges[x["kind"], x["body"]].append((*where, x["want"], x["got"]))
                else:
                    other.append((*where, x))
    for (kind, body), hits in sorted(diverges.items()):
        seeds = sorted({h[0] for h in hits})
        print(f"== {kind}: {len(seeds)} seeds {seeds[:6]} in {' '.join(sorted({h[1] for h in hits}))}")
        print("    " + body)
        print(f"  off: {hits[0][2][:150]}\n  jit: {hits[0][3][:150]}")
    for msg, hits in sorted(internal.items(), key=lambda kv: -len(kv[1])):
        seeds = sorted({h[0] for h in hits})
        print(f"== internal: {len(seeds)} seeds {seeds[:6]} in {' '.join(sorted({h[1] for h in hits}))}\n  {msg[:250]}")
    for seed, where, x in other:
        detail = {k: v for k, v in x.items() if k not in ("what", "bin", "mode")}
        print(f"== {x['what']}: seed {seed} in {where}\n  {detail}")


if __name__ == "__main__":
    main()
