#!/bin/sh
# gate.sh NEW [OLD] [OUT]: the per-commit cost gate (jit-perf-fixes2.md G3).
# Runs the operation table (opcost.py) for NEW, and OLD when given, then the
# corpus (tools/corpus) and the LJR regressions and Are We Fast Yet programs
# with NEW, NEW's interpreter and OLD interleaved, two rounds each. Results
# go to OUT (default: a fresh temporary directory). Copy the binaries out of
# target/ first: a rebuild during the run would replace them.
#
# The program runs use tools/ljr.py; LJR is expected at
# ~/repos/luajit-remake/luabench (LJR=dir overrides).
set -e
NEW=$1; OLD=$2; OUT=${3:-$(mktemp -d -t gate)}
HERE=$(cd "$(dirname "$0")" && pwd)
LJR=${LJR:-$HOME/repos/luajit-remake/luabench}
ROUNDS=${ROUNDS:-2}
mkdir -p "$OUT"
echo "gate: results in $OUT" >&2

PROGRAMS=bounce,cd,deltablue,havlak,json,list,nbody,permute,queen,richard,storage,towers
PROGRAMS=$PROGRAMS,chameneos,series,coroutine-ring,mandel-metatable,revcomp,fixpoint-fact
PROGRAMS=$PROGRAMS,recursive-fib-uv,binary-trees-num

if [ -n "$OLD" ]; then
  python3 "$HERE/opcost.py" --bin "$OLD" --rounds "$ROUNDS" --json "$OUT/opcost-old.json" \
    > "$OUT/opcost-old.txt" 2>/dev/null || true
  CMP="--compare $OUT/opcost-old.json"
  MODES=jit,interp,old
  TCVM="--tcvm old=$OLD"
else
  CMP=""
  MODES=jit,interp
  TCVM=""
fi
# shellcheck disable=SC2086
python3 "$HERE/opcost.py" --bin "$NEW" --rounds "$ROUNDS" --json "$OUT/opcost.json" $CMP \
  > "$OUT/opcost.txt" 2>/dev/null || true
cat "$OUT/opcost.txt"
# shellcheck disable=SC2086
python3 "$HERE/ljr.py" --suite "$HERE/corpus" --bin "$NEW" $TCVM --modes $MODES \
  --rounds "$ROUNDS" --json "$OUT/corpus.json" > "$OUT/corpus.txt" 2>/dev/null
cat "$OUT/corpus.txt"
# shellcheck disable=SC2086
python3 "$HERE/ljr.py" --suite "$LJR" --bin "$NEW" $TCVM --modes $MODES \
  --rounds "$ROUNDS" --only "$PROGRAMS" --json "$OUT/ljr.json" > "$OUT/ljr.txt" 2>/dev/null
cat "$OUT/ljr.txt"
