#!/bin/sh
# perop.sh FILE N1 N2 [BIN]: instructions and cycles per unit of the
# benchmark's size argument, from runs at N1 and N2 (the difference cancels
# startup and compile time). Env vars (TCVM_JIT=off, ...) pass through.
FILE=$1; N1=$2; N2=$3; BIN=${4:-$(dirname "$0")/../target/release/tcvm-cli}
run() {
  /usr/bin/time -l "$BIN" -f "$FILE" "$1" 2>&1 >/dev/null |
    awk '/instructions retired/ {i=$1} /cycles elapsed/ {c=$1} END {print i, c}'
}
set -- $(run "$N1") $(run "$N2")
awk -v i1="$1" -v c1="$2" -v i2="$3" -v c2="$4" -v n1="$N1" -v n2="$N2" 'BEGIN {
  d = n2 - n1
  printf "%.1f instructions, %.1f cycles per unit\n", (i2 - i1) / d, (c2 - c1) / d
}'
