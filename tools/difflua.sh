#!/bin/sh
# difflua.sh FILE [ARGS...]: run FILE under lua5.5 and under tcvm-cli with
# every region compiled at its first count, and diff their stdout.
BIN=${TCVM_BIN:-$(dirname "$0")/../target/release/tcvm-cli}
F=$1; shift
A=$(mktemp); B=$(mktemp)
lua5.5 "$F" "$@" > "$A" 2>&1
TCVM_JIT_HOT_CALL=1 TCVM_JIT_HOT_LOOP=1 "$BIN" -f "$F" "$@" > "$B" 2>&1
if diff "$A" "$B"; then echo "same"; fi
rm -f "$A" "$B"
