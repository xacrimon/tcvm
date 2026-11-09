# LJR benchmark suite

The 43 programs from luajit-remake's `luabench/` (commit a1275ae) used by
`tools/ljr.py`; `life.lua` is left out because its source is Latin-1. Added
here: `particles.lua`, a game-script-style particle simulation (method
dispatch, per-call allocation, closures, string keys, library calls).

The programs are edited for Lua 5.5 subtypes (integer work in integers, float
work in floats) while staying LuaJIT-compatible. Where 5.5 syntax (`//`, bit
operators) is cleaner, `NAME-5.5.lua` is the same program rewritten with it;
`tools/ljr.py` and the tools built on it run that file under tcvm and `lua`,
and `NAME.lua` under LuaJIT.

Run them with `tools/ljr.py` (this directory is its default suite). Sizes (luajit-remake's
`run_bench.sh`; particles' own), as in `ARGS` in `tools/ljr.py`:

| program | arguments |
|---|---|
| array3d | `300 packed` |
| binary-trees-name | `15` |
| binary-trees-num | `16` |
| bounce | `3000` |
| chameneos | `10000000` |
| coroutine-ring | `20000000` |
| fannkuch | `11` |
| fasta | `5000000` |
| fixpoint-fact | `1000` |
| heapsort | `1 3000000` |
| k-nucleotide | `5000000` |
| linear-sieve | `30000000` |
| mandel-metatable | `256` |
| mandelbrot | `3000` |
| nbody | `5000000` |
| nsieve | `12` |
| partialsums | `30000000` |
| particles | `10000` |
| pidigits-nogmp | `5000` |
| qt | `14` |
| quadtree-2 | `14` |
| queen | `12` |
| ray | `9` |
| ray-prop | `9` |
| recursive-fib-gv | `40` |
| recursive-fib-uv | `40` |
| revcomp | `5000000` |
| scimark-fft | `10` |
| scimark-lu | `5` |
| scimark-sor | `5` |
| scimark-sparse | `300` |
| series | `5000` |
| spectral-norm | `2000` |
| table-sort | `5000000` |
| table-sort-cmp | `1000000` |

`k-nucleotide` and `revcomp` read the output of `fasta.lua 5000000` on
stdin. Programs without an entry run with no arguments.
