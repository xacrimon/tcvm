# LJR benchmark suite

The 43 programs from luajit-remake's `luabench/` (commit a1275ae) used by
`tools/ljr.py`; `life.lua` is left out because its source is Latin-1.

Run them with `tools/ljr.py --suite bench/ljr`. Sizes (luajit-remake's
`run_bench.sh`), as in `ARGS` in `tools/ljr.py`:

| program | arguments |
|---|---|
| array3d | `300 packed` |
| binary-trees-name | `15` |
| binary-trees-num | `16` |
| bounce | `3000` |
| chameneos | `1e7` |
| coroutine-ring | `2e7` |
| fannkuch | `11` |
| fasta | `5e6` |
| fixpoint-fact | `1000` |
| heapsort | `1 3000000` |
| k-nucleotide | `5e6` |
| linear-sieve | `3e7` |
| mandel-metatable | `256` |
| mandelbrot | `3000` |
| nbody | `5e6` |
| nsieve | `12` |
| partialsums | `3e7` |
| pidigits-nogmp | `5000` |
| qt | `14` |
| quadtree-2 | `14` |
| queen | `12` |
| ray | `9` |
| ray-prop | `9` |
| recursive-fib-gv | `40` |
| recursive-fib-uv | `40` |
| revcomp | `5e6` |
| scimark-fft | `10` |
| scimark-lu | `5` |
| scimark-sor | `5` |
| scimark-sparse | `300` |
| series | `5000` |
| spectral-norm | `2000` |
| table-sort | `5e6` |
| table-sort-cmp | `1e6` |

`k-nucleotide` and `revcomp` read the output of `fasta.lua 5000000` on
stdin. Programs without an entry run with no arguments.
