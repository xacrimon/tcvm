# tcvm-fuzz

A differential fuzzer for tcvm. It generates Lua programs and runs each one under

- reference Lua 5.5 (`lua` on `PATH`),
- tcvm with the JIT off,
- tcvm in several JIT configurations drawn from the `TCVM_*` knobs,

and reports where they disagree, crash, hang, or where a JIT compile fails on a compiler error.

## Setup

The fuzzer needs a `tcvm-dev` build of `tcvm-cli`; it refuses a binary without one.

```sh
cargo build --release --workspace --features tcvm/tcvm-dev
```

Rebuild after changing tcvm: the fuzzer runs whatever binary is at `target/release/tcvm-cli`.

Every command below is `cargo run --release -p tcvm-fuzz -- <command>`, or
`target/release/tcvm-fuzz <command>` once built. `--help` on any command lists its options.

## Fuzzing

```sh
tcvm-fuzz run            # until Ctrl-C, on all cores but one
tcvm-fuzz run -j 10      # on 10 cores
tcvm-fuzz run --hours 8  # stop by itself
```

A run goes on where the last run into the same output directory stopped, so stopping and
starting again loses nothing. `--from N` starts at a seed of your choice, `--count N` stops
after N seeds.

It prints a status line every 30 seconds:

```
0.5h: 54000 seeds (30.1/s), 8100 with findings, 212 causes, at seed 54012
```

### What a run leaves behind

In `target/fuzz/` (`--out DIR` for another place):

| File | What it holds |
|---|---|
| `summary.tsv` | One line per distinct cause: its kind, how many seeds showed it, the seeds whose programs are kept, the configurations it showed in, and its signature. Rewritten every 30 seconds. |
| `findings.tsv` | One line per finding of a kept program: seed, kind, binary, configuration, unit, the unit's kind, and the differing lines or the error. |
| `prog/p<seed>.lua` | The kept programs: the first 5 seeds of each cause (`--keep N`). |
| `state` | The next seed. |
| `log.txt` | A line per session (start seed, tcvm commit, binaries) and every status line. |

`tcvm-fuzz triage` prints `summary.tsv` readably, the most seen causes first.

To start over, delete the output directory. Do that after fixing bugs too: the counts in
`summary.tsv` are of the tcvm that ran.

### Kinds of finding

| Kind | Meaning |
|---|---|
| `diverges` | A JIT configuration printed something else than JIT-off, or exited differently. |
| `crash` | A JIT configuration died of a signal. |
| `hang` | A JIT configuration ran out of time where JIT-off did not. |
| `internal` | A compile failed on a verifier, checker or backend error, or a panic. The region falls back to the interpreter, so the output is right. |
| `ref` | JIT-off printed something else than reference Lua. |
| `interp` | The JIT-off run itself failed: a panic, a crash, a timeout. |
| `reference` | Reference Lua failed on the program. This is a bug in the generator, not in tcvm. |

A timeout is retried once with three times the limit before it counts. Timeouts under
`TCVM_GC_STRESS` are not findings.

### Configurations

Every program runs with the default thresholds (`default`), with everything compiled at its
first count (`hot`), and in three configurations drawn at random. A configuration's name lists
what it sets: `hc4,hl20,noinline,deopt,check,far,gc64` is `TCVM_JIT_HOT_CALL=4`,
`TCVM_JIT_HOT_LOOP=20`, `TCVM_JIT_INLINE=0`, `TCVM_JIT_DEOPT_ALL=1`, `TCVM_JIT_CHECK=1`,
`TCVM_JIT_FAR=1`, `TCVM_GC_STRESS=64`. Some programs also run as `off,gcN`: JIT off under GC
stress.

All runs of a program share one `TCVM_HASH_SEED` (the seed modulo 1000, plus 1), so a run by
hand needs the same one to repeat.

### Several binaries

```sh
RUSTFLAGS="-C debug-assertions" cargo build --release --workspace \
    --features tcvm/tcvm-dev --target-dir target/dbg
tcvm-fuzz run --bin rel=target/release/tcvm-cli --bin dbg=target/dbg/release/tcvm-cli
```

Each binary is compared with itself only. A build with debug assertions checks the
interpreter's and the collector's invariants as it runs, which a release build compiles out;
it is slower, so a run with both does about half as many seeds. With several binaries the
configurations in `summary.tsv` carry the binary's name (`dbg:hot`).

## Looking at a finding

Take a seed and its configuration from `triage` or `findings.tsv`.

```sh
tcvm-fuzz gen 38 > p38.lua                          # the program again, if it was not kept
lua p38.lua > want.txt
TCVM_JIT=off TCVM_HASH_SEED=39 target/release/tcvm-cli -f p38.lua > got.txt
diff want.txt got.txt
```

Output lines start with the unit they belong to (`U1`, `U2`, …):

- `U1w H key=count …`: the warm-up, every distinct result with how many calls gave it.
- `U1p3 H …`: two calls with other values.
- `U1q4 …`: one call.
- `U1z H …`: the warm-up's values again.

A result reads `#2,i5,f0x1.8p+1` (two values, the integer 5 and the float 3.0) or
`E<L12: message>` (an error raised at line 12), followed by `|value` for each `probe` the call
ran.

### Shrinking

```sh
tcvm-fuzz reduce 38 --what ref
tcvm-fuzz reduce 159 --what diverges --mode hc4,hl20,check
tcvm-fuzz reduce 67 --what internal --mode hot --grep "used before any definition"
```

This writes `prog/p<seed>.min.lua` in the output directory. It removes whole units, then
statements, then replaces expressions by their operands, and at last removes lines of the
text. `--grep TEXT` keeps it on one finding where a program shows several: the differing
line, or the JIT log's error, must contain the text. Give `reduce` a file instead of a seed
to shrink by lines only.

The reduced program still carries most of the prelude. It is small enough to finish by hand.

## Differences from reference Lua that are accepted

tcvm's error messages differ from Lua's in ways that are known and not fixed yet. They are
taken out of both outputs before the comparison:

| Rule | Lua | tcvm |
|---|---|---|
| `variable` | `attempt to index a nil value (field 'x')` | no variable named |
| `no-value` | `got no value` | `got nil` |
| `qualified` | `bad argument #1 to 'string.rep'` | `'rep'` |
| `callee` | names a native by its call site (`'a'`, `'integer index'`) | by its own name |
| `arg-number` | numbers a method call's arguments without the receiver | with it |

`--enforce variable,callee` reports the named ones as findings after all. Once tcvm matches
Lua on one, remove its rule from `Accepted` in `src/oracle.rs`.

The differences `CLAUDE.md` lists as deliberate are avoided by the generator instead: a
*strict* unit, the only kind compared with reference Lua, writes no float as text (tcvm
writes `%.14g`), takes no length of a table with holes, and never yields outside a coroutine.
A unit's header line says whether it is strict: `do -- U1 free risk=0 roles=[Num, Str, Tab] strict`.

## What the programs reach

```sh
tcvm-fuzz cover 0 600
```

Runs the seeds with `TCVM_STATS` and prints the opcodes the interpreter never dispatched and
the JIT's events (regions compiled by entry kind, failed compiles by class, exits by tag,
passes that changed something). Use it after changing the generator, to see that a new form
reaches what it was written for.

## Changing the generator

`src/generate/mod.rs` states what a generated program must hold to: it ends, its strings stay
bounded, and what it prints depends on nothing but its text. A new statement or expression
form has to keep all of that, or runs start timing out and reference Lua starts disagreeing
for reasons that are no bugs.

- `expr.rs`, `stmt.rs`: the typed forms of free units. Add a statement form to `FORMS` in
  `mod.rs` and to the `match` in `Gen::stmt`.
- `templates.rs`: the hand-written shapes that each aim at one thing the JIT speculates on.
- `driver.rs`: how a unit's function is called, and with which values.
- `src/prelude.lua`: the value pool and the output functions, at the head of every program.

Programs are a function of their seed, so a change to the generator changes what every seed
produces: kept programs stay valid as files, but `gen` and `reduce` on an old seed give the
new program.

After a change, check that reference Lua accepts what comes out before fuzzing with it:

```sh
tcvm-fuzz run --out /tmp/check --count 2000
tcvm-fuzz triage /tmp/check | grep -A1 "^== reference"
```

No `reference` lines means every program ran to its end under `lua`.
