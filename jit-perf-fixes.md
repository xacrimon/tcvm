# JIT performance fixes

Why LuaJIT's compiled code beats ours across the LJR suite, what in our code
generation causes it, and what changes make our machine code match or beat
LuaJIT's. Measured 2026-10-11 at eacebf4 (branch `jit2`) on the M4 Pro against
LuaJIT 2.1.1788856981 and clang 21 (`-O2`), all 44 LJR benchmarks, three
interleaved rounds, median cycles. Section references are to `jit-design.md`.

Contents

1. Where we stand
2. Three kernels, instruction by instruction
3. How LuaJIT, V8 and JSC avoid each cost
4. Problems and fixes (P1 to P9)
5. Order and expected results
6. Appendix: side-by-side code

## 1. Where we stand

Over the 38 benchmarks that run, LuaJIT's compiler is 1.44× faster than ours
(geomean of LuaJIT cycles / tcvm cycles 0.696) and ahead on 26 of 38. The
16-benchmark set that section 17 was written from (geomean 0.90) hid the worst
group: numeric loops over arrays, where we are 4 to 10× behind.

| Group | Benchmark | tcvm JIT | tcvm interp | LuaJIT | LJ / tcvm |
|---|---|---|---|---|---|
| numeric loops, arrays | spectral-norm | 5.050 | 7.213 | 0.479 | 0.095 |
| | array3d | 3.208 | 4.445 | 0.323 | 0.101 |
| | permute | 2.923 | 6.448 | 0.535 | 0.183 |
| | scimark-lu | 0.952 | 2.010 | 0.236 | 0.248 |
| | partialsums | 5.431 | 11.285 | 1.386 | 0.255 |
| | linear-sieve | 2.371 | 3.251 | 0.658 | 0.277 |
| | towers | 1.093 | 3.203 | 0.380 | 0.348 |
| | series | 0.625 | 0.558 | 0.302 | 0.483 |
| | scimark-sor | 0.712 | 1.738 | 0.369 | 0.518 |
| | nsieve | 5.291 | 11.045 | 2.914 | 0.551 |
| | queen | 0.664 | 1.795 | 0.402 | 0.605 |
| | scimark-fft | 1.063 | 2.186 | 0.704 | 0.662 |
| | bounce | 0.993 | 1.565 | 0.658 | 0.663 |
| | list | 1.324 | 3.482 | 0.906 | 0.684 |
| | scimark-sparse | 1.237 | 2.033 | 0.908 | 0.734 |
| | nbody | 1.108 | 10.420 | 1.050 | 0.948 |
| | heapsort | 3.032 | 5.082 | 2.949 | 0.973 |
| calls, methods | ray-prop | 7.824 | 12.096 | 2.084 | 0.266 |
| | ray | 8.472 | 13.189 | 2.935 | 0.346 |
| | recursive-fib-uv | 3.047 | 5.321 | 1.283 | 0.421 |
| | mandel-metatable | 0.485 | 0.899 | 0.239 | 0.493 |
| | recursive-fib-gv | 2.161 | 3.950 | 1.204 | 0.557 |
| | table-sort-cmp | 3.094 | 3.470 | 3.113 | 1.006 |
| | json | 2.412 | 4.347 | 2.609 | 1.082 |
| | qt | 4.824 | 9.608 | 5.295 | 1.098 |
| | quadtree-2 | 4.483 | 9.776 | 6.634 | 1.480 |
| | deltablue | 1.274 | 1.536 | 2.642 | 2.073 |
| | richard | 1.179 | 3.615 | 2.842 | 2.411 |
| coroutines, natives | k-nucleotide | 21.971 | 21.704 | 5.834 | 0.266 |
| | chameneos | 5.366 | 4.685 | 3.060 | 0.570 |
| | table-sort | 12.135 | 12.533 | 7.320 | 0.603 |
| | coroutine-ring | 2.011 | 1.707 | 1.327 | 0.660 |
| allocation | binary-trees-num | 2.221 | 3.061 | 3.232 | 1.455 |
| | storage | 1.338 | 1.658 | 2.752 | 2.056 |
| | binary-trees-name | 0.816 | 1.325 | 2.323 | 2.847 |
| | havlak | 2.267 | 2.926 | 6.547 | 2.888 |
| | fixpoint-fact | 0.404 | 0.574 | 1.909 | 4.721 |
| | cd | 2.633 | 3.973 | 22.869 | 8.685 |

(Gcycles. cd is a LuaJIT pathology, 245,000 traces; its interpreter runs it in
7.96. richard's LuaJIT time varies by run: 1.04 in the section 17 measurement,
2.84 here. series and chameneos run slower compiled than interpreted; so does
coroutine-ring.)

Six benchmarks do not run: fannkuch, fasta, mandelbrot, pidigits-nogmp and
revcomp use the Lua 5.1 globals `unpack` and `loadstring`, life.lua is not UTF-8.
These are harness compatibility, not JIT work (P9), but mandelbrot and fannkuch
are exactly the float-loop and array-shuffle kernels the JIT is for.

Per-operation costs, from instruction and cycle deltas between two problem
sizes (deterministic for instructions):

| Kernel | Unit | tcvm instr / cycles | LuaJIT instr / cycles | C, clang -O2 |
|---|---|---|---|---|
| recursive-fib-uv | one call | 78.9 / 9.5 | 36.0 / 3.8 | 16.5 / 3.5 |
| spectral-norm | inner iteration, `a = a + x[j] * A(i, j)` | 222 / 31.5 | 23.6 / 3.0 | 10 instr |
| queen | inner iteration of `isplaceok` (hot-path instructions; cycles are the whole run over its 45.4M iterations) | 65 / 14.6 | 17 / 8.9 | 13 instr |
| nbody | one body pair | 170 / 22.1 | 185 / 20.8 | 26 instr |
| array3d | one `array_set` | 838 / 102 | 59 / 12.7 | |
| linear-sieve | one outer iteration | 315 / 74 | 127 / 24 | |
| partialsums | one iteration | 1260 / 181 | 337 / 44.7 | |
| coroutine-ring | one resume/yield round trip | 692 / 97.6 | 469 / 68.6 | |

nbody is the one kernel where the loop is fully typed and has no calls, and
there we are at parity. Everything else is a multiple. The next section says
where the instructions go.

## 2. Three kernels, instruction by instruction

### 2.1 spectral-norm: `for j = 1, N do a = a + x[j] * A(i, j) end`

LuaJIT's loop is 24 instructions: `x[j]` is a typed array load (`ldr`, tag
compare, branch to exit), `A` is inlined (its closure guarded once before the
loop), `a` is a float PHI in `d15`, the loop counter in `w28`, and `ABC` is
hoisted. Our region for the same loop is 3.5 KB of code (894 instructions, 69
basic blocks) and executes 222 per iteration. Its IR (after the passes) is:

```
b3(v11: val:tab, v12 .. v17: val:any, v18: val:small|big|float, v19 .. v22: i32, ...):
  gccheck                                   ; every iteration
  v386: val:small = box v22                 ; j boxed, for the helper arms
  v27: i32 = tabasize v24 ; br (j < asize) -> b14 -> b15
b15: v40 = call.index v11, v386 ; guard.false (failed)     ; out of bounds: helper
b14: v30 = arrload v29, v22 ; br (nil) -> b16 -> b17
b16: v32 = tabmmbits v24 ; ... br (INDEX bit) -> b18 -> b19
b18: v37 = call.index v11, v386 ; guard.false (failed)     ; nil: helper
b11(v59: val:any, v432: val:any):
  store r0 v11 ... store r7 v18             ; 8 invariant registers stored
  store r8 (box v19) ... store r11 v386     ; 4 loop registers boxed and stored
  store r12 v59 ; store r17 v17 ; store r18 v386 ; store r13 v432   ; 17 stores
  call r13 nargs=2 c=2 pc14 -> b20()
b20() resume:
  v60 = resume c=2
  guard val:float v60 ; load r12 ; guard val:float ; load r7
  load r8 ; guard small ; load r9 ; guard small ; load r10 ; guard small
  load r0 .. load r6                        ; 11 loads, 5 guards
  v66: f64 = fmul v63, v65 ; v68 = box v66
  br (istype small v67) -> b22 -> b23       ; `a + v`: three arms
b23: br (istype small|float v67) -> b24 -> b25
b25: v93 = call.arith.add v67, v68 ; guard.false   ; helper arm
b24: ... select ; fadd ; box                         ; float arm
b22: unbox ; itol ; ladd ; box                       ; integer arm
b21(v102: val:small|big|float, ...)
```

Where the 222 instructions go, per iteration:

| Cost | Instructions | Why |
|---|---|---|
| call protocol in the caller: 17 home-slot stores with boxing, 4 header stores, callee kind test, `blr` | about 45 | R1 and R6 (5.3): nothing lives across a call |
| callee `A`: prologue, entry guards, 3 `subs`/`b.vs`, `smull`, 2 `scvtf`, `fmul`, `fadd`, `fdiv`, box, result store, nil default, epilogue, `ret` | about 40 | a real call; LuaJIT inlines it (13, I1) |
| resume: `base` and `closure` reloads, 11 slot reloads, 5 type guards, 3 unboxes | about 35 | every value is reloaded and re-guarded after a call (5.4) |
| `a + v`: two `istype` tests, a `select`, a branch, the float arm's own `istype`, unbox, `scvtf`, `select`, `fadd`, box | about 18 | `a` is `small|big|float`: the add site's feedback saw the integer `0` of iteration 1 (P2) |
| `x[j]`: box `j`, `tabasize`, bounds branch, `arrload`, nil test, join | about 10 | two helper arms, 7.6 (P1) |
| `gccheck`, counter, spills and reloads of invariants around the helper `bl`s, branch-to-branch | about 20 | the arms' `bl` clobbers caller-saved registers (P1c), `prune_gc_checks` keeps the check for the arms (P5) |
| LuaJIT's loop | 24 | |

### 2.2 queen: `if (a[i] == c) or (a[i] - i == c - n) or (a[i] + i == c + n)`

Our inner loop, hot path only (the taken arms excluded), 65 instructions
against LuaJIT's 17 and clang's 13. The three `a[i]` reads each cost:

```
   ac: ldr  x16, [x23, #0x1030]      ; gccheck: 4 instructions, a dependent load chain
   b0: ldp  x16, x17, [x16]
   b4: cmp  x16, x17
   b8: b.hs exit
   bc: orr  x2, x21, #0xffffffff00000000   ; box i for the helper arms
   c0: str  x2, [x29, #0x30]          ; and spill it
   c4: ldr  w0, [x28, #0x28]          ; asize
   c8: cmp  w21, w0
   cc: b.hs 0x130                     ; -> call.index arm
   d0: ldr  x0, [x28, #0x10]          ; array
   d4: ldr  x0, [x0, w21, uxtw #3]    ; a[i]
   d8: mov  x16, #-0x200000000        ; NIL, materialized per test
   dc: cmp  x0, x16
   e0: b.ne 0x12c
   ...                                ; nil arm: class bits, helper
  148: lsr  x16, x0, #32              ; guard small on the loaded value
  14c: cmn  w16, #0x1
  150: b.ne exit
  154: ldr  x1, [x29, #0x20]          ; c, reloaded from a spill slot
  158: cmp  w0, w1
  15c: b.eq 0x2d8
  160: str  x1, [x29, #0x20]          ; and spilled again
```

LuaJIT's loop body:

```
->LOOP:
  ldr   x27, [x4, w28, uxtw #3]      ; a[i]: array pointer hoisted, ABC hoisted
  cmp   x5, x27, lsr #32             ; int tag
  bne   ->7                          ; anything else exits
  mov   w27, w27
  cmp   w27, w3 ; beq ->8            ; == c
  subs  w26, w27, w28 ; bvs ->9      ; a[i] - i
  cmp   w26, w2 ; beq ->10           ; == c - n (hoisted)
  adds  w27, w27, w28 ; bvs ->11
  cmp   w27, w1 ; beq ->12
  add   w28, w28, #1
  cmp   w28, w0
  ble   ->LOOP
```

Everything LuaJIT keeps in a register we reload from a spill slot (`c`, `n`,
the constant 1, the limit), because the `bl jit_index` on the two arms
clobbers every caller-saved register and the allocator spills around it; the
three loads of `a[i]` are not shared because `call.index` is declared to write
every table; `c - n` and `c + n` are recomputed with overflow checks because
their blocks do not dominate the loop's exits (LICM's spine rule, 9.5).

### 2.3 recursive-fib-uv: one call

| | tcvm (J10 fast call) | LuaJIT (up-recursion trace 4) | C |
|---|---|---|---|
| instructions per call | 78.9 | 36.0 | 16.5 |
| cycles per call | 9.5 | 3.8 | 3.5 |

Our call site is 41 instructions before the `blr`: the argument boxed and
stored, the callee value stored, three header words (`adr`, `str`, `str base`,
`ldr` literal, `str`), then the dynamic callee test (tag, vararg byte,
prototype, `fast_entry`), the stack-depth test, `blr`. The callee's entry does
the window and arity checks again, opens its frame and re-guards `n`. The
return stores the result to a home slot, pops, and the resume reloads `base`
and `closure`, defaults the result to nil (5 instructions), then re-guards it.
LuaJIT's trace knows the callee (one `EQ` guard per level), passes `n` in a
register, unrolls two levels, and returns through its frame link.

## 3. How LuaJIT, V8 and JSC avoid each cost

The same problems exist in every dynamic-language JIT. The three engines
settle them the same way; we settle them differently in four places.

| Concern | LuaJIT | V8 Maglev / Turbofan | JSC DFG | tcvm today |
|---|---|---|---|---|
| a value of an unexpected type on a hot path | a guard that exits the trace: every `>` instruction in the IR (`lj_asm_arm64.h:86` `asm_guardcc`); no type dispatch in a trace | `CheckedSmiUntag`, `CheckMaps`: `EmitEagerDeoptIf` (`maglev-ir.cc:1896`); generic nodes (`GenericAdd`) only when feedback is `kAny` (`maglev-graph-builder.cc:1599`) | `speculationCheck(Overflow, ...)` under `Int32Use` (`DFGSpeculativeJIT.cpp:4854`); `UntypedUse` only when the `BinaryArithProfile` saw non-numbers | the quickened forms give a guard, every other site gets 2–3 inline arms plus a helper call that stays on the hot path (`emit_generic`, `generic_cmp`, `emit_gettable`) |
| where a type is decided | at the definition: `SLOAD` carries the type seen when recording, `ALOAD`/`HLOAD` guard their result type once | at the definition from feedback, and at phis: `MaglevPhiRepresentationSelector` untags a phi whose inputs and uses agree (`maglev-phi-representation-selector.h:50`) | `FixupPhase` chooses a `UseKind` per edge from profiles (`DFGFixupPhase.cpp:629` `attemptToMakeIntegerAdd`, else `DoubleRepUse`) | at the use, from the site's form and byte (7.1, 9.1); block parameters that join `small` and `float` stay `Val` and their uses stay generic (9.2's noted gap) |
| a rare slow path that needs a runtime call | the trace exits; the interpreter runs the case | deferred code after the function body with a `RegisterSnapshot` of what is live (`maglev-assembler-inl.h:257` `JumpToDeferredIf`, `AllocateSlow`) | `SlowPathGenerator` emitted out of line; `silentSpillAllRegisters` saves only around that call (`DFGSpeculativeJIT.h:307,394`) | the helper `bl` is in the body; `regalloc2` sees its clobbers on the hot path and spills (10.4's noted gap) |
| integer vs float | all numbers are doubles; `NARROW` picks integers only where it pays and strips overflow checks (`lj_opt_narrow.c`) | `Int32AddWithOverflow` with eager deopt; on overflow the feedback becomes `kNumber` and the site becomes `Float64Add` | `Int32Use` with `Overflow` exit, then `DoubleRepUse` | correct for Lua 5.5 (`I32` with exits, `I64` after overflow), but an `I64` value makes every float consumer and intrinsic generic (P2) |
| values across a call | inlined, or the trace exits | spill slots of the frame, typed; a deopt reads them through the frame state | spill slots; `silentSpill` | every live value is stored to its home slot, then reloaded and re-guarded (5.3 R1/R6, 5.4) |
| loop-invariant guards | copy-substitution unrolling: the first iteration is the pre-roll, every guard the body repeats is CSE'd (`lj_opt_loop.c:24-80`) | loop peeling and check hoisting | | peeling up to `PEEL_LIMIT = 48` instructions (`opt/peel.rs:10`), LICM only from blocks that dominate every loop exit |
| the collector | one `asm_gc_check` per trace at its head or `LOOP` (`lj_asm_arm64.h:1845`) | allocation sites poll; no per-loop check | | one per loop header whenever any arm of the loop may allocate (`prune_gc_checks`, `opt/mod.rs:81`) |
| a builtin without compiled code | `coroutine.resume`/`yield`: trace stitching, the C function runs and a new trace starts after it (`lj_ffrecord.c:110`) | `CallKnownJSFunction`, builtins called by their ABI | | the generic `enter` → `native_enter` → `drive` path; the interpreter's own fast entries are never reached from a region (5.4) |

Four differences carry the gap: we dispatch on type inline instead of
speculating and exiting; types live at sites instead of on values; slow paths
share the hot path's register allocation; and nothing survives a call. The
fixes below are organized around those four, then the optimizations that
only make sense once they hold (inlining, sinking, range analysis).

## 4. Problems and fixes

Each item: the evidence, the cause in our code, the change, and the code it
yields. Benchmarks in brackets are the ones it moves most.

### P1. Generic arms on the hot path

**Evidence.** The compiled suite has helper arms nearly everywhere: the
linear-sieve loop region has 320 blocks and 69 helper calls, scimark-sor's
223 blocks and 48, heapsort's 496 and 92; queen's `a[i]` costs three arms per
read (2.2); spectral's `a + v` has three (2.1). The arms rarely run, but they
cost the hot path in four ways:

1. the type dispatch itself: `istype`, `select`, branch, join (6–10
   instructions per site);
2. their `bl` clobbers x0–x17 and d0–d7, d16–d31, so every hot value live
   across the arm is spilled: 35 spill loads in queen's region, 291 in
   spectral's, 1873 in ray's;
3. their declared effects (`call.index` is `MAY_ALLOC | W_SHAPE | W_META`)
   end GVN's and LICM's reuse, so queen loads `a[i]` three times and nothing
   hoists;
4. `MAY_ALLOC` on an arm keeps a `gccheck` at the loop header (4
   instructions and a dependent load chain per iteration, in every loop of
   the suite that reads an array).

**Cause.** `emit_generic` (`build/emit.rs:829`) emits the typed path only
when both operands' kinds are known and otherwise the full three-arm
dispatch; `generic_cmp` and `generic_eq` the same for compares;
`emit_gettable` (`build/tables.rs:634`) gives every array read a nil arm and
an out-of-bounds arm that call `jit_index`; `helper_arms` puts the metamethod
call inline once the byte has `MM`. The design chose arms over exits so that
"everything compiles" (goal 1) and no site deopts repeatedly; LuaJIT, V8 and
JSC get the same guarantee from feedback plus recompilation, with the generic
code out of line.

**Fix.** Three rules in the builder and one in the backend:

- *Speculate or exit.* A site compiles the one arm its feedback supports and
  exits on anything else, with a widenable tag so a hot exit recompiles it
  (6.4 already does this for the forms). The feedback byte's only job is to
  pick the guards that establish each operand's kind (`I32`, `I64`, `F64`,
  `STR`, `TAB`, ...); the operation is then a pure function of the two
  kinds, chosen from the tables below, and never dispatches at run time.
  Nothing calls `jit_arith`, `jit_lt`, `jit_le` or `jit_eq` any more and
  they are deleted; the one call that can remain at an arithmetic or compare
  site is the metamethod arm, which is a call to Lua, not a compare. This is
  LuaJIT's recorder, where every operand has an `IRType` and `BC_ISLT`
  emits one typed `LT` (`lj_record.c:2354`), and TurboFan's
  `ReduceJSComparison`, which picks `NumberLessThan` or `StringLessThan`
  from the input types (`js-typed-lowering.cc:907`). Array reads: the bounds
  miss and the nil result are exits
  (LuaJIT's `ABC` and typed `ALOAD`) unless the site's byte saw them (a new
  `NIL`/`OOB` feedback bit the exit records), and the absence of `__index` on
  the table's class is a guard on `TabMmBits` (an `R_META` read GVN shares
  and LICM hoists), not a live test in the nil arm. Array stores likewise.
- *Cold arms when feedback demands them.* A site whose byte did see the case
  (a nil read in a sparse array, a metamethod) keeps the arm, but the arm is
  cold: its `bl` goes to a per-helper trampoline that saves and restores the
  caller-saved registers itself (about 40 instructions, only on the arm), and
  the lowering declares its clobbers as `{x0, x16, x17}` only. The hot path's
  allocation never sees the call, as Maglev's deferred code and the DFG's
  slow-path generators never do. Values only the arm needs (the boxed key)
  are computed in the arm (code sinking at lowering, or the builder emitting
  the box inside the arm).
- *Precise effects.* `jit_index` and the array helpers are
  read-only (they run no Lua; a metamethod makes them return `FAIL`): declare
  them `R_TAB | R_META`, so a later read of the same table and key forwards,
  the shape and class guards survive, and `prune_gc_checks` sees no
  allocation.
- *Lowering.* `Br(IsType)` and `Br(B1)` as `cmp; b.cond` / `tbz` (no `cset`,
  no `select` for `both`), `NIL` and `BOX` shared within a call-free span.

**Arithmetic by operand kind.** Verified on Lua 5.5 and tcvm (identical
output): `9007199254740993 + 0.0` is `9007199254740992.0`, so int to float in
arithmetic is a rounding cast and needs no exactness guard for `I32` or
`I64`; `math.maxinteger + 1` wraps; `7 // 0` and `7 % 0` raise; `3.5 | 0`
and `2^63 | 0` raise "number has no integer representation"; `"10" + 1` is
`11`. The integer representation starts at `I32`: a site compiles `I64`
only when its byte has `BIGINT` or `OVERFLOW` or an operand already has
`I64` kind from its load guard, which is LuaJIT's narrowing with `ADDOV`
guards (`lj_opt_narrow.c:526`) and Maglev's hint ladder (`kSignedSmall` to
`Int32AddWithOverflow`, eager deopt on `vs`, `maglev-ir-arm64.cc:77`;
`kNumber` to `Float64Add`). The `I32` overflow check is a representation
guard, not a semantic one: Lua integers wrap, so `I64` ops carry no check.

| operands | code |
|---|---|
| `I32`, `I32`: `+ - *` | `adds` / `subs` / `smull` + `cmp`, `b.vs exit`; the exit records `OVERFLOW` and the recompile widens that value to `I64` |
| `I64`, `I64`: `+ - *` | wrapping `add` / `sub` / `mul`, no check |
| int, `F64` (either side), any op | `scvtf` the int, then the float op; one instruction for `I32` and `I64` alike (LuaJIT's `CONV num.int`, `lj_record.c:2369`) |
| `/`, `^`, any numeric kinds | always float: `scvtf` the int operands, `fdiv`; `x^2` is `fmul`, small constant exponents unroll, the rest `bl pow` (LuaJIT does the same) |
| `//`, `%`: int, int | `sdiv` and Lua's floor adjustment (about 6 instructions); a zero divisor exits, since it raises; AArch64 `sdiv` gives `mininteger` for `mininteger // -1`, which is Lua's answer |
| `//`, `%` with an `F64` | `scvtf`, `fdiv` + `frintm`; `fmod` with the sign fix |
| bitwise, int kinds | the native op; shifts with Lua's rule for counts of 64 and more |
| bitwise with an `F64` | `FToIExact` (`fcvtzs`, `scvtf`, `fcmp`, `b.ne exit`); the exit is the "no integer representation" error |
| a `STR` operand | exit; the coercion and the error both happen in the interpreter; a byte that is hot with `STR` gets a cold arm |
| `MM` in the byte | the cold arm |

**Compares by operand kind.** Verified the same way: `1 == "1"`, `true == 1`
and `nil == false` are `false`; `1 < "a"` and `true < false` raise;
`9007199254740993 > 9007199254740992.0` is `true` (a rounding cast would
say `false`), `1 < 0/0` and `0/0 == 0/0` are `false`. Only compares need the
exact int to float conversion, and only when the int is `I64`: an `I32` is
always within 2^53. tcvm's `num::lt_int_float` follows Lua's `LTintfloat`
for the same reason, and LuaJIT skips the guard only because its integers
are 32-bit.

| operands | code |
|---|---|
| `I32`, `I32` or `I64`, `I64` | `cmp; b.cond` |
| `F64`, `F64` | `fcmp` and `mi` / `ls` / `gt` / `ge` / `eq` / `ne`; every one is false on unordered except `ne`, which is Lua's result for NaN under every operator, so no unordered fix-ups (LuaJIT's `irop ^= 4`) |
| `I32`, `F64` | `scvtf; fcmp`, no guard |
| `I64`, `F64` | the 2^53 guard below, then `scvtf; fcmp`; the same guard gives `==`, since an int that fits equals the float iff the doubles are equal |
| `STR`, `STR`: `==`, `~=` | `cmp` of the pointers (every string is interned; the JEQS form already does this) |
| `STR`, `STR`: `<`, `<=` | the inline word loop below, a pure `StrCmp` op with no clobbers; tcvm's `Ord` is bytewise like LuaJIT's `lj_str_cmp` (`lj_str.c:19`), not reference Lua's locale `strcoll`, so this changes nothing |
| different kinds, `==` / `~=` (other than int, float) | the constant `false` / `true`; in 5.5 `__eq` only fires for two tables or two userdata. TurboFan folds `ReferenceEqual` when the types are disjoint (`typed-optimization.cc:461`); LuaJIT's `lj_record_objcmp` returns "two different types are never equal" and emits nothing (`lj_record.c:244`) |
| different kinds, booleans or nil under `<` / `<=` | exit; the interpreter raises (LuaJIT aborts the trace there, `lj_record.c:2378`) |
| `TAB`, `TAB` or `UD`, `UD`: `==` | pointer compare under a `TabMmBits` guard that the class has no `__eq`, the same guard the nil arm uses for `__index`; with `__eq` in feedback, the cold arm |
| `TAB` / `UD` ordering | the cold arm when the byte has `MM`; otherwise exit, since it raises |

The `I64` vs `F64` compare, `x1 < d0`, six instructions and nothing
clobbered (`cls` counts the leading bits equal to the sign bit, so a count of
10 or more means the value is within ±2^53 and the cast is exact):

```
cls   x16, x1
cmp   x16, #10
b.lo  exit            ; |i| >= 2^53: the interpreter decides
scvtf d16, x1
fcmp  d16, d0
b.mi  taken
```

String ordering, LuaJIT's `lj_str_cmp` widened to 8-byte words: `a < b`
with `x0`, `x1` the string objects, `LEN` and `DATA` the header offsets. The
loop body is six instructions; the whole sequence about twenty. Like
LuaJIT's, it reads up to 7 bytes past the shorter string, so string storage
needs that much readable padding (LuaJIT's "innocuous access up to end of
string + 3").

```
ldr   w2, [x0, #LEN]
ldr   w3, [x1, #LEN]
cmp   w2, w3
csel  w4, w2, w3, lo      ; n = min(len a, len b)
add   x0, x0, #DATA
add   x1, x1, #DATA
mov   x5, xzr
1:
cmp   x5, x4
b.hs  3f                  ; common prefix equal: the shorter is smaller
ldr   x6, [x0, x5]
ldr   x7, [x1, x5]
add   x5, x5, #8
cmp   x6, x7
b.eq  1b
rev   x6, x6              ; first byte into the high bits
rev   x7, x7
subs  x8, x5, x4          ; bytes read past n, if the word straddles the end
csel  x8, x8, xzr, hi
lsl   x8, x8, #3
lsr   x6, x6, x8          ; drop them
lsr   x7, x7, x8
cmp   x6, x7
b.eq  3f
cset  w0, lo              ; `ls` for <=
b     4f
3:
cmp   w2, w3
cset  w0, lo              ; `ls` for <=
4:
```

Both LuaJIT (`IRCALL_lj_str_cmp`) and TurboFan (`StringLessThan` builtin,
`machine-lowering-reducer-inl.h:3002`) call out for string ordering and
inline only the equality fast paths; the call clobber is exactly the cost
this problem removes, so the loop goes inline.

**Code after**, queen's `a[i] == c` with the array pointer and bounds hoisted
(P5): `ldr x0, [x4, w21, uxtw #3]; lsr x16, x0, #32; cmn w16, #1; b.ne exit;
cmp w0, w3; b.eq out` — 6 instructions, LuaJIT's are 5. The whole iteration:
22 instructions against LuaJIT's 17 before P5 hoists `c - n` and `c + n`, 17
after. [queen, linear-sieve, scimark-lu/sor/fft/sparse, heapsort, nsieve,
array3d, bounce, towers, list; the spill traffic of ray and qt]

### P2. Types live at sites, not on values

**Evidence.** spectral's accumulator `a` is `small|big|float` through the
whole loop because `local a = 0` is an integer and the add site's byte saw
`SMALL` on the first iteration and `FLOAT` after (2.1). partialsums: `k3 =
k2 * k` overflows i32 at k = 1291, the recompile makes the site `I64`, and
every float consumer of `k`, `k2` and `k3` goes generic: `(2/3)^k` becomes
`call.arith.pow`, `1/(k3*sk*sk)` `call.arith.mul`, and `sin(k)`, `cos(k)`,
`sqrt(k)` lose their intrinsics and become calls (36 call sites in the
region; 1260 instructions per iteration against LuaJIT's 337, which keeps
`k` an int with `MULOV` and converts with `CONV num.int`). series recompiles
its loop three times on `Type` exits and ends slower than the interpreter.

**Cause.** 9.1 types a value from its uses' feedback, and a use whose byte is
polymorphic contributes nothing; 9.2 leaves a block parameter that joins
`I32` and `F64` as `Val NUM` ("that is the case block duplication would
address later", J4). Feedback bytes conflate a transient first-iteration
kind with steady-state polymorphism. Overflow widening (7.2) widens every
value that flows from the site, and `F64` ops and the intrinsics accept
`I32` but not `I64` operands (`intrinsics.rs`, `Operand::f64`).

**Fix.**

- *Representation selection on block parameters* (Maglev's phi untagging,
  Turbofan's representation selection): a parameter whose incoming values
  are `I32` and `F64`, or `I32` constants and `F64`, and whose uses are all
  float arithmetic, float compares, stores and snapshots, becomes `F64`, the
  integer edges convert (`scvtf`, exact for i32). A snapshot entry of such a
  parameter records `F64`, so a deopt sees a float where the interpreter
  would have had one. The zero-trip case (the loop never runs, `a` stays
  the integer `0`) is correct because the entry edge's value is only
  converted on the edge into the loop header, and the exit edge from the
  header before any iteration keeps the integer. The same selection narrows
  a parameter that joins `small` on every edge to `I32` (today's `narrow`
  does this, `opt/infer.rs:109`), and one that joins `I64` and `F64` to
  `F64` when the `I64` side is proven within 2^53 by range analysis (P5),
  else stays generic.
- *Peel more.* `PEEL_LIMIT = 48` instructions excludes every loop in section
  2 (spectral's body is about 100 IR instructions, nbody's 150). Peel
  innermost loops up to 400 instructions, and outer loops whose inner loop
  is small; the peeled iteration takes the entry's kinds and the loop proper
  joins only steady-state values, which is what makes the first-iteration
  `0` disappear from the loop's parameter. LuaJIT pays this unconditionally
  (its pre-roll is a whole iteration, `lj_opt_loop.c`).
- *`I64` is a representation, not a disease.* Float ops and intrinsics
  take `I64` operands through `LToF` (`scvtf x`), which is the interpreter's
  own conversion: Lua's int to float in arithmetic is a rounding cast (P1's
  arithmetic table, verified), so there is no exactness condition and an
  intrinsic with an `I64` argument is still the intrinsic. Only a compare
  against an `F64` needs the 2^53 guard from P1. The partialsums case then
  needs no special handling: `k` widening to `I64` costs one `scvtf` per
  float use. Range analysis (P5) narrows `I64` values back to `I32` where
  bounds allow, so a loop counter stays `I32` when only `k3` leaves the
  range.
- *Site feedback with a sense of time.* A `Type` exit that recompiles a site
  records the kind it saw; a site whose byte is `SMALL | FLOAT` but whose
  operand the representation pass proved `F64` compiles float code without
  consulting the byte. Where the byte is the only information, treat
  `SMALL | FLOAT` as `NUM`, not as polymorphic: an int operand costs one
  `scvtf` before the `fadd`, and only `OTHER`/`STR`/`MM` are polymorphic.

**Code after**, spectral's add: `fmadd d15, d14, d13, d15` (one instruction,
after P1 removed the arms and P3/P5 put `a` in a register); partialsums'
`1/(k3*sk*sk)`: `scvtf d, x_k3; fmul; fmul; fdiv` with `sin`/`cos` as
`bl` to libm as LuaJIT's `CALLN`. [spectral-norm, partialsums, series,
scimark-*, bounce, qt]

### P3. Nothing survives a call

**Evidence.** spectral: 17 stores before the call, 11 loads and 5 type
guards after it, every iteration (2.1). array3d: three `assert` calls per
`array_set`, each with `store r9, r10, r5` and the full protocol, and the
`self.nx`/`self.ny`/`self.image` loads redone after each (838 instructions
per call; LuaJIT's trace does the whole `array_set` in 59). fib: 79
instructions per call, 41 at the call site alone (2.3). coroutine-ring's
main loop reloads five registers and re-guards three after every `resume`.

**Cause.** 5.3's R1 and R6 and 5.4: before a call, every live register is
in its home slot as a boxed `Value`; after it, the builder reads every
register again with `Load` and the resume re-guards them, because the frame
was popped (before J10) and because a callee could in principle have changed
the slots (D4, `debug.setlocal`). J10 keeps the native frame across a fast
call but the protocol still assumes nothing survives. The header is written
in full at every call, the callee is tested dynamically, its entry repeats
`enter`'s checks, the result lands through memory with a nil default.

**Fix**, in four steps, each shipping on its own:

1. *Trust the slots across a call.* A resume keeps the types of the
   registers it reloads: no guards, no unboxing of a value the region
   already had unboxed (the `Load` after the resume is typed by the value
   stored before the call, and a `Load` of a slot whose boxed value is also
   needed is one `ldr`). The only writer of a suspended frame's registers is
   `debug.setlocal`; it runs as a native, so the chain was reset and the
   caller resumes through header word 1: `debug.setlocal` on a frame whose
   continuation is a region resume point rewrites word 1 to that resume
   point's *checking* twin (every resume block gets one, emitted in the cold
   section, that reloads and re-guards as today). Nothing else changes.
   Removes the 5 guards and 3 unboxes per iteration in spectral, the 3
   re-guards in coroutine-ring, the `n` re-guard in fib.
2. *Values stay in the frame across fast calls.* The frame is kept (J10), so
   a value live across the call lives in the region's spill slots or in a
   callee-saved register instead of its home slot: `regalloc2` sees a fast
   call as a clobber of the caller-saved set, nothing more. The home slots
   are written only on the way out of compiled code: each fast-call site
   gets a *call-site snapshot* (the exit snapshot encoding of Appendix C:
   register → spill slot or callee-saved register, with its representation),
   and `leave()` (`backend/aarch64/emit.rs:346`), before it resets the stack
   to dispatch's entry, walks the native frame chain it is about to drop
   (x29 links; each frame's return address finds its call site in the
   region's table) and writes every frame's live registers to their home
   slots, boxing as the snapshot says. Callee-saved registers need the
   standard unwind step: each region records which of x19–x28, d8–d15 its
   prologue saves and where, and the walk restores them frame by frame from
   the inside out. R1 then holds exactly when it is needed (whenever the
   interpreter, the collector, a native or the unwinder can see the frame),
   which is V8's and JSC's model: the frame holds the values, a stack map
   says where. The collector runs only between `Lua::enter`s (5.7), always
   after a `leave()`, so nothing new needs rooting. The slow call path
   (`enter` for a callee without a fast entry) is itself a `leave()`.
   Removes the 17 stores and 11 loads per iteration in spectral and the
   store/reload of `a` and the tables in array3d.
3. *Known callees.* A callee known at compile time (a constant upvalue of
   any type, a global through `_ENV`'s IC, an own or `_PROTO` slot, a closure
   the region created, the live frame's value for a register at a loop
   entry, guarded once at the entry as LuaJIT's `rec_call_specialize`
   guards every call (`lj_record.c:796`), or per-site callee feedback, D1)
   with a function region is called with `bl` to a *typed entry* past the
   callee's prologue guards: the identity guard hoisted, the window checked
   once per region entry against the deepest known call, the arity known,
   the arguments' kinds proven by the caller, no stack-depth test for a
   non-recursive callee. Header word 0 is written in its final form; words
   1–3 stay (walkers need them) until step 4.
4. *Results in registers, headers lazily.* One result in `x0` (a `Val`) or
   `d0`, two in `x0`/`x1`; a callee whose every `RETURN` yields exactly one
   value sets no nil default. Header words 1–3 become part of the call-site
   snapshot and are written by the same `leave()` walk (the callee, pc and
   caller base are compile-time constants of the site), so a fast call
   writes word 0 only. The resume computes `base` by subtraction and reloads
   `closure` only before a use.

**Code after**, fib's inner node (steps 1–4 and P5's range analysis on the
`n >= 2` edge, before inlining):

```
fib_typed:                      ; w0 = n (i32), proven small by the caller
  stp  x29, x30, [sp, #-32]!
  stp  x19, x20, [sp, #16]
  mov  w19, w0
  cmp  w19, #2
  b.lt ret_one
  sub  w0, w19, #2              ; no overflow check: n >= 2
  str  x24, [x22, #a*8]         ; header word 0 (the callee)
  bl   fib_typed
  mov  w20, w0
  sub  w0, w19, #1
  str  x24, [x22, #b*8]
  bl   fib_typed
  adds w0, w20, w0
  b.vs exit
  ldp  x19, x20, [sp, #16]
  ldp  x29, x30, [sp], #32
  ret
```

About 20 instructions per call, LuaJIT's 36 (it re-guards `n` and the
closure per level), C's 16.5. With I1's two-level unrolling of the recursion
it drops below LuaJIT's. [recursive-fib-uv/gv, spectral-norm, array3d,
permute, towers, ray, richard, deltablue, json, qt, bounce, list]

### P4. Builtins and coroutines from compiled code

**Evidence.** coroutine-ring: 97.6 cycles per round trip compiled, 85
interpreted, LuaJIT 68.6. chameneos the same shape. A `resume` from a
region takes `fast_call`'s slow path (the callee is a native: `FN_IS_VARARG`
is nonzero), `leave()`, `enter` → `native_enter` (44 straight-line
instructions) → the full builtin through `drive`, `native_act`,
`resume_into`, `prepend_true`; the interpreter's `CALL` reaches `ff_resume`
(`vm/coro.rs:250`), 31 cycles for the switch. `pcall`, `coroutine.wrap`,
`select`, `tostring`, `string.format`, `table.insert` all take the same
generic path from compiled code.

**Cause.** 5.4: "in milestone 2 the fast entries of `ff.rs` are not
reached"; `enter`'s native arm goes straight to `native_enter`. The fast
entries read the CALL word's `a`, `b`, `c` from `insn`, which a region does
not have.

**Fix** (17.3 C3, unchanged in substance): every fast entry gets a core that
takes a written header (callee in word 0, continuation in word 1, arguments
at word 4, `nargs` in `insn`); `enter`'s native arm tails the core when the
native has one (one load, one branch); a region whose callee is a known
native (P3 step 3's sources) without an intrinsic calls the core directly
with `bl`, keeping its frame when the core cannot switch threads
(`select`, `tostring`, `string.*`) and through `leave()` when it can
(`resume`, `yield`, `wrap`, `pcall`, `xpcall`). `resume_switch!` and
`ff_yield` return through header word 1, the region's resume point, so both
sides of a switch stay compiled. Expected: coroutine-ring 97.6 → about 70
cycles per round trip (LuaJIT 68.6), chameneos 107 → about 70.

LuaJIT does no better here by design: its traces stitch around
`coroutine.resume` (`lj_ffrecord.c:110`), the C function runs, a new trace
starts after it. Beating it needs the resumer's native frame to survive the
switch, which J10 left open (one native stack per coroutine); that is the
only way below about 60 cycles and is not in this plan. [coroutine-ring,
chameneos, json, k-nucleotide and table-sort partly: their time is in
natives and `print`, 17.8 R1–R4]

### P5. The optimizations that LuaJIT runs and we do not

Each is small once P1–P3 hold; today they cannot run because arms and calls
cut every loop into pieces.

- *Inlining* (13, I1). spectral's `A`, fib's recursion two levels,
  array3d's `array_set` through `self.set`, ray's `sphere_distance`/`vsub`/
  `dot`, richard's scheduler, mandel-metatable's `__add`/`__mul`, json's
  `self:read()`. Callee identity from P3 step 3; budget per 13. This is the
  item that takes spectral from about 80 instructions per iteration (after
  P1–P3) to about 25.
- *Allocation sinking* (9.10, A4): mandel-metatable's three temporaries per
  iteration, ray's vectors, fixpoint-fact's closures (`lj_opt_sink.c`).
- *Bounds checks against the loop range* (9.7, Q7): an index that is the
  loop counter or counter ± k checks `asize` once in the preheader
  (LuaJIT's `ABC` with scalar evolution, `lj_record.c:1398`); the array
  pointer is loaded once per loop when the loop stores no new key.
- *Range analysis* (9.7, Q8): `n - 1`, `n - 2` on `n >= 2`; the loop
  counter's bounds; narrowing `I64` back to `I32` (P2).
- *Invariant arithmetic with exits* (Q6): queen's `c - n` and `c + n` sit
  in blocks that do not dominate the loop's exits, so LICM's spine rule keeps
  them. Hoist an invariant `IAdd`/`ISub`/`IMul` by computing it in `I64`
  (no exit) in the preheader and narrowing with one `cmp x, w, sxtw` guard
  there; the guard can only fire where the interpreter would have boxed a
  big integer, so it never changes a result, only where the loop runs.
- *Rematerialization* (Q4): constants, `NIL`, `BOX`, shape and tag words,
  unboxes of slots are re-emitted at uses rather than spilled (queen reloads
  the constant 1 from a spill slot every iteration; nbody's loop reloads the
  limit, the constant 1 and the `BOX` bound).
- *The collector check* (C2, Q2): with P1's effects a loop that allocates
  nothing has no check; a loop that does has one, at the header, as
  LuaJIT's trace has one.
- *Branch layout* (Q5): thread empty join blocks (six branch-to-branch
  jumps per iteration in queen today).

### P6. Constants the builder does not see

**Evidence.** array3d's `assert` is read from `_ENV` through an `_AUX` IC
entry (`tabload { loc: 0x80000080 }`) and is not refined to the constant, so
the intrinsic (12) is missed and the call is real: three per `array_set`.
partialsums' `sin`, `cos`, `sqrt` are locals of the main chunk, live
registers at the loop entry, typed `fun` by the entry guard but never
refined, so their intrinsics are missed too (every one is a call in the
region). richard and qt reload and re-test constant upvalues after every call
(S7).

**Fix.** Refine any loaded function, string or table to the constant seen at
compile time when a use wants it, from an `_AUX` slot as from an `_INL` one;
at a loop entry, refine a live register holding a function to the frame's
value with one identity guard in the prologue (LuaJIT guards the function
of every call, `lj_record.c:826`); constant upvalues of every type for
upvalues never assigned after capture (`rec_upvalue_constify`,
`lj_record.c:1760`; the compiler sets the flag). [array3d, partialsums,
richard, qt, recursive-fib-gv]

### P7. Code size and compile time

The arms are most of the IR: spectral's loop region is 894 instructions and
69 blocks for a three-line loop (LuaJIT's trace: 83 instructions), heapsort's
`sift` 496 blocks. `regalloc2` costs 0.4–0.8 µs per IR instruction (17.7)
and its spill decisions are driven by the arms' clobbers. P1 and P3 shrink
the IR 3–5× on loop regions and remove the clobbers, which is the compile-time
fix: the passes and the allocator cost per instruction, and a region with a
fifth of the instructions and no hot-path clobbers compiles in a fifth of the
time with fewer spills. The remaining items of 17.3 T1–T4 stand; none is
needed to reach LuaJIT's code quality.

### P8. Things that already work and must not regress

nbody (170 instructions per pair against LuaJIT's 185) shows the typed path is
right: `_FF` forms, `F64` block parameters, slot loads at constant offsets,
one shape guard per receiver, barriers only on heap stores. The allocation
benchmarks (binary-trees, havlak, storage, fixpoint-fact, deltablue) are
ahead by 1.5–4.7× because closure creation compiles and allocation is cheap;
inlining and sinking must keep their budgets (13) so these do not pay for
ray's gains in compile time.

### P9. The harness

`unpack` and `loadstring` (five benchmarks) and a Latin-1 source file (life)
keep six benchmarks from running at all. The LJR harness is Lua 5.1; the fix
belongs in the benchmark runner (a prelude that aliases `unpack =
table.unpack`, `loadstring = load`, and reads the file as bytes), not in the
VM. Until then mandelbrot and fannkuch, the two kernels most sensitive to P1
and P2, are not measured.

## 5. Order and expected results

The order follows dependencies: P1 and P2 change what a loop body is made
of and are prerequisites for everything that reasons about loops; P3 changes
the call protocol that inlining (P5) then removes from the hot kernels; P4
is independent and small.

| Step | Items | Kernels it fixes | Expected (instructions per unit; LuaJIT in brackets) |
|---|---|---|---|
| 1 | P1 (speculate or exit, cold trampolines, precise effects, lowering), P6 | queen 65 → 22 (17); linear-sieve 315 → about 150 (127); array reads everywhere; the spills of ray and qt | scimark-*, heapsort, nsieve, towers, list, bounce level with or ahead of LuaJIT; queen, linear-sieve within 1.3× |
| 2 | P2 (parameter representation, peeling, `I64` operands, feedback as `NUM`) | spectral's add, partialsums 1260 → about 400 (337), series | partialsums and series within 1.2×; spectral's non-call part gone |
| 3 | P3 steps 1–2 (trusted slots, values in the frame), P4 | spectral 222 → about 110; fib 79 → about 55; coroutine-ring 97.6 → 70 cycles (68.6); array3d 838 → about 300 | chameneos, coroutine-ring level; fib within 1.5× |
| 4 | P3 steps 3–4 (typed entries, results in registers, lazy headers) | fib 55 → about 20 (36); spectral → about 80; array3d → about 150 | fib ahead of LuaJIT; ray, richard, deltablue, json faster by the call share (10–16% of their cycles, 17.2) |
| 5 | P5 (inlining, sinking, ABC/range, invariant hoisting, remat, GC check, layout) | spectral 80 → about 25 (24); queen 22 → 17 (17); array3d → about 60 (59); ray 1266 → about 300 (273); mandel-metatable 272 → about 40 cycles (135); fixpoint-fact's closures | the suite's numeric and call-heavy halves within 1.1× of LuaJIT or ahead; allocation half keeps its 1.5–4.7× |
| 6 | P7 | compile time 2–5× lower per region from the smaller IR | deltablue, havlak, json compile shares under 2% |

Step 1 alone moves 17 of the 38 benchmarks. Steps 1–4 are the structural
changes the user's question is about; each is measured against the same
per-operation table as section 1, and the per-kernel targets above come from
LuaJIT's traces and the clang kernels, which show the code shape is
attainable from our IR once arms, reloads and the call protocol are gone.

## 6. Appendix: side-by-side code

### spectral-norm inner iteration

C (clang -O2, the scalar remainder loop of `Av`):

```
  ldr   d1, [x17], #8          ; x[j]
  sub   w4, w16, #1
  mul   w4, w16, w4            ; ij * (ij - 1)
  scvtf d2, w4
  fmadd d2, d2, d0, d22        ; * 0.5 + i
  fdiv  d2, d3, d2             ; 1.0 / ...
  fmadd d23, d1, d2, d23       ; a += x[j] * A
  add   w16, w16, #1
  subs  x3, x3, #1
  b.ne  loop
```

LuaJIT, trace 2's loop (24 instructions; `A` inlined, int `ij` with overflow
exits, `a` in `d15`):

```
->LOOP:
  ldr   x27, [x2, w28, uxtw #3]   ; x[j]
  cmp   x3, x27, lsr #32          ; int tag
  bne   ->2
  mov   w27, w27
  adds  w25, w28, w1 ; bvs ->2    ; i + j
  subs  w25, w25, #1 ; bvs ->2    ; ij
  subs  w26, w25, #1 ; bvs ->2    ; ij - 1
  smull x26, w25, w26
  cmp   x26, w26, sxtw ; mov w26, w26 ; bne ->2
  scvtf d14, w26
  fmul  d14, d14, d2              ; * 0.5
  fadd  d14, d14, d1              ; + i
  fdiv  d13, d0, d14
  scvtf d14, w27
  fmul  d14, d14, d13
  fadd  d15, d14, d15
  add   w28, w28, #1
  cmp   w28, w0
  ble   ->LOOP
```

tcvm today: 222 instructions, section 2.1.

tcvm after steps 1–5 (P1: typed array load with the bounds and array pointer
hoisted; P2: `a` an `F64` parameter; P3: no stores or reloads; P5: `A`
inlined, `i + j - 1` with range-proven arithmetic, counter in a register):

```
loop:
  ldr   x9, [x4, w28, uxtw #3]    ; x[j]
  lsr   x16, x9, #32 ; cmn w16, #1 ; b.ne exit   ; small int
  add   w10, w28, w1              ; i + j (ranges: no overflow)
  sub   w10, w10, #1
  sub   w11, w10, #1
  mul   w11, w10, w11             ; proven < 2^31 for N <= 46340, else smull form
  scvtf d14, w11
  fmadd d14, d14, d2, d1          ; * 0.5 + i
  fdiv  d13, d0, d14
  scvtf d12, w9
  fmadd d15, d12, d13, d15        ; a += x[j] * A
  add   w28, w28, #1
  cmp   w28, w0
  b.le  loop
```

17 instructions. Each line maps to a fix: the typed load and hoisted bounds
to P1 and P5, the `fmadd` on `d15` to P2, the absence of stores and reloads
to P3, the inlined body to P5, the overflow-free `add`/`sub` to range
analysis.

### queen inner iteration

LuaJIT's 17 instructions are in 2.2. tcvm after steps 1 and 5:

```
loop:
  ldr   x0, [x4, w21, uxtw #3]    ; a[i]
  lsr   x16, x0, #32 ; cmn w16, #1 ; b.ne exit
  cmp   w0, w3 ; b.eq false       ; == c
  sub   w1, w0, w21               ; a[i] - i (range-proven, else subs/b.vs)
  cmp   w1, w2 ; b.eq false       ; == c - n (hoisted)
  add   w1, w0, w21
  cmp   w1, w5 ; b.eq false       ; == c + n (hoisted)
  add   w21, w21, #1
  cmp   w21, w19
  b.le  loop
```

14 instructions; LuaJIT keeps the overflow checks (17).

### Reproducing the measurements

```
# suite: /usr/bin/time -l on each binary, cycles elapsed and instructions retired
tcvm-cli -f luabench/X.lua ARGS ; TCVM_JIT=off tcvm-cli ... ; luajit ... ; luajit -joff ...
# our code
TCVM_JIT_LOG=1 TCVM_JIT_DUMP=opt,asm tcvm-cli -f luabench/X.lua ARGS 2> X.log
# LuaJIT's traces, IR and machine code
LUAJIT_DUMPFILE=X.dump luajit -jdump=im luabench/X.lua ARGS ; luajit -jv luabench/X.lua ARGS
# per-operation costs: the delta of instructions retired between two sizes,
# divided by the delta of the unit count (fib: 2*F(n+1)-1 calls; spectral: 40 N^2)
```
