# TCVM JIT: SSA IR and bytecode lowering

Status: design, not yet implemented.

## Decisions

| Axis | Choice |
| --- | --- |
| Compilation model | CFG SSA with basic-block versioning (not a linear trace) |
| Region scope | One `Prototype`. Calls stay real calls. `FrameState` carries a parent chain so inlining is additive later. |
| IR levels | One optimizing IR, lowered **in place** (high `tabget` → low `load.val`), then a dumb machine IR (VCode) for regalloc + encoding |
| Metamethod invalidation | Watchpoints: compiled code declares dependencies; metatable mutation invalidates dependents |
| Guard failure | Deopt to the interpreter. Exit descriptors carry `(pc, framestate, entry context)` so lazy stubs can be added later. |

Consequence worth naming: with deopt-only exits, block versioning is **eager** — versions are
discovered at compile time by propagating the entry type context through the CFG, not lazily
minted at runtime on guard failure. A truly polymorphic site deopts and re-profiles rather than
converging on two versions. Mitigation is exit counters plus recompile-with-widened-context.

## What the existing runtime forces on the IR

These are not incidental; each one shows up as a concrete IR feature.

**`Value` is a 16-byte tagged pair** (`kind: u8`, `data: u64`), not NaN-boxed, and Lua 5.5
Integer and Float are distinct types. So: type tests are byte compares, but a value in unpacked
form (a bare `i64` in a register) needs explicit `pack`/`unpack` nodes to become a `Value`, and
arithmetic bifurcates into `i64` and `f64` op families that specialization must resolve.

Note that **nothing in this IR boxes in the allocating sense**. Lua 5.5 numbers never heap-allocate,
and strings/tables are already heap objects whose pointer the tagged pair merely carries. `pack.int`
is a tag write plus a payload write — pure, no GC, no rooting. The ops are named `pack`/`unpack`
rather than `box`/`unbox` precisely so nobody "fixes" their effect summary to include `may_gc`.

**Shapes are a ready-made guard primitive, and self-invalidating.** `transition_add_prop` hands a
table a *new* `Shape` pointer when a new string key appears, and deletion migrates it to dict
mode (again a different shape). So `guard.shape` — a pointer compare — proves the property slot
layout. Two caveats:
  - **Never specialize on a dict-mode shape.** The dict sentinel is *shared* across every dict
    table with the same `MtCache` (`MtCache::ensure_dict_sentinel`), so a shape match there proves
    nothing about layout. The compiler must reject `is_dict()` shapes as guard subjects.
  - Overwriting an *existing* key does not transition, so a guarded slot store is safe; writing a
    *new* key does, so it cannot be compiled as a slot store.

**A shape guard does not prove metamethod absence.** `MtCacheData::bits` is a `Cell` that
`maybe_update_mt_bit` mutates in place, so `mt.__index = f` changes the meaning of `t.x` without
changing any shape pointer. This is why we have watchpoints (below).

**The GC is incremental mark-and-sweep, non-moving, with explicit barriers.** Non-moving is a
gift: a raw interior pointer (e.g. the `properties` Vec's data pointer) stays valid as long as
the owning table is reachable, so rooting the *base object* suffices and derived pointers need no
rooting. But:
  - Any store of a `Gc` into a GC object from compiled code must emit a barrier
    (`Mutation::backward_barrier` / `forward_barrier`), so barriers are explicit IR nodes.
  - `ThreadState`'s `Collect` impl traces `stack[..live_top]`. A live `Value` held only in a
    machine register across a GC point is **invisible to the collector**. Hence the rooting rule
    below.
  - The `properties`/`array` Vecs can reallocate, so a cached data pointer is killed by any store
    that can grow them. That is an aliasing fact, not a GC fact, and is handled by effect classes.

**Open upvalues address stack slots by index.** `UpvalueState::Open { thread, index }`. A local
captured by a closure is pinned to its canonical stack slot and cannot live purely in SSA/registers.

**Compiled code runs on the native stack, and Lua calls can yield or raise.** A callee that yields
cannot suspend our native frame. Handled by making calls deopt-on-suspend (below).

## Structure

Index-arena style, Cranelift-shaped:

```rust
pub struct Func {
    blocks:  PrimaryMap<Block, BlockData>,
    insts:   PrimaryMap<Inst,  InstData>,
    values:  PrimaryMap<Val,   ValData>,
    states:  PrimaryMap<FsRef, FrameState>,
    exits:   PrimaryMap<ExitRef, Exit>,
    pool:    ConstPool,      // the only thing that touches 'gc
    entry:   Block,
}

pub struct BlockData {
    params: Vec<Val>,        // block params, NOT phi nodes
    insts:  Vec<Inst>,
}
```

**Block params, not phi nodes.** They make the LBBV entry type context *literally* the types of a
block's parameters, which is exactly what we key version lookup on.

**No `'gc` in the IR.** GC references (constants, shapes, prototypes, interned key strings) live in
a `ConstPool<'gc>` that implements `Collect`; the IR holds typed indices (`ConstRef`, `ShapeRef`,
`ProtoRef`, `StrRef`) into it. This keeps the IR plain data — hashable for GVN, printable, unit-
testable without an arena — and confines rooting to one structure if compilation ever moves off the
mutator thread.

Instructions may produce **multiple results** (`CALL`, `TFORCALL`), so `InstData` carries a result
list, not a single value.

## Types

Two orthogonal axes per SSA value.

**Representation** — how the value is physically held:

```
Val   full 16-byte tagged Value (tag + payload)
I64   raw integer, no tag
F64   raw float, no tag
B1    boolean condition (not a Lua value)
Ptr   raw untagged GC pointer or interior pointer
```

**Lua type set** — what we've proven, as a bitset:

```
NIL FALSE TRUE INT FLOAT STR TAB FUN THR UDATA
```

Splitting `FALSE`/`TRUE` rather than a single `BOOL` gives truthiness precision for free —
`falsy = NIL|FALSE` — which lets `TEST`/`TESTSET` branches fold whenever the set is disjoint from
one side.

**Refinement** — optional, at most one:

```
Const(ConstRef)     value is a known constant
Shape(ShapeRef)     implies TAB; a concrete non-dict shape
Proto(ProtoRef)     implies FUN; a known prototype
```

So `Ty = { rep, set, refine }`. `Ty::any()` is `{ Val, all kinds, None }`.

**Guards produce refined values.** `v3 = guard.type v2, INT` yields a *new* SSA value with the
narrower type, rather than mutating `v2`'s type in place. Type facts then ride the use-def graph and
the block-entry context is just "the types of the block params" — no side table, no flow-sensitive
type map to keep in sync.

## Effects and aliasing

Every op carries an effect summary. Without this, GVN and LICM are either unsound or useless.

Flags: `may_gc`, `may_raise`, `may_call`, `may_yield`, `terminator`.

Memory is partitioned into alias classes rather than one monolithic heap edge:

```
TabProps(t)    string-keyed slot storage of table t
TabPropsPtr(t) the properties Vec's data pointer (killed by any shape transition on t)
TabArray(t)    array part of t
TabArrayPtr(t) the array Vec's data pointer (killed by any array growth on t)
TabHash(t)     misc-hash part of t
Shape(t)       t's shape field
Upval(c)       an upvalue cell
Stack          the thread's value stack (open upvalues alias this)
```

v1 disambiguates by class only, not by base value — two `TabProps` accesses on different tables are
assumed to may-alias unless one of the tables is a known-local allocation. Escape analysis upgrades
this later. `may_call` ops clobber everything.

## Guards, assumptions, deopt

**Guard** — a check with local failure semantics. Carries an `ExitRef`.

```
guard.type  v: Val, TypeSet          -> Val   (refined)
guard.shape v: Val{TAB}, ShapeRef    -> Val{Tab<S>}
guard.cond  b: B1                             (bounds checks, div-by-zero, loop bounds)
```

**Assumption** — a global fact, backed by a watchpoint. Emits **no code**; records a dependency in
the code object.

```
assume.no_mm  ShapeRef, MmBits
```

**FrameState** — the abstract interpreter state at a program point:

```rust
pub struct FrameState {
    pc:     u32,                  // bytecode pc to resume at
    regs:   Vec<Option<Val>>,     // one entry per Lua register
    parent: Option<FsRef>,        // always None in v1; inlining fills it
}
```

Varargs and the below-base region need no representation: the region never writes them, so they
stay on the stack untouched.

**Exit**:

```rust
pub struct Exit {
    fs:    FsRef,
    ctx:   TypeContext,   // entry context observed here; for widening on recompile
    count: Cell<u32>,     // hot-exit counter -> triggers recompile with widened ctx
}
```

**Safepoint** — `safepoint(FsRef)`. A GC poll *and* an invalidation poll. Placed at loop back-edges
and after every call.

### GC rooting rule

At any op with `may_gc`, every live value with rep `Val`/`Ptr` must be **anchored**: either it is a
Lua register named by the FrameState (and therefore spilled to its canonical stack slot, where
`ThreadState::trace` will find it), or it is *derived* from an anchored value (which is sound only
because the collector does not move objects).

Anything else — a live `Value` temporary that isn't a Lua register — spills to a small
`Collect`-traced JIT spill area on `ThreadState`. Lowering enforces this; the verifier checks it.

### Invalidation

The code object owns a dependency set. Metatable mutation looks up dependents by `MtCache`
identity and marks them invalid. An executing region notices at its next safepoint poll and deopts
through the FrameState it is already carrying.

Poll placement must be sound against self-inflicted invalidation: compiled code that stores to a
table which is *itself* adopted as a metatable can invalidate its own assumptions. `maybe_update_mt_bit`
only fires when `mt_cache.is_some()`, so the rule is: poll after every call, and after any heap
store whose target table is not proven to have `mt_cache == None`.

## Op set

Grouped by level. Ops below the line in each group are what specialization/lowering produces.

**Constants / repr**
```
kconst ConstRef -> Val        iconst i64 -> I64       fconst f64 -> F64     bconst bool -> B1
pack.int I64 -> Val            pack.float F64 -> Val    pack.bool B1 -> Val
unpack.int Val -> I64          unpack.float F64         unpack.ptr Val -> Ptr
tag Val -> I64                is.type Val, TypeSet -> B1
```

**Generic Lua ops** — the escape hatch when types are unknown. `may_call`, `may_raise`, `may_gc`,
clobbers all memory. Lowers to a runtime helper call.
```
lua.arith Op, Val, Val -> Val      lua.cmp Op, Val, Val -> B1     lua.eq Val, Val -> B1
lua.concat Val, Val -> Val         lua.len Val -> Val
lua.getindex Val, Val -> Val       lua.setindex Val, Val, Val
```

**Specialized arithmetic** — pure.
```
add.i64 sub.i64 mul.i64 idiv.i64 mod.i64 neg.i64
band bor bxor shl shr bnot
add.f64 sub.f64 mul.f64 div.f64 idiv.f64 mod.f64 pow.f64 neg.f64
sitofp I64 -> F64        fp_to_int_exact F64 -> I64   (guarded)
icmp CC, I64, I64 -> B1  fcmp CC, F64, F64 -> B1      is.falsy Val -> B1
```
`idiv.i64`/`mod.i64` take a `guard.cond` on a nonzero divisor — matching `ArithOp::INT_ZERO_DIVISOR_RAISES`.

**Tables**
```
tab.new hint -> Val{TAB}                              may_gc
tab.props   Val{Tab<S>} -> Ptr                        reads TabPropsPtr
slot.get    Ptr, imm slot -> Val                      reads TabProps
slot.set    Ptr, imm slot, Val                        writes TabProps  (+ barrier)
tab.arr     Val{TAB} -> Ptr                           reads TabArrayPtr
tab.arr_len Val{TAB} -> I64                           reads TabArray
arr.get     Ptr, I64 -> Val                           reads TabArray   (bounds-guarded)
arr.set     Ptr, I64, Val                             writes TabArray  (+ barrier)
tab.hash_get Val{TAB}, Val -> Val                     reads TabHash    (helper, no gc)
gc.barrier_back Ptr        gc.barrier_fwd Ptr, Val
```
A `slot.set` is only legal for a key **already present** in the guarded shape. Adding a new key
transitions the shape, so it must go through `lua.setindex`.

**Upvalues and pinned registers**
```
upval.cell imm idx -> Ptr        (closure->upvalues[idx]; immutable per closure)
upval.get  Ptr -> Val            reads Upval|Stack
upval.set  Ptr, Val              writes Upval|Stack  (+ barrier)
upval.close imm start            writes Stack|Upval, may_gc
stack.get imm reg -> Val         reads Stack     — for stack-pinned registers
stack.set imm reg, Val           writes Stack
```
Any register captured by a `CLOSURE` in this prototype (a `ParentLocal` upvalue descriptor) is
**stack-pinned**: the frontend pre-scans for these and routes all reads/writes of them through
`stack.get`/`stack.set` rather than SSA, because an open upvalue can observe the slot at any time.

**Calls**
```
call.lua    Val{FUN}, args.. , imm nret -> (results.., Status)
call.native Val{FUN}, args.. , imm nret -> (results.., Status)
closure.new ProtoRef, upvals.. -> Val{FUN}            may_gc
```
Calls are `may_call | may_gc | may_raise | may_yield` and clobber all memory. They take a
`FrameState`, materialize the frame before transferring control, and return a `Status`. The
frontend emits a branch on it:

- `Ok` — results are in the result values, continue in compiled code.
- `Suspended` — the callee yielded. We cannot suspend a native frame, so write the FrameState back
  as a real `LuaFrame` with `pc` = *after* the call, return from the region, and let the executor's
  normal machinery resume us **in the interpreter**. Correct because `op_call` already leaves the
  caller's `pc` past the `CALL` and `op_return` lands results at `func_idx`.
- `Error` — unwind: pop our frame, return the error to the executor.

So a yield inside a JIT'd loop costs a drop back to the interpreter, and the next hot back-edge
re-enters via OSR. That is the price of not inlining, and it's cheap to pay.

**Control**
```
jump Block(args..)                       br B1, Block(args..), Block(args..)
ret vals..                               deopt ExitRef
safepoint FsRef
```

## Not supported in v1 (compilation aborts; region stays interpreted)

Because exits deopt, "abort" is always a legal answer — we simply don't compile.

- **Multi-return / MULTRET** (`CALL`/`TAILCALL` with `args == 0` or `returns == 0`, `VARARG` with
  `count == 0`, `SETLIST` with a dynamic tail). Needs a `vals` pseudo-type and stack-window ops.
- **`TBC`** — to-be-closed slots interact with error unwinding in ways not worth modelling yet.
- **Varargs functions** — `VARARGPREP`/`VARARG`. `VARARGGET` alone (the below-base fast form) is
  tractable and can come early.
- **Coroutine-specific opcodes** — none exist; yields are handled via the call status path above.

## Frontend: bytecode → SSA

The frontend is *simultaneously* the SSA builder and the block-versioning driver. It does not build
a generic SSA function and then specialize it; it abstract-interprets the bytecode with a symbolic
register state and emits typed IR directly.

**1. Bytecode CFG.** Leaders are: pc 0, every jump target, every instruction after a
jump/compare/return. Compare and test ops (`EQ`/`LT`/`LE`/`TEST`/`TESTSET`) conditionally skip the
next instruction, which is always a `JMP` — each such pair becomes one two-way edge. `FORPREP`/
`FORLOOP` and `TFORPREP`/`TFORCALL`/`TFORLOOP` become the loop's edges.

**2. Pre-scan.** Determine the stack-pinned register set (registers captured as `ParentLocal` by any
`CLOSURE` in this prototype) and check the abort list.

**3. Symbolic execution with versioning.** Worklist of `(bytecode_pc, TypeContext)`. State is
`reg -> Val` (SSA value) plus each value's `Ty`. For each bytecode op, consult the type state:

- Types known and favourable → emit specialized ops directly, no guard.
- Types unknown → emit a `guard.type` seeded from feedback, then specialize on the refined value.
- No feedback → emit the generic `lua.*` op.

At a branch, the successor is looked up as `(target_pc, ctx)`. Hit → `jump` to the existing version,
passing block params. Miss → mint a new version and push it on the worklist. Version count per pc is
capped; on overflow the context is **widened** (types joined toward `any`) so the block count stays
bounded.

**4. Where type feedback comes from.** Notably, we need almost none up front:

- **The entry context is read from the live stack at compile time.** Compilation is triggered from a
  running frame — at a hot back-edge (OSR) or a hot call — so the *actual* types of every live
  register are right there. That is the seed context, and LBBV propagates it. No profiler required
  to get numeric loops fully specialized.
- **Existing inline caches** (`Prototype::ic_table`) give `GETFIELD`/`SETFIELD`/`GETTABUP`/`SETTABUP`
  their shape → `guard.shape` + `assume.no_mm` + `slot.get`.
- Anything else (what a call returns, what's in a table) gets a guard or stays generic.

The risk is that a once-observed entry context is unrepresentative, which produces deopt thrash;
the exit counters and context widening exist for exactly that.

**Sketch** of `for i = 1, n do s = s + t.x end` after the frontend, given an OSR entry with
`i: INT, n: INT, s: INT, t: Tab<S_A>`:

```
BB0(i0: Val{INT}, n0: Val{INT}, s0: Val{INT}, t0: Val{TAB}):
    t1 = guard.shape t0, S_A          ; from the GETFIELD IC
         assume.no_mm  S_A, INDEX     ; watchpoint; emits nothing
    p0 = tab.props t1                 ; loop-invariant, LICM hoists
    i1 = unpack.int i0
    n1 = unpack.int n0
    s1 = unpack.int s0
         jump BB1(i1, s1)

BB1(i: I64, s: I64):                  ; loop header
         safepoint FS(pc=12)
    x  = slot.get p0, 3               ; no guard, no tag check
    x1 = guard.type x, INT            ; deopt if the field isn't an int
    x2 = unpack.int x1
    s' = add.i64 s, x2
    i' = add.i64 i, 1
    c  = icmp le, i', n1
         br c, BB1(i', s'), BB2(i', s')

BB2(i: I64, s: I64):
    ...  pack.int, write back, ret
```

Everything the interpreter does per iteration — two tag checks, an IC shape compare, a metamethod
bit test, a tagged add, a tagged compare — is gone from the loop body.

## Pipeline after the frontend

The frontend already emits reasonably specialized code (that's the point of versioning), so the
passes are mostly about hoisting and cleanup:

1. **GVN / redundant guard elimination** — a `guard.shape` dominated by an identical one is dead.
2. **Load elimination** — using the alias classes; kill `slot.get` redundant with a dominating
   `slot.get`/`slot.set` on the same class.
3. **LICM** — hoist loop-invariant guards, `tab.props`, `tab.arr`, bounds checks. This is where the
   real win is: the guard cost moves from per-iteration to per-entry.
4. **Narrowing** — eliminate `pack.*` feeding a use that immediately unpacks it again.
5. **DCE**, then **lowering** to the machine op subset, then VCode + regalloc + encode.

Later, and unlocked by the FrameState parent chain: **inlining**, then **escape analysis /
scalar replacement** (a table that never escapes the region is never allocated).

## Module layout

```
src/jit/
  mod.rs
  ir/
    mod.rs       Func, Block, Inst, Val arenas
    ty.rs        Rep, TypeSet, Refine, Ty, TypeContext (lattice + join/widen)
    op.rs        opcode enum + effect summaries
    pool.rs      ConstPool<'gc> (the only 'gc-aware part), Collect impl
    build.rs     builder, block param plumbing
    verify.rs    SSA/type/effect/rooting invariants
    print.rs     textual form (the sketch above)
  frontend/
    cfg.rs       bytecode CFG
    lower.rs     symbolic execution + block versioning
  opt/           later
  codegen/       later
```
