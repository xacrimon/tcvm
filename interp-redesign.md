# Interpreter core redesign

Status: design agreed 2026-10-07 on branch `ljr-prototype` at fd5d79b. Implementation
not started. This document is the specification for whoever implements it, in
stages (section 15). Read it whole once before stage 0; afterwards each stage names
the sections it depends on.

Every statement about the current code was checked against the source and the
release binary at fd5d79b on 2026-10-07. Every statement about the new design is a
decision (marked D1..D14) or a consequence of one. Estimates are labelled as
estimates. Numbers about hardware behaviour come from measurements recorded in
earlier sessions (section 3) and must be re-measured on this design before being
trusted for it.

Rules for the implementer:

- Deviating from this document is allowed only with a written entry in section 17
  (what, why, cost) made in the same commit. Silent simplifications have cost this
  project real time; do not take a shortcut without saying so.
- A stage is not done until every gate in its checklist passes. "Mostly passing" is
  not done.
- Do not run benchmark suites unless a stage's checklist calls for them or the user
  asks. Report test status and offer measurement instead.
- Comments in code are terse: one line for the contract, at most one sentence for a
  non-obvious invariant. Rationale lives here and in commit messages.
- No attribution trailers in commits.

## Contents

1. Goals and scope
2. The current interpreter, measured
3. Hardware behaviour the design must respect
4. Decisions
5. Handler ABI
6. The runtime struct
7. Value stack and frame header
8. Calls, returns and continuations
9. Natives
10. Errors and unwinding
11. Coroutines and the executor boundary
12. Instruction set
13. Compiler changes
14. Garbage collector interaction
15. Module layout and code conventions
16. Implementation stages and gates
17. Deviations log
18. Open questions
19. Appendix A: baseline numbers
20. Appendix B: register assignments

## 1. Goals and scope

The LuaJIT Remake (LJR) skeleton adopted in October 2026 stays: frames carry a
continuation that decodes the caller's instruction, every control transfer between
handlers is a guaranteed tail call under the `rust-preserve-none` calling convention,
natives that call Lua keep their state in stack slots and continue through a
continuation function. Everything around that skeleton is rewritten:

- the handler signature and how state passes between handlers and slow paths;
- the frame representation (headers move into the value stack);
- the call, return, metamethod and native call protocols;
- the error, unwind and coroutine machinery that sits on frames;
- the instruction set, which becomes adaptive (generic forms specialize themselves);
- the file and macro structure of `src/vm`.

Goals, in the order the user gave them: code quality (minimal, consistent, careful
macro use), an instruction set that lets fast handlers do one guard and one
operation, a principled way to pass data to tail-called slow paths, call and
arithmetic efficiency, code size of the executed handler set, consistency, and raw
performance on both aarch64 and x86-64.

Out of scope: a JIT, changes to the value representation beyond the NaN rule in D9,
the garbage collector itself, the compiler beyond what section 13 lists, folding of
never-assigned plain locals (the user keeps `<const>` as the fold trigger), and
library rewrites beyond the fast entries in section 9.6.

Lua semantics are the reference, not luac's register layout. Where this design
diverges observably from Lua 5.5 (debug library effects of by-value upvalues, error
positions of tail-called natives) the divergence is listed in section 18 and must be
noted in the commit that introduces or keeps it.

## 2. The current interpreter, measured

`src/vm/interp.rs` is 7465 lines holding about 170 handler functions behind 122
opcodes (71 base forms, 19 immediate arithmetic forms, 15 inline-cache forms, 3
shared-cell upvalue forms, 2 CALL variants, 7 `_NUM` and 3 metamethod arithmetic
forms). Each handler repeats a nine-argument signature and invokes `helpers!`, which
defines fourteen nested macros per body.

### 2.1 Handler arguments today

```
instruction  Instruction           the current word (also abused: a raw Value in
                                   meta_call, a result count in continuations)
ctx          Context<'gc>          two words: &Mutation, &State
thread       &mut ThreadState
registers    *mut Value            the frame's register window
ip           *const Instruction    (also abused: a stack-slot pointer in
                                   continuations and meta_call)
handlers     *const ()             the static dispatch table
ds           &mut DispatchState    fault, native closure, meta_call's ret, current
frame        *mut LuaFrame         the frame record in a separate frame stack
closure      LuaFn                 the current closure
```

Ten words. On aarch64 that is fine (17 argument registers). On x86-64 the
`preserve_none` convention has 12 argument registers, leaving three non-argument
temporaries for the whole handler body. The current shape cannot port.

### 2.2 Fast-path instruction counts (release build at fd5d79b, objdump)

Counts include the four-instruction dispatch tail.

| Handler | Path | Instructions | Where it goes |
|---|---|---|---|
| MOVE | | 8 | floor |
| GETUPVAL | by-value upvalue | 9 | floor |
| ADD | float, float | 21 | each operand loaded twice (tag test then `ldr d`), `fmov`+compare+branch NaN check after the store |
| ADD | small int, small int | 21 | the float test is first and taken |
| ADD_NUM | int, float | about 30 | two tests and a branch per operand |
| JLT | float, float | 17 to 19 | same double load |
| GETFIELD_OWN | hit, inline slot | 26 | tag test, IC table load, entry load, shape compare, inline/aux branch, nil test, `__index` bit test |
| CALL_R1 | Lua callee, arity satisfied | 39 | seven stores (pc on the caller frame, five frame fields, frames top), stack base reloaded, `ds.native` path on natives |
| RETURN1 | | 8 | |
| ret_call1 | | 13 | closure, pc, base reloaded from the frame; stack base reloaded |
| CALL_R1 + RETURN1 + ret_call1 | | 60 | LJR: 49 plus 5 for tier-up counting |
| op_call_native | plain native, excluding its body | about 70 | stack frame, `ds.native` reload, vectorized result copy loop, gc check |
| SETTABLE | array store | has a stack frame | the write barrier's slow path `lighten` is a call |
| quickened arithmetic metamethod | handler side | 104 (traced) | |

### 2.3 Structural problems

1. **Side channels.** `DispatchState` has four fields (`fault`, `native`, `ret`,
   `current`); the instruction register carries a raw `Value` or a count; `ip`
   carries a stack slot. None of it is typed or documented in one place.
2. **Native glue.** Three native kinds with three entries; a 16-byte
   `Result<CallbackAction, Error>` returned through memory (`sret`); a four-phase
   `drive_natives` state machine, partly duplicated in `ret_native`; six frame flags
   (`NATIVE`, `PROTECTED`, `HANDLER`, `PASS`, `PASS_TRUE`, `BASE`); elided pcall
   frames that the unwinder, TAILCALL and the frame walker re-materialize
   (`restore_protected`, `FrameRef::Elided`, `protected_frame`).
3. **Executor frames.** `ExecKind::{Start, WaitThread, Error}` live in a second
   vector interleaved with Lua frames by depth; `pending_ret` and `pending_action`
   carry work between dispatch and the executor.
4. **Type dispatch inside handlers.** 41 arithmetic opcodes each test float/float,
   then int/int, then bail; the `_NUM` form exists because mixed sites otherwise hit
   the slow path every time.
5. **Calls into Rust on fast paths.** Any handler that calls a function (barrier
   slow path, `walk_index_chain`, allocation) gets a stack frame on its fast path too.

### 2.4 What the user measured as costly in practice

From earlier sessions (memory notes): Lua call 16.6 cycles versus LuaJIT's 22.7, but
a plain native call 20 to 24 cycles versus LuaJIT's 6 to 9; metamethod call sites
through `schedule_meta_call` at 22% of an `__index` call; mixed int/float arithmetic
sites falling to the slow path (see `test-files/mandel_bench2.lua`: the pixel setup
is `SUB ff; MUL int*float; DIV float/int; ADD ff`, twice per call, because the plain
locals holding 800 and 1000 are not folded).

## 3. Hardware behaviour the design must respect

These are measured facts from the M4 Pro work (memory notes `perf-methodology-apple-
silicon`, `ljr-gap-fixes`, `arith-quickening`) and they shape handler code. x86-64 has
not been measured yet; section 16 stage 6 covers it. Each rule names the structural
mechanism that enforces it; the implementer must keep the mechanism and verify in the
disassembly.

### 3.1 Branch prediction keys on the branch's address

A conditional branch has one predictor entry per branch instruction, and an indirect
`br`/`jmp` has one per site. Consequences:

- **One dispatch `br` per outcome.** A conditional handler (compares, FORLOOP) must
  end each outcome in its own copy of the dispatch tail, so the taken and not-taken
  successors are predicted at different sites. LLVM tail-merges the two copies unless
  stopped; the `branch_if!` macro stops it with an opaque `asm!("/* {0} */",
  inout(reg) ip, options(nomem, nostack, preserves_flags))` barrier in both arms.
  Merging them cost 5 to 6% on collatz and primes2. Keep the barrier in both arms.
- **Never let a conditional `ip` update become a `csel`.** LLVM if-converts
  `if cond { ip += off }` into a select, making the next instruction-word load
  data-dependent on the compare: about 15 cycles exposed per execution, 2x on mandel
  when it happened. Same barrier. Check the disassembly for `csel ... x_pc` before a
  dispatch load.
- **Shared handlers widen a site's target set.** The generic family handlers of the
  new ISA are reached from many opcodes' dispatch sites; that is fine because they run
  only during warm-up and on misses. Specialized handlers are one per opcode.

### 3.2 Store-to-load forwarding and memory-dependence prediction

- **Do not reload a slot the previous handler just stored.** `op_call_native`
  re-read `R[func]` after a MOVE had written it; the load sometimes ran ahead of the
  store and replayed, doubling the cost of `x = max(i, 3)`. The fix was to hand the
  value through a register. The new ABI passes the native closure pointer in a slot
  for the same reason (section 5.3).
- **A load pair must not half-overlap a fresh store.** Making `LuaFrame` `repr(C)`
  with `pc` next to `closure` let `ret_call1` reload them with one `ldp` over the
  `pc` CALL had just stored: +15% cycles on a call-and-return loop while instruction
  count fell 10%. A narrow load inside a wide store forwarded fine. The header word
  order in section 7.2 is chosen so that the pairs the continuation loads (caller
  base, caller pc) were written at CALL time, long before, and the word RETURN loads
  (ret) is alone.
- **The memory-dependence predictor is PC-indexed.** Moving identical handler bytes
  changes which result-store/next-load pairs alias; on mandel this is a ±4% lottery
  at `op_mul +0x7c / op_load +0x34 / op_move +0x1c`. Only avoiding forwarding through
  the stack in tight sequences fixes it structurally. Treat |Δ| < 2.5% on mandel as
  noise even with counters.
- **Loop-carried chains through the stack.** A slot loaded, modified and stored each
  iteration forwards cheaply only while nothing else loads it; a second load costs
  about 3 cycles per iteration. The numeric-for layout (hidden control slot plus a
  store-only visible copy) exists for this; keep it.

### 3.3 Front end

- **Align every dispatch target to 32 bytes** (`#[rustc_align(32)]`, feature
  `fn_align`). An unaligned entry raised `MAP_DISPATCH_BUBBLE_SLOT` tenfold for about
  4% wall clock. Keep it on every handler, continuation and slow path.
- **Handler size is a layout variable.** Any size change shifts every later handler.
  Compare builds with counters, interleaved runs, never A-then-B.
- **Taken branches are not free on the fallthrough-heavy paths.** Put the expected
  case on the fallthrough; one guard per handler in the specialized ISA makes this
  automatic. In the current generic arithmetic, floats on the fallthrough and ints
  behind a taken branch cost primes 7% while the reverse cost mandel 15%; the
  adaptive ISA removes the choice.

### 3.4 Register files and moves

- **GPR to FPR moves.** `Value::read_float` is a volatile load so LLVM issues a
  separate `ldr d` rather than reusing the integer load and inserting `fmov`. The
  write side uses a volatile `str d`. Keep both; re-measure on x86-64 where `movq`
  is cheap and the double load may lose.
- **The NaN check after every float result** (`fmov`, compare, branch, second store)
  goes away under D9.

### 3.5 Stack frames

A handler that calls any Rust function, however cold the path, gets a prologue and
epilogue on its fast path too (`op_settable` pays about 12 instructions for the
`lighten` barrier call it almost never makes). Rules:

- Fast handlers make no calls. Cold work is a tail-called handler with its own frame.
- A write barrier that needs the collector runs as "lighten, then retry the
  instruction" in a tail-called handler (section 14.2).
- `fill_nil` and `copy_values` stay loops with an opaque pointer barrier so LLVM
  cannot turn them into `memset`/`memcpy` calls; fixed-count continuations
  (`ret_call0/1/2`) have no loop at all.
- Stage 0 adds a script that fails the build check if a handler outside an allowlist
  has a prologue.

### 3.6 x86-64 specifics to verify in stage 6

Twelve argument registers in `preserve_none` (r12, r13, r14, r15, rdi, rsi, rdx,
rcx, r8, r9, r11, rax), three non-argument temporaries (rbx, rbp, r10). Dispatch is
`mov r_insn, [r_pc]; add r_pc, 8; movzx eax, r_insn_b; jmp [r_rt + rax*8]`. Watch
for: spills (`[rsp]` references in a fast handler are a failure), partial-register
writes, `cmp/jcc` fusion kept adjacent, 32-byte alignment, and the double-load
versus `movq` question above. The thread pointer is not pinned on x86-64 (D3); the
handlers that touch `top`, the stack limit or the upvalue list pay one load.

## 4. Decisions

- **D1. Frame headers live in the value stack.** Four words below each frame's
  base, written by the call site. No separate frame stack, no `frame` handler
  argument. (Section 7.)
- **D2. Five portable handler slots**: `insn`, `pc`, `base`, `rt`, `closure`, each
  with a primary role in opcode handlers and documented secondary roles in
  continuations and slow paths. (Section 5.)
- **D3. `thread` is a sixth slot on aarch64 only**; x86-64 loads it from `rt`. The
  difference is confined to the declaration macro and one accessor.
- **D4. The dispatch table is embedded at offset zero of the runtime struct**, so
  `rt` doubles as the table base; the table is per Lua instance and swappable. The
  mutation pointer is cached in the runtime so a `Context` is one word. (Section 6.)
- **D5. Every call writes a header, natives included**, and every callee, Lua or
  native, returns through the header's continuation. One landing path, uniform
  frames for error locations, no framed-or-unframed native distinction.
  (Sections 8, 9.)
- **D6. One shared `enter`** makes a call from a written header: Lua callee inline,
  native to its entry, callable non-function to the cold `__call` hop. CALL inlines
  the Lua arm of the same code. (Section 8.3.)
- **D7. Catch points are identified by the frame's continuation and flags only.**
  `pcall` and `xpcall` never own a frame; the callee's header overlays the hidden
  slots of the pcall call; the unwinder lands the failure result itself and continues
  through the CALL's continuation. Native frames that catch are flagged. (Section 10.)
- **D8. The ISA is adaptive**: each polymorphic family has one fat generic handler
  that executes any types and rewrites the site to a thin specialized form; a failing
  guard returns to the generic, which counts misses and locks the site after a few.
  The compiler emits generic forms only. Tier 1 is 197 opcodes. (Section 12.)
- **D9. Canonical NaN.** `Value::float` maps every NaN to the canonical quiet NaN;
  hardware add, sub, mul, div results store unchecked; libm-backed operations keep
  the check. (Section 12.4.)
- **D10. Frame state is published lazily**: the top frame's base and pc, and `top`
  where a producer defines it, are written to the thread only at exits (native calls,
  collector checks, errors, coroutine switches, growth). One `sync!` macro used at
  every exit. (Section 7.6.)
- **D11. One-word native results** (`NativeOut`): return, error pointer, or a
  packed action descriptor. The fast path handles return inline; everything else
  tail-calls one cold handler. (Section 9.2.)
- **D12. Continuation natives are indexed**, not pointed to: a native frame stores a
  continuation index in its header word 0, and the index selects from a crate-internal
  table. Public natives are plain or async only (decided 2026-10-05). (Section 9.4.)
- **D13. The executor's own frame kinds go away.** A not-yet-started coroutine is a
  stack with an unwritten header; a waiting resumer is a native frame; an uncaught
  error is a thread field; results delivered by the host re-enter through the yield
  frame's continuation. (Section 11.)
- **D14. The interpreter is split into modules** under `src/vm/` with one declaration
  macro for handlers and one table macro for specialized families. (Section 15.)

## 5. Handler ABI

### 5.1 Signature

Every dispatch target has the same signature, which is what `become` requires:

```rust
pub(crate) type Handler = for<'gc> extern "rust-preserve-none" fn(
    insn: Slot,                 // the instruction word, or a role-dependent word
    pc: *const Instruction,     // next instruction, or a role-dependent pointer
    base: *mut Value<'gc>,      // register window of the running frame
    rt: Context<'gc>,           // &Runtime: dispatch table at offset 0 (D4)
    closure: Slot,              // the running closure, or a role-dependent word
    #[cfg(target_arch = "aarch64")]
    thread: *mut ThreadState<'gc>,
) -> Exit;
```

`Slot` is `#[repr(transparent)] struct Slot(u64)` with typed constructors and
accessors (`Slot::insn(Instruction)`, `.insn()`, `Slot::nret(usize)`, `.nret()`,
`Slot::closure(LuaFn)`, `.closure()`, `Slot::native(&NativeClosure)`, `.native()`,
`Slot::raw(u64)`, `.raw()`). A `Slot` is never dereferenced without going through an
accessor, and an accessor is only valid under the role table below. It is a plain
integer in the ABI, so a payload that is not a pointer is sound to pass through it,
which a `LuaFn`-typed argument would not be.

`Exit` is unchanged: `End`, `Gc`, `Pending`.

Argument order is register order. On aarch64 the first nine arguments go in x20 to
x28, so `insn` is x20, `pc` x21, `base` x22, `rt` x23, `closure` x24, `thread` x25.
On x86-64 they are r12, r13, r14, r15, rdi. The native entry must move `rt` out of
rdi before a SysV call; that is one `mov` and only on the native path.

### 5.2 Slot roles

| Slot | Opcode handler | Continuation | Slow path / entry |
|---|---|---|---|
| `insn` | instruction word | result count `nret` | payload named by the target; the instruction is reloaded from `pc - 1` |
| `pc` | next instruction | pointer to the first result | next instruction, or the header pointer once `pc` has been saved into that header |
| `base` | register window | the finished callee's base, so `base - 4` is its header and the header's caller word is the frame to resume | register window |
| `rt` | runtime | runtime | runtime |
| `closure` | current closure | unspecified; reloaded from the caller's header | payload named by the target, most often the native closure |
| `thread` (aarch64) | current thread | current thread | current thread |

Rules:

1. A target's roles are declared with the target, by the kind keyword of the
   declaration macro (section 5.4): `op`, `cont`, `slow(payload...)`, `entry`.
   The macro binds names of the right types, so a slow path that wants the
   instruction writes `let insn = insn_at!(pc)` and cannot read the payload as one.
2. A payload travels in `insn` first, in `closure` second. A slow path that uses the
   `closure` slot as payload reloads the real closure from the header
   (`hdr_closure!(base)`, two instructions) if it needs it.
3. `pc` may be repurposed only after the handler has stored it into a header word 3
   (the call being made) or into the published thread state (an exit).
4. The dispatch tail passes `insn` reloaded, `pc` advanced, the rest unchanged.
5. Nothing ever lives in `DispatchState`; the type is deleted. The one piece of
   cross-handler scratch that does not fit a slot, the pending fault record, is a
   cold field of the runtime (section 10.1).

### 5.3 Why these five, and what the removed arguments became

- `frame` is `base - 4` (D1).
- `handlers` is `rt` (D4). The dispatch tail is `ldr insn, [pc], #8; and t, insn,
  #0xff; ldr h, [rt, t, lsl #3]; br h` on aarch64, four instructions as today, with
  no table register.
- `ctx`'s second word, the mutation pointer, is a field of the runtime refreshed on
  every `enter` (section 6).
- `ds.native` is the `closure` slot on the tail call into a native entry, which is
  also the fix for the load-replay hazard of re-reading the function slot (3.2).
- `ds.ret` is written straight into the staged header's word 1 by the metamethod
  site (section 8.4).
- `ds.fault` is `rt.fault` (10.1). `ds.current` is the runtime's cached thread
  pointer, which coroutine switches update (section 11).
- `thread` stays pinned on aarch64 because register pressure there is not a concern
  and the multret producers, `top` readers, upvalue list and `tbc` list touch it.
  On x86-64 it costs one of ten temporaries and is loaded from `rt.thread` instead.

### 5.4 Declaration macro

One macro, `handler!`, generates every dispatch target. Sketch:

```rust
handler! {
    /// `R[dst] = R[src]`
    op fn op_move(dst: a, src: b) {
        reg![dst] = reg![src];
        next!();
    }

    /// Continuation of CALL_R1: `R[A] = first result or nil`.
    cont fn ret_call1 {
        let (caller, cpc) = caller!();          // ldp of header words 2 and 3
        let call = insn_at!(cpc);
        let v = if nret > 0 { unsafe { values.read() } } else { Value::nil() };
        unsafe { caller.add(call.a() as usize).write(v) };
        resume!(caller, cpc);                   // closure from the caller's header, dispatch
    }

    /// Native entry: the native closure arrives in the `closure` slot.
    entry fn native_call(nc: native) { ... }

    /// Error raise: the fault is in `rt.fault`.
    slow fn impl_error { ... }
}
```

What the macro does for each item:

- emits `#[inline(never)] #[rustc_align(32)] extern "rust-preserve-none" fn` with the
  full signature, under both `cfg(target_arch)` variants;
- binds the slot names for the kind: `op` binds `insn: Instruction` decoded by the
  listed operand accessors (`a`, `b`, `c`, `d`, `e`, `imm`, `h`); `cont` binds `nret:
  usize`, `values: *const Value`, `base`; `entry` binds the payload with the declared
  accessor; `slow` binds nothing extra and offers `insn_at!(pc)`;
- defines the body macros: `next!()` (dispatch), `tail!(target, payloads...)`,
  `reg![i]`, `k![i]` (constant), `upval![i]`, `thread!()`, `sync!()`, `raise!(kind)`,
  `throw!(err)`, `branch!(cond, offset)` (the two-arm barrier form), `gc_check!()`,
  `hdr!(base)` (header accessors), `caller!()`, `resume!(base, pc)`, `enter!(...)`;
- registers the handler in the dispatch table when the item is `op` and names an
  opcode.

A second macro, `family!`, generates specialized handlers from rows
(section 12.6). Nothing else expands into handler bodies. Shared logic is an
`#[inline(always)]` function returning a small value (`Option<Value>`, `bool`, a two-
variant enum), never a macro that expands a body into many handlers; the current
`get_slow_body!`, `set_slow_body!`, `call_action!`, `call_plain!`, `land_native!`,
`tailcall_lua!`, `return_to_ret!`, `resume_switch!`, `protected_call!` and
`get_quick!`/`set_quick!` have no successors.

The per-architecture difference is confined to: the signature emitted by `handler!`,
the `thread!()` accessor (slot or `rt.thread()` load), and `sync!()`/`switch!()`
which store or reload the thread pointer. No handler body contains a `cfg`.

### 5.5 Dispatch and entry trampolines

`run_thread(ctx, thread) -> (Exit, Thread)` becomes `dispatch::enter(rt, thread) ->
Exit`: it reads the published top frame (7.6), decides how to continue, and calls
the first handler with a plain call (not `become`), as today's `enter` does:

- top frame is a Lua frame with a published pc: call `op_nop` with the frame's
  state, which dispatches;
- top frame is a native frame whose continuation is pending (async native woken,
  or results delivered by the host): call `ret_native` as if the awaited call had
  just returned, with `nret` and `values` from the published `top`;
- the yield frame of a coroutine resumed by the host: call that frame's header
  continuation (section 11.4);
- an unstarted thread: call `enter` with header 0 (section 11.1).

The current thread after dispatch is read back from `rt.thread`.

## 6. The runtime struct

Today's `State` (in `src/lua/mod.rs`) becomes `#[repr(C)]` with the dispatch tables
first, and `Context<'gc>` becomes a one-word wrapper around `&'gc State<'gc>`:

```rust
#[repr(C)]
pub struct State<'gc> {
    /// Indexed by opcode byte. Offset 0 so `rt` is the table base.
    dispatch: [Handler; 256],
    /// CALL continuations by the `returns` operand (0 = MULTRET).
    rets: [Handler; 256],
    /// Per current thread, refreshed on switch and growth (section 7.5):
    thread: Cell<*mut ThreadState<'gc>>,
    stack_end: Cell<*const Value<'gc>>,   // physical end of the current stack
    mutation: Cell<*const Mutation<'gc>>, // refreshed by Lua::enter
    gc_due: Cell<*const GcTrigger>,       // what gc_check! compares
    fault: Cell<Option<Fault<'gc>>>,      // section 10.1, cold
    ... existing State fields (globals, symbols, interner, root shapes, next_fn, ...)
}
```

- The 4 KiB of tables per instance is accepted; they are copied from a `const` at
  construction. A per-instance table allows instrumentation (opcode histograms,
  future hooks) by swapping entries, as LuaJIT does with its per-`global_State`
  dispatch table.
- `ctx.mutation()` is one load. The GC allocation paths already take `&Mutation`;
  they get it from the context as before.
- `stack_end` is the physical end of the current thread's value stack as a pointer,
  so the CALL window check is `new_base + max_stack <= stack_end`: one load and one
  compare on both targets. It is updated by stack growth and by thread switches. The
  logical limit (`STACK_LIMIT`, headroom for message handlers) stays a thread field
  consulted only by the grow path.
- `gc_due` points at the metrics pair `gc_check_due()` reads today, so a check is
  one pointer load, one pair load and a compare; unchanged in count, one fewer
  dependent load than through `ctx.mutation().metrics()`.

## 7. Value stack and frame header

### 7.1 Layout

A thread's value stack is one `Vec<Value>` as today. A frame occupies, from low to
high addresses: its vararg extras (vararg functions only), its four-word header, its
register window. The header is at `base - 4` for every frame, Lua or native:

```
            ┌───────────────┬──────────────────────┬────────────────────────────────┐
  slot      │ base-4-nv ..  │ base-4   base-3   base-2   base-1 │ base .. base+max  │
            │ extras (nv)   │ func     ret      caller   pc     │ registers         │
            └───────────────┴──────────────────────┴────────────────────────────────┘
```

The call site writes the header into the function slot and the three hidden slots the
compiler reserves after it (13.1), so for a CALL at register `a` the callee's base is
`base + a + 4`. A vararg callee moves its header and fixed parameters up past the
extras on entry (7.4), which keeps the header at a fixed offset from base at the cost
of a rotate on vararg entry only.

### 7.2 The four words

| Word | Name | Lua frame | Native frame |
|---|---|---|---|
| 0 | `func` | closure pointer (48 bits) `\| nv << 48` | native closure pointer `\| at << 48 \| cont_idx << 56 \| ok << 62` |
| 1 | `ret` | continuation handler `\| flags` (low 5 bits) | same |
| 2 | `caller` | caller's base pointer, null for a thread's bottom frame | same |
| 3 | `pc` | caller's resume pc | same |

Flags (word 1 low bits; handler addresses are 32-byte aligned so five bits are free):

```
NATIVE      1   word 0 is a native closure, word 3 is still the caller's pc
HAS_OPEN    2   a CLOSURE in this frame captured a local by reference
HAS_TBC     4   a TBC in this frame registered a to-be-closed slot
PROTECTED   8   native frame: errors reaching it go to its continuation
HANDLER    16   native frame: window slot 0 holds the xpcall message handler
```

Word 0 fields: `nv` is the vararg count (16 bits suffice: the stack is at most 65500
slots). For native frames `at` is the window offset of the awaited call (8 bits),
`cont_idx` indexes the continuation table (6 bits, section 9.4) and `ok` is the
pass-through mode (2 bits, section 9.5). The closure pointer is recovered with the
same 48-bit mask that turns any boxed `Value` into a pointer, so reading word 0 as a
closure costs what reading the function slot cost before: one load and one `and`.

Why this order: CALL writes words 0 and 1 as one pair store and words 2 and 3 as
another. RETURN reads word 1 alone. A continuation reads words 2 and 3 as one pair,
both written at CALL time, long before, so the pair never half-overlaps a fresh store
(3.2). A continuation reloads the caller's closure from the caller's own word 0.

### 7.3 Invariants

1. `base - 4` is a valid header for every frame, with the thread's bottom frame's
   header at slot 0 of the stack and its `caller` null.
2. For a Lua frame, word 3 is the pc *of the caller*. The running frame's own pc is
   in the `pc` slot and is published only at exits (7.6). For a frame below the top,
   its pc is word 3 of the frame above it.
3. The register window of a Lua frame is `[base, base + max_stack)`; its header and
   extras are below `base`; nothing of this frame is above the window except a
   multret producer's values up to `top`.
4. Frames nest: the callee's header starts at or above the caller's base, inside the
   caller's window or at its staging slot (`base + max_stack`) for metamethod calls.
5. Every word of a header is written before the callee runs (no lazy words). The
   unstarted-thread exception is section 11.1.
6. A native frame's window is `[base, top)`; `at` points inside it; the awaited call's
   header, if any, is at `base + at`.
7. Header words are not `Value`s. Nothing may read them as values; the stack tracer
   skips them (14.3); `Stack` (the native view) never exposes them.

### 7.4 Varargs

On entry to a vararg function with `nv > 0` extras, the arguments are
`[H][p_0 .. p_np-1][e_0 .. e_nv-1]` from `base - 4`. The entry rotates the block
`[base - 4, base + np + nv)` so that it reads `[e_0 .. e_nv-1][H][p_0 .. p_np-1]`,
then `base += nv` and word 0 gets `nv`. `VARARG` reads `[base - 4 - nv, base - 4)`.
The original function slot, where results land, is `base - 4 - nv`; no handler on
the return path needs it because continuations derive the destination from the
caller's instruction (8.5). `VARARGPREP`'s materialized vararg table is unchanged.

TAILCALL from a vararg frame keeps `nv` in word 0 so the frame's results still land at
the original slot; TAILCALL *to* a vararg callee is a slow path that relocates the
header to `F + nv' * 8` where `F` is the original slot (8.6).

### 7.5 Stack growth

`grow_slots` reallocates the `Vec`. Everything that holds a pointer into the stack is
rebased: open upvalues (as today), every header's `caller` word (walk from the
published top frame down the chain), the published `thread.base`, and
`rt.stack_end`. The handler that triggered growth recomputes its `base` from an index
it took before growing. Growth is confined to the cold handlers `call_grow`,
`enter_grow`, `vararg_grow` and the native paths; a fast handler never grows.

`top` stays an index, so it needs no rebasing.

### 7.6 Publication (D10)

The thread records, for the top frame only: `top_base: *mut Value`, `top_pc:
*const Instruction`, plus `top: usize` as today. Dispatch keeps them in registers and
writes them with `sync!()` at:

- every native call (the native may inspect frames, raise with a position, or grow
  the stack);
- `gc_check!` exits and every `return Exit::*`;
- `raise!`/`throw!` before unwinding;
- coroutine switches (both sides);
- the grow paths.

Readers of frame state outside dispatch (unwinder, debug walkers, `Stack::frames`,
the tracer) start at `top_base` and follow `caller` words. A debug-build assertion in
`next!()` compares the registers against the published state after any handler that
called `sync!()`.

### 7.7 Frame walking

`FrameWalk` iterates headers from `top_base` down: each item yields `base`, the
header words, whether it is native, and its pc (the published `top_pc` for the top
frame, the frame above's word 3 otherwise). It replaces `frames_rev`, `FrameRef`,
`top_lua`, `top_lua_ptr`, `frames.last()`, `Stack::lua_frames` and the executor's
depth bookkeeping. `where_prefix`, `local_name`, `frame_line` and `message_handler`
are rewritten on it (10.4).

### 7.8 ThreadState after the change

Removed: `frames`, `exec_frames`, `pending_ret`, `pending_action`, `call_limit`.
Added: `top_base`, `top_pc`, `started: bool`, `uncaught: Option<Error>`. Kept:
`stack`, `top`, `open_upvalues`, `tbc_list`, `no_yield`, `main`, `status`,
`thread_handle`, `yield_bottom` (now the yield frame's base index, section 11.3),
`death_error`, `stack_limit`, `resumer`, `resume_depth`, the async fields.
`FrameStack`, `LuaFrame`, `ExecFrame`, `ExecKind`, `CallSite`, `PendingRet`,
`PendingAction`, `PendingKind`, `frame_flags` are deleted with their uses.

## 8. Calls, returns and continuations

### 8.1 CALL

`CALL a b c` with `b = nargs + 1` (0 = MULTRET, count from `top`) and `c = wanted + 1`
(0 = MULTRET, publish `top`). `CALL_R0` and `CALL_R1` are the same with a constant
continuation. The handler, in order:

```
f    = R[a]
if !is_function(f)              → tail call_meta                 (cold)
p    = f & PTR_MASK
if kind(p) == Native            → tail (p.entry) with closure slot = p
hdr  = base + a                 (the function slot)
nb   = hdr + 4
if nb + p.max_stack > rt.stack_end → tail call_grow              (cold)
if b <= p.fixed_arity           → tail call_fixup                (missing params, varargs, MULTRET)
hdr[0], hdr[1] = p, rets[c]      (pair store; CALL_R0/R1: constant handler)
hdr[2], hdr[3] = base, pc        (pair store)
closure = p; base = nb; pc = p.code
next
```

Estimate: about 29 instructions including dispatch, against 39 today. The savings
are the frames-top store, the separate caller-pc store, the stack-base reload, and
the five single stores becoming two pairs. `fixed_arity` keeps today's meaning
(`num_params`, or 255 for a vararg function, so a vararg callee always takes
`call_fixup`).

`call_fixup` nil-fills missing parameters, computes `nv` for a vararg callee, does
the vararg rotate (7.4), writes the header with `nv` in word 0, and enters. `call_grow`
grows (7.5), recomputes `base`, and re-tails `CALL`. `call_meta` resolves the
`__call` chain (shifting arguments up one slot and inserting the callable, as today),
then re-tails `CALL`.

### 8.2 RETURN

`RETURN0` and `RETURN1` test nothing. The assembler already emits them only in
functions where no local is cell-captured and no to-be-closed variable exists
(`src/compiler/defs.rs`, the `RETURN0`/`RETURN1` rewrite), so `HAS_OPEN` and
`HAS_TBC` can never be set on their frame, and `NATIVE`/`PROTECTED`/`HANDLER` are
native-frame flags. A debug assertion checks word 1's low bits are zero.

```
RETURN1 a:   ret = hdr[1]; pc = &R[a]; insn = 1; become ret
RETURN0:     ret = hdr[1]; insn = 0;            become ret
```

About 6 instructions. `base` is left as the callee's base; the continuation uses it
to find the header.

Generic `RETURN a b` tests the flags and goes to `return_close` if any is set
(closes upvalues from `base`, runs `__close` calls through `ret_return` as today).
`b = 0` reads `top` for the count.

### 8.3 `enter`

`enter` is the shared "call from a written header" path, reached by tail call with
the header pointer in the `pc` slot and the argument count in `insn`. The header's
words 1 to 3 have been written by the stager (continuation, caller base, caller pc);
word 0 holds the callee as a raw `Value` so `enter` can check it:

```
v = hdr[0]
if !is_function(v)              → tail enter_meta                (cold: __call chain)
p = v & PTR_MASK; hdr[0] = p
if kind(p) == Native            → tail native_enter with closure slot = p, pc slot = hdr, insn = nargs
if hdr + 4 + p.max_stack > rt.stack_end → tail enter_grow
if nargs < p.num_params or vararg → tail enter_fixup
closure = p; base = hdr + 4; pc = p.code
next
```

CALL does not go through `enter`; it inlines the Lua arm with its own operand
decoding (8.1). Metamethod sites, TFORCALL's iterator call, `__close` calls, the
unwinder's runner frames, the thread entry trampoline and `CallThen` all stage a
header and tail `enter`. Writing word 0 twice (raw value, then pointer) costs one
store on these paths and keeps the stager free of the tag check.

### 8.4 Metamethod staging

A metamethod call from an opcode handler stages above the window:

```
hdr = base + closure.max_stack      (slot index base + max_stack; the scratch area)
if hdr + 4 + 3 > rt.stack_end       → tail <op>_grow (grow, then retry the instruction)
hdr[0] = mm (raw Value)
hdr[1] = <continuation for this site kind>
hdr[2] = base
hdr[3] = pc
hdr[4..] = arguments
tail enter with pc = hdr, insn = nargs
```

The continuation is a constant chosen by the site: `ret_store_a` (arithmetic, unary,
concat, `__index`), `ret_discard` (`__newindex`), `ret_cond` (comparisons),
`ret_tfor` (iterator), `ret_close` (CLOSE's `__close`), `ret_return` (RETURN's
`__close`). The metamethod lookup is two loads through the shape's metamethod cache
(`shape.mt_cache()` then `mm_at(index)`), as the quickened arithmetic forms do today.

Estimate for a quickened arithmetic metamethod site, handler side: about 55
instructions (guard 8, lookup 3, header 4, arguments 2, `enter`'s Lua arm 14,
callee's RETURN1 6, `ret_store_a` 18) against 104 traced today.

### 8.5 Continuations

A continuation receives `nret` in `insn`, the first result's address in `pc`, and the
callee's base in `base`. It never needs the callee's closure. Shape of `ret_call1`:

```
(caller, cpc) = ldp hdr[2], hdr[3]
call = *(cpc - 1)
R'[call.a] = nret > 0 ? *values : nil
closure = *(caller - 32) & PTR_MASK
base = caller; pc = cpc
next
```

About 15 instructions. The destination comes from the caller's CALL instruction, so
the vararg count is never on the return path. Generic `ret_call` lands `min(nret,
wanted)` values and nil-fills the rest; for `c = 0` it publishes `top = dst + nret`.
Fixed-count continuations do not touch `top` (today's rule; the collector bounds the
trace by the window, 14.3).

The complete set: `ret_call`, `ret_call0`, `ret_call1`, `ret_call2`, `ret_store_a`,
`ret_discard`, `ret_cond`, `ret_tfor`, `ret_close`, `ret_return`, `ret_native`
(9.4), `ret_pcall`, `ret_xpcall` (10.3), `ret_coroutine_end` (11.3), `ret_exit`
(11.1). `ret_cond` keeps today's two-arm branch form.

### 8.6 TAILCALL

Fast path: the callee is a Lua function, not vararg; the current frame's word 1 has
no `HAS_OPEN`/`HAS_TBC` flag; the callee's window fits from `base`. Then: copy the
arguments down from `base + a + 4` to `base` (ascending copy, destinations below
sources), nil-fill missing parameters, `hdr[0] = p | (hdr[0] & NV_MASK)` (keep the
frame's own `nv` so results still land at the original slot), `pc = p.code`,
`closure = p`, dispatch. Words 1 to 3 are untouched: the callee returns to this
frame's caller.

Native callee: `tailcall_native` runs the native on the window `base + a + 4 .. top`
and then `become hdr[1].ret` with `nret = top - window`, `pc = window`, `base` as it
is: the results are returned from this frame without copying and without
re-dispatching a RETURN. An action result converts this frame into the native's frame
(9.4) in place, keeping words 1 to 3.

Everything else (`__call` chain, open upvalues or tbc to close, stack growth, vararg
callee relocating the header to `F + nv' * 8` with `F = hdr - nv * 8`) is
`tailcall_slow`.

### 8.7 MULTRET and `top`

Unchanged in meaning: producers with the 0 sentinel (`CALL c = 0`, `VARARG`,
`RETURN b = 0`, native results) publish `top`; consumers (`CALL b = 0`, `RETURN b =
0`, `SETLIST`, `TAILCALL b = 0`) read it. `top` lives in the thread; on x86-64 that
is a load of the thread pointer first. The fixed-count paths never touch it.

### 8.8 Budgets

Fast-path instruction budgets the stage gates check (section 16), dispatch included:

| Path | Budget |
|---|---|
| CALL_R1, Lua callee, arity satisfied | 32 |
| RETURN1 | 8 |
| ret_call1 | 16 |
| enter, Lua arm, from a staged header | 18 |
| TAILCALL, Lua callee fast path, 2 args | 36 |

## 9. Natives

### 9.1 Kinds and signatures

Public: plain natives `fn(ctx, &NativeClosure, Stack) -> Result<(), Error>` and
async natives (`AsyncFn`, unchanged). Crate-internal: continuation natives, which
return a `NativeOut` and name a continuation by index. This is the split decided on
2026-10-05; the only API change for public natives is none, and `Context` shrinking
to one word is invisible to them.

### 9.2 `NativeOut`

One `u64` returned in a register, tag in the low 3 bits:

```
0 Return     results are window .. top; no payload
1 Error      the Error pointer in bits 3..63 (8-aligned)
2 CallThen   protect in bits 3..4, at in bits 48..55, cont_idx in bits 56..61,
             ok in bits 62..63
3 Resume     at, cont_idx, ok as above
4 Yield      no payload
5 YieldThen  at, cont_idx as above
6 Async      no payload
7 Pending    no payload
```

`at` is the window slot of the staged call (8 bits), `cont_idx` indexes the
continuation table (6 bits, 9.4), `ok` is the pass-through mode (9.5), `protect` is
none, errors, handler or base as today's `Protect`. Plain natives return
`Result<(), Error>` (already 8 bytes) and the entry converts: `Ok` is 0, `Err(e)` is
`e | 1`. `vm/native.rs` owns the constructors and accessors; nothing else touches
the bits.

### 9.3 Entries

`native_call` is the generic `NativeClosure::entry` reached from CALL and TAILCALL
with the native closure in the `closure` slot:

```
call = *(pc - 1)                      (a, b, c)
hdr  = base + call.a; win = hdr + 4
nargs = b ? b - 1 : top - win
sync!()                               (top_base, top_pc; the native may look)
hdr[0], hdr[1] = nc, rets[c]; hdr[2], hdr[3] = base, pc
top = win + nargs
out = f(rt, nc, Stack { thread, bottom: win })
if out == 0  → become rets[c] with nret = top - win, pc = win, base = win
             (the generic landing; the frame is "returned from")
else         → tail native_act with insn = out, pc = hdr          (cold)
```

`native_enter` is the same from a staged header (`enter`'s native arm): the header
is written, `nargs` is in `insn`, the window is `hdr + 4`. Both share one
`#[inline(always)]` body. The entry handler has a stack frame because it calls the
native; nothing else about it is heavy: no result copy loop (the continuation lands
results), no re-read of the function slot, no `CallbackAction` through memory.

A `TAILCALL` of a plain native (8.6) returns through the *current* frame's header
and writes no header of its own. Its error position therefore names the Lua frame,
as today and as PUC does for a tail-called C function's `luaL_where(1)`. Listed in
section 18.

Estimate: native `CALL_R1` plus landing, excluding the native's body, about 50
instructions against about 70 today. The two pair stores for the header are the cost
accepted in D5; they buy one landing path and uniform frames.

### 9.4 Native frames and `ret_native`

When a native returns `CallThen`, `Resume`, `YieldThen` or `Async`, `native_act`
converts the native's frame in place: word 0 gets `at`, the continuation index and
`ok`; word 1 gets `NATIVE` plus `PROTECTED`/`HANDLER` from `protect`; words 2 and 3
stay (the caller is unchanged). Then it makes the requested call: the callee sits at
`win + at` with its arguments after the hidden slots (`win + at + 4 ..`), so the
native staged it in call layout; `native_act` writes that header's words 1 to 3
(`ret_native`, `win`, null pc) and tails `enter`.

`ret_native` is the continuation of a call a native frame made. It receives the
callee's base; `hdr[2]` of that callee is the native's window `win`, and the native
frame's header is `win - 4`:

```
nf = win - 4; at, cont_idx, ok = unpack(nf[0])
copy results to win + at .. ; top = win + at + nret
if ok != Cont  → pass-through (9.5)
out = CONT_TABLE[cont_idx](rt, nc, Stack { thread, bottom: win }, Ok(()))
if out == 0    → become nf[1].ret with nret = top - win, pc = win, base = win
else           → tail native_act with insn = out, pc = nf
```

`native_act` for a frame that already is a native frame updates word 0 and makes the
next call. There is no `framed: bool`, no `drive_natives`, no `NativeState`,
`NativeStep`, `run_natives!` or `native_step!`.

The continuation table (`CONT_TABLE: [NativeCont; N]`) lists the crate's
continuation functions: `pcall_cont`, `xpcall_cont`, `resume_cont`, `wrap_cont`,
`wrap_close_cont`, the sort and gsub steps, `tostring`, `pairs`, `dofile`, string
arithmetic `trymt`, the close runners (`close_entry_cont`, `close_running_cont`),
`handler_cont`, `close_cont`, `async_cont`. Six bits allow 64; a `const` assertion
checks the count. `NativeCont`'s signature is unchanged except for returning
`NativeOut`.

### 9.5 Pass-through modes

`ok` in word 0: `Cont` (call the continuation), `Return` (the call's results are the
native's results), `ReturnTrue` (prepend `true`). `ret_native` handles `Return` and
`ReturnTrue` without calling the continuation, writing `true` into the slot below
the results, which is the finished callee's function slot and free. Used by
`coroutine.wrap` and `coroutine.resume` (11.2) and `pcall`/`xpcall`'s native fallback
when the fast entry does not apply.

### 9.6 Fast entries

`NativeClosure::entry` keeps its role: a builtin may provide a handler that handles
the common argument shape inline from the CALL's registers and never errors, falling
back to `native_call` for everything else. Today's `ff_sqrt`, `ff_sin`, `ff_cos`,
`ff_abs`, `ff_floor`, `ff_ceil`, `ff_pairs`, `ff_ipairs`, `ff_pcall`, `ff_xpcall`,
`ff_resume`, `ff_wrap`, `ff_yield` stay. A `builtin!` macro declares the Rust
function and the entry together and wires the fallback (`tail!(native_call)` with the
native closure already in the `closure` slot), so adding a fast entry is one item.
Fast entries write no header when they complete inline; `ff_pcall`, `ff_xpcall`,
`ff_resume`, `ff_wrap` and `ff_yield` write headers because they create frames or
suspension points (10.3, 11.2, 11.3).

Candidates for new fast entries, to be justified by the histogram, not added
blindly: `type`, `select('#')`, `rawget`/`rawset`, `string.sub`, `string.byte`,
`table.insert` (append form), `math.max`/`math.min` with two numbers, `tostring` of
a string or integer.

### 9.7 Errors from natives

A plain native's `Err(e)` arrives as `NativeOut` tag 1; the entry publishes the
frame state (already done before the call) and tails `impl_error` with the error.
The native's frame exists (its header was written), so `locate` counts it as level 0
and the Lua caller as level 1, which is PUC's numbering. `native_overflowed()`
(results pushed past the limit without `check_stack`) is checked in the entry as
today and converted to the stack overflow error.

### 9.8 Async natives

`invoke_async` keeps its structure: spawn, first poll inline from the frame, requests
through `Env`. The frame is the native's own header converted with `cont_idx =
ASYNC_CONT`; `Pending` publishes and returns `Exit::Pending`; the trampoline re-polls
through `ret_native` on the next `enter`. `Env.base` is the window index as today.
The task arena, `Local` roots and epochs are unchanged.

## 10. Errors and unwinding

### 10.1 Raising

`raise!(kind)` stores the `OpError` into `rt.fault` and tails `impl_error`;
`throw!(err)` stores an `Error` the same way. `impl_error` publishes the frame state,
renders the message (10.4) and runs the unwinder. The fault cell is cold, typed and
consumed immediately; it exists because an `OpError` holds up to two values and a
discriminant, which no slot carries, and because errors are not worth a wider ABI.

### 10.2 The unwinder

`unwind(rt, thread, err)` walks headers from `top_base`:

```
loop over frames from the top:
  if NATIVE:
      if PROTECTED (and the error is not an exit the frame doesn't catch):
          catch: close detached tbc first if any (runner frame), else
          out = CONT_TABLE[cont_idx](rt, nc, Stack { win }, Err(err))
          continue as ret_native would with `out`
      else pop (drop its task if async)
  else (Lua):
      if ret == ret_pcall or ret == ret_xpcall:
          catch: see 10.3
      close upvalues >= base; detach tbc entries at base; pop
  bottom reached: coroutine death or host (11.3, 11.5)
```

"Pop" means `base = hdr[2]`; nothing is stored, the frames above are dead. The
message-handler search (`luaD_throw`'s `errfunc`) walks the same chain looking for the
nearest `PROTECTED|HANDLER` native frame or a `ret_xpcall` frame, whose handler sits
in the slot below that frame's header (10.3). The runner frames for the message
handler and for `__close` of detached variables are native frames pushed at
`live_top` with the `unwind` native in word 0 and continuations `handler_cont`,
`close_cont` from the table; their logic is today's `unwind.rs` unchanged except for
how frames are read and written. `Protect::Base` (the coroutine close runner catching
exits) becomes the `PROTECTED` flag plus the runner's own continuation deciding on
exit kinds, as today.

### 10.3 pcall and xpcall without frames

`ff_pcall` at `CALL a b c`: the callee `f` is at `R[a+4]` and its arguments at
`R[a+5]..`. The callee's header is written at `R[a+1]..R[a+4]`, the three hidden slots
plus `f`'s own slot, so the callee's base is `R[a+5]` and its arguments are already
in place; nothing moves. Word 1 is `ret_pcall`, word 2 `base`, word 3 `pc`. Then the
Lua arm of `enter` inline. `pcall`'s own function slot `R[a]` is untouched and the
continuation finds the CALL through word 3 as usual.

`ret_pcall` writes `true` into the slot below the results (the callee's function
slot, dead), and continues through `rets[call.c]` of the CALL at `cpc - 1` with
`nret + 1`, as today.

`ff_xpcall` at `CALL a b c`: `f` at `R[a+4]`, handler at `R[a+5]`, arguments from
`R[a+6]`. The entry moves the handler to `R[a+1]` and writes the callee's header at
`R[a+2]..R[a+5]`, base `R[a+6]`, word 1 `ret_xpcall`. The unwinder reads the handler
from `hdr - 1` of a `ret_xpcall` frame.

A catch at a `ret_pcall`/`ret_xpcall` frame: the unwinder writes `false, errv` at
the dead callee base `b`, then `become rets[call.c]` with `nret = 2`, `pc = b`,
`base = b`, which lands them at the CALL's destination. For `c = 0` the continuation
publishes `top`. No frame is re-materialized and no `protected_frame`,
`restore_protected` or `FrameRef::Elided` exists.

Callees that are not Lua functions, or windows that do not fit, go to the native
`pcall`/`xpcall`, which call with `ok = ReturnTrue` through a native frame (9.5).

### 10.4 Locations and messages

`where_prefix(level)` walks `FrameWalk`: level 0 is the top frame when it is native
(the raiser), level 1 the frame below, and so on; a native frame at the requested level
yields the empty prefix, as PUC does for C functions. `frame_line` uses the frame's
pc (7.7). `op_error_message` is unchanged except that the top frame is read through
the walker; the variable-name suffix stays unimplemented. Tail-called plain natives
have no frame (9.3), so `error("x")` in tail position names the Lua caller's line,
which is what PUC prints too.

### 10.5 Exits

`Exit::End` after an uncaught error leaves `thread.uncaught = Some(err)` and
`status = Stopped`; the executor reads it where it read `ExecKind::Error`. Process
exits (`os.exit`) and the coroutine `Exit::Clean`/`Failed` kinds keep their current
handling in the unwinder and the executor.

## 11. Coroutines and the executor boundary

### 11.1 Thread entry

`Executor::start(f, args)` seeds a stack as `[f, _, _, _, args...]` with `started =
false`, status `Suspended`. The first `enter` from the trampoline writes header 0:
word 1 `ret_exit`, word 2 null, word 3 null, and tails `enter` with `pc = hdr 0`,
`insn = nargs`. A native `f` goes through `native_enter` like any other. `ret_exit`
copies results to slot 0, sets `top`, status `Result { bottom: 0 }`, returns
`Exit::End`. `ExecKind::Start` and `schedule_call_at` are deleted.

### 11.2 Resume in dispatch

`ff_resume` and `ff_wrap` (fast entries) apply when the target is suspended at a
yield frame (11.3), the depth limit allows, and the values fit. They write the
resumer's native frame header at `base + a`: word 0 `nc | ok << 62 | cont_idx <<
56` with `ok = ReturnTrue` for resume and `Return` for wrap, word 1 `rets[c] |
NATIVE | PROTECTED`, words 2 and 3 `base`, `pc`; publish the resumer; copy the
arguments into the target's yield window; switch (`rt.thread`, `rt.stack_end`, the
`thread` slot, statuses, `resumer`, `resume_depth`); then `become` the yield frame's
`hdr[1].ret` with `nret = n`, `pc = window`, `base = yield base`. Anything else goes
to the native `resume`/`wrap`, which return `NativeOut::Resume` and are handled by
`native_act` as a switch with the same frame shape.

### 11.3 Yield in dispatch

`ff_yield` at `CALL a b c`, when the resumer waits in dispatch on a native frame
whose `ok` is a pass-through mode: write a header for the yield frame at `base + a`
(word 0 the yield native, word 1 `rets[c]`, word 2 `base`, word 3 `pc`), set
`yield_bottom = Some(base + a + 4)` (the yield frame's base index), status
`Suspended`, publish; copy the values into the resumer's native window at `at`; switch
to the resumer; then act as `ret_native` does for that frame (pass-through writes
`true` below the values for `ReturnTrue`), i.e. `become nf[1].ret`. Resuming later
(11.2) delivers values through the yield frame's `hdr[1].ret`, which is the CALL's
own continuation: the yield behaves as a call that returned the resume arguments.

Yields the resumer cannot take in dispatch (resumer not waiting there, `no_yield`,
the main thread yielding to the host) go to the native `yield`, which returns
`NativeOut::Yield`; `native_act` sets `yield_bottom`, publishes, and returns
`Exit::End` with status `Suspended`. The executor reports `Yielded(values)`.

`ret_coroutine_end` is the bottom continuation of a coroutine body: status
`Result`, hand the results to the resumer's native frame as 11.3 does, or `Exit::End`
if the resumer is not in dispatch. An error reaching a coroutine's bottom frame with
a resumer waiting in dispatch switches to the resumer and delivers `Err` to that
native frame's continuation (`resume_cont` makes `(false, err)`), as today's unwinder
does.

### 11.4 Host boundary

`Executor::resume(args)` after a host yield writes the arguments into the yield
window and calls `dispatch::enter`, which sees `yield_bottom` and calls the yield
frame's `hdr[1].ret` with the count: no `pending_ret`, no `land_call_results`,
no `deliver`. `propagate_inner_to_resumer`, `schedule_thread_resume`,
`apply_pending_action`, `follow_switches` and `run_natives` in the executor are
deleted; the executor's thread stack is derived from the `resumer` chain when the
host needs it (`Execution::is_main`, `coroutine.running`).

### 11.5 Limits

`MAX_RESUME_DEPTH = 200` is enforced by `ff_resume` and `native_act`'s resume arm on
`resume_depth`. `no_yield` and `coroutine.close`'s seeded close (`close.rs`) keep
their logic: `seed_thread_close` seeds the thread with the close entry native as
11.1 seeds any thread.

## 12. Instruction set

### 12.1 Word format

Unchanged: a 64-bit word, opcode in the low byte, operands `a b c` as bytes 1 to 3,
and either `d:u16 e:u16` or `imm:i32` in the high half (`src/instruction.rs`). The
`instructions!` table stays the single declaration of opcodes, constructors and
shapes; the dispatch table is built from it with `Op::table`, so a missing handler
is a build error.

Changes:

- **Adaptive bits.** Each specializable shape has a documented spare field holding
  `misses: u2` and `locked: u1`:

  | Shape | Where | Used by |
  |---|---|---|
  | `Abc` | bits 32..34 (`d` low byte) | register arithmetic, bitwise |
  | `AbcImm` | `c` bits 1..3 (`c` bit 0 is `flipped`) | immediate arithmetic |
  | `Abde` | `c` byte | constant-key table access (replaces today's megamorphic count) |
  | `AbImm` | `c` byte | register compares |
  | `AhImm` | bits 32..34; the branch offset narrows to 24 bits in bits 40..63 | immediate compares |
  | `AImm`, `Ab` | `b` or `c` byte as free | FORLOOP, TFORCALL (no counter: chosen by the prep instruction, guarded) |

  `AhImm` gets `imm24()`/`set_imm24()`; the assembler rejects a conditional branch
  whose offset exceeds 23 bits with the "control structure too long" compile error.
- **Metamethod forms** carry the original opcode in `e`'s low byte (`Abc`) or as a
  4-bit index into the immediate-op list in `c` bits 4..7 (`AbcImm`).
- `Instruction::quickenable`, `with_no_quicken`, `with_mm_form`, `mm_form_op` and
  `Op::unquickened`/`num_form`/`is_reversed` are replaced by `OP_INFO` (12.5) and
  three accessors: `adaptive()`, `with_adaptive(misses, locked)`, `with_op`.
- **SELF** writes the receiver to `R[dst + 4]`, the first argument slot under the
  hidden-slot layout (13.1).

### 12.2 Mechanism (D8)

Every polymorphic family has one generic handler, shared by all opcodes of the
family and reached both by dispatch (the compiler emits generic opcodes) and by the
guard failure of any specialized form. It:

1. reloads the instruction from `pc - 1` and reads the opcode byte: a specialized
   opcode means this is a miss, a generic one a first execution or a locked site;
2. executes the operation completely, for every type, including metamethods
   (staging and `enter`) and errors;
3. if the site is not locked: on a miss increments `misses` and, at 3, writes the
   generic opcode with `locked`; otherwise computes the specialized form for the
   operand kinds it saw (`OP_INFO[op].form(kinds)`) and writes it if one exists, else
   locks the site.

A specialized handler is: load operands, one guard combining its kind tests into one
branch, the operation, the store, dispatch. Its guard failure is `tail!(generic)`.
Specialized forms keep the generic form's operand layout, so a rewrite is a store of
the word with a new opcode byte and adaptive bits; the store is one aligned 64-bit
write into the `Code` cell and any form is correct for any operand types (guards),
so interleaved execution of the same code by coroutines is safe.

Policy constants: `MISSES_TO_LOCK = 3`. A locked site stays generic for the life of
the prototype. These are tunable by the histogram (12.7).

### 12.3 Tier 1 opcodes

197 opcodes. The table order is the numbering; specialized forms follow their
generic so `OP_INFO` is table-driven. "guard" is what the specialized handler tests
in one branch; `small` means an inline i32, `float` a non-boxed double.

**Control and miscellaneous (20).** `NOP STOP JMP CALL CALL_R0 CALL_R1 TAILCALL
RETURN RETURN0 RETURN1 CLOSURE VARARG VARARGGET VARARGPREP CLOSE TBC ERRNNIL
SETLIST NEWTABLE CONCAT`. Semantics as today except CALL's layout (13.1).

**Loads and moves (5).** `MOVE LOAD LFALSESKIP` as today; `LOADI a imm` sets
`R[a]` to the small int `imm` (the compiler emits it for integer constants that fit
i32); `LOADNIL a b` sets `R[a .. a+b)` to nil.

**Upvalues (3).** `GETUPVAL GETUPVAL_REF SETUPVAL` as today.

**Unary (4).** `UNM BNOT NOT LEN` as today (float-first UNM, small-first BNOT).

**Loops (9).**
- `FORPREP FORLOOP TFORPREP TFORCALL TFORLOOP` as today.
- `FORLOOP_I`: guard `step`, `idx`, `last` all small (one `and` chain and compare);
  integer step with the `idx != last` termination; written by `FORPREP` into the
  loop's FORLOOP when its three values are small. `FORLOOP_F`: guard `step` float;
  written when the loop is a float loop. Both bail to `FORLOOP`, which handles boxed
  ints and anything `debug.setlocal` did to the hidden slots. `FORPREP` rewrites the
  FORLOOP at `pc + offset - 1` on every execution (one store).
- `TFORCALL_NEXT`: guard `R[base+1]` is a table and `R[base+3]` is a non-negative
  small int; the inline `next` step, today's `op_tforcall` next arm. `TFORCALL_IPAIRS`:
  guard table and the ipairs marker. Written by `TFORPREP` into the TFORCALL at its
  jump target; both bail to `TFORCALL`.

**Register-key tables (2).** `GETTABLE SETTABLE` as today (array fast path inline,
general path and metamethods in `gettable_generic`/`settable_generic`).

**Constant-key get (22).** `GETFIELD GETTABUP SELF` generic, each with six forms,
plus `GETTABUP_REF`:

| Form | Guard | Fast path |
|---|---|---|
| `_INL` | receiver is a table, live shape == cached shape | load the inline slot, store |
| `_INL_MT` | same | load, nil test, on nil fall to generic (the shape has `__index`) |
| `_AUX` | same | load the spill cell pointer, load the slot, store |
| `_AUX_MT` | same | as `_AUX` with the nil test |
| `_ABSENT` | same | nil; if the shape's `__index` is a function, stage its call inline (today's behaviour) |
| `_PROTO` | receiver shape == cached, metatable's `__index` is still the cached holder, holder's shape == cached | load the holder's slot; nil falls to generic |

The IC table (`Prototype::ic_table`, `InlineCache`) stays the shape cache; the
form only fuses what the entry already says (slot kind, metatable presence) into the
opcode so the hit path has no slot-kind branch and no `has_mm` test when the shape
has no `__index`. `GETTABUP` reads the receiver from a by-value upvalue, `SELF` also
writes the receiver to `R[dst+4]`. Estimate: 26 to about 23 on `_INL`.

**Constant-key set (11).** `SETFIELD SETTABUP` generic, each with four forms, plus
`SETTABUP_REF`: `_INL` and `_AUX` (own slot exists; a nil old value on a shape with
`__newindex` falls to generic), `_TRANS` (cached from-shape, room in the cell: push
the key, move to the to-shape), `_ABSENT` (shape lacks the key and `__newindex` is a
function: stage its call). The write barrier is the retry form of 14.2.

**Register arithmetic (30).**
- Generic: `ADD SUB MUL DIV MOD IDIV POW`.
- `_II` for `ADD SUB MUL DIV MOD IDIV`: guard both small (`and`, compare, one
  branch); integer op with overflow check for add/sub/mul, zero divisor to generic
  for mod/idiv; `DIV_II` converts both and divides as floats.
- `_FF` for all seven: guard both float (compare, `ccmp`, one branch); the hardware
  op; unchecked store (D9). `MOD_FF` and `POW_FF` call libm and keep the NaN check.
- `_IF` and `_FI` for `ADD SUB MUL DIV`: guard one small and one float (compare,
  `ccmp`, one branch); one `scvtf`; the float op; unchecked store.
- `ARITH_MM`: guard lhs is a table whose shape cache has the metamethod for the
  original opcode; stage and `enter`. `ARITH_MM_R`: rhs table, lhs a number, numbers
  have no metatable.

Estimates: `_II` and `_FF` about 17 (21 today), `_IF`/`_FI` about 18 (about 30 today
through `_NUM`).

**Register bitwise (10).** `BAND BOR BXOR SHL SHR` generic and `_II` each. Floats
with integral values and metamethods go through `arith_generic`.

**Immediate arithmetic (33).** Generic `ADDI SUBI MULI MODI POWI DIVI IDIVI RSUBI
RMODI RPOWI RDIVI RIDIVI`. Forms by (register kind, immediate kind):
- `_I` (small register, integer immediate, integer result): `ADDI SUBI MULI MODI
  IDIVI RSUBI`.
- `_F` (float register, immediate of either kind converted in one or two
  instructions): `ADDI SUBI MULI MODI IDIVI RSUBI POWI DIVI RDIVI`.
- `_IF` (small register, float immediate, or integer immediate for a float-result
  op): `ADDI SUBI MULI DIVI RDIVI`.
- `ARITH_MMI`: the register operand is a table with the metamethod.
Estimate: `ADDI_I` about 9 (12 today).

**Immediate bitwise (12).** Generic `BANDI BORI BXORI SHLI SHRI RSHLI RSHRI`; `_I`
for the first five.

**Compares (36).** The 22 current branch opcodes stay generic. Forms:
- `JLT_II JNLT_II JLE_II JNLE_II`: guard both small; integer compare.
- `JEQ_II JNEQ_II`: guard both small.
- `JLTI_F JNLTI_F JLEI_F JNLEI_F JGTI_F JNGTI_F JGEI_F JNGEI_F`: guard float
  register; the 15-bit immediate converted with one `scvtf`.
The int-first immediate compares already have one guard for the small case, so no
`_I` forms. Mixed register compares are tier 2 (12.7).

### 12.4 Canonical NaN (D9)

`Value::float(f)` maps any NaN (`f != f`) to `CANONICAL_NAN`; this replaces the
box-space test and also catches signalling NaNs, which could otherwise quiet into
box space. Every float that enters the VM goes through `Value::float` or a path that
guarantees canonical NaN (immediates cannot be NaN; `read_float` of a slot that holds
a canonical float; results of hardware add, sub, mul, div, neg, abs of canonical
inputs, which propagate a canonical payload or produce the default NaN, both outside
box space because tag 0 is unused). Therefore `write_float` becomes a plain store for
those operations. `fmod`, `pow` and any other libm result keep a check, as do
conversions from bits (`string.unpack`), which already go through `Value::float`.
A debug assertion in `write_float_unchecked` verifies the stored bits are not in
box space.

### 12.5 `OP_INFO`

A `static OP_INFO: [OpInfo; 256]` built `const` from the table: for every opcode its
generic opcode, its family, the arithmetic kind and metamethod index, whether it is
the swapped immediate form, the specialized form for each operand-kind pair, and
the location of its adaptive bits. The generic handlers consult it instead of
matching on opcodes; the listing (`compiler/format.rs`) prints specialized forms by
their generic name with a suffix.

### 12.6 `family!`

The specialized arithmetic, bitwise and compare handlers are generated from rows:

```rust
family! {
    arith_reg: generic = arith_generic;
    ADD_II = (ADD, Small, Small) => int_add,
    ADD_FF = (ADD, Float, Float) => float_add,
    ADD_IF = (ADD, Small, Float) => float_add,
    ...
}
```

Each row produces one `handler!` item whose body is the guard for the two kinds,
the named operation, the store, dispatch. The guard and conversion code for each
kind pair is written once in `arith.rs` as `#[inline(always)]` functions; the table
form keeps the source to a few lines per opcode and makes the ISA table and the
handler set the same thing.

### 12.7 Histogram and tiers 2 and 3

A `cfg(feature = "op-stats")` build counts dispatches per opcode in `next!()` into
an array in the runtime; the CLI prints it sorted with `--op-stats`. Over the
benchmark corpus (section 16, stage 0) and the test suites it answers which forms
are hot, which generics stay hot after warm-up (sites that lock), and whether the
counters flip.

Tier 2 candidates, each needing a histogram showing the site hot and monomorphic:
mixed register compares (`JLT_IF`/`_FI` and friends, 8), dict-mode `GETFIELD`/
`GETTABUP` forms probing the hash part inline (2), `JEQI_F`/`JNEQI_F` (2),
`RPOWI_F` (1). Tier 3: a shape id embedded in the instruction instead of the IC
table load (needs u32 shape ids and a pc-keyed IC lookup on misses), array forms of
`GETTABLE`/`SETTABLE`, pre-decoded operands in the dispatch tail on aarch64 (an ABI
experiment, not an ISA one). 59 opcodes remain free.

## 13. Compiler changes

### 13.1 Hidden call slots

`compile_expr_func_call` (and the method-call and statement-call paths that share
it) allocates the function register, then reserves three registers, then the
arguments. `CALL a b c` keeps its meaning with `b = nargs + 1`; the handler finds
the first argument at `a + 4`. `SELF dst obj key` writes the method to `R[dst]` and
the receiver to `R[dst+4]`. `TAILCALL` likewise. Multiple-result producers feeding a
call (`f(g())`) still work because the inner function slot is exactly where the
outer call's arguments start. The per-function register limit is unchanged; a
function that overflows 255 registers because of nesting gets the existing "too
many registers" error, as in LJR. Snapshot tests of listings change accordingly.

`TFORCALL` keeps its layout (`base`, `base+1`, `base+2`, `base+3`, variables from
`base+4`); the generic handler stages the iterator call above the window as a
metamethod call.

### 13.2 New emitted forms

- `LOADI` for integer constants in i32 range; `LOAD` keeps strings, floats, big
  integers, booleans and nil as today unless `LOADNIL` applies.
- `LOADNIL a b` for `local a, b, c` and explicit nil runs.
- `FORLOOP_I` emitted directly when the loop's init and step are integer literals
  (stage 5, optional); otherwise `FORPREP` decides at run time.

### 13.3 Branch offsets

`AhImm` branches carry a 24-bit offset; `emit_jump_instr`/`patch_to` go through
`set_imm24` for those opcodes and the assembler raises "control structure too long"
past the range. Other shapes keep 32 bits.

### 13.4 Assembler

The `RETURN0`/`RETURN1` rewrite (close check and vararg exclusion) is unchanged and
is what makes 8.2 sound; add an assertion in the interpreter's debug build. The
`_REF` upvalue rewrites are unchanged. `Op::unquickened()` callers in the listing use
`OP_INFO`.

### 13.5 Unchanged

The constant-key IC allocation (`alloc_ic_slot`, one entry per site), the templates
for `NEWTABLE`, `TFOR_VARS`, `VARARGPREP`/`VARARGGET` emission, and the `<const>`
folding.

## 14. Garbage collector interaction

### 14.1 No collection during dispatch

Unchanged: the collector runs only between `Lua::enter` calls, so handlers hold raw
`Value`s and header words without roots. Allocating handlers (`NEWTABLE`, `CONCAT`,
`CLOSURE`, the generic arithmetic when it boxes an i64, every native call, `VARARG`
with a table) run `gc_check!()` after allocating: one load of `rt.gc_due`, one pair
load, one compare; on due, `sync!()` and `return Exit::Gc`. The executor defers the
check and reports `Pending` as today.

### 14.2 Write barriers without calls

`Gc::write_if_clean` stays the fast test. When it fails the handler tails
`barrier_retry` with the object pointer in the `closure` slot: `barrier_retry` calls
`lighten` (it has a frame; it is cold), then rewinds `pc` by one instruction and
dispatches, so the store handler runs again and now passes the test. The retry costs
one extra execution of the instruction only when the object was black, and removes
the prologue from every store handler (`SETTABLE`, `SETFIELD*`, `SETTABUP*`,
`SETUPVAL`, `SETLIST`). IC fills (`fill_ic`) run in the generic get/set handlers,
which are allowed a frame.

### 14.3 Tracing a thread's stack

`ThreadState::trace` walks frames from `top_base` (published; during tracing
dispatch is not running, so the publication is current):

```
live = max(top, top_base + top_closure.max_stack)      (native top frame: top)
nil-fill [live, stack.len())                           (dead slots, as today)
b = top_base
loop:
    hdr = b - 4
    trace hdr[0] & PTR_MASK as a Function (Lua closure or native closure)
    upper = previous frame's header start minus its nv, or `live` for the top frame
    trace [b, upper) as Values                          (registers, in-flight values,
                                                        and the callee's extras)
    b = hdr[2]; stop when null
```

Header words 1 to 3 are skipped. A native frame's window `[b, upper)` is traced the
same way. `open_upvalues`, `tbc_list`, `thread_handle`, `resumer`, the async locals
and `uncaught` are traced as today; `yield_bottom` is an index.

### 14.4 Code rewriting

`Code` keeps instructions in `Cell`s; adaptive rewrites, `FORPREP`'s and
`TFORPREP`'s choices and IC form changes write through them. A rewrite never
introduces a `Gc` pointer into the code, so no barrier is needed; cached shapes stay
in the IC table, which has its barrier in `fill_ic`.

### 14.5 Headers and `Gc` pointers

Word 0 holds a `Gc` pointer to a `FunctionKind`, kept alive by the trace above and
by the function slot it came from. Word 2 is a stack pointer (rebased on growth,
7.5). Word 3 is a code pointer into a `Prototype::code` the frame's closure keeps
alive, or a continuation fn pointer. None needs a barrier: a header is written by
the mutator into the thread's own stack, and the thread is re-grayed by the
`state_mut` barrier dispatch already emits when it starts running a thread.

## 15. Module layout and code conventions

### 15.1 Files

```
src/vm/
  mod.rs            module list, re-exports
  abi.rs            Handler, Slot, Exit, the `handler!` macro and its body macros,
                    the per-arch signature, dispatch tail, sync!/switch!
  frame.rs          header layout constants, Hdr accessors, FrameWalk, publish,
                    stack-growth rebasing, vararg rotate
  dispatch.rs       table construction (Op::table), rets table, trampolines
                    (enter, return-into, poll)
  ops/
    mod.rs          the handler table rows
    data.rs         MOVE LOAD LOADI LOADNIL LFALSESKIP GETUPVAL* SETUPVAL
    field.rs        constant-key get/set families, IC fill and form selection
    table.rs        GETTABLE SETTABLE NEWTABLE SETLIST LEN CONCAT
    arith.rs        arithmetic and bitwise families, kind guards and conversions,
                    arith_generic, arithi_generic
    compare.rs      compare families, JT JF JTSET JFSET, cmp_generic, eq_generic
    call.rs         CALL* TAILCALL RETURN* enter call_grow call_fixup call_meta
                    ret_call* budgets
    control.rs      JMP FOR* TFOR* CLOSE TBC CLOSURE VARARG* ERRNNIL NOP STOP
    meta.rs         metamethod staging, ret_store_a ret_discard ret_cond ret_tfor
                    ret_close ret_return, __index/__newindex chain walks
  native.rs         NativeOut, native_call native_enter tailcall_native native_act,
                    ret_native, CONT_TABLE, builtin! macro, fast entries ff_*
  unwind.rs         raise/impl_error, unwinder, catch points, runner frames
  coro.rs           ff_resume ff_wrap ff_yield, ret_coroutine_end, switch helpers
  async_native.rs   kept, adapted to the frame accessors
  close.rs          kept, adapted
  debug.rs          kept, rewritten on FrameWalk
  num.rs            kept; adds the kind guards' conversions and the branchless
                    mixed path
```

Target sizes: no file over 1500 lines; `interp.rs` is deleted.

### 15.2 Conventions

- Every dispatch target is declared with `handler!`; every specialized family
  member with `family!`. No hand-written `extern "rust-preserve-none" fn`.
- Shared logic is an `#[inline(always)]` function returning a small value. A
  function a fast handler calls must be inlined; the stage 0 frameless check catches
  the ones that are not.
- Cold work is a tail-called handler. `#[cold] #[inline(never)]` functions are
  called only from handlers that already have a frame (generics, entries, unwinder).
- Payload roles are documented on the target in one line: `/// slow: insn = native
  closure`.
- Comments: one line for the contract (`R[dst] = R[src]`), at most one sentence for a
  non-obvious invariant or a why-not-the-obvious-alternative. No codegen narration.
  Rationale goes in this document and commit messages.
- `unsafe` blocks carry the one-line reason, as today.
- Debug builds keep the bounds-checked register access, the publication assertion in
  `next!()`, the header-shape assertions in `FrameWalk`, the `RETURN0/1` flag
  assertion and the `write_float_unchecked` assertion.
- Formatting: `rustfmt --edition 2024 <file>` on touched files; `cargo fmt` reformats
  unrelated files in `src/compiler`.

## 16. Implementation stages and gates

Each stage is a set of commits on `ljr-prototype` (or a branch off it, the user
decides at the start). Commits inside a stage may leave tests failing; a stage's end
commit must pass its gates. Benchmarks run only where a gate names them. Compare
binaries built with `cargo build --release`, not those `cargo test --release`
produces (dev features). Alternate the two binaries run by run, at least 15 runs,
and compare distributions (3.3).

Reference binaries: keep `fd5d79b`'s release build as `tc-base` in the scratchpad or
`target/compare/` for every A/B.

### Stage 0: tooling and baselines (no interpreter change)

1. `tools/frameless.py BIN`: disassemble every symbol under `tcvm::vm::` with
   `objdump --disassemble-symbols`, fail if a function not on the allowlist contains
   a prologue (`sub sp`/`stp x29, x30` on aarch64; `push rbp`/`sub rsp` on x86-64).
   Allowlist: the generic family handlers, native entries, `native_act`,
   `impl_error`, the unwinder, grow and fixup handlers, barrier_retry, IC fill.
2. `tools/hotpath.py BIN SYM...`: count straight-line instructions from a handler's
   entry along the fallthrough path to its first dispatch `br`/`jmp`, reporting the
   count and the branches taken off it. Approximate, but it is what the budgets in
   8.8 and 12.3 are checked against.
3. `op-stats` feature and `--op-stats` flag (12.7).
4. Benchmark corpus list in `tools/bench.txt`: the LJR suite files used before
   (`ljrb.py` list), `test-files/{mandel_bench,mandel_bench2,collatz_bench,
   primes2,fft,...}.lua`, and the microbenchmarks regenerated from memory into the
   scratchpad (`call`, `callnat`, `meth`, `getf`, `setf`, `glob`, `fadd`, `iadd`,
   `mixed`, `mmadd`, `idxfn`, `pcall`, `coyield`, `gsubfn`, `sortfn`, `tostr`). The
   metamethod benches stay out of the repo (user decision 2026-09-24).
5. Record the baselines of Appendix A for the new binaries, with cycles from
   `/usr/bin/time -l` or xctrace, into `tools/baseline-fd5d79b.txt`.

Gate: the three tools run on the current binary and report; the frameless check
lists today's offenders (`op_settable` and friends) as expected failures.

### Stage 1: runtime, ABI, headers, calls

The big-bang stage; nothing works until it is complete. Order inside it:

1. `State` becomes the runtime (D4): tables at offset 0, cached fields, one-word
   `Context`. `Lua::enter` refreshes the mutation pointer. Everything compiles with
   the old interpreter still in place.
2. `abi.rs`, `frame.rs`, `dispatch.rs` and the `handler!` macro, with the old
   interpreter still compiled (the new modules are dead code until step 5).
3. Compiler: hidden slots (13.1), `SELF` destination, `LOADI`/`LOADNIL`, 24-bit
   offsets (13.3). Snapshot tests updated. The old interpreter is now wrong; from
   here tests fail until step 7.
4. `ThreadState` per 7.8; `Executor` per 11.1 and 11.4; `close.rs`, `debug.rs`,
   `builtin/coroutine.rs`, `lua/mod.rs` adapted to `FrameWalk` and the trampolines.
5. Port the handlers family by family into `ops/`, same opcode set as today (the
   existing `_OWN/_ABSENT/_PROTO/_TRANS`, `_NUM` and `_MM` forms included, keyed to
   the new adaptive bits but with today's rewrite policy), with `enter`, the
   continuations, metamethod staging, `impl_error` and the unwinder on headers,
   `native_call`/`native_enter`/`native_act`/`ret_native` with `NativeOut` and the
   continuation table, pcall/xpcall per 10.3, coroutines per 11.
6. Delete `interp.rs`, `DispatchState`, `drive_natives`, `ExecKind`, the pending
   fields, `CallbackAction` (replaced by `NativeOut`), `LuaFrame`, `FrameStack`.
7. Tests, debug-assertion tests, official suite.

Gates:
- `cargo test`, `cargo test` with debug assertions in the interpreter crate, and the
  official 5.5.1 suite run (`off.sh` style, compared with the baseline file) all at
  parity with fd5d79b, except where section 18 lists an intended change.
- `tools/frameless.py` passes with the stage's allowlist.
- `tools/hotpath.py` within the budgets of 8.8 for `op_call_r1`, `op_return1`,
  `ret_call1`, `enter`, and within 110% of today's counts for `op_move`,
  `op_getupval`, `getfield_own`, `op_add`, `op_jlt` (they are not redesigned yet).
- Benchmarks: `call`, `callnat`, `meth`, `pcall`, `coyield`, `mmadd`, `idxfn`
  microbenchmarks and the LJR suite, interleaved against `tc-base`. Expected: call
  and metamethod paths faster, nothing slower beyond noise except possibly `callnat`
  by the header stores, which must be reported either way.
- Deviations log filled.

### Stage 2: native protocol cleanup and fast entries

1. `builtin!` macro; port the existing `ff_*` entries; port the continuation natives
   to `NativeOut` and the index table (sort, gsub, tostring, pairs, dofile, trymt,
   close runners, handler/close conts, async_cont).
2. Delete the remaining `Protect`/`OnOk` enums in favour of the packed bits, or keep
   them as the constructors' parameters only.
3. Histogram run over the corpus to pick fast-entry candidates (9.6); add at most
   the top three with measured wins.

Gates: tests and suite at parity; frameless check; `tostr`, `gsubfn`, `sortfn`,
`callnat` microbenchmarks not slower than stage 1; natbench-style per-call cycles
reported.

### Stage 3: adaptive mechanism, arithmetic, canonical NaN

1. `OP_INFO`, adaptive bits accessors, the generic family handlers for register and
   immediate arithmetic and bitwise ops with the branchless mixed path, `family!`,
   the tier 1 forms of 12.3 for arithmetic and bitwise, removal of `_NUM` and today's
   `binop_slow`.
2. D9: `Value::float` canonicalizes, `write_float_unchecked` for the hardware ops,
   checks kept on libm results; a test with payload NaNs through `string.unpack`,
   `0/0`, `math.huge - math.huge`, `-(0/0)`, `math.abs(0/0)`, `(0/0) % 1`, `2 ^
   (0/0)` asserting no value is ever in box space (debug assertion plus a
   `tostring`/`==` test).
3. Tests for every form's guard failure (type flip, boxed ints, zero divisors,
   metamethods on both sides, string coercion) and for the miss counter (a site that
   flips three times locks and still computes correctly).

Gates: tests, suite; frameless; `hotpath` budgets: `ADD_II` 18, `ADD_FF` 18,
`ADD_IF` 19, `ADDI_I` 10, `MULI_IF` 12; benchmarks `fadd`, `iadd`, `mixed`,
`mandel_bench`, `mandel_bench2`, `primes2`, `collatz`, `fft`, LJR `ray`,
`mandel-metatable`; `op-stats` on the corpus showing the expected forms hot and no
flipping sites.

### Stage 4: constant-key table forms

The six get forms and four set forms per 12.3, `get_generic`/`set_generic` with
`fill_ic` and form selection, `barrier_retry` (14.2), removal of `get_slow`/
`set_slow`/`ic_get`/`ic_set` as they exist today.

Gates: tests (including `field_ic.rs`), suite; frameless (store handlers now pass);
`hotpath`: `GETFIELD_INL` 24, `SETFIELD_INL` 22, `SELF_PROTO` within 110% of
today's `self_proto`; benchmarks `getf`, `setf`, `glob`, `meth`, LJR `richard`,
`deltablue`, `havlak`, `json`.

### Stage 5: compares, loops, loads

Compare forms, `cmp_generic`/`eq_generic`, `FORLOOP_I`/`_F` with `FORPREP` writing
them, `TFORCALL_NEXT`/`_IPAIRS` with `TFORPREP` writing them, `LOADI`/`LOADNIL`
handlers (emitted since stage 1 but executing as `LOAD` until now, or emitted here;
the implementer picks and notes it), optional static `FORLOOP_I` emission.

Gates: tests, suite; `hotpath`: `FORLOOP_I` 23, `JLT_II` 17, `JLTI_F` 15,
`TFORCALL_NEXT` within 90% of today's next arm; benchmarks `primes2`, `collatz`,
`mandel_bench2`, LJR `queen`, `fixpoint-fact`, `pairs`-heavy `json`.

### Stage 6: x86-64

Build and run the test suite on x86-64 (the Docker emulation used for the LJR build
is acceptable for correctness; it is not for timing). Verify the argument registers
of `preserve_none` match Appendix B, the frameless check and the spill check (no
`[rsp]` in fast handlers), 32-byte alignment, the dispatch sequence, `cmp/jcc`
adjacency in the compare handlers, and the double-load versus `movq` choice with a
microbenchmark if real hardware is available. Record findings in Appendix A.

### Stage 7: counter-driven tuning

With xctrace on the M4 Pro (memory note `perf-methodology-apple-silicon`): PMI
sites for mispredicts and memory-order violations on `mandel_bench`, `primes2`,
`call`, `getf`; load-pair checks over fresh stores; placement experiments only with
counters. Every change here is measured interleaved and recorded. This is where the
"things may need to be adjusted once benchmarking happens" expectation lands; the
design above is the starting point, not the end state, and the implementer should
expect to adjust header word order, guard forms and dispatch-tail shapes from what
the counters say, logging each adjustment in section 17.

## 17. Deviations log

Entries are added by the implementer in the commit that deviates. Format: date,
commit, section deviated from, what was done instead, why, cost or risk, whether it
is temporary.

(none yet)

## 18. Open questions and intended divergences

Decisions the user has not made yet; the implementer asks before the stage that
needs them:

1. Where tooling lives: `tools/` in the repo is this document's assumption.
2. Branch: continue on `ljr-prototype` or a branch off it.
3. x86-64 timing hardware for stage 6 and 7.
4. Whether `LOADI`/`LOADNIL` are emitted in stage 1 (needs their handlers then) or
   stage 5.

Intended divergences from Lua 5.5 that this design keeps or introduces, to be
listed in the commit messages that touch them and in the memory note
`intentional-divergences`:

- `debug.setupvalue`/`upvaluejoin`/`upvalueid` on by-value upvalues (existing).
- A tail-called plain native has no frame, so an error it raises at level 1 names
  the Lua caller's line (existing behaviour, now stated).
- Conditional branch offsets in immediate compares are limited to 23 bits (new; no
  real function reaches it; the compile error message matches Lua's).
- Message handlers may yield (existing).

## 19. Appendix A: baseline numbers

From the release build at fd5d79b on the M4 Pro unless stated. Instruction counts
are from objdump of the fallthrough path including the dispatch tail; cycle numbers
are from the memory notes and are per operation with the loop subtracted.

| Item | Today | LJR x86-64 (memory) | Target after redesign |
|---|---|---|---|
| MOVE | 8 | 8 | 8 |
| GETUPVAL | 9 | 9 to 11 | 9 |
| ADD float/float | 21 | 13 | 17 |
| ADD int/int | 21 | | 17 |
| ADD int/float (`_NUM`) | about 30 | | 18 |
| ADDI int | about 12 | | 9 |
| JLT float/float | 17 to 19 | 14 (LT+JMP) | 17 |
| GETFIELD own hit | 26 | 17 | 23 |
| global get (GETTABUP own) | about 27 | 14 | 24 |
| CALL_R1 Lua | 39 | | 29 |
| RETURN1 | 8 | | 6 |
| ret_call1 | 13 | | 15 |
| call + return | 60 | 49 (+5) | 50 |
| native CALL + landing, no body | about 70 | | 50 |
| arith metamethod, handler side | 104 | | 55 |
| Lua call, cycles | 16.6 | LuaJIT 22.7 | |
| plain native call, cycles | 20 to 24 | LuaJIT 6 to 9 | |
| pcall(f, i), cycles | 17 | LuaJIT 21 | |
| wrap resume+yield, cycles | about 40 | LuaJIT 35 | |
| tostring with `__tostring`, cycles | 50 | LuaJIT 29 | |
| gsub with function, cycles | 184 | LuaJIT 93 | |
| sort with comparator, cycles | 123 | LuaJIT 87 | |
| LJR suite geomean, tcvm / luajit -joff | about 0.88 at d82483c | | |

Code: `interp.rs` 7465 lines, about 170 handlers, 122 opcodes; `src/vm` 10612 lines.

## 20. Appendix B: register assignments

`rust-preserve-none` argument registers, in order, as observed (aarch64, from the
fd5d79b disassembly) and as defined in LLVM's `X86CallingConv.td` (x86-64; verify in
stage 6):

| Arch | Argument registers | Non-argument temporaries |
|---|---|---|
| aarch64 | x20 x21 x22 x23 x24 x25 x26 x27 x28 x0 x1 x2 x3 x4 x5 x6 x7 | x8 to x15, x16, x17 |
| x86-64 | r12 r13 r14 r15 rdi rsi rdx rcx r8 r9 r11 rax | rbx rbp r10 |

Slot assignment:

| Slot | aarch64 | x86-64 |
|---|---|---|
| `insn` | x20 | r12 |
| `pc` | x21 | r13 |
| `base` | x22 | r14 |
| `rt` | x23 | r15 |
| `closure` | x24 | rdi |
| `thread` | x25 | loaded from `rt` |

Dispatch tail: aarch64 `ldr x20, [x21], #8; and x8, x20, #0xff; ldr x9, [x23, x8,
lsl #3]; br x9`; x86-64 `mov r12, [r13]; add r13, 8; movzx eax, r12b; jmp [r15 +
rax*8]`.
