# TCVM JIT design

Branch `jit2`, written 2026-10-09 against main at bdd8c75 and revised the same day
after a review against the code. This is the document to implement from; it
supersedes the design notes of the old `jit` branch (PR #127, issues #123 and #128).
File references are to main at that commit. Revised again on 2026-10-11 after
milestone 2's measurements against LuaJIT: section 17 is the performance plan the
remaining milestones (15) follow, Appendix F its evidence, and the sections it
changes say so.

## Contents

1. Goals and scope
2. Decisions
3. What the old JIT taught
4. The runtime the JIT targets
5. Execution model
6. Hotness, entries and tiering
7. Feedback and speculation, by opcode
8. The IR
9. The optimizer
10. The aarch64 backend
11. Runtime data structures and lifetimes
12. Intrinsics
13. Inlining (milestone 5)
14. Tooling and testing
15. Milestones
16. Risks and open questions
17. Performance plan (after milestone 2)
- Appendix A: register assignment
- Appendix B: header words written by a region
- Appendix C: snapshot encoding
- Appendix D: opcode coverage matrix
- Appendix E: x86-64 extension points
- Appendix F: the milestone 2 measurements, per benchmark

## 1. Goals and scope

The JIT is a method compiler for the adaptive interpreter on main. It compiles a
prototype from its entry or from a hot loop header, speculating on what the
quickened bytecode and the inline caches have already observed, and deoptimizes to
the interpreter when a speculation fails. Its floor is the interpreter up to bounded
costs: compile time, and at most `EXIT_HOT` deopt round trips before each
recompile, `MAX_RECOMPILES` times per entry (section 6.4). Compiled frames are
ordinary frames whenever anything else looks at them (section 5.3), so the JIT adds
no other slow path.

Goals, in priority order:

1. Everything compiles. No operation declines compilation; an operation without a
   fast path lowers to a helper call or to a deoptimization. Coverage gaps were the
   first problem with the old JIT and they are a design property, not a backlog.
2. Loop and straight-line numeric code runs at native speed: unboxed i32 and f64 in
   registers, guards hoisted out of loops, no dispatch.
3. Field access, method calls, metamethod arithmetic, iterators and allocation are
   compiled against the shape system: one shape compare per receiver per region,
   slot loads at compile-time offsets, class identity for `__index` chains.
4. Calls cost no more than they do in the interpreter, and known callees cost less:
   direct jumps to compiled entries, inlined builtins, the builtins' fast entries
   for the rest. Milestone 2 does not meet this for builtins: a coroutine switch
   from compiled code costs twice the interpreter's (17.2).
5. Compile time stays in the tens of microseconds for ordinary functions, so the
   JIT can trigger early and compile often. Budgets in section 15.
6. The second target, x86-64, can be added without touching the IR, the optimizer,
   the runtime protocol or the allocator contract. Appendix E lists the seams.
7. Faster than LuaJIT by a wide margin: the LJR suite at least 2× LuaJIT's
   compiler (geomean), at a compile time per program well under LuaJIT's trace
   compiler's. Milestone 2 is at 0.88×; the targets and the plan to reach them
   are section 17.

Non-goals for this design: a tracing tier, a background compiler thread, a
baseline (non-speculating) tier, debug-hook support inside compiled code, and
compiled coroutine switches (a yield is a tail-out like any call).

Benchmarks the design is measured against: `test-files/{primes2,collatz_bench,
mandel_bench,fft,nbody,particles_bench,iterative_fib_bench}.lua` and the LJR suite
in the scratchpad, against the interpreter on main and against LuaJIT with the JIT
off and on.

## 2. Decisions

Each was settled with the user after a tradeoff round; the rest of the document
elaborates them.

- **J1. Compiled code is a dispatch target.** A region is machine code with the
  handler ABI (`src/vm/abi.rs`): entered by `become` with the handler slots in
  registers (six on aarch64, five on x86-64), leaving by `become` into a handler. It makes calls by writing a frame
  header and tailing `enter`; it returns through the header continuation like
  `RETURN1`; it deoptimizes by tailing into the interpreter. No native frame
  survives a tail-out. Rejected: regions as C functions with a re-entrant dispatch
  loop underneath. (Section 5.)
- **J2. `regalloc2` is the register allocator.** The in-house Braun–Hack spiller
  and colouring scan are not ported. (Section 10.4.)
- **J3. One machine instruction set per target**, lowered from the shared SSA IR by
  a per-target backward matcher. The IR carries no target assumptions. aarch64
  first; x86-64 later but designed for from the first commit. (Section 10,
  Appendix E.)
- **J4. No basic-block versioning.** Per-site speculation from the quickened forms
  and the inline caches, optimistic type inference, loop peeling, and
  recompilation driven by hot exits. Block duplication at type-divergent joins is
  kept in reserve. (Sections 7 and 9.)
- **J5. Hotness is counted at function entry and at loop back-edges** through one
  table of 16-bit counters in `State`: calls count the callee's compiler-emitted
  `FUNC` at pc 0, `while` and generic-`for` loops count at their back edge (a
  counting `JMP_BACK`, `TFORLOOP`), `repeat` loops at their `LOOP`, and numeric
  loops in the `FORLOOP` forms. Entries are installed only over dedicated words, as
  LuaJIT patches `FUNCF`/`LOOP`/`FORL`: `FUNC`, the `LOOP` at a `while`/`repeat`/
  generic-`for` header, a backward `goto`'s `JMP_BACK` and the `FORLOOP` forms.
  (Sections 5.2, 6.)
- **J6. Integers are `I32` or `I64` in compiled code.** A site whose feedback is
  small ints only gets `I32` with overflow guards that deoptimize; a site that saw
  boxed integers or an i32 overflow gets `I64`, which needs no overflow checks
  since Lua's integers wrap, and is boxed only when it escapes (an allocation when
  it leaves i32). The feedback comes from a per-prototype byte per instruction that
  the interpreter's slow paths and the JIT's exits write (4.8). Blocks that never
  executed compile to a deoptimization, as V8 does for `kNone` feedback. A range analysis
  later removes overflow checks and narrows `I64` to `I32` where bounds allow.
  (Sections 4.8, 6.3, 7.2.)
- **J7. Compilation is synchronous**, on the trigger, so a loop entry seeds the
  types of the live registers from the actual frame. Regions belong to the
  prototype, not the closure; upvalue and global constants are guarded per use.
- **J8. Snapshots support inlined frames from the first version**, so inlining
  changes no exit protocol. (Appendix C, section 13.)
- **J9. After a deoptimization the frame runs interpreted only until the next
  entry point** (a loop entry or a call), and a hot exit recompiles with its site
  widened at the entry's next arrival; a bounded number of recompiles blacklists
  the entry. (Section 6.4.)
- **J10. Calls between compiled functions are native calls whose frames can be
  dropped** (2026-10-11, extends J1). A compiled caller writes the callee's
  header and its live values to their slots exactly as J1 requires, then calls
  the callee's fast entry with `blr`, and the callee returns with `ret`. The Lua
  frame stays the record of the call: header word 1 holds the caller's resume
  point for every other way back. The native frames are a cache: any way out of
  compiled code into the interpreter (a deopt, a call through `enter`, a native,
  a dispatch, a collector exit) first resets the native stack to where dispatch
  was entered, so the interpreter, coroutine switches, unwinding and the
  collector never see a compiled frame, and each dropped frame resumes through
  its callee's header. Rejected for now: one native stack per coroutine (V8 and
  JSC style native frames), which J10 keeps open, since with per-coroutine stacks
  the same protocol holds and only the reset points move. (Section 5.4.)

## 3. What the old JIT taught

Facts from reading branch `jit` (head 800eb09, 406 main commits behind) and PR
#127; the numbers are the branch's own, from `PLAN.md` and the PR.

What it built: a method JIT with eager basic-block versioning (cap 5), two-phase
lowering (a type-only analysis to discover versions, then emission, both over one
transfer function), Braun on-the-fly SSA with block parameters, a `FrameState` per
guard that cloned the 256-slot register map, a shared "dumb" machine IR with four
`Vec`s per instruction, a Braun–Hack spiller plus pre-split colouring, aarch64 and
x86-64 encoders with a per-guard exit stub that stored every live register, regions
as C functions called from `op_call` after 64 calls with an `accepts()` type loop
and an `assumptions_hold()` loop per call, and no loop entry.

What compiled: constants, pack and unpack, type, shape and condition guards,
`tab.props` plus `slot.get`, pinned stack reads and writes, int and float
arithmetic, compares, branches, returns. Refused: any region with a call or an
allocation (`may_gc`), field stores, array access, upvalues, globals, `NEWTABLE`,
`CONCAT`, `CLOSURE`, every generic `lua.*` op, every vararg, `MULTRET` or `TBC`
function. Only `fib`-shaped functions and the loops of `mix`/`mix2` ran native.

| Measurement, old branch | Value |
|---|---|
| keepalive operands on one `mix2` guard | 35 |
| lower to encode, `mix` | 76 µs |
| allocate, whole-value scan vs split pipeline, `mix` | 39 µs vs 76 µs |
| allocate, split pipeline, `mix2` | 287 µs |
| malloc and memset share of allocate | about 30% |
| exit stub cost per live register | 3 to 5 instructions |

Root causes, mapped to this design:

- Calls, builtins and allocation were impossible because the region lived on the
  native stack under which the interpreter could not run a callee that might
  yield, and GC rooting of native-stack values was never built. J1 removes the
  native frame; section 5.6 removes the rooting problem.
- Instruction selection was weak because the shared machine IR had only
  `Load{off}`, `Store{off}`, `Alu`, `AluImm` and one fused compare-branch; the
  encoders could only transcribe. J3 gives each target its real forms.
- The machine IR was slow because of the per-instruction `Vec`s and the cloned
  `FrameState`s. Section 8.2 makes every IR structure a flat pool with ranges;
  section 8.6 makes snapshots data with liveness pruning.
- Deopt stubs were huge because each guard listed every live value as a keepalive
  operand, which also pinned constants in registers across loops (#128's
  "constant 1 pinned in r11"). Section 5.5 replaces stubs with a two-instruction
  trampoline and data snapshots that reference constants and unchanged slots
  without operands.
- Versioning multiplied blocks, needed a cap, and threw all specialization away at
  the fallback version (`and/or false` nil polymorphism in #128). J4.

What is kept, by file of the old branch:

| Old file | Fate |
|---|---|
| `ir/mod.rs`, `ir/op.rs`, `ir/ty.rs` | ideas kept (block params, const pool, effect classes, `Rep`/`TypeSet`/refinement); rewritten on flat storage (section 8) |
| `ir/simplify.rs`, `ir/verify.rs`, `ir/print.rs` | ported to the new storage |
| `frontend/cfg.rs` (block boundaries, liveness, `reg_uses`/`reg_defs`/`edge_defs`) | ported to main's ISA (204 opcodes) |
| `frontend/ssa.rs` (Braun builder) | ported |
| `frontend/lower.rs`, `sink.rs` (two phases, `TySink`) | dropped; one-pass builder (section 8.7) |
| `backend/mach.rs`, `isel.rs` | dropped; per-target `MachInst` and lowering (section 10) |
| `backend/regalloc.rs`, `spill.rs`, `nextuse.rs`, `spillcost.rs` | dropped for `regalloc2` |
| `backend/order.rs` (loop forest, loop-contiguous layout) | kept for layout and loop passes |
| `backend/aarch64_asm.rs` | kept, extended (section 10.7) |
| `backend/x64_asm.rs` | kept for later |
| `backend/code.rs` (dual-mapped W^X memory, icache sync), `alloc.rs` (segments, 64-byte units) | kept as is |
| `backend/layout.rs` (offsets by `offset_of`, tests against reality) | kept, re-derived for the 8-byte `Value` and the new `TableState` |
| `region.rs`, `op_call` glue, `Status` word, `Declined` | dropped |
| exec tests, differential runtime tests, pipeline bench | patterns kept (section 14) |

## 4. The runtime the JIT targets

Facts the design depends on, with where they live. Anything the JIT emits inline
reads these layouts through a `layout` module of `offset_of` constants with tests
against live objects, as the old `backend/layout.rs` did.

### 4.1 Values (`src/env/value.rs`)

Eight bytes, NaN-boxed. A float is its raw bits; every NaN produced by `Value::float`
is the canonical `0x7FF8_0000_0000_0000`, and hardware add, sub, mul, div, neg and
abs of canonical operands are stored unchecked (`write_float_unchecked`); libm
results go through `write_float`, which remaps a NaN in box space. Boxed values are
`0xFFF8_0000_0000_0000 | tag << 48 | payload`:

| Tag | Value |
|---|---|
| 1 | userdata pointer |
| 2 | heap-boxed `i64` pointer |
| 3 | string pointer |
| 4 | table pointer (`Gc<RefLock<TableState>>`) |
| 5 | function pointer (`Gc<FunctionKind>`) |
| 6 | thread pointer |
| 7 | immediates: small int `0xFFFF_FFFF_0000_0000 | i32 as u32`, nil `0xFFFF_FFFE_0000_0000`, false nil+1, true nil+2 |

Tests the JIT emits, all on the 64-bit word:

- float: `bits < BOX` where `BOX = 0xFFF9_0000_0000_0000`, i.e. unsigned compare
  against a constant;
- small int: `bits >> 32 == 0xFFFF_FFFF` (`lsr; cmn w, #1`-style, or `cmp x, x, lsr`);
  both small: `(a & b) >= SMALL_INT` unsigned;
- nil or false (falsy): `bits - NIL < 2` unsigned;
- a pointer tag: `bits >> 48 == 0xFFF8 | tag`;
- box an i32: `orr x, w_zext, #0xFFFF_FFFF_0000_0000` (one logical immediate) after a
  zero extension; unbox: `sxtw` (`ldrsw` from memory);
- box a pointer: `orr x, p, #(QNAN_NEG | tag << 48)` (logical immediate, since the
  payload is 48 bits); unbox: `and x, v, #0xFFFF_FFFF_FFFF`.

Heap-boxed integers are produced by `Value::integer` for anything outside i32 and
compare and hash by value. Compiled code treats tag 2 as "not small" everywhere;
only the generic helpers see it.

### 4.2 Frames (`src/vm/frame.rs`)

Four header words below each base in the value stack:

| Word | Content |
|---|---|
| `base-4` func | Lua: closure pointer, `nv << 48`; native: closure, `at << 48`, `cont << 56`, `ok << 62` |
| `base-3` ret | continuation handler address (32-byte aligned) with flags in the low five bits: `NATIVE 1`, `HAS_OPEN 2`, `HAS_TBC 4`, `PROTECTED 8`, `HANDLER 16` |
| `base-2` caller | the caller's base, null for the bottom frame |
| `base-1` pc | the caller's resume pc |

A `CALL a b c` places the callee at `R[a]`, its header is `R[a..a+4)` and its base
is `base + a + 4`; arguments start at `R[a+4]`. The header is written by
`write_hdr` as four single volatile stores (the M4 does not forward a pair store to
the continuations' single loads). `ThreadState` publishes `top_base`/`top_pc`
lazily (`sync!`) at exits only; `top` bounds multires values. The collector walks
frames from `top_base` and traces every window `[base, upper)` as values
(`ThreadState::trace`), so every slot inside a live window must hold a valid
`Value` at any time the collector can run. Stacks grow by reallocation
(`ensure_frame_slots`), after which `frame::rebase` fixes the caller words and the
open upvalues; dispatch reloads `base` from the thread after any grow path.

### 4.3 Handlers and dispatch (`src/vm/abi.rs`, `src/vm/dispatch.rs`)

Every dispatch target is `extern "rust-preserve-none" fn(insn: Slot, pc, base,
rt: Context, closure: Slot, thread: *mut ThreadState) -> Exit` on aarch64,
`#[rustc_align(32)]` and `#[inline(never)]`, entered by `become`. The slots are x20
to x25 in that order, from LLVM's aarch64 `preserve_none` argument order; the source
names no registers (Appendix A). x86-64 handlers take five slots and read the thread
through `rt.thread_ptr()` (`abi.rs:99-116`). `Context` is one word, `&State`, whose
first 256 words are the opcode table and the next 256 the CALL continuations
(`rt.handler(op)`, `rt.ret(c)`). Other `State` fields are reached through the
layout module's `offset_of!` constants (`Runtime` is not `repr(C)`, so offsets are
per build), and `enter`, `native_enter` and the other `slow` routines exist only
as Rust symbols, in no table. `Exit` is `End`, `Gc` or `Pending`; dispatch
leaves by returning it through `dispatch::enter` → `run` → `call_handler`, a single
plain call of the first handler. `Jump` is how cold Rust routines tell a handler
where to go next. The thread is reloaded after any Rust routine that may switch
coroutines (`reload_thread!`).

### 4.4 Calls, returns and continuations (`src/vm/ops/call.rs`)

`CALL`'s Lua arm: window check against `thread.stack_end`, arity check against
`fixed_arity`, `write_hdr(hdr, lua_func_word(callee, 0), handler_bits(ret), base,
pc)`, then `closure = callee; base = nb; pc = callee.code; next!()`; it publishes
nothing. `enter` is the same from a written header whose word 0 holds the callee as
a raw value, with `pc` the header and `insn` the argument count; a native goes to
`native_enter`, a non-function to `enter_meta`. Its slow paths publish the caller
themselves (`enter_sync!`, `call.rs:57-64`): `top_base` from the `base` register,
`top_pc` from the staged header's word 3. `RETURN1` is `become hdr.ret(nret = 1,
values = &R[a], base)`. `ret_call` reads `a` and `c` from the CALL word at
`caller_pc - 1`, `ret_call1` and `ret_call2` only `a`, `ret_call0` nothing; each
lands the results and `resume!(caller, cpc)`, which reloads the closure from the
caller's header. Metamethods are staged at `base + max_stack_size`
with `ret_store_a`, `ret_cond_t/f`, `ret_discard`, `ret_tfor`, `ret_close`,
`ret_return`. Budgets (interp spec 8.8): `CALL_R1` 32, `RETURN1` 8, `ret_call1`
16, `enter` 18 instructions.

### 4.5 Natives (`src/vm/native.rs`, `src/vm/ff.rs`)

`enter` of a native goes to `native_enter`, which writes the native header, sets
`top_base = window`, `top = win + nargs`, calls the Rust function and lands a plain
return through the frame's continuation, or goes to `native_act` for `call_then`,
`resume`, `yield`, `async`. Fast entries (`ff_sqrt`, `ff_floor`, `ff_pairs`,
`ff_pcall`, …) are reached only from a `CALL` or `TAILCALL` instruction through
`NativeClosure::entry`; `enter` bypasses them. A `pcall` from `enter` runs the full
builtin, which stages its callee with `call_then` and a `PROTECTED` native frame.
Natives may grow the stack; after a native returns, every pointer into the stack is
stale until reloaded from the thread.

### 4.6 The collector (`src/dmm/`)

Non-moving Immix. The collector runs only between `Lua::enter`s: an allocating
handler runs `gc_check!()` after the allocation (one pair load of
`Metrics.gc_check.{allocated_bytes_total, gc_check_at}` through `rt.metrics`, one
compare) and on due does `sync!(); return Exit::Gc`; natives reach the same exit
through `gc_exit` (`native.rs:395-409`). The executor returns `Pending`, the
collection runs as `Lua::enter` exits (`collect_debt`, `lua/mod.rs:145,175-217`),
and the next step resumes from the published frame. Allocation itself never
collects, and an mmap failure aborts (`heap.rs:180-181,422-458`). Write barrier:
`Gc::is_gray` (bit 0 of the box header's tagged vtable word) means no barrier, and
it is the whole fast-path test; otherwise `barrier_retry` runs
`backward_barrier_erased` out of line and re-executes the instruction. Nothing in a
frame window is rooted separately: tracing the window is the rooting.

### 4.7 Tables, shapes, classes (`src/env/table/mod.rs`, `src/env/shape/mod.rs`)

`TableState` is 48 bytes, drop-free, in a `RefLock` cell with its inline slots and a
constructor-sized array part trailing it. Fields: `shape`, `spill` pointer,
`spill_cap`, `inline_len`, `array` pointer, `asize`, `len_hint`, `aux`. A `SlotLoc`
is a byte offset from the table's lock pointer for an inline slot, or `SPILLED |
byte offset` into the spill cell; `has_room(loc)` is `!spilled || off <
spill_cap * 8`. The spill cell is a separate GC object (`Slots`) the table marks; a
store into it barriers the table (`ops/mod.rs:256-258`). Shapes are
`Gc<ShapeData>`, compared by pointer; a shape carries the key layout, `slot_count`,
`inline_cap`, `is_dict`, and the metatable class `mt_cache: Option<MtCache>`. A
class (`MtCacheData`) is content-shared across metatables with the same
metamethods and holds `bits: MetamethodBits`, `weak`, `__index`/`__newindex`
inline (`index_table()` is the address of the `__index` table or 0), the other
metamethods by `MmIndex` (`mm_at`), and a `stale` flag.

A shape pins its class pointer, not the class's contents. A write of a
metamethod key into a metatable goes through `member_write`
(`shape/mod.rs:727-750`): a class with one member is updated in place (`mirror`,
`:856-885`, sets `bits`, `index`, `newindex`); a shared class is made stale
(`make_stale`, `:713-721`: every bit set, `index`/`newindex` cleared), and its
tables move to a fresh class only lazily, when a slow path calls `Table::meta`.
`Shape::has_mm(bit)` is a live `Cell` read (`:1079-1084`). So a table keeps its
shape while `has_mm(INDEX)` or `has_mm(NEWINDEX)` turns from false to true, and
every fast path re-reads the bits at run time (`ops/mod.rs:149-265`): a shape
compare proves layout and class identity, never the absence of a metamethod. That
is the invariant every IC and every JIT guard rests on. The SET fill refuses to
cache a mirrored key on an adopted shape (`field.rs:351-353`), which keeps
metatable writes off the `SETFIELD_INL` fast path.

Inline caches (`InlineCache`, 32 bytes, `align(32)`, one per constant-key site):
`Own{shape, loc}`, `Absent{shape}`, `Transition{from, to, loc}`, `ProtoLoad{recv,
holder: weak, holder_shape, loc}`, `Empty`. There is no megamorphic state. Most
misses refill, except: dict shapes are never cached, mirrored keys on adopted
shapes are not cached for stores, an `Absent` hit with an `__index` function does
not refill, and `ProtoLoad` is filled only when the holder has the key; the
collector empties a `ProtoLoad` whose holder died (`function.rs:186-197`).
Array part: Lua key `k` is at index `k` (index 0 holds key 0, as in LuaJIT);
`array_get(i)` for `i < asize`, nil-holed; `set_array_at` after a barrier.

### 4.8 The adaptive ISA (`src/instruction.rs`)

A 64-bit word `op:8 a:8 b:8 c:8 ext:32`, 204 opcodes, every branch offset at bits
40..63. The compiler emits generic forms; the handlers rewrite sites in place
(`Code` is cells) to the forms for what they saw: arithmetic `_II/_FF/_IF/_FI/_NN`,
immediate `_I/_F/_IF`, `ARITH_MM/_MM_R/_MMI`, compares `_II` and `_F`, field
access `_INL/_AUX/_ABSENT/_PROTO/_TRANS`, loops `FORLOOP_I/_F`,
`TFORCALL_NEXT/_IPAIRS`, upvalue `_REF` forms chosen by the assembler. Adaptive
bits (`misses:2`, `locked:1`) sit at `OP_INFO[op].adaptive_shift`; three misses lock
a site to its generic opcode or its `_NN` form. A form's guard failure tails to the
generic handler; an overflow or a boxed operand does not count as a miss. `CALL_R0/
R1`, `CALLS*` fuse the continuation and the callee move; statement calls are
fused, so a `CALLS_R0/R1` can be the first instruction of a loop body. Only
`Instruction::generic_op` recovers the emitted opcode for every form;
`Op::unquickened(ARITH_MM_R)` is `ARITH_MM`.

A site's word is not only read through `insn`. Every slow path re-reads it with
`insn_at!()` (`*pc.sub(1)`, `abi.rs:374-378`, 23 uses); `ret_call`/`ret_call1/2`,
`ret_store_a`, `ret_cond_t/f`, `ret_tfor`, `ret_close`, `ret_return` and
`stage_grow` decode or re-execute the word at `caller_pc - 1` (`call.rs:540-570`,
`meta.rs:68-134`); `ret_pcall`, `ret_xpcall` and the unwinder decode the CALL that
called `pcall` (`native.rs:625-646`, `unwind.rs:169-178,312`). Quickening writes
sites: `specialize` (`arith.rs:308-344`) writes a word built from `insn` or from a
fresh `insn_at!()`, field `quicken` read-modify-writes the word (`field.rs:69-91`),
`FORPREP` and `TFORPREP` rewrite their loop instruction (`control.rs:158-169,
382-389`). Section 5.2 depends on this list.

On main, absence of quickening is not absence of execution. `specialize` counts
misses only on a site already in a form (`arith.rs:317-322`), so boxed integers,
overflow, string coercion and number kinds with no form (`POW`/`MOD`/`IDIV`
mixes, `RMODI`, `RSHLI`, …) leave an executed site generic with `misses = 0`
(`arith.rs:594-633`). The generic `JLT` family is itself the two-float fast path
and never adapts on floats or small/float mixes; generic `JLTI` is the small-int
fast path and never adapts on ints (`compare.rs:96-150`). An `Empty` IC is not
"never executed" either: dict shapes and mirrored keys on adopted shapes are never
cached (`field.rs:223-230,351-353`).

Milestone 1 adds a feedback store beside the opcode, so that the JIT can tell
"never ran" (V8's `kNone`, `v8/src/objects/type-hints.h:18-43`) and "ran with
boxed integers" from the rest. V8 and JSC write their feedback on every execution:
an OR into a feedback-vector slot (`v8/src/codegen/code-stub-assembler.cc:13848-13863`),
an OR into an arith profile or a value-profile bucket store
(`JavaScriptCore/llint/LowLevelInterpreter.asm:2968-2973`,
`LowLevelInterpreter64.asm:77-81`). Here the opcode stays the feedback on every
fast path, and only paths that are already slow write the store.

The store is `Prototype.feedback: Box<[Cell<u8>]>`, one byte per instruction,
allocated zeroed with the prototype. Its bits are the kinds of the operands seen
(`SMALL`, `BIGINT`, `FLOAT`, `STR`, `TAB`, `OTHER`), `OVERFLOW` (small operands
gave a result outside i32) and `MM` (a metamethod ran). It is an OR, so it only
grows. Writers: `arith_slow`, `specialize(None)` on a generic site, `cmp_slow`,
`eq_slow`, `forloop_slow`, the field slow paths, and `jit_exit` (5.6). A writer
reaches the byte through `closure.proto` and indexes it by `pc - closure.code`:
two loads and an OR on a path that already boxes, allocates or calls. One byte per
eight-byte instruction is 12.5% of code size; allocating it lazily at hotness, as
V8 does, would leave the executions before it unrecorded.

Arithmetic and field sites always leave a record when they run: a quickened
form, a miss or a lock, a filled IC, or a byte written by the slow path they take
when nothing fits. Compares do not: the generic `JLT` family is itself the
two-float fast path, generic `JLTI` the small-int one, and `JEQ`/`JNEQ` settle
floats and identical bits inline, all without a write. Section 6.3 therefore
decides "never executed" per block, from the sites that do record, and the types
of compare operands come from inference and use-driven speculation (9.1), not
from the compare. A slow path stores the byte only when its bits change, so a
site that keeps taking it pays a load and a compare. The interpreter cost of the
byte is measured with the counting cost in milestone 1.

### 4.9 Closures, upvalues, prototypes (`src/env/function.rs`)

`FunctionKind::{Lua(LuaClosure), Native(NativeClosure)}` inline in the cell with the
upvalue slots trailing it (`LuaFn::upvalue_ptr`). `LuaClosure` holds `proto` and
pointers into the prototype's `code`, `constants` and `ic_table` boxes
(`function.rs:316-323`), so a write into `Code` is seen by every closure of the
prototype, plus copies of `max_stack_size`, `num_params`, `is_vararg`,
`fixed_arity`. `Code` is a `Box<[Cell<Instruction>]>` from the global allocator,
8-byte aligned by type and 16 by the system malloc. An upvalue slot is the value
itself when the descriptor is `by_value` (never reassigned after initialization),
else a shared `UpvalueCell { v, closed, thread }` whose `v` targets the stack slot
while open and `closed` after. `CLOSURE` capturing by reference inserts into
`open_upvalues` (sorted by slot) and sets `HAS_OPEN` on the frame when it creates
a new open cell. Prototypes hold `code`, `constants`, `prototypes`,
`upvalue_desc`, `ic_table`, `templates`, `lineinfo`, `locvars`.

### 4.10 Threads (`src/env/thread.rs`)

`ThreadState` holds the value stack (`stack`, `stack_end`, `top`), the published
frame (`top_base`, `top_pc`), `open_upvalues`, `tbc_list`, `status`, `yield_bottom`,
`resumer`, `stack_limit` (65300 slots, 65500 while a message handler runs). On
aarch64 the thread pointer is a handler slot (x25); coroutine switches replace it.

## 5. Execution model

### 5.1 A region is a dispatch target

A region is a block of machine code with one entry and the handler signature. The
interpreter enters it with `become`, exactly as it enters a handler: `insn` in x20,
`pc` in x21, `base` in x22, `rt` in x23, `closure` in x24, `thread` in x25, the
native stack pointer where the trampoline left it, x29/x30 holding the
trampoline's frame and return address. The region may use every other register
freely. It leaves only by a tail jump into a handler with the same six registers
set for that handler's role, with the native stack pointer, x29 and x30 restored to
their values at entry. It never returns to the trampoline itself.

Consequences that the rest of the design relies on:

- A region can be entered from, and leave to, any point of the interpreter, since
  the interpreter's state is entirely in the six registers plus the Lua stack.
- The interpreter's correctness properties about frames, suspension, unwinding
  and tracing hold for frames whose code happens to be compiled, because such
  frames are ordinary frames (4.2) at every moment the interpreter or the
  collector can look at them (5.3).
- The native stack holds nothing across a Lua call. A callee that yields, errors,
  or triggers a collection unwinds nothing of the region's.

Four handler registers are pinned for the region's whole body: `base` (x22), `rt`
(x23), `closure` (x24) and `thread` (x25) keep their values, are not allocatable,
and are read directly by every slot access, helper call and tail-out. All four are
callee-saved in the C ABI, so Rust helper calls preserve them for free. `insn`
(x20) and `pc` (x21) are free in the body and loaded at each tail-out.

### 5.2 Entries

Entries are installed only over instructions that exist for that purpose, as in
LuaJIT, where `FUNCF`, `LOOP`, `FORL` and `ITERL` become `JFUNCF`, `JLOOP`, `JFORL`
and `JITERL` (`LuaJIT/src/lj_bc.h:175-201`). An arbitrary header word cannot be
overwritten: its slow paths, continuations and quickening re-read or rewrite it
from `Code` (4.8). Five opcodes are added to the ISA:

- `FUNC`, emitted by the compiler at pc 0 of every prototype (before
  `VARARGPREP`). Every way into a Lua function (`CALL`, `enter`, `TAILCALL`, a
  coroutine's first resume) reads it (`call::start!`): a `FUNC` counts the call
  (6.1) and the callee starts at pc 1, so `FUNC` is not dispatched; a `JIT_ENTRY`
  is dispatched. Its own handler counts and falls through, for the JIT's paths
  that run an entry's original word.
- `LOOP`, emitted at the header of every `while`, `repeat` and generic `for`
  loop: a `repeat` loop's counts an iteration and falls through. A `while` loop's
  back edge is a `JMP_BACK` and a generic `for`'s a `TFORLOOP`, both counting,
  which jump past the `LOOP` (V8 counts at `JumpLoop`, its back edge,
  `v8/src/interpreter/interpreter-generator.cc:2453`), so the `LOOP` runs once per
  entry into a `while` loop and never for a generic `for` until an entry is
  installed on it: the install points the loop's back edges at the `LOOP`
  (`Entry::back_edges`, 11.1), and retiring the entry points them past it again.
  A generic `for` gets one, where LuaJIT patches `ITERL`, because here its
  back-jump is taken inside `TFORCALL_*` (`tfor_finish!` reads the `TFORLOOP`
  word, `control.rs:138-140`), so `TFORLOOP` cannot be patched.
- `JMP_BACK`, shape `AImm { hot: u8, offset }`: a counting jump, emitted as a
  `while` loop's back edge and as a backward `goto` (LuaJIT emits `BC_LOOP` at
  the `goto`, `lj_parse.c:2804`: a one-pass compiler learns that a label heads a
  loop only when the backward `goto` arrives). A `goto`'s `JMP_BACK` is its
  loop's entry word, so a `goto` loop's entry is at its latch, as a numeric
  loop's is; one whose label directly follows a `LOOP` is a back edge of that
  `LOOP`'s loop.
- `JIT_ENTRY`, shape `Ad { entry: u16 }`, never emitted: written over a `FUNC`
  whose function entry was compiled.
- `JIT_LOOP`, same shape, never emitted: written over a `LOOP`, over a numeric
  loop's `FORLOOP`/`FORLOOP_I`/`FORLOOP_F` (the back-edge, as LuaJIT's `JFORL`) or
  over a `goto`'s `JMP_BACK`, whose loop entry was compiled. A region entered at a
  `FORLOOP` starts with the loop step, one at a `JMP_BACK` with the jump.

`FUNC`, `LOOP` and `JMP_BACK` have no slow path, no adaptive bits and no
continuation that decodes them, so nothing but dispatch and `call::start!` (which
tells `FUNC` from `JIT_ENTRY`) reads them. A `FORLOOP` has two readers,
both taught to recognize a JIT word: `write_forloop_form` (`control.rs:157-169`)
leaves it alone (the `FORLOOP_I` region's entry guards and `FORLOOP_I`'s own
fallback to the generic `FORLOOP`, `control.rs:259-267`, cover a changed loop
kind), and `forloop_slow` decodes the word `op_forloop` passed it, never the one
in `Code`. That word is always the original: every path that would dispatch the
word at a pc (a `jit_exit` resume, 5.6; `jit_entry_fail`) dispatches the original
instead, which also keeps an exit `Before` a loop header from re-entering the
region it just left.

The overwritten word is kept in the prototype's `JitState.entries[i].original`
(section 11). The entry's code address is in a
table reached from `State` (11.1), so the handlers are two loads and a branch:

```
op fn op_jit_entry / op_jit_loop {
    let code = rt.jit_entries()[insn.d()];       // ldr t, [rt, #JIT_ENTRIES]; ldr x, [t, d, lsl #3]
    tail!(code)                                  // become into the region with pc = this word + 1
}
```

A failed entry guard runs the original: the entry-fail stub pops the frame and
tails `jit_entry_fail` (6.4) with the region in `insn`, which records the kinds
of every register the entry guards test (the region keeps their list), counts and
then does
`tail!(rt.handler(original.opcode()), insn = original)` with `pc` already pointing
past the word, the exact state the original handler expects. A blacklisted or retired entry
restores the original word in `Code` and frees its table slot, so `op_jit_*` is
never reached for it again.

The prologue, in order:

1. Open the native frame (section 10.5): `stp x29, x30, [sp, #-F]!; mov x29, sp`,
   with `F` the spill area plus 16, 16-byte aligned. Frame size is per region.
2. Entry guards: for each live register at the entry pc whose entry type is
   narrower than `Any`, load the home slot and test the type; a failure jumps to
   the single entry-fail stub, which pops the frame and tails `jit_entry_fail`.
   For a `JIT_ENTRY` the live registers are the parameters, typed by use-driven
   speculation (9.1): the type their uses' feedback agrees on. For a `JIT_LOOP`
   they are the registers live at the entry: one whose uses expect a set keeps it
   when the kind of the live frame's value at compile time (J7) and the kinds
   failed entries saw fit it, and is left untyped otherwise; one whose uses expect
   nothing is typed by the frame's kind and the seen ones.
3. Unbox the guarded values into the representations the body wants.

A `JIT_ENTRY` region compiles the whole function from pc 0; a `JIT_LOOP` region
compiles from its `LOOP` or `FORLOOP` and covers everything reachable from it,
including the code after the loop and the returns, with the enclosing loops
rebuilt as real loops by peeling the rest of their current iterations (8.7). A prototype may hold up to `MAX_ENTRIES = 4`
live regions (one function entry, three loop entries); a fifth evicts the deepest
loop entry whose loop lies inside another live loop entry's loop, since the outer
region already contains the inner loop and the inner entry only saves the
interpreted remainder of one outer iteration after a deopt inside the inner body;
when no entry is nested in another, the oldest loop entry goes. Each entry keeps
the headers of its loop nest for this. Both kinds coexist: an interpreted frame reaching a `JIT_LOOP`
word enters the loop region; a call reaching `JIT_ENTRY` enters the function
region; compiled code never reads bytecode, so neither word is seen by a region.

### 5.3 Register discipline

Lua registers are SSA values inside a region and live in machine registers or
native spill slots. The home slot of a register (`base[r]`) is written only:

- before a tail-out, for every register that is live across it and whose home slot
  does not already hold its current value (the builder tracks per register which
  SSA value the slot holds, section 8.7);
- before a helper that reads home slots through open cells (`close_upvalues`,
  `jit_closure`), for the captured registers (5.11).

Rule R1: **at every tail-out, every register that is live after the tail-out holds
its current value in its home slot, encoded as a `Value`.** This is the whole
rooting story: the collector, the unwinder, a coroutine switch and the interpreter
after a deopt all read frames, and frames are correct whenever the region is not
running.

Rule R2: **a home slot never holds anything but a valid `Value`.** Unboxed i32 and
f64 live in registers or in the native frame. A dead register's slot keeps whatever
`Value` it last held, as in the interpreter, so a window traces as values at any
moment.

Rule R3: **the native frame holds nothing live across a tail-out.** It is popped
before every tail-out and re-opened by the prologue or by a resume (5.4). Spill
slots are therefore never seen by the collector.

Rule R4: **no pointer into the value stack survives a tail-out.** `base` is
reloaded from the frame header on resume (5.4), and derived register addresses are
recomputed from it.

Rule R5: **no Rust helper called from a region may grow the value stack, switch
threads, run Lua code, or raise.** Anything that needs one of those is a tail-out
(a call) or a deopt. Helpers may allocate; the collector checks of 5.7 cover them.

Rule R6: **no virtual register is live across a tail-out.** R3 pops the frame
that would hold a spill, and the callee clobbers every register. A value used on
both sides of a call is either a Lua register, reloaded from its home slot (R1),
or rematerializable (a constant, a pool load, an unbox of a reloaded slot), and is
rematerialized after the resume point. Passes that move or share values (GVN,
LICM, the prologue's hoisted constants) create such uses, so the call-boundary
pass of 9.8 re-materializes them, and the verifier (8.8) rejects anything left.

### 5.4 Calls

Every call made by a region, whether to a Lua function, a native, or a metamethod,
is a tail-out through `enter`, and the callee returns straight into the region:
the resume point is the callee's continuation. The sequence for `CALL a b c` with
`nargs` arguments already in `R[a+4..]` (they are stored to home slots as part of
R1; the callee slot `R[a]` holds the callee value):

```
  ; R1 write-backs for registers live after the call
  str  func_value, [base + a*8]                        ; hdr word 0
  adr  x9, resume_K
  str  x9,         [base + a*8 + 8]                    ; word 1: the resume point, flags 0
  str  base,       [base + a*8 + 16]                   ; word 2: caller base
  ldr  x9, =pc_after_call
  str  x9,         [base + a*8 + 24]                   ; word 3: bytecode pc after the CALL
  ldp  x29, x30, [sp], #F                             ; pop the native frame
  mov  x20, #nargs                                    ; insn = nargs
  add  x21, base, #a*8                                ; pc = header
  b    enter                                          ; closure (x24) and base (x22) are the caller's
```

Header stores are four single stores as `write_hdr` requires (4.2). The header is
exactly what an interpreter caller writes, except that the continuation is region
code: word 3 is the bytecode pc after the CALL (a literal: regions belong to the
prototype, whose `Code` all closures share, 4.9), so every reader of word 3
works unchanged: `frames()`, `enter_sync!`, `pop_frame` (`unwind.rs:258`),
`xpcall_handler`, the frameless-`pcall` catch. Nothing identifies a frame by its
continuation's address except the `pcall`/`xpcall` continuations
(`unwind.rs:139-154`, `frame.rs:387-389`, `call.rs:91-92`), so a region resume
point is never mistaken for anything. Word 1's flags are 0; `enter` adds `NATIVE`
for a native callee, and the callee's own `CLOSURE`/`TBC` add theirs, so a resume
point is 32-byte aligned like every continuation. Its padding follows the call's
unconditional `b enter` and never executes. The region publishes nothing, as the
interpreter's CALL publishes nothing: `enter`'s slow paths publish the caller from
`base` and the header (4.4).

`enter` does the rest exactly as for an interpreter caller: window check, arity
fixup, `__call` resolution, growth, natives through `native_enter`.

A return reaches the resume point as it reaches any continuation (`RETURN1` is
`become hdr.ret(...)`), with `nret` in x20, the result pointer in x21 and the
callee's base in x22. The resume point does what a continuation's `caller!()` and
`resume!` do, then re-opens the native frame and lands the results itself, in
registers:

```
resume_K:
  ldr  x22, [x22, #-16]        ; caller base (callee header word 2), rebased by any growth
  ldr  x24, [x22, #-32]
  and  x24, x24, #PTR_MASK     ; closure (caller header word 0)
  stp  x29, x30, [sp, #-F]!
  mov  x29, sp
  (land results; continue)
```

| CALL form | Resume code |
|---|---|
| `CALL_R0` (`c == 1`) | nothing |
| `CALL_R1` (`c == 2`) | `cbz x20, nil; ldr v, [x21]` into the vreg of `R[a]`; nil otherwise |
| `c == 3` | two conditional loads |
| `c >= 4` | helper `jit_land(dst = base + a, values, nret, wanted)`; the registers are then reloaded from their slots on demand |
| `c == 0` (MULTRET) | helper `jit_land_multret(thread, base, a, values, nret)` copies and sets `top`; consumers of `top` are the MULTRET forms of CALL, RETURN, SETLIST and VARARG, which take `top` from the thread |

Everything the region kept in registers before the call that is still live after
it was written to its home slot (R1) and is reloaded by the body on first use after
the resume (the builder emits `Load` for it; section 8.7). Values that are not
live across the call cost nothing. So a call-heavy region pays, per call, the
stores of its live-across values plus the header, which is at most what the
interpreter pays on every instruction for the same registers.

Metamethod calls (`ARITH_MM`, `__index` functions, `__newindex`, `__eq`, `__lt`,
`__le`, `__concat`, `__len`, `__close`, `__call` chains) use the same sequence with
the header at `base + max_stack_size` as `stage_mm!` does, and a resume point that
lands one result (or a truthiness test for the compare continuations). The window
check against `stack_end` is emitted inline; its failure is a deopt to the
instruction, where the interpreter's `stage_grow` handles growth.

Native callees take the same path through `enter` into `native_enter`. In
milestone 2 the fast entries of `ff.rs` are not reached; their work is done by
intrinsics (section 12) when the callee is known at compile time, and by the full
builtin otherwise. That makes a builtin without an intrinsic slower to call from
a region than from the interpreter, whose `CALL` reaches the fast entry: a
`coroutine.resume`/`yield` switch costs about 70 cycles from a region and 31
interpreted (17.2). Milestone 3 gives each fast entry a core that takes a written
header as `enter` does (the callee in word 0, the continuation in word 1, the
arguments from the header plus 4, `nargs` in `insn`), with today's CALL-word
adapter in front of it for the interpreter. `enter`'s native arm tails the core
when the native has one (a load and a branch), and a region whose callee is a
known native without an intrinsic tails the core directly. A fast entry that
returns does so through header word 1, which for a region is its resume point,
so `coroutine.resume`, `yield`, `pcall`, `xpcall` and `coroutine.wrap`'s
functions keep compiled code on both sides of the switch, as LuaJIT's traces
stitch around its assembler fast functions (17.3 C3).

**Fast calls (J10, milestone 3).** Every call above pays `enter` (31
instructions: the callee's kind, the window, the arity), the callee's `JIT_ENTRY`
dispatch and `op_jit_entry` (4 more, two dependent indirect branches), a frame pop
and push, and a return through an indirect `br` that the return-address stack
never predicts: 5-32% of the cycles of most benchmarks (17.2). Between compiled
functions a call is a native call instead:

- *Dispatch's entry.* The executor enters dispatch through `dispatch_entry`
  (`backend/aarch64/abi.rs`), which records the stack pointer, frame pointer and
  return address every handler runs with in `JitRuntime::native` (`NativeBase`),
  and a stack limit `FAST_STACK` (256 KiB) below it; a nested dispatch records
  its own and the outer one's are restored when it returns. Every way out of a
  region into the interpreter (calls through `enter`, `Dispatch`, `TAILCALL`'s
  dispatch, metamethod calls, the entry-fail stub, `exit_common`) restores those
  three instead of popping its own frame, which drops the frames of any compiled
  callers under it. So the interpreter, natives, coroutine switches, the unwinder
  and the collector always run at dispatch's entry with no compiled frame below
  them, as before.
- *The caller.* After the header words (word 0, the callee value, already in its
  slot as R1 stores it), a `CALL` tests that the callee is a function, a Lua
  closure without varargs (the niche byte, 5.5) whose prototype has a fast
  entry (`Prototype::jit_fast`, set while a function region at pc 0 is installed
  and cleared when it is marked for recompilation, retired or blacklisted),
  rewrites word 0 to the closure as `enter` would, advances `base`, sets
  `closure` and `insn` (the argument count) and calls the fast entry with `blr`,
  keeping its own frame. Anything else is the slow path above.
- *The callee's fast entry*, before the region's prologue, checks what `enter`
  would: the window against `max_stack_size`, the arity against `num_params`,
  and the stack pointer against the limit. A miss puts word 0 back to the callee
  value, restores the caller's `base` and `closure`, resets the native stack and
  takes the caller's slow path through `enter`, which grows the stack or fills
  the missing parameters; the caller then resumes through its header.
- *Returns.* A frame a compiled caller entered holds that caller's return
  address; any other holds dispatch's. `RETURN` pops the frame and compares: a
  compiled caller's frame returns with `ret`, `insn` and `pc` set as for a
  header return; any other returns through header word 1 as before.
- *Deep chains.* The return address stack holds about 48 entries on the M4: a
  recursion 64 calls deep costs 32 cycles per call with `ret` against 13 with
  an indirect `br`, which the branch predictor follows. Below `RAS_STACK`
  (512 bytes) under dispatch's entry a caller therefore branches with `br` and
  a return address tagged in bit 0, and a tagged frame returns with `br`; the
  return address stack then stays right for the shallower frames' `ret`s.
  (`ret` alone made fixpoint-fact, about 100 calls deep, 8% slower than the
  old protocol; with the tag it is level, and binary-trees keeps `ret`'s 13%.)
- *Metamethod calls and tail calls.* `CallMm` takes the same fast path from
  its header at `max_stack_size`. A `TAILCALL` whose callee has a fast entry
  pops this frame and branches to it with the frame's return address, so the
  callee returns where this frame would have.
- *Resume.* Each resume block has a second label past its frame push, where a
  fast callee's `ret` lands (the caller's frame is still open). Both reload
  `base` and `closure` from the callee's header and land the results the same
  way, and nothing but constants lives across the call (R6), so the fast path
  needs no other change to the IR.

Measured 2026-10-11 against the milestone 2 protocol (LJR suite, median of 3
interleaved rounds, f9b7039 and baad547 against dc2266a): 4.5% fewer cycles over
the suite (geomean of 16); recursive-fib-uv 1.42× (12.9 to 9.2 cycles per call),
binary-trees 1.17×, recursive-fib-gv 1.15×, richard 1.07×, ray 1.05×, json
1.04×. Two lose a little, both where the chain cannot stay fast:
coroutine-ring 3.5% (every `resume`/`yield` call pays the callee test before its
slow path, until the builtins' header cores below exist) and mandel-metatable
about 2.5% (its metamethods call `setmetatable`, a native, which drops the chain
the metamethod call built). deltablue and havlak are within their run-to-run
spread. What remains per call is the dynamic callee test, the fast entry's
checks and the reloads after the resume, which a known callee and values kept
across the call remove; keeping frames across a visit to the interpreter (16)
keeps chains that call natives fast:

- *Known callees* (direct calls): a constant upvalue (7.5, any by-value upvalue
  never assigned after capture), a global read through `_ENV`'s IC, a method
  from a `_PROTO` holder's or a receiver's own slot, a closure the region itself
  created, and per-site callee feedback (17.4 D1). The identity is guarded once
  and hoisted when invariant; the window and arity are checked against the
  callee's known sizes in the caller, and the call goes to a second entry past
  the checks.
- *Values across calls.* With the frame kept, values could stay in callee-saved
  registers and spill slots across a fast call and skip the reload and re-guard
  after it; that needs a resume edge that carries values (R6 holds only for the
  slow resume) and the `debug.setlocal` flag of 17.4 D4.
- *Later*: a typed entry past the entry guards for callers that prove the
  arguments' kinds; one result in `x0`.

The header and its resume point are the same whichever way the callee was
entered, so walkers, the unwinder and `pcall` see no difference.

### 5.5 Returns and tail calls

`RETURN0`, `RETURN1`, and `RETURN a b` with a constant `b` follow the handlers:
store the result values to `R[a..]` (R1 makes them home slots), pop the native
frame, and `become hdr.ret(nret, values = base + a*8, base)`, which is a load of
header word 1, a mask of the flag bits, and a branch. A frame whose `HAS_OPEN` or
`HAS_TBC` flag may be set tests the flags word: `HAS_TBC` deopts to the `RETURN`
instruction, since closing may call `__close`; `HAS_OPEN` alone calls the
`close_upvalues` helper (which runs no Lua) after the captured registers are
flushed (5.11), then returns. A function that creates closures would otherwise
take a deopt on every return. `RETURN b == 0` (up to `top`) is a deopt to the
instruction in milestone 1 and from milestone 2 leaves for the interpreter's
`RETURN` handler, as the slow cases of `TAILCALL` below do.

`TAILCALL`'s fast arm (non-vararg Lua callee, window fits, no `HAS_OPEN`/`HAS_TBC`
flags) is compiled: store the arguments to `R[0..]` (their SSA values, read before
the first store), nil-fill missing parameters, rewrite header word 0 to the callee
(keeping `nv`), then dispatch the callee's first instruction with `base` unchanged,
which is its `JIT_ENTRY` when it has a region. Whether the callee is such a closure
is one byte: `FunctionKind` keeps its tag in the niche of `LuaClosure::is_vararg`
(a native's variant ends before that byte), so the byte is 0 exactly for a Lua
closure without varargs; a layout test pins this. Every other case (a native, a
vararg callee, `__call`, a frame with something to close, a window that needs the
stack grown) leaves for the interpreter's `TAILCALL` handler with the frame in its
slots (`Dispatch`, R1), which does exactly the work that remains, with no exit to
count or snapshot to restore.

A callee whose prototype has a fast entry (J10, 5.4) is entered there after the
frame rewrite, with this region's frame popped and its return address kept, so
the callee returns where this frame would have; the argument count passed is
the parameter count after the nil fill. Only a callee without one dispatches
its first word (the opcode-table load, the indirect branch, `op_jit_entry`'s
entry-table load and a second indirect branch, five dependent loads that were
on fixpoint-fact's hot path).

### 5.6 Exits

Every guard is `b.cond` to a stub; a stub is two instructions:

```
exit_K:  mov  w16, #K
         b    exit_region        ; the region's three-instruction trampoline into exit_common (10.6)
```

`exit_common` is one naked function in the binary (section 10.6). It stores x0–x15,
x19–x21, x26–x28, d0–d31 and the region's spill slots to `State.jit.exit_regs` (a
fixed buffer reached through `rt`), pops the native frame, and tails `jit_exit`, a
`slow` handler, with `insn` packing the region pointer and `K`. `jit_exit`:

1. Finds `exit K` in the region's exit table: its snapshot (whose header holds
   the resume pc and kind), its tag, its counter.
2. Interprets the snapshot (Appendix C): for each entry, materializes the `Value`
   from the recorded location (a register image, a spill slot of the frame just
   popped, read before the pop into the same buffer, a constant from the region's
   pool, an inline nil or boolean, or an unboxed i32/i64/f64 to box) and stores it
   to the home slot. Entries for
   unchanged slots are not in the snapshot; dead registers are not in it.
3. For a snapshot with inlined frames (Appendix C), writes the inlined callees'
   headers from the outermost in, so the interpreter sees ordinary frames.
4. On every `Before` exit, records the kinds of the site's operands into its
   feedback byte, reading them from the frame it just restored (they are all in
   home slots by then); an `After` exit follows a completed instruction and
   records nothing. It counts the exit when it is widenable (6.4), and requests a
   recompile or a blacklist when a threshold trips. A collector check's exit
   (tag `Gc`, 5.7) then leaves through `Exit::Gc` and resumes at its pc after the
   collection.
5. `sync!`s the frame and `become`s the handler of the instruction at the resume
   pc with `insn` loaded from `Code` (which may by now be a quickened form, which is
   fine), or for an "after" snapshot the instruction following it. A
   `JIT_ENTRY`/`JIT_LOOP` word there is replaced by its entry's original (5.2):
   resuming at a loop entry must run the loop's own instruction, not re-enter the
   region that just exited.

Exit kinds:

| Kind | Resume | Used by |
|---|---|---|
| `Before(pc)` | re-execute the instruction at `pc` in the interpreter | type and shape guards, overflow, generic fallbacks, unsupported ops, the collector checks |
| `After(pc)` | continue at `pc + 1`; the instruction's effects are complete | definition guards (9.1), a helper that completed but reported "slow path needed afterwards" |

A deopt never raises an error: an operation that would error in the interpreter
(a nil index, a non-callable, a bad `for` value) deopts `Before` and the
interpreter raises it with its own message and position.

Only widenable exits count toward recompilation: type, shape and overflow exits,
which a recompile can cover, and failed entry guards (6.4). A never-executed exit (6.3) is
widenable too, and stronger: it can only fire because its block did run, so its
first firing records the kinds and marks the entry for recompilation at once,
outside the `MAX_RECOMPILES` budget. Each block flips from never-executed to
executed at most once, so this terminates. `Gc` exits,
`GuardFits`, a helper's `FAIL` for an error or a metamethod the site already
handles generically, and `HAS_TBC` returns are uncountable: no recompile changes
them, and counting them would blacklist an allocating loop after about
`EXIT_HOT × MAX_RECOMPILES` collections.

Guard code is emitted in the body, stubs in a cold section after the body (10.6).
A region of 300 guards costs 2.4 KiB of stubs and one shared routine.

### 5.7 The collector

Because the collector runs only between `Lua::enter`s, nothing in a region's
registers needs rooting: no collection can happen while the region's registers
hold anything. The region must only make sure the collector gets its chance, which
it does once per section, as LuaJIT does (`asm_gc_check` at a trace's head and its
`LOOP`, `lj_asm.c:1691,2587`): at the header of every loop that allocates (an
allocating helper, a boxed integer outside i32, an inline allocation) and at the
region's entry when code outside every loop allocates. A call is not an
allocation: the callee checks its own, as the interpreter's allocating handlers
and `native_enter` do. Milestone 2 counts calls, which costs every calling loop
and every calling function region a check that never fires on its account
(fib's entry, 6.7% of qt's `clip`, 7.8% of richard's main loop; 17.3 C2), and
milestone 3 stops. Every loop header gets a check from the builder, LICM (9.5)
borrows its snapshot for the exits it hoists, and `prune_gc_checks` removes the
checks a section does not need only after LICM. The check is
the interpreter's check `ldp x9, x10, [metrics]; cmp x9, x10; b.hs gc_exit_K`, a
`Before` exit at the header's or the entry's pc that leaves through `Exit::Gc`.
Garbage then waits at most one iteration or one entry, and no allocating
instruction needs a snapshot of its own (boxes are inserted after the builder, where none exists). Natives
returning through a resume point have already had their check in `native_enter`.

The collector may free objects the region holds only pointers to: between
tail-outs nothing is freed, since no collection runs; across a tail-out every live
value is in a home slot (R1) and traced. Derived pointers (a table's slot cell, an
array part, a spill cell) are valid between tail-outs for the same reason, and are
recomputed after every tail-out (R4).

Write barriers: a store of a possibly-heap value into a table, an upvalue cell or a
thread emits `ldr x9, [obj - HEADER]; tbnz x9, #0, ok; bl jit_barrier(obj)` with the
helper running `backward_barrier_erased`. Stores of values the type lattice proves
non-heap (small ints, floats, nil, booleans) emit no barrier. Helpers that store on
the region's behalf run their own barriers.

### 5.8 Frame walkers

A JIT frame is a Lua frame whose callee's header has a resume point in word 1 and
the bytecode pc after the CALL in word 3 (5.4). Walkers read word 3 and never
call the continuation, so `frame::frames` (the unwinder, `where_prefix`, error
locations, `Stack::lua_frame_count`), `enter_sync!`/`enter_raise!`, `pop_frame`
and `check_sync` need no change, and `Frame::pc_index` and `line()` are exact. The
only frame they cannot describe is the region's own frame while it runs, which
nothing walks: the region publishes and tails out before anything else runs on
the thread.

### 5.9 Coroutines, `pcall`, unwinding

Nothing changes. A yield from a callee of a JIT frame stores `yield_bottom` and
switches threads; the JIT frame waits with its state in home slots; a resume lands
values in the waiting native frame, which eventually returns into the region's
resume point. An error unwinds frames by their headers; a JIT frame is popped like
any Lua frame (its `HAS_OPEN` upvalues are closed by `close_upvalues`, its TBC list
by the unwinder) and, if it is the catching frame's callee, the catch lands
through the native frame's continuation, which is the region's resume point.
`pcall(f)` from a region runs the builtin (no fast entry through `enter`), which
stages `f` with a `PROTECTED` frame and `ret_native`; the results land in the
region's resume point. Line numbers of JIT frames come from word 3 as for any
frame (5.8).

### 5.10 Stack growth

The value stack moves only on paths that are tail-outs from the region (`enter`'s
grow paths, `stage_grow`, natives, metamethod staging via the interpreter after a
deopt). The region's own staging checks `stack_end` and deopts when the staged
header would not fit, so the region never grows the stack. After any tail-out the
`base` register is reloaded from the callee header at the resume point (5.4) or is
the entry's (a `JIT_LOOP` entry after a deopt and re-entry), both rebased by
`frame::rebase`.

### 5.11 Captured registers, varargs, TBC, hidden slots

- A register captured by reference (`CLOSURE` with a non-`by_value` descriptor for
  a `ParentLocal`) can be read or written through its open cell only by code that
  runs while the region is not running: a callee, a coroutine resumed by a call,
  or the `close_upvalues` helper copying it into the cell. Helpers run no Lua (R5),
  and a closure's upvalues point into its parent's frame, never its own. So a
  captured register stays an ordinary SSA value between tail-outs; the builder
  treats it as live at every tail-out (R1 stores it even when the region no longer
  reads it), stores it before `close_upvalues` and `jit_closure`, and reloads it
  after every call, since the callee may have written it. A loop without calls
  pays nothing. The set is static per prototype and conservative (every register
  any child prototype captures by reference, section 8.7); a register in the set
  that no closure has captured yet costs one extra store per tail-out.
- Varargs: `VARARG` with a fixed count and `VARARGGET` read `nv` from header word 0
  and copy from `base - 4 - nv`; `VARARG` with `count = 0` sets `top` through a
  helper. `VARARGPREP` with `needs_vararg_table` is a helper that allocates (5.7).
- `TBC` deopts (it registers a to-be-closed slot, rare in hot code); `CLOSE` with no
  TBC entries is a helper call to `close_upvalues`, with TBC entries a deopt.
- `FORPREP` and `TFORPREP` hidden slots are ordinary registers with known types
  after the prep (section 7.6).

## 6. Hotness, entries and tiering

### 6.1 Counters

`State` gains `hot: [Cell<u16>; 64]`. Each counting instruction carries its
counter index in a free operand byte (`FUNC`, `LOOP` and `JMP_BACK` in `a` by
construction, `FORLOOP` and `TFORLOOP` in the `b` byte their `AImm` shape leaves
free), assigned round-robin by the compiler; each handler reads its own byte
(`count_hot!`).
A hash of the pc would waste half the table on `FUNC` sites, since `Code` is
16-byte aligned (4.9). Counters decrement by one; one that reaches zero tails
the cold handler `jit_hot`, which resets it to `HOT_START` and decides.
Collisions between sites only make a site hot earlier.

| Site | Handler | Cost |
|---|---|---|
| function entry | the call (`call::start!`): the callee's word 0 is loaded as dispatch would; a `FUNC` is tested for, counts with `ubfx; add; ldrh; sub; strh; tst; b.eq` on `rt + HOT + idx * 2`, and the callee starts at pc 1 | 11 per call, no dispatch |
| numeric loop | `FORLOOP_I`, `FORLOOP_F`, `FORLOOP` on the taken branch, before the index is written (so `jit_hot` sees the frame at the instruction) | 7 |
| `while` loop, backward `goto` | `JMP_BACK`, before the jump | 7 |
| generic `for` | `TFORLOOP`, and `TFORCALL_*`'s inline back-jump (`tfor_finish!`), on the taken branch | 7 |
| `repeat` loop | `LOOP` at the header | one extra dispatch per iteration plus 7 |

No compare handler counts, so the sign tests on backward branches in 22 compare
handlers are gone; a `repeat` loop's back edge is one of them, so it keeps the
`LOOP` dispatch. The milestone 1 measurement with the JIT off found the first
version, which counted at a dispatched `FUNC` and at a `LOOP` on every
iteration, 6.2% slower than main over the program benchmarks (geomean cycles),
most of it those dispatches and an 18-instruction counter handler that decoded
its byte by opcode; this layout is 4.0% slower (16).

Defaults, all tunable through `TCVM_JIT_HOT_CALL` and `TCVM_JIT_HOT_LOOP`:
`HOT_START = 200` for both kinds. A function called 200 times or a loop iterating
200 times compiles. Compile cost (section 15 budgets) is tens of microseconds; 200
iterations of a small loop is a few microseconds, so the earliest compile is paid
back by a loop of a few thousand iterations, which is the shape every benchmark
and most hot code has.

### 6.2 `jit_hot`

A `slow` handler, entered from a counting instruction with the frame at that
instruction (for a call, at the callee's `FUNC`). It:

1. Finds or creates the prototype's `JitState`.
2. Refuses quickly when the entry is blacklisted, already compiled, or compiling.
3. Runs the compiler (section 15 budget) synchronously, with the live frame
   available for seeding a loop entry's types. The entry pc is the counting
   instruction itself, except for a back edge that jumps past a `LOOP`: its
   entry is that `LOOP`, where the frame is the same, since a `LOOP` changes
   nothing. A back edge an installed entry pointed at its `LOOP` refuses at once.
4. On success, installs the region (11.3), writes the `JIT_ENTRY`/`JIT_LOOP` word
   over the entry instruction, points the loop's back edges at a `LOOP` entry,
   and enters the region at once, since the frame is exactly at the entry.
5. On a compiler failure (an internal limit, never a semantic decline) counts a
   strike, dispatches the original instruction; three strikes blacklist the entry.
6. Refuses, without a strike, a region in which no path does work in compiled
   code before it leaves: reaches a back edge, a return (from a function entry),
   or an exit a recompile could widen (inside a loop, for a loop entry, whose exit
   and return happen once per run of the loop); a call continues in its resume
   block. Such a region only adds an entry and an exit to the interpreter's work,
   and since unwidenable exits never recompile (5.6), nothing would fix it: a loop
   whose every iteration hits an op the milestone does not compile would pay both
   on each iteration. A region that has a useful path but whose hot path still
   always deopts unwidenably is not caught; telling it apart needs a rate
   (unwidenable exits per entry), which waits until milestone 2 shows which
   unwidenable exits remain.

### 6.3 What a region covers

A function-entry region covers every instruction reachable from pc 0; a loop-entry
region everything reachable from the header. Unreachable code is not compiled.
Code that has never executed compiles to a `Deopt(Before)` (J6): the interpreter
runs it, quickens it, and the hot-exit path recompiles with real feedback. The
decision is per basic block of the bytecode CFG, made before emission:

- A block is **known executed** if any site in it left a record (a quickened form,
  `misses > 0` or `locked`, a filled IC, a nonzero feedback byte), if it dominates
  a known-executed block (a block cannot run without its dominators), or if it is
  the region's entry block.
- A block is **known never executed** if it is not known executed, has at least
  one recording site (register or immediate arithmetic, a constant-key field
  site), and none of them recorded anything. Compares never decide it (4.8).
- Every other block is assumed executed.

A never-executed block compiles to one `Deopt(Before)` at its first instruction,
and blocks reachable only through it are not emitted. In an executed block a site
without feedback compiles from what it has: a byte with integer kinds only gives
`I64` code (7.2), any other byte or no byte generic code (the generic handler's
inline cases plus a helper), and a compare whose operands inference or 9.1 typed
compiles to the typed compare. Assuming a block executed when it was not costs
little: its compares are generic inline code, and its arms still prune themselves
at their own recording sites. A misjudgment the other way, a block whose sites ran
but recorded nothing, cannot happen for recording sites; the never-executed exit
of 5.6 remains as a safety net.

A generic `TFORCALL` is not a signal: `TFORPREP` keeps a generic `TFORCALL` for any
iterator other than `next` and `ipairs`. A generic `FORLOOP` is one only with its
byte: `FORPREP` records `BIGINT` at a `FORLOOP` it makes generic and
`forloop_slow` on each of its iterations, so a generic `FORLOOP` that it and its
`FORPREP` recorded nothing at never iterated and compiles to the never-executed
deopt.

### 6.4 Exits, recompilation, blacklisting

Each exit has a 32-bit counter in the region's exit table. Every `Before` exit
records what it saw (5.6, 7.1): the kinds of the guarded values and an overflow into
the site's feedback byte, the receiver shape for a shape guard (up to four per
site) into `JitState.shapes_seen`. On the `EXIT_HOT = 10`th taking
of one widenable exit (5.6) `jit_exit` marks the entry for recompilation by
pointing its entry-table slot (11.1) at `jit_recompile`, a `slow` handler. The
next arrival at the `JIT_ENTRY`/`JIT_LOOP` word lands there with the frame
exactly at the entry, compiles with the widened feedback and the live frame's
types (J7), replaces the region (11.4) and enters it. Counting stops while an
entry is installed (its `FUNC`/`LOOP`/`FORLOOP`/`JMP_BACK` is overwritten; a
`while` or generic `for` loop's back edges still count when the interpreter takes
them after an exit, and `jit_hot` refuses them at once), so no counter could
trigger the recompile, and none is needed. After `MAX_RECOMPILES = 4`
recompiles of one entry the entry is blacklisted: its word is restored, its
regions retired, and `jit_hot` refuses it until the prototype dies. Evictions
(5.2) re-create entries with fresh counts, so a prototype also has a lifetime cap,
`MAX_COMPILES = 16` regions of any kind; past it `jit_hot` refuses the prototype,
which bounds the retired-region memory of 11.2.

Widening is monotone: a site that saw small ints and floats compiles its float
path; one that saw three shapes compiles a three-way chain; one that saw more than
four, or a dict shape, goes generic (a helper call). Two pieces of state it reads
are kept monotone although the interpreter's are not. A site's shapes keep the
shape its inline cache held when a compile used it, since the interpreter refills
the cache after an exit; and a field site also tests the shapes earlier sites
tested on the same receiver value, while under `MAX_SHAPES`, since the receiver
most likely has that shape there too. An arithmetic or compare site whose form
failed a type guard is recorded in `JitState.kind_exits` and compiles as generic
code from then on (7.2): its form can change kinds more than once before the
interpreter locks it. A hot type or shape exit that taught the prototype something
since its region was compiled (the region records how much of both records its
compile saw) recompiles outside `MAX_RECOMPILES`, as a never-executed block does:
each widens a finite number of times, and `MAX_COMPILES` still bounds the
prototype. A
`JIT_LOOP` region whose entry guards fail repeatedly (the loop is entered with
other types later in the run) is widened the same way: the entry-fail stub tails
`jit_entry_fail`, a `slow` handler that records the kinds of every register the
region's entry guards test into the entry's `seen` (the next compile of the entry
reads them, 9.1) and the entry instruction's operand kinds into its byte, counts
on the region toward the same `EXIT_HOT`, and dispatches the original
instruction.

### 6.5 Interaction with the adaptive interpreter

Compiled code never rewrites bytecode or inline caches. The interpreter keeps
quickening, so by the time a recompile happens the deopted instruction has been
re-specialized by the interpreter's own rules, and the builder reads the newer
form. Retired entries restore the original word: a `FUNC`, `LOOP` or
`JMP_BACK`, or a `FORLOOP` form, which `FORPREP` rewrites again on the loop's
next start as it would have; a `LOOP` entry's back edges jump past it again.

A form lasts only while its site sees the form's kinds. A slow path that sees
kinds no form covers (strings, tables, metamethod kinds, big integers at a float
form; floats or other non-small operands at an equality form) drops the form and
counts a miss, as a change between forms does, so no form outlives the kinds its
site sees: the builder compiles a form as what it says (7.2, 7.3), and its
guards exit only on kinds that are new since the last compile. A small overflow
and a big integer at an integer form keep it: those are the integer form's `I64`
code (7.2).

## 7. Feedback and speculation, by opcode

The quickened form of a site is its type feedback and the guard the region emits
is exactly the form's own predicate. This section is the contract per family; the
full per-opcode table is Appendix D.

### 7.1 Feedback sources

1. The opcode of the site in `Code` (its quickened form) and its adaptive bits.
2. The inline cache entry of a field site.
3. The constant pool, upvalue descriptors and templates of the prototype.
4. For a loop entry, the actual values of the live registers in the frame.
5. The site's feedback byte (4.8), written by the interpreter's slow paths and by
   exits: operand kinds, `OVERFLOW`, `MM`. A value's uses combine into the type
   it is speculated at where it is defined (9.1). Receiver shapes seen at hot shape
   exits are in `JitState.shapes_seen`, keyed by pc; exit counts are in the
   region's exit table.
6. Global facts: whether a number metatable is set (`State.type_metatables[2]`),
   read at compile time and guarded by a region-level watch (11.5).

### 7.2 Numbers

Representations in compiled code: `I32` (sign-extended in an x register, or a w
register), `I64` (an x register), `F64` (a d register), `Val` (the boxed word).
Which integer representation a site gets comes from its form and its feedback
byte:

| Site | Representation |
|---|---|
| `_II` form, byte without `BIGINT`/`OVERFLOW` | `I32`, overflow guarded |
| `_II` form with `OVERFLOW` or `BIGINT` in the byte, or a generic site whose byte has only `SMALL`/`BIGINT` | `I64` |
| byte with `FLOAT` and integer kinds, no form | generic code |

| Site form | Speculation and code |
|---|---|
| `ADD_II`, `SUB_II`, `MUL_II` | both operands small (`Guard SMALL` on each, usually already refined); `adds/subs/smull+cmp` with `b.vs`/`b.ne` to a `Before` exit; result `I32` |
| `MOD_II`, `IDIV_II` | both small; divisor zero and `MIN/-1` deopt; expanded to `sdiv`, `msub`, sign-fix `csel` (section 10.3); result `I32` |
| `DIV_II`, `POW_II` | both small; convert, `fdiv`/`pow` helper; result `F64` |
| `_FF` | both float (`Guard FLOAT`: unsigned compare against `BOX`); hardware op; `MOD_FF` inline with a helper fallback (below), `POW_FF` by helper with a canonical-NaN fixup except `x ^ 2`, which is `x * x` (both are correctly rounded; 2M random doubles agree under lua 5.5.1, while `x ^ 0.5` and `sqrt` differ); result `F64` |
| `_IF`, `_FI` | one small, one float; `scvtf` the small one; result `F64` |
| `_NN` and locked generic | no speculation on the operands: the handler's inline cases (both small: `I64` arithmetic, boxed; both numbers: `F64`), then helper `jit_arith(rt, kind, a, b) -> Value or FAIL`, whose `FAIL` is as at a generic site; the helper boxes big integers, which the section's collector check covers (5.7) |
| generic, in a never-executed block (6.3) | `Deopt(Before)` |
| generic, byte with integer kinds only | `I64` code (below) |
| generic, any other nonzero byte | generic code: the handler's inline small/float cases, then `jit_arith`; its `FAIL` (a metamethod or an error) exits with the widenable tag `Mm`, which records `MM` in the byte, and a site whose byte has `MM` calls the metamethod `jit_mm` finds instead (5.4). From milestone 4 a byte with numbers only (`SMALL`, `FLOAT`, no `MM`, no `OTHER`) compiles the small and float arms and a widenable type exit for the rest instead of the helper arm, whose call clobbers registers on the hot path (17.3 S2) |
| a form whose guard a type exit failed (`kind_exits`, 6.4) | generic code |
| immediate forms `_I`, `_F`, `_IF` | one register operand guarded as the form says; the immediate is a constant (`Imm`: 31-bit int or f32-exact float); `ADDI_I` becomes `adds w, w, #imm` when the immediate fits 12 bits |
| `ARITH_MM`, `_MM_R`, `_MMI` | the operand with the metamethod `Guard TAB`, its metamethod loaded from its class (`ClassMm`: the shape's class, the class's metamethod array, each possibly null, then the slot at `mm_at(idx)`), guarded not nil, and a call through 5.4 with `ret_store_a` semantics (one result into `R[a]`). With the number on the left (`_MM_R`, `_MMI` with the immediate on the left) it is guarded a number and the numbers' metatable cell must be empty (11.5). Nothing records the metamethod a site called, so it is loaded rather than compared with a compile-time constant; a direct call (13) would add that. A failed guard is a type exit, and the site compiles generically after (6.4) |
| `UNM` | small with overflow check, or float `fneg`; else helper |
| `BNOT`, `NOT`, `LEN` | small `mvn`; falsy test; `LEN` in 7.6 |
| `CONCAT` | helper that builds the string from two values (numbers coerced), `FAIL` deopts; an allocation for 5.7 |
| `FORPREP` | the loop's kind is its `FORLOOP`'s form, unless the bytes of the `FORLOOP` and the `FORPREP`, written at their exits, saw floats (a float loop) or big integers (an `I64` loop); one that saw both integer and float loops deopts, since a loop's kind is visible (the index's `math.type`) and no one representation runs both. Integer prep inline when init and step are small; a float limit, known from its type or from the `FORPREP` byte's `FLOAT_LIMIT` (the `OTHER` bit, which no `FORPREP` operand sets otherwise), is floored or ceiled inline as `for_limit` does, clamped with an out-of-range skip and NaN taken as too small; `last` outside i32 deopts, as the interpreter then writes generic `FORLOOP` (it writes `FORLOOP_I` only when `last`, step and init all fit, `control.rs:221-224`); float prep inline for floats; writes the four hidden slots (`R[a]` last, `R[a+1]` step, `R[a+2]` index, `R[a+3]` the visible copy; a float loop keeps the limit in `R[a]`) as typed SSA values |
| `FORLOOP_I` | `cmp idx, last; b.eq exit_loop; add idx, idx, step` with the step a constant when `FORPREP` saw a constant; no overflow check needed (`FORPREP` chose `last` in range) |
| `FORLOOP_F` | `fadd; fcmp; b.cond`, direction chosen at compile time when the step is a constant |
| `FORLOOP` generic with integer kinds (`FORPREP` and `forloop_slow` write `BIGINT`; with nothing recorded the loop never iterated, 6.3) | an `I64` loop with the reference's unsigned trip count: `last`, step and index unboxed once at `FORPREP` or the entry, `cmp; b.eq; add` on x registers, the visible copy boxed only when it escapes |

**`I64` code.** Lua integers wrap (`math.maxinteger + 1 == math.mininteger`,
`(1 << 62) * 4 == 0` in lua 5.5.1), so `add`, `sub`, `mul`, the bitwise ops and
`lslv`/`lsrv` on x registers are the exact semantics, with no overflow check.
`//` and `%` deopt on a zero divisor (the interpreter raises) and handle
`mininteger // -1` (`mininteger`) and `% -1` (0) with a `csel`, since `sdiv`
returns the dividend there without trapping. Values enter `I64` by
`Unbox{I64}` (a small int's `sxtw`, or a boxed int's load from its box, after a
two-way kind test) and leave by `Box{I64}`, which keeps the interpreter's
invariant that a value in i32 is always a small int (`value.rs:144-149`): a range
test, then the small-int `orr`, or the helper `jit_box_i64` when the value is
outside i32, an allocation for 5.7. A boxing that allocates happens only where the value
escapes: a store to a table or upvalue, an argument or result, a register live at
a tail-out or a snapshot. A loop that keeps a big integer in a register allocates
nothing.

Overflow policy: an `I32` site's overflow exit records `OVERFLOW`, and the
recompile gives that site, and the values that flow from it, `I64`. The range
analysis of section 9.7 removes checks and narrows `I64` back to `I32` where
bounds prove it.

**Float `%`.** Lua 5.5's float modulo is C's `fmod` with the result moved to the
divisor's sign (`luai_nummod`), so `-4.0 % 2` is `-0.0` and `5.5 % -math.huge`
is `-inf` (lua 5.5.1), where LuaJIT's `a - floor(a / b) * b` gives `0` and `nan`.
The helper's libm `fmod` costs about 90 cycles, 12.9% of richard (Appendix F).
From milestone 4 the site computes `q = frintm(a / b)` and `r = fmsub(q, b, a)`
(one rounding, exact when `q` is the true floored quotient), corrects `q` by one
when `r` falls outside the half-open range between 0 and `b`, gives a zero result
the dividend's sign as `fmod` does, and calls the helper when an operand is not
finite or `|a / b| ≥ 2^52`. It ships only after a differential run against
`fmod` over generated edge cases (signed zeros, subnormals, quotients near the
limit).

`collatz_bench` meets this policy in milestone 1: 948 of its 1.5M starting values
leave i32, in 7,408 of 203,419,330 steps (measured with lua 5.5.1). The first
compile is `I32` (the interpreter has seen no overflow by then); the first
overflow exit records `OVERFLOW` on `3*n+1`, and the recompile keeps `n` in an x
register for the whole loop. The interpreter heap-boxes those 7,408 values; the
compiled loop boxes none, since `n` never escapes the loop.

### 7.3 Compares and branches

| Site form | Code |
|---|---|
| `JLT_II` … `JNEQ_II` | both small; `cmp w, w; b.cond` fused |
| `JLTI_F` … `JNGEI_F` | float register; `fcmp d, #imm-constant` (`fmov` of the 15-bit immediate hoisted); unordered falls to the not-taken side as the handler's `<`/`<=` do |
| generic register compare (`JLT`, `JNLT`, `JLE`, `JNLE`) | operands typed by inference or use-driven speculation (9.1): `fcmp`, `cmp w` or `I64` `cmp` with `b.cond`; one typed operand speculates the other to its kind unless the byte saw it differ; byte with integer kinds only: `I64` `cmp`; otherwise generic inline in the handler's order: both floats, both small, then the helper `jit_lt`/`jit_le` (mixed numbers, big integers, strings), which returns 1, 0 or `FAIL` for a metamethod or an error, and `FAIL` deopts `Before`. An `I64` against an `F64` goes to the helper boxed, since converting the integer is not exact |
| generic `JLTI` … `JNGEI` | a register typed `I32`: `cmp w, #imm; b.cond`; typed `F64`: as `_F`; otherwise as the register compare, with the immediate a constant |
| `JEQ`/`JNEQ` generic | in the handler's order (`compare.rs:67-71`): two floats by `fcmp` first, since a NaN has its own bits and must not equal itself; then bit identity; then both small (unequal); then `jit_eq` (boxed integers, mixed numbers), whose `FAIL` lets the interpreter run an `__eq`. From milestone 4 two more inline cases come before the helper: different type tags other than an integer against a float are unequal (Lua consults `__eq` only for two tables or two full userdata; richard's `x ~= NO_TASK` against a nil upvalue is 50 instructions through `jit_eq` today), and a small against a float compares as `scvtf` and `fcmp` (an i32 converts exactly; `20000000 == 2e7` is true in lua 5.5.1) |
| `JEQI`, `JNEQI`, `JEQS`, `JNEQS` | inline: small compare, float compare, or bit identity against the interned constant; no metamethod |
| `JT`, `JF`, `JTSET`, `JFSET` | `sub x, v, x_nil; cmp x, #2; b.lo/hs` with `NIL` hoisted; the `SET` forms assign on one edge (an edge block param, as the old `edge_defs` did) |

A compare whose operands the inference proves `I32` or `F64` loses its guards; the
compare of a loop counter against its `last` is the common case.

Compare sites record kinds only on their slow paths (4.8), and the fast paths of
the generic compares handle two floats, so a compare that has only ever seen
floats has a zero byte: it compiles to the three-arm generic compare with a
`jit_lt`/`jit_le` arm, whose call makes the allocator spill around it. Every
compare in qt's `clip` and table-sort-cmp's `lhs.x ~= rhs.x` are such sites
(Appendix F). From milestone 4 their operands are typed from their definitions
where 9.1 can (field and table loads that other uses type, parameters whose
entry types come from the live frame at a loop entry and from the callers'
direct-call argument kinds), and a compare whose operands nothing types records
its operands' kinds once, on its first execution, with the cost measured with
the JIT off as 6.1's counters were (17.3 S1).

### 7.4 Fields

| Site form and IC entry | Code |
|---|---|
| `GETFIELD_INL`, `GETTABUP_INL`, `SELF_INL` with `Own{shape, loc}` | receiver `Guard TAB`; `ldr s, [t + SHAPE]; cmp s, #shape; b.ne exit`; `ldr v, [t + loc]`; a nil result reads the class bits live (`ldr w, [class + BITS]; tbnz w, #INDEX, slow`, with the class address a constant since the shape pins the class pointer, 4.7) and takes the slow helper or a deopt when `INDEX` is set; only the nil path pays |
| `_AUX` with a spilled `loc` | as above through `ldr spill, [t + SPILL]; ldr v, [spill + off]` |
| `_ABSENT` with `Absent{shape}` | shape guard; class bits read live; result nil without `INDEX`; with `INDEX` and the class's `__index` (read live) a function, a metamethod call (5.4); otherwise the `jit_index` walk through `__index` tables, whose `FAIL` (a function further up) exits |
| `_PROTO` with `ProtoLoad{recv, holder, holder_shape, loc}` | receiver shape guard; `ldr cls, [shape + MT_CACHE]; ldr it, [cls + INDEX_TABLE]; cmp it, #holder`; `ldr hs, [holder + SHAPE]; cmp hs, #holder_shape`; load `loc` from the holder; a nil there takes the `jit_index` walk on past it |
| `SETFIELD_INL/_AUX`, `SETTABUP_*` with `Own` | shape guard; load the existing value, and when it is nil read the class bits live and deopt on `NEWINDEX`; barrier (5.7) unless the stored value is proven non-heap; store. The SET fill never caches a mirrored key on an adopted shape (4.7), so this store never changes class bits |
| `_TRANS` with `Transition{from, to, loc}` | shape guard on `from`; class bits read live, `NEWINDEX` deopts (the shape pins the class, not its bits); value nil deopts (stores nothing in the interpreter, which the deopt reproduces); `has_room(loc)` inline for a spilled `loc` (`ldr cap, [t + SPILL_CAP]; cmp`), true for an inline one; barrier; store the value; store `to` into `t + SHAPE` (the shape pointer is a GC object the table now references: the barrier above covers it, since the table is the parent) |
| `_ABSENT` store | class bits read live; an `__newindex` function on the class: a metamethod call; else deopt |
| `GETFIELD`/`SETFIELD` generic with `Empty` and a zero byte | a recording site that never ran: its block is never executed (6.3), deopt |
| a field site with an `Empty` IC and a nonzero byte (it ran, never cacheable: a dict shape, a mirrored key) | the generic helper (`jit_index`/`jit_newindex`) |
| generic with a filled IC that no longer matches the form | the IC was refilled after the form was written; the builder uses the IC entry, which is authoritative, and the form only as a hint |
| `GETTABUP_REF`, `SETTABUP_REF` | receiver loaded through the cell (`ldr cell, [closure + UPVAL i]; ldr p, [cell + V]; ldr t, [p]`), then as the `_INL` cases using the IC entry |

A site that saw several shapes (6.4) compiles a test per shape (`ShapeIs`), each
resolved at compile time: an own slot, a holder's slot as `_PROTO` reads it, an
absent key; a store to an absent key follows the transition a table already took
from that shape when there is one (`Shape::prop_child`), else the generic store,
and a store that would tell a metatable's class (a mirrored key on an adopted
shape) is the generic store. Another shape exits as a shape guard does.

A shape compare covers later uses of the same receiver until something may change
a shape (`W_SHAPE`: a transition, a generic store helper, `setmetatable`, a call;
8.4). GVN removes the repeats in between, and LICM hoists it out of a loop whose
receiver is loop-invariant and whose body writes no shape, which a loop of stores
into existing fields does not (section 9.5). The class-bit reads are `R_META` and
hoist the same way.

**Chains resolved at compile time (milestone 4).** `_PROTO` covers one hop. A
deeper chain (deltablue's `c:output()` two hops, `c:inputs_known()` three) and a
nil own slot under a metatable with `__index` (deltablue's `v.determined_by`)
take the `jit_index` walk on every execution: 23% of deltablue's cycles, 354
instructions for a two-hop `o:get()` against LuaJIT's 6 (Appendix F). Each hop
compiles as `_PROTO`'s does: the class's `__index` loaded and compared with the
table found at compile time, that table's shape guarded, then the key's slot
loaded or, absent, the next hop; an absent key at the end of the chain is nil
without a call. The guards are loads and compares of `R_META` and `R_SHAPE`
that GVN shares between sites on the same receiver and LICM hoists. A chain that
changes at run time fails a guard, a shape exit that records and widens as any
other. V8's validity cells (one load and compare per site, invalidated by any
write to a prototype's shape) are the alternative if guard chains grow long.

**Several shapes.** A site that saw two shapes tests both at every access on
the receiver, because the paths merge again at each join: deltablue's
`EqualityConstraint:execute` is 2304 bytes after three compiles, json's
`Parser:read` 3072. Tail duplication (9.11) after the first dispatch on a
receiver keeps one shape per path until the next call or loop header.

**Aux slots.** Objects built by `setmetatable({}, mt)` and then filled keep all
their fields in aux storage, one more load per access. Sizing a `NEWTABLE`'s
template from the transitions its tables take (allocation-site slack tracking)
puts them inline; this is runtime work outside the JIT (17.3 F3).

### 7.5 Globals, upvalues, constants

`GETTABUP` on `_ENV` is the field path of 7.4 with the receiver loaded from the
upvalue. A by-value upvalue (`upval!(value i)`) is a load from the closure cell at
a constant offset; the region treats it as a loop-invariant `Val` and may refine it
to a constant when the loaded value is a function, string or table and the site
wants that (a direct call, an intrinsic, a `_PROTO` holder): the refinement is
guarded by one pointer compare per region entry, hoisted to the prologue when
every path uses it. The same holds for a global read through a `_INL` IC entry on
`_ENV`'s shape: the loaded function is compared against the constant seen at
compile time. `SETUPVAL` stores through the cell with the thread-or-cell barrier
(`UpvalueCell::barrier_target`) done by a helper.

From milestone 4, as LuaJIT's `rec_upvalue_constify`: an upvalue the compiler
marks never assigned after its capture (a per-upvalue flag in the prototype) is a
constant of the closure for every type, not only functions, strings and tables;
it is guarded once per entry and never reloaded or re-tested after a resume.
richard and qt reload such upvalues and re-test their types after every call
today, about 15 instructions each (Appendix F). `SETUPVAL`'s barrier tests the
mark bit inline, as 5.7's table barrier does, and calls the helper only when the
target is black (chameneos calls it on every store today). A global read after a
call reloads `_ENV`, re-tests its tag and shape and reloads the slot (13
instructions, twice per node in recursive-fib-gv); a cell per global that holds
the value and is invalidated on write (V8's `PropertyCell`) would make it a
constant; that needs its own design (17.3 F5).

`LOAD`, `LOADI`, `LOADNIL`, `LFALSESKIP`, `MOVE` are constants and copies in SSA
and emit nothing by themselves.

### 7.6 Tables, arrays, iteration

| Site | Code |
|---|---|
| `GETTABLE` with a key proven `I32` | receiver `Guard TAB`; `ldr n, [t + ASIZE]; cmp idx, n; b.hs slow; ldr arr, [t + ARRAY]; ldr v, [arr, idx, lsl #3]` (key `k` is at index `k`, 4.7); a nil result reads the class bits live (`TabMmBits`: shape, class, bits, 0 without a class) and goes to the slow helper `jit_index(rt, t, k) -> Value or FAIL` when `INDEX` is set; other keys: the helper |
| `SETTABLE` with an `I32` key | bounds check, load old value, a nil old value with the live `NEWINDEX` bit takes the helper, else barrier, store; other keys the helper `jit_newindex`, which returns `FAIL` for a metamethod |
| `NEWTABLE` | helper `Table::from_template`; an allocation for 5.7 |
| `SETLIST` | helper; a constant count has the values in home slots (R1 stores them first) |
| `LEN` | a string's length inline (one load); a table's class bits read live: without `LEN` the raw-length helper (the full border search; `raw_len_hint` alone misses after each append), with it, and for other values, the generic helper with its metamethod arm (7.2) |
| `TFORPREP` | the form it wrote is the feedback: `_NEXT` means `next` over a table from position 0; `_IPAIRS` the array walk |
| `TFORCALL_NEXT` | the interpreter's walk (`table/mod.rs:1467-1579`): parts in the order array, integer hash, named shape slots, dict strings, misc hash, the position a small int `part << 28 \| index + 1` in `R[a+3]` (`next` walks the same order, so a deopt mid-loop continues correctly); the array part (`ArrNext`, one machine instruction scanning for the next non-nil slot) and the named slots (`NamedNext`, the same over the shape's slots, inline then spilled, the metatable's slot skipped, then `NamedKey` and `NamedVal`) inline, the step from one part to the next and the other parts by helper (`jit_tfor_next`, through the slots); the position stays in the hidden slot's encoding at every tail-out and snapshot; `TFORCALL_NEXT` itself does the following `TFORLOOP`'s back-jump (`tfor_finish!`, `control.rs:138-140`), so the loop variables are SSA values and the nil test on the first ends the loop |
| `TFORCALL_IPAIRS` | `i + 1`, bounds check against `asize`, load, nil ends the loop |
| `TFORCALL` generic | a call (5.4) through `ret_tfor` semantics landing `count` results |
| `CLOSURE` | helper `jit_closure(rt, thread, base, proto_idx)` that does what `op_closure` does, including open upvalue creation and the `HAS_OPEN` flag; an allocation for 5.7; captured registers are stored before it (5.11) |

Changes from the measurements (17.3; milestone in brackets):

- **Array reads [4].** The nil and out-of-bounds arms call `jit_index`, which is
  declared `MAY_ALLOC | W_SHAPE | W_META` but only reads (raw gets and the
  `__index` walk; a function `__index` returns `FAIL`). Declared read-only, its
  arm keeps no collector check at the loop header and ends no fact. The builder
  reuses an earlier `GETTABLE` site's result for the same table and key with no
  write between them, as `field_loads` does for fields, so queen's three `a[i]`
  reads per iteration become one. The arms' calls go through 10.5's cold stubs,
  and a value only an arm uses (the boxed key) is computed inside it. An index
  that is a loop counter checks `asize` against the loop's `last` once before
  the loop (9.7).
- **Appends [6].** `SETTABLE` at `asize` with capacity left stores and bumps the
  size inline instead of calling `jit_newindex` (havlak's `Vector:append`).
- **`NEWTABLE` [6].** A template of known size is bump-allocated inline: the
  header, the template's shape and its initial values are written directly, with
  no `from_template` call, no nil fill of slots the following stores overwrite
  (`memset_pattern16`, 12.5% of binary-trees), and no barrier or shape guard on
  the fresh table, which is white and has the template's shape. A
  `setmetatable` with a constant metatable right after it allocates with the
  shape that transition gives (12). The allocation is an allocation for 5.7.
- **`CLOSURE` [6].** A closure whose upvalues are all by-value or already open
  is allocated inline (prototype, header, upvalue words); creating a new open
  upvalue stays in the helper. fixpoint-fact's `jit_closure` is 42% of its
  cycles, 17 cycles a closure.
- **Sinking [6].** A table or closure that does not escape the region after
  inlining is not allocated at all (9.10).

### 7.7 Calls

| Site | Code |
|---|---|
| `CALL*`, `CALLS*` with an unknown callee | the sequence of 5.4; the callee value is loaded from its SSA value and stored to `R[a]` |
| callee a compile-time constant native with an intrinsic | section 12 |
| callee a known native without an intrinsic | milestone 3: its fast entry's header core, else `enter` (5.4) |
| callee a known Lua closure (5.4's sources) | direct call (5.4, milestone 3); inlined when 13's conditions hold (milestone 5) |
| `b == 0` (arguments up to `top`) | the count from `thread.top` (`jit_top_args`); when the arguments end in a call whose callee is known to return one value, the inner call is compiled with `c == 2` and the outer one with a constant count, so `math.sqrt(f(x))` reaches its intrinsic (17.3 B2) |
| `TAILCALL` | 5.5 |
| `RETURN*` | 5.5 |

### 7.8 Everything else

`VARARG`, `VARARGGET`, `VARARGPREP`, `CLOSE`, `TBC`, `ERRNNIL` (an `is_nil` test
whose failure deopts so the interpreter raises), `NOP`, `STOP` (deopt), `JMP`
(control flow only). `CONCAT` and `LEN` are in 7.2 and 7.6.

## 8. The IR

### 8.1 Shape

A region is one `Func`: a control-flow graph of basic blocks with block parameters
(no phi instructions), instructions in SSA form, a constant pool outside the IR,
and snapshots in a side table. Every structure is a flat arena indexed by a
`u32` newtype, and every list is a run of a pool; nothing inside an instruction
or a block is a `Vec`.

```rust
pub struct Func<'gc> {
    pub blocks: Vec<BlockData>,        // params: run of ppool, insts: run of layout, resume/dead/peeled
    pub insts:  Vec<InstData>,         // op, args: run of vpool, results: Val range, edges: run of edges, snap, tag, block
    pub vals:   Vec<ValData>,          // ty: Ty, def: ValDef (Inst(i, k) | Param(b, k))
    pub vpool:  Vec<Val>,              // operands and edge arguments
    pub ppool:  Vec<Val>,              // block parameters
    pub layout: Vec<Inst>,             // blocks' instruction lists
    pub edges:  Vec<BlockCall>,        // target: Block, args: run of vpool
    pub snaps:  Vec<SnapData>,         // pc, kind, entries: run of snap_pool (8.6)
    pub snap_pool: Vec<(u8, Val)>,
    pub pool:   Vec<Value<'gc>>,       // the heap values the code embeds; the only `'gc` holder
    pub entry:  Block,
    pub meta:   RegionMeta,            // entry pc and kind, max_stack_size
    cfg:        RefCell<Option<Rc<CfgInfo>>>, // RPO, predecessors, incoming edges, dominators
}
```

An instruction's arguments are a run of `vpool` and its results a range of
values, so an instruction with any arity costs no allocation; a block's
parameters and instructions and an edge's arguments are runs too. A run that
grows (a pass inserting an instruction, the builder adding a parameter or an
edge argument at a join) and does not end its pool moves to the pool's end; a
copy of the old handle keeps reading the old elements, so a pass may walk a
block's old list while it inserts. Instruction ids are stable, so nothing is
renumbered. The control-flow facts every pass wants (RPO, predecessors, the
incoming edges of each block, dominators) are computed once per CFG and shared,
and any change to an edge's target or to the set of blocks drops them. The
first measurement of the pipeline (section 15) found 40% of the passes' time in
malloc; this storage and the shared facts took the passes on `mandel`'s inner
loop from 137 µs to 65 µs.

### 8.2 Types

```rust
pub struct Ty { rep: Rep, set: TypeSet, refine: Refine }

pub enum Rep { Val, I32, I64, F64, B1, Ptr }

bitflags TypeSet: u16 {
    NIL, FALSE, TRUE, SMALL, BIGINT, FLOAT, STR, TAB, FUN, THR, UDATA
    // derived: BOOL, INT = SMALL|BIGINT, NUM = INT|FLOAT, HEAP = STR|TAB|FUN|THR|UDATA|BIGINT, FALSY, TRUTHY, ANY
}

pub enum Refine {
    None,
    Const(PoolIdx),       // the exact value (an immediate, or an object by identity)
    Proto(PoolIdx),       // a Lua closure of this prototype (identity unknown)
}
```

A refinement is a fact about the value, so it must hold for as long as the value
exists: a type tag, an identity, a closure's prototype. A table's shape is not one
of those. A transition, a generic store helper, `setmetatable` or a call changes
the shape of every alias of the table, so shape knowledge is never part of `Ty`:
it lives in `GuardShape` instructions, which read the `R_SHAPE` effect class and
are removed only by GVN (8.4, 9.4).

`Rep::I32` implies `SMALL`, `I64` implies `INT`, `F64` implies `FLOAT`, `B1` implies `BOOL`, `Ptr` a
single heap type. `Val` carries any set. The lattice join is the union of sets
with the refinement kept only when both sides agree. Every value has a `Ty`; the
builder assigns the speculated type, inference (9.2) narrows it, and guards
produce refined copies: `Guard(v, set) -> v'` where `v'` has `set ∩ ty(v)` and the
same `Rep`; `Unbox` changes `Rep`.

### 8.3 Operations

Grouped; each has a fixed arity, a result type rule, and an `Effects` value.
`Op` payloads are in brackets.

Constants and moves: `Const[pool]`, `Nil`, `Bool[b]`, `I32[imm]`, `F64[pool]`.

Register file: `Load[r]` (home slot, `Val`), `Store[r](v)`, `EntryLoad[r]` (as
`Load`, only in the prologue), `Sync[pc]` (publish `top_base`/`top_pc`),
`SetTop(v)`, `GetTop`.

Boxing and tests: `Unbox{I32|I64|F64|Ptr}(v)`, `Box{I32|I64|F64|B1|Ptr[tag]}(v)`,
`ToF64(v) -> F64` (a small integer or float `Val`; anything else exits),
`IsType[set](v) -> B1`, `IsFalsy(v) -> B1`, `SameBits(a, b) -> B1` (bit
identity), `Select(c, a, b)`.

Guards (all carry a snapshot, all deopt `Before` unless noted): `Guard[set](v) ->
v'`, `GuardShape[pool](t) -> t'`, `GuardConst[pool](v) -> v'` (pointer or bit
identity), `GuardIndexTable[pool](shape)` (class `__index` identity),
`GuardCond(b1)`, `GuardNotNil(v) -> v'`, `GuardFits(v)` (`stack_end` room for a
staged header), `Deopt` (unconditional, `Before`), `GcCheck` (tag `Gc`, `Before`,
leaving through `Exit::Gc`, 5.7).

Integer (`I32` in, `I32` out): `IAdd`, `ISub`, `IMul` (overflow deopts; carry a
snapshot), `IAddNo`, `ISubNo`, `IMulNo` (proven not to overflow, 9.7), `INeg`,
`IDivFloor`, `IModFloor` (zero divisor deopts; expanded before lowering, 10.3),
`IAnd`, `IOr`, `IXor`, `INot`, `IShl`, `IShr` (Lua count semantics, expanded),
`ICmp[cc] -> B1`, `IToF -> F64`.

Integer (`I64` in, `I64` out, wrapping, no snapshot): `LAdd`, `LSub`, `LMul`,
`LNeg`, `LAnd`, `LOr`, `LXor`, `LNot`, `LShl`, `LShr` (Lua count semantics),
`LDivFloor`, `LModFloor` (zero divisor deopts; `mininteger`/`-1` by `csel`),
`LCmp[cc] -> B1`, `LUDiv` (unsigned, for a loop's trip count), `IToL` (`sxtw`),
`LToI` (deopts outside i32; used by range narrowing and array keys),
`LToF -> F64`.

Float: `FAdd`, `FSub`, `FMul`, `FDiv`, `FNeg`, `FAbs`, `FSqrt`, `FFloor`, `FCeil`,
`FFloorDiv` (`fdiv` then `frintm`), `FMod` and `FPow` (helpers, canonical-NaN
fixup on the result), `FCmp[cc] -> B1` (unordered is false), `FToIExact -> I32`
(deopts when not an exact i32; used by `floor`/`ceil` intrinsics, array keys and
float loop limits), `FToL -> I64` (`fcvtzs`, saturating; a float limit of an
`I64` loop).

Tables (`Ptr` receivers, offsets from the layout module): `TabShape(t) -> Ptr`,
`TabLoad[loc](t) -> Val`, `TabStore[loc](t, v)`, `TabSpill(t) -> Ptr`,
`TabArray(t) -> Ptr`, `TabASize(t) -> I32`, `ArrLoad(arr, i) -> Val`,
`ArrStore(arr, i, v)`, `TabSetShape[pool](t)`, `TabHasRoom[loc](t) -> B1`,
`ShapeClass(shape) -> Ptr`, `ClassBits(class) -> I32`, `ClassMm[idx](class) ->
Val`, `ClassIndexTable(class) -> Ptr`, `Barrier(obj)`, `NewTable[template]`,
`Len(t)`.

Upvalues and closure: `UpvalValue[i] -> Val`, `UpvalCell[i] -> Ptr`,
`CellLoad(c) -> Val`, `CellStore(c, v)` (helper for the barrier), `Closure[proto]`.

Calls and frames: `Call[site]{a, nargs, wanted}(callee, args…) -> results` (a
tail-out and resume; `wanted` results as `Val`), `CallMm[site]{cont}(f, args…) ->
Val`, `Return[a, n](vals…)`, `TailCall[a, nargs](callee, args…)`, `Helper[fn](args…)
-> results` with the helper's declared effects and `may_fail` (a `FAIL` return
deopts `Before`).

Control: `Jump(target, args)`, `Br(c, then, else)` with block-call arguments on
both edges, `Exit[kind]`.

Added in milestone 2, by their implementation names: `NumVarargs`, `VarargAt[k]`,
`VarargIdx` (7.8); `ClassMm[idx](t)` (a table's class metamethod, nil without),
`TabMmBits(t)` (a table's live class bits, 0 without a class), `NumMtSet` and
`TypeMtIs[slot, word]` (a type's metatable cell, 11.5); `ArrNext`, `NamedNext`,
`NamedVal`, `NamedKey` (7.6); `StrLenOf(s)`; `FnFixedLua(f)`, `FnFits(f)`,
`TailEnter[nargs](f)` (5.5); `Dispatch[pc]` (a tail-out to the interpreter's
handler of the instruction at `pc`, the frame in its slots); `HasFlag[mask]` and
`GuardNoFlag[mask]` (frame flags); `Refine[set](v)` (a type a dominating test
proved; rematerializable after a call, 9.8).

Generic Lua operations as helpers: `LuaArith[kind]`, `LuaConcat`, `LuaLen`,
`LuaIndex`, `LuaNewIndex`, `LuaForPrep`, `LuaNext`, `LuaVararg`. Each is a
`Helper` with a fixed signature. The compares are `Helper[Lt|Le|Eq](a, b) -> I32`
(`jit_lt`, `jit_le`, `jit_eq`: 1, 0, or `FAIL` = 2, tested by a `GuardFalse` that
deopts `Before`, 7.3).

### 8.4 Effects and alias classes

```rust
bitflags Effects: u32 {
    MAY_DEOPT, MAY_ALLOC, TAILOUT, TERMINATOR,
    R_SLOT, W_SLOT,          // home slots: explicit Store/Load
    R_TAB, W_TAB,            // named slots, spill cells, array parts (keyed, below)
    R_SHAPE, W_SHAPE,        // shape words
    R_META, W_META,          // class bits and metamethod slots
    R_UPVAL, W_UPVAL,
    R_TOP, W_TOP,
    R_GC, W_GC,              // gc_due counters (allocations write)
}
```

A `Call`, `CallMm`, `TailCall` or a helper flagged `RUNS_LUA` writes every class.
A `NewTable` writes `W_GC` only. `TabShape` and `GuardShape` read `R_SHAPE`;
`TabSetShape` writes `W_SHAPE`; generic store helpers (`jit_newindex`, `rawset`)
and `setmetatable` write `W_TAB | W_SHAPE | W_META`. A `TabStore` through an IC
entry writes `W_TAB` only: it stores into an existing slot or follows a
transition whose `TabSetShape` is separate, and it never stores a mirrored key on
an adopted shape (4.7), so it changes neither shapes nor class bits.
`ShapeClass(shape)` of a constant shape is a constant (the shape pins the class
pointer); `ClassBits`, `ClassMm` and `ClassIndexTable` read `R_META`.

`R_TAB`/`W_TAB` are disambiguated by key, as LuaJIT's `aa_ahref` does
(`LuaJIT/src/lj_opt_mem.c:126`): `TabLoad`/`TabStore` carry the site's constant
key, two accesses with different constant keys never alias, and named-slot
accesses never alias array-part accesses (`ArrLoad`/`ArrStore`), which are
string keys and integer keys. Accesses with the same key alias whenever their
receivers may be the same table. So a loop doing `p.x = p.x + p.vx` still forwards
and hoists a `math.sqrt` lookup.

GVN and load forwarding use these: a load is forwarded across an instruction that
does not write its class (and key); a guard is removed when a dominating equal
guard exists and no write to its class intervenes.

### 8.5 The constant pool

`ConstPool<'gc>` holds `Value`s, shapes, prototypes, strings, helper addresses and
code addresses, interned by identity, and is the only structure in the compiler
with a `'gc` lifetime. The region keeps it after compilation as its root set
(section 11.2): everything a region's code embeds as an immediate pointer is in the
pool and traced from the region.

### 8.6 Snapshots

A snapshot describes how to rebuild the Lua frame(s) at an exit. It is built by
the builder at every instruction that may deopt, from the register map and
liveness at that pc, and pruned to live registers whose slot does not already hold
their value. In the IR a snapshot is a pc, a kind and a run of entries in one
pool; a snapshot equal to the one before it is shared (most guards of one
instruction share one), and a pass that changes one an instruction may share
copies it first. The region's are one `u32` arena, with exits whose encodings are
equal sharing one; Appendix C has the encoding. Before
lowering, each snapshot's value references are ordinary IR uses (they keep values
alive and are the allocator's `Any` uses on the guard instruction); after
allocation they are replaced by locations. A value that is a constant or a
rematerializable unbox of an unchanged slot is recorded as such and is not an
allocator use, which is what avoids the old design's pinned constants.

### 8.7 The builder

One pass over the bytecode from the entry pc:

1. **CFG**: block boundaries from branch targets and fallthroughs (the ported
   `cfg.rs`), with `JIT_ENTRY`/`JIT_LOOP` words resolved to their originals and
   each back edge that jumps past a `LOOP` pointed at it, so a loop's header is
   its `LOOP` whether or not an entry pointed its back edges there, and
   the `SET`-form branches (`JTSET`/`JFSET`) and `TFORCALL` edge definitions
   handled as edge-block assignments. Loop headers and the loop forest come from
   `order.rs`. Bytecode liveness per block (backward dataflow over `reg_uses`/
   `reg_defs`) gives the live-out sets snapshots and tail-outs need.
   **Loop entries inside a nest** are built as V8 builds an OSR entry
   (`AdvanceToOsrEntryAndPeelLoops`, `v8/src/compiler/bytecode-graph-builder.cc:1390-1460`).
   Entered at loop `L_n` inside `L_{n-1} ⊂ … ⊂ L_0`, the builder emits `L_n` from
   its entry, then the rest of `L_{n-1}`'s iteration; at `L_{n-1}`'s back-edge it
   emits no edge but continues at a fresh copy of `L_{n-1}`'s header, built as a
   whole loop (containing a fresh `L_n`); and so on outward until `L_0` is built
   whole with the rest of the function. Entering at `L_n`'s header without this
   would make that header dominate the whole region: the outer back-edges would
   stop being back-edges, the nest would collapse into one loop with no inner
   preheader, and LICM and peeling would lose the inner loop. The copies cost code
   size: `L_n` appears once per enclosing level plus once. Peeling stops at
   `OSR_PEEL_LIMIT = 150` duplicated bytecode instructions, about 400 IR
   instructions at the three per bytecode instruction the builder emits: past it
   the outermost remaining levels stay merged, their back edges entering the
   instances already made, so they collapse into the entry's loop. The builder
   counts in bytecode because it fixes the copies (its instance graph) before it
   emits any IR. An entry at the entry loop's header has run a whole iteration
   before the loop proper, which peeling (9.3) does not repeat; one at its latch
   (a `FORLOOP`, a backward `goto`'s `JMP_BACK`) has run only the latch.
2. **Captured registers**: the set of registers any `CLOSURE` in the function
   captures by reference, from the child prototypes' `upvalue_desc`
   (`ParentLocal(r)` with `by_value == false`), for the whole function. They
   are ordinary SSA values with the extra stores and reloads of 5.11.
3. **Emission** in reverse post-order with the Braun SSA builder (`ssa.rs`):
   `read_var(block, r)` gives the SSA value of register `r`, creating block
   parameters at joins lazily; `write_var` defines. Each instruction is emitted
   by a table-driven function keyed on the opcode (Appendix D) that reads the
   site's feedback (7.1) and emits guards, the operation, and the result write.
   Snapshot references are taken from the register map at the instruction's pc,
   pruned by live-out.
4. **Slot state**: alongside the register map, the builder tracks for each
   register which SSA value its home slot holds (`SlotHolds[r] = Some(val) |
   Unknown`). A `Load` sets it; a `Store` sets it; a tail-out or a deopt invalidates
   nothing (slots are written by the tail-out's own stores, which set it). At a
   tail-out the builder emits `Store[r](v)` for each live register with
   `SlotHolds[r] != Some(v)`. At a join, `SlotHolds` is the meet (equal or
   `Unknown`). This replaces the old design's whole-frame sync and is what keeps
   loops free of stores: a loop-carried value in a register whose slot is stale is
   stored once per tail-out, not once per iteration.
5. **Calls** produce the `Call` instruction and, after it, every register still
   live is read again with `Load` on first use (the register map is reset to
   "in slot" for every register, captured ones included, since the callee may
   have written them through an open cell), since the resume brings nothing in
   registers but the results.
6. **Never-executed blocks** (6.3, decided after step 1 from the CFG and its
   dominator tree) emit one `Deopt(Before)` at their first instruction; blocks
   reachable only through them are not emitted.
7. **Sealing**: blocks are sealed when all predecessors are emitted; loop headers
   seal at the back-edge, which is when Braun's algorithm completes their
   parameters; trivial parameters are removed by the existing `simplify`.

The result is verified (8.8) before any pass runs.

### 8.8 Verifier and printer

The verifier checks SSA dominance, operand types against each op's rule, block
parameter arity at every edge, that every `MAY_DEOPT` instruction has a snapshot
whose referenced values dominate it, that no instruction follows a terminator,
that every `Store` is a tail-out write-back or a captured-register store before
a closing helper, that no virtual register is live across a tail-out (R6), and
that every tail-out is followed only by its resume block. The printer emits one line per
instruction with types and snapshots, which the snapshot tests (`insta`) record.

## 9. The optimizer

Passes in order; each is a function `Func -> Func` or an in-place rewrite, with a
verifier run after each in debug builds.

### 9.1 Use-driven speculation

Values enter a region without a type: parameters, the live registers at a loop
entry, field and table loads, upvalues and globals, call results, `VARARG`. JSC
types such values where they are defined, from value profiles the LLInt writes on
every execution (argument profiles, `JavaScriptCore/bytecode/CodeBlock.h:420-424`;
load and call results). tcvm has feedback only at uses (the consumers' forms and
feedback bytes), so this pass reconstructs a definition's type from its uses and
guards the value once, where it is defined, instead of once per use.

1. **Expectation per use**, from the consumer's feedback:

   | Use | Expectation |
   |---|---|
   | operand of an `_II`/`_I` form | `SMALL` |
   | operand of an `_FF`/`_F` form | `FLOAT` |
   | operand of `_IF`/`_FI` | `SMALL` on one side, `FLOAT` on the other |
   | operand of a generic site whose byte has only integer kinds | `INT` (an `I64` candidate) |
   | receiver of a field form | `TAB` |
   | callee of a call | `FUN` |
   | generic compare, store, return, call argument | none (uninformative) |

   Uses in blocks the never-executed rule cuts (6.3) do not count.
2. **Backward propagation** to a fixpoint over the SSA uses: a value's expectation
   is the intersection of its informative uses' sets, and a block parameter passes
   its expectation back to its incoming arguments, so loop-carried values take
   part. This is the direction of V8's truncation propagation
   (`v8/src/compiler/simplified-lowering.cc:64-72,714-722`) and JSC's backwards
   propagation of use flags (`JavaScriptCore/dfg/DFGNodeFlags.h:62-69`), which
   carry how a value is used rather than a type to guard.
3. **Conflict**: an empty intersection (one use quickened to `_II`, another to
   `_FF`) means the value is polymorphic; it gets no definition guard and keeps
   per-use guards.
4. **Placement**: the guard goes at the earliest point after the definition from
   which every path reaches an informative use expecting that set before the
   value's last use, the anticipability ("down-safety") condition of partial
   redundancy elimination. A guard placed there deopts only executions that a
   per-use guard would deopt later, so it never adds a deopt, only moves one
   earlier. Its snapshot is `After` the defining instruction, so the
   interpreter runs again what lies between: when that point is not right after
   the definition, it is the start of each of the earliest blocks the definition
   dominates from which the condition holds, with the guard testing the uses that
   block dominates, and only where no path from the definition to it writes or
   calls. Uses no guard covers keep their own.
5. **Entries are the exception.** Parameters and the live registers at a loop
   entry are guarded in the prologue even where that point is not down-safe: a
   failed entry guard costs one interpreted run of the original instruction
   (`jit_entry_fail`, 6.4), and the prologue needs the types. At a loop entry a
   register's expectation holds only when the kind of the live frame's value (J7)
   and the kinds failed entries saw fit it, and the register is not speculated
   otherwise; the expectation is kept rather than narrowed to the frame's one
   kind, since it may already cover kinds this frame does not show (an `INT` from
   a site that saw big integers). A register whose uses expect nothing is typed
   by the frame's kind and the seen ones, which gives generic compares, truth
   tests and arithmetic on it a type.
6. **After the guard** the value is refined and unboxed once; inference (9.2)
   carries the type to every use, which deletes their own guards.

A wrong speculation corrects itself through the interpreter: after a definition
guard deopts, the interpreter runs the uses with the other type, their forms miss
and re-quicken or go generic, and the next compile sees conflicting or widened
expectations and leaves the value to per-use guards. The cost is one backward
fixpoint over the uses and one bit-vector anticipability dataflow with a bit per
candidate value.

Example, `primes2`'s `is_prime(n)`: `n` is used by `n < 2` and `n == 2`
(uninformative), `n % 2` (a small-int form), `i * i <= n` (`JLE_II`) and `n % i`
(`MOD_II`). They agree on `SMALL`, the prologue guards `n` once, and all three
compares are `cmp w`. Counter-example, `local v = t.x; if v > 0 then s = s + v end`
with the add `_FF`: the path where `v > 0` fails never uses `v` as a float, so the
guard stays at the add and the compare stays generic.

### 9.2 Type inference

Optimistic forward dataflow to a fixpoint over the SSA graph: every value starts
at the bottom type, instructions apply their result rules, block parameters join
their incoming arguments, guards narrow. Because speculation already typed each
site, inference's job is to carry types through copies, block parameters and
loop-carried values so that a loop counter typed `I32` by `FORPREP` keeps its
`I32` through the back-edge, an accumulator typed `F64` by its first `_FF` add
stays `F64`, and a value guarded `TAB` once stays `TAB`. Where inference proves a
type guard's set already holds, the guard is deleted. Shape guards are not types
(8.2) and are left to GVN. Where a block parameter joins `I32` and `F64` the result is `Val`
`NUM`, and its consumers keep their guards; that is the case block duplication
would address later (J4).

### 9.3 Loop peeling

Each innermost loop of the region whose body is at most `PEEL_LIMIT = 48`
instructions is peeled once: the body is copied before the header, the copy's exit
edges join the original loop's exits, and the original header's parameters are
fed only by the back-edge and the copy's end. Inference then runs again on the
loop, where the steady-state types are no longer joined with the entry types, and
guards that held in the peeled iteration and whose values are loop-invariant are
proven by GVN in the loop (9.4). Peeling is what makes the `_II` counter loop and
the `_FF` accumulator loop guard-free in their steady state. Outer loops are not
peeled here; the peeling of enclosing iterations at a loop entry (8.7) is a
different transformation and runs in the builder. The one exception is the loop a
region enters at its header: the builder already put a whole iteration in front
of it (8.7). A loop entered at its latch is peeled like any other.

Constant folding of integer and float arithmetic (with the interpreter's floor
division and modulo, and only where the operation would not exit) runs in the
simplifier, so the peeled iteration's operations on the loop's start values fold.

### 9.4 Global value numbering with guard elimination

Dominator-order hashing of pure instructions by op and arguments. Guards hash by
op, argument and payload, and a guard is redundant when an equal guard dominates
it with no intervening write to the class it reads (8.4). Loads (`TabLoad`,
`Load`, `UpvalValue`, `CellLoad`, `ClassMm`) are value-numbered with the last
write to their class as an extra key, which gives load forwarding for free within
a block and across blocks along the dominator tree.

### 9.5 Loop-invariant code motion (milestone 2)

For each loop, instructions whose arguments are defined outside the loop and
whose effects read only classes the loop does not write are moved to the
preheader. Guards are moved under the same rule, and only from blocks that
dominate every latch of the loop: a guard in a conditional arm would deopt
iterations that never reach it. A hoisted guard takes a new snapshot, `Before`
the loop header with the preheader's state, and deopts at the loop entry, which is
correct because the loop's first iteration would have deopted there too (peeling
guarantees the first iteration's guards are present before the loop). Allocating
instructions and calls are never hoisted. A memory read moves only while no exit
that stays in the loop comes before it in the body, since that exit may be what
makes the read valid (a shape guard before a slot load). In a loop that calls
(`Call`, `CallMm`), only what the call-boundary pass rematerializes after a call
moves (9.8): constants, pool loads, slot unboxes, refinements.

LICM takes the snapshot of a hoisted exit from the header's `GcCheck`, which
the builder puts at every loop header and `prune_gc_checks` removes only after
LICM, so pruning the checks of loops that only call (5.7) does not stop hoisting.
queen's `c - n` and `c + n` (an `ISub`/`IAdd` with an overflow
exit, both operands parameters of the function) stay in its inner loop today;
milestone 4 finds which condition keeps them (17.3 Q6).

### 9.6 Dead code elimination and narrowing

Unused pure instructions are deleted; `Box` of an `Unbox` and `Unbox` of a `Box`
fold; a `Val` that is only ever unboxed to `I32` is kept unboxed from its
definition when the definition is a boxed `I32` producer (narrowing of block
parameters: a parameter whose every incoming argument is `Box(I32)` becomes an
`I32` parameter with the boxes moved to the uses that need `Val`).

### 9.7 Range analysis (milestone 4)

Interval analysis over `I32` values: constants, `FORPREP` bounds (`last` and the
step give the counter's range), compares on dominating edges, `asize` bounds for
array indices. An `IAdd`/`ISub`/`IMul` whose operand ranges keep the result in
i32 becomes the `No` form without the overflow branch; an `I64` value whose range
fits i32 is narrowed to `I32`, which boxes with one `orr` instead of a range test;
an array index proven in `[0, asize)` by a dominating compare loses its bounds
check. This is V8's
`maglev-range-analysis` idea scoped to what Lua loops need. An index that is a
loop counter, or the counter plus a constant, has its bounds check replaced by
one check of the loop's range against `asize` in the preheader when the loop
writes no array size (LuaJIT's `rec_idx_abc`, `lj_record.c:1398-1435`, with the
scalar evolution of the `FORL` counter); the
exit is the header's, so a loop that would go out of bounds deopts at its entry
and runs in the interpreter. fib's `n - 1` and `n - 2` lose their overflow checks
on the `n >= 2` edge.

### 9.8 Call boundaries, snapshot pruning and critical edges

The call-boundary pass enforces R6. For each tail-out that resumes (`Call`,
`CallMm`) it computes the values live across it; each must be a constant, a pool
load or an unbox of a slot reloaded after the resume, and is re-emitted after the
resume point with its uses below rewritten. Anything
else live across a tail-out is a builder bug, and the verifier rejects it. This
is what lets GVN and LICM share constants freely and lets the backend keep
`NIL`, `BOX` and shape literals in registers between calls (10.3).

Milestone 1 has the constant part: the builder reloads every register after a
resume and GVN's scopes restart at resume blocks, so only constants can be live
across a call, which happens when GVN merges a constant into one that dominates
a join a resume path also reaches. Each such constant is re-emitted after the
`Resume` of every resume block it is live into, and its later uses get the
definition that reaches them (Braun et al.'s lookup, with block parameters where
both meet); afterwards the verifier allows nothing live into a resume block.
Milestone 2 adds pool loads, slot unboxes and refinements, which LICM moves out of
loops that call.

A rematerializable value used after resumes in several blocks is re-emitted at
each use rather than merged into block parameters: richard's main loop header
carries eight such parameters (an upvalue, three booleans derived from it, its
unboxed float, the constants 2 and 0), which the allocator keeps in spill slots
(one reload is 9.6% of that region's samples; 17.3 Q4).

Snapshots are then re-pruned against the final code: an exit's snapshot drops a
register whose value is a load of that register's own slot (through boxing, guards
and refinements) that no path from the load to the exit writes (a path through
the load's block again loads again): the slot still holds the value. Values the
builder's slot tracking could not see were unchanged (guarded copies that
speculation, GVN and peeling put in loop snapshots) then die at their last other
use; about a tenth of the entries of richard's and deltablue's exits go. A
constant entry is encoded in place (Appendix C) rather than kept in a register.
Critical edges are split by inserting empty blocks
so that `regalloc2`'s requirement holds and edge-block parameters have a home.

### 9.9 Block layout

`backend/order.rs` (the old JIT's, ported) lays out blocks loop-contiguously: each
loop collapses to its header, the collapsed graph is laid out in reverse
postorder, and each loop's own layout is spliced in where its header landed.
Blocks that end in a deopt run at most once per entry and go after the rest, so
loop bodies fall through; the exit stubs follow in the cold section (10.6).
Irreducible graphs, which Lua's `goto` can produce, are rejected before, by the
bytecode CFG (`cfg.rs`): such a region fails compilation with an internal limit
and is counted as a strike (6.2), never mis-compiled.

An empty block that only jumps is threaded away before layout: branches to it
go to its target. Milestone 2 leaves branch-to-branch chains at the joins that
the arms of generic sites and array reads make (six taken jumps per iteration
of queen's inner loop; 17.3 Q5).

### 9.10 Allocation sinking (milestone 6)

After inlining (13), a table or closure the region allocates (7.6's inline
`NEWTABLE` and `CLOSURE`) whose result does not escape is not allocated: its uses are loads and stores of constant slots (a template's
named slots, array slots with constant keys), `setmetatable` with a constant
metatable, the identity tests of inlined calls, and snapshots. LuaJIT's
`lj_opt_sink.c` is the model: it sinks three of the four tables per iteration of
mandel-metatable and records `setmetatable` as one store. In tcvm:

- *Analysis.* A value escapes when it is stored anywhere but into a sunk
  object's slot, passed to a call or a helper, returned, stored to a home slot at
  a tail-out (R1), compared by identity with anything but a constant, or carried
  around a loop's back edge (a loop-carried table, like mandel's `z`, stays
  allocated, as in LuaJIT). The analysis is a fixpoint over the values' uses,
  linear in the IR.
- *Replacement.* Each slot of a sunk object becomes an SSA value; GVN's store
  forwarding (9.4) already gives every load its stored value, so the loads
  disappear and the stores become dead.
- *Snapshots.* An exit whose snapshot refers to a sunk object materializes it:
  the snapshot gets an object entry (Appendix C: kind table or closure, the
  template or prototype, then one entry per slot), and `jit_exit` allocates and
  fills the object before writing the home slots. Objects referring to each other
  are allocated first and linked after. Nothing needs rooting, since no
  collection runs between tail-outs (5.7).
- *Order.* After inlining, GVN and LICM; before the call-boundary pass, which
  must not see a sunk value live across a tail-out (a sunk value live there
  escapes and is not sunk).

Estimates (Appendix F): mandel-metatable from 272 to 30-40 cycles per iteration
(LuaJIT 135, C with value structs 9), fixpoint-fact's two closures per
recursion level once its calls are inlined.

### 9.11 Tail duplication (milestone 4)

J4 kept block duplication in reserve; the measurements call for one use of it. A
receiver that a site dispatched on two or three shapes is tested again at each
later access, because the arms join after every access. From the first dispatch
on a receiver to the next call, loop header or join that brings in a different
receiver, the blocks are duplicated per shape, so each path keeps one shape
(GVN then removes the later tests). The duplication is bounded by a budget of
IR instructions per region (13's budget counts it), and a site over the budget
keeps the per-access dispatch.

## 10. The aarch64 backend

### 10.1 Structure

```
src/jit/
  mod.rs            region lifecycle, jit_hot, jit_exit, jit_recompile, jit_entry_fail
  layout.rs         offset constants with tests
  ir/               func.rs types.rs ops.rs pool.rs snap.rs verify.rs print.rs
  build/            cfg.rs ssa.rs builder.rs emit/*.rs (one file per opcode family)
  opt/              infer.rs peel.rs gvn.rs licm.rs dce.rs range.rs snapprune.rs
  backend/
    vcode.rs        target-neutral VCode container, VReg, operand model, block layout
    regalloc.rs     regalloc2 adapter (Function impl, MachineEnv per target, edits)
    code.rs alloc.rs  reused
    aarch64/
      inst.rs       MachInst enum
      lower.rs      IR -> MachInst
      abi.rs        pinned registers, prologue, tail-outs, exit routine, helper calls
      emit.rs       MachInst -> bytes via asm.rs
      asm.rs        the old encoder, extended
```

`vcode.rs` and `regalloc.rs` are generic over a `Target` trait: `type Inst`,
`fn operands(&Inst) -> &[Operand]`, `fn clobbers(&Inst) -> PRegSet`,
`fn is_branch/is_ret`, `fn machine_env() -> &MachineEnv`, `fn emit_move(from, to,
class)`. Appendix E lists what a second target implements.

### 10.2 `MachInst`

One enum with real aarch64 forms, operands as `VReg` (allocatable) or `PReg`
(pinned or fixed), immediates checked at construction:

- loads and stores: `Ldr{dst, mem, size}`, `Str`, `Ldp`, `Stp`, `LdrF`, `StrF`, with
  `Mem = Offset{base, imm12 scaled | imm9 signed} | Index{base, idx, lsl shift} |
  Literal{label}`;
- ALU: `AluRRR{op: Add|Sub|And|Orr|Eor|Lsl|Lsr|Asr|Mul|SDiv|...; dst, a, b, w/x}`,
  `AluRRI{op, dst, a, imm12 | logical imm}`, `AluRRRShift{..., shift}`,
  `AluRRRExtend`, `Madd`, `Msub`, `Smull`, `Adds/Subs` (flag-setting), `Cmp`,
  `Cmn`, `Tst`, `Csel{cond}`, `Cset`, `Ccmp`, `MovZ/MovK/MovN` sequences, `Adr`,
  `Sxtw`, `Uxtw`;
- float: `FAlu{op: Add|Sub|Mul|Div|Neg|Abs|Sqrt|Rintm|Rintp}`, `FCmp`,
  `FMovToGpr/FromGpr`, `Scvtf`, `Fcvtzs`, `FMovImm`;
- control: `B{label}`, `BCond{cond, label}`, `Cbz/Cbnz`, `Tbz/Tbnz`, `Br{reg}`,
  `Bl{helper}` (C ABI call), `TailOut{target, args}` (loads x20/x21 and branches),
  `ExitStub{id}`, `Ret`;
- pseudo: `Move{dst, src, class}` (allocator edits), `Prologue{frame}`,
  `Epilogue`, `GcCheck{exit}`, `Nop`.

Each variant knows its operands with constraints (`Use`/`Def`, `Any`/`Reg`/`Fixed`/
`Reuse`, early/late) and its clobbers (`Bl` clobbers the C caller-saved set,
x0–x17 and d0–d7, d16–d31).

### 10.3 Lowering

A backward pass over each block (so a use is seen before its def, which is what
lets a single-use def be folded into the use): for each IR instruction with
remaining uses, match the largest pattern rooted at it. Patterns, with what they
fold:

| IR pattern | aarch64 |
|---|---|
| `Br(ICmp[cc](a, b))`, single use | `cmp; b.cond` |
| `Br(ICmp[cc](a, I32[k]))`, k in imm12 | `cmp w, #k; b.cond` (negative k becomes `cmn`) |
| `Br(FCmp[cc](a, b))` | `fcmp; b.cond` with the unordered sense folded into `cond` |
| `Br(IsFalsy(v))` | `sub x9, v, x_nil; cmp x9, #2; b.lo` with `NIL` (`0xFFFF_FFFE_0000_0000`) materialized once per call-free span (`movz/movk`, or a load from the island; 9.8 re-materializes it after each resume point) |
| `Br(IsType[FLOAT](v))` | `cmp v, x_BOX; b.lo` with `BOX` in a register shared within a call-free span when used more than once, else `movz/movk` |
| `Br(IsType[SMALL](v))` | `lsr x9, v, #32; cmn w9, #1; b.eq` |
| `Br(IsType[tag](v))` | `lsr x9, v, #48; cmp w9, w_k; b.eq` with `k = 0xFFF8 \| tag` (too wide for imm12: a `movz` shared within a call-free span) |
| `Guard*` | the test and `b.cond` to the stub; the refined result is the same register (no move) |
| `IAdd/ISub(a, b)` with overflow | `adds w; b.vs stub` |
| `IMul` with overflow | `smull x; cmp x, w, sxtw; b.ne stub` |
| `IAdd(a, I32[k])` | `adds w, w, #k` or `subs` for negative k |
| `IDivFloor(a, b)` | guard `b != 0` (`cbz`), guard not (`MIN, -1`) (`cmn w, #1; ccmp`), `sdiv q; msub r = a - q*b; cmp r, #0; ccmp (a ^ b) < 0; sub q1 = q - 1; csel` |
| `IModFloor` | `sdiv; msub; add r2 = r + b; tst/eor; csel` |
| `IShl/IShr(a, b)` | Lua semantics on the 64-bit value (`num.rs:473-482`): `a >> n` is `a << -n`, a count with \|n\| ≥ 64 gives 0 (`cmp/csel`, since `lslv/lsrv` take the count mod 64), a negative count shifts right **logically** (`lsrv`, never `asrv`: `-1 >> 1` is `0x7FFF_FFFF_FFFF_FFFF`); the operand is `sxtw`-extended first and the result range-checked back to i32 by `cmp x, w, sxtw; b.ne stub`, which deopts a negative operand shifted right |
| `IToF` | `scvtf d, w` |
| `FToIExact` | `fcvtzs w, d; scvtf d2, w; fcmp d, d2; b.ne stub` |
| `Box{I32}(v)` | `orr x, x_v_zext, #0xFFFF_FFFF_0000_0000` after `mov w, w` (zero extend; folded when the producer was a w-register op) |
| `Unbox{I32}(Load[r])` | `ldrsw w, [base, #r*8]` |
| `Unbox{I64}(v)` | `lsr x9, v, #32; cmn w9, #1; b.ne big; sxtw x, w_v`; `big:` tag-2 test (`lsr x9, v, #48`, compare with `0xFFFA`, else the stub), `and p, v, #0xFFFF_FFFF_FFFF; ldr x, [p]` (the box's payload is at offset 0, `value.rs:190-198`) |
| `Box{I64}(x)` | `cmp x, w_x, sxtw; b.ne big`, then the small-int `orr`; `big:` `bl jit_box_i64` and a `GcCheck` exit (`After`), out of line in the cold section |
| `LAdd`, `LSub`, `LMul`, `LAnd`… | the 64-bit `add`/`sub`/`mul`/`and`… on x registers, immediates folded as for `I32` |
| `Unbox{F64}(Load[r])` | `ldr d, [base, #r*8]` |
| `Box{F64}(v)` → `Store[r]` | `str d, [base, #r*8]` (no GPR trip) |
| `Box{Ptr[tag]}(p)` | `orr x, p, #k` with `k = QNAN_NEG \| tag << 48` as a logical immediate when `k`'s ones are contiguous (tags 4 and 7), else `orr x, p, x_k` with `k` hoisted |
| `Unbox{Ptr}(v)` | `and x, v, #0xFFFF_FFFF_FFFF` |
| `TabLoad[loc](t)` | `ldr v, [t, #loc]` (inline, `loc` < 4096·8) or `ldr s, [t, #SPILL]; ldr v, [s, #off]` |
| `ArrLoad(arr, i)` | `ldr v, [arr, w_i, uxtw #3]` |
| `TabASize`/bounds | `ldr w, [t, #ASIZE]; cmp w_i, w; b.hs stub` |
| `GuardShape[s](t)` | `ldr x9, [t, #SHAPE]; cmp x9, x_s; b.ne stub` with `s` materialized once per call-free span (a literal load `ldr x, label` from the constant island, or `movz/movk` ×3; prefer the literal: one load, shared) |
| `Barrier(obj)` | `ldr x9, [obj, #-HDR]; tbnz x9, #0, skip; bl jit_barrier; skip:` |
| `GcCheck` | `ldr x9, [rt, #METRICS]; ldp x10, x11, [x9, #GC_CHECK]; cmp x10, x11; b.hs stub` |
| `Select(c, a, b)` from a compare | `cmp; csel` |
| `Call[site]` | the 5.4 sequence; the resume label is a block entry |
| `Helper[fn](args)` | moves into x0.. with `Fixed` constraints, `bl`, result in x0/d0 `Fixed` |
| `Store[r](v)` | `str x, [base, #r*8]`; `r*8` always fits the scaled imm12 |
| `Load[r]` | `ldr x, [base, #r*8]` |
| `UpvalValue[i]` | `ldr x, [closure, #UPVALS + i*8]` |

Expansions before lowering (shared, in the IR, section 9.6 runs after them):
`IDivFloor`, `IModFloor`, `IShl`, `IShr`, `FFloorDiv`, `IsFalsy` into primitive
compares and selects, so the second target lowers only primitives. The patterns
that fuse compare and branch, immediates and addressing modes are the per-target
part.

What milestone 2's code shows the lowering still lacks (Appendix F; milestone 4):

- A branch on a boolean, or on a select of type tests, materializes it
  (`cset; cmp; b.eq`, `cset`/`csel` chains in generic `==`); it should branch
  directly (`tbz`/`cbz`, or one conditional branch per test). GVN folds the
  selects the generic arms leave (`select(c, c, false)`, `select(c, k, k)`)
  before lowering.
- `Guard(SMALL | BIGINT)` followed by `Unbox{I64}` tests smallness twice; the
  pair lowers as one test.
- Wide constants (`BOX`, `NIL`, tag words) are not shared within a call-free
  span as the table above says: one ray region materializes the `BOX` bound 25
  times. A per-span constant cache in the lowering, invalidated at calls.
- A compare that folded its immediate leaves the `mov` of the constant dead.

### 10.4 Register allocation with `regalloc2`

The VCode implements `regalloc2::Function`: instructions in block order,
`block_succs`/`block_preds`, `block_params` as the IR's block parameters mapped
to `VReg`s, `branch_blockparams` on `B`/`BCond` from the edge arguments,
`inst_operands` from each `MachInst`, `inst_clobbers` from `Bl`, `is_ret` for
`TailOut`/`ExitStub`/`Ret`, two register classes (`Int`: x0–x15, x19–x21, x26–x28;
`Float`: d0–d30), `scratch_by_class` x17 and d31 for parallel-move cycles, which
regalloc2 requires to be outside the allocatable sets
(`regalloc2-0.15.2/src/lib.rs:1535-1551`), and `spillslot_size` 1 for both.
`MachineEnv.preferred_regs_by_class` lists the C caller-saved registers so values
that do not live across `bl`s stay cheap, and `non_preferred_regs_by_class` the
callee-saved ones (x19–x21, x26–x28, d8–d15). `fixed_stack_slots` is empty. A
tail-out that resumes is modeled as a branch to its resume block with nothing
live across it (R6).
The output's `allocs` give each operand's register or spill slot, `edits` give the
moves and spill/reload to insert at program points, and `num_spillslots` sizes
the native frame. `Algorithm::Ion` by default; `Fastalloc` behind
`TCVM_JIT_FASTALLOC=1` for compile-time comparisons. The `checker` feature runs in
debug builds and under `TCVM_JIT_CHECK=1`.

Snapshot references after allocation: for each guard, its snapshot entries that
were allocator uses are rewritten from the guard's `allocs` (a register image
slot index or a spill slot); constant and rematerializable entries are untouched.

Measured at milestone 2 (17.7): `Ion` costs 0.4-0.8 µs per IR instruction, about
a third of compile time; `Fastalloc` is 4-13× faster but its code retires
10-51% more instructions, so it stays a comparison tool. Two things inflate the
allocator's work and its spills: helper calls on cold arms, which clobber the
caller-saved registers at every hot-path value live across them (fixed by the
cold stubs of 10.5), and constants and block parameters that could be
rematerialized (9.8). If `Ion` still costs too much after both, the option is a
linear scan for straight-line function regions with `Ion` kept for loops.

### 10.5 Prologue, frame, tail-outs

```
prologue:
  stp  x29, x30, [sp, #-F]!      ; F = 16 + 8*num_spillslots rounded to 16
  mov  x29, sp
  (entry guards; unboxing)
tail-out (call, return, deopt, dispatch):
  (R1 stores; header stores; sync)
  ldp  x29, x30, [sp], #F
  mov  x20, ... ; mov x21, ...   ; (x22-x25 are already right)
  b    target
resume_K:
  stp  x29, x30, [sp, #-F]!
  mov  x29, sp
  (land results; continue)
```

Spill slots are `[x29 + 16 + i*8]`. Helper calls (`bl`) are made with sp 16-byte
aligned (`F` is a multiple of 16) and x29 a valid frame pointer, so a Rust panic
or a profiler walking from inside a helper sees a normal chain. The exit routine
reads spill slots through x29 before popping.

The frame is pushed at entry and at every resume through a header. Since J10 a
tail-out does not pop it: it resets the stack pointer, frame pointer and return
address to dispatch's entry (`NativeBase`, 5.4), which drops this frame and those
of any compiled callers under it; a fast call keeps it, and the callee's `ret`
lands past the resume block's frame push. A region's `RETURN` pops its own frame
and returns with `ret` when the frame holds a compiled caller's return address.
Helper calls on cold arms go through out-of-line stubs that save the
caller-saved registers the hot path keeps live, so the allocator sees no clobber
on the hot path and does not spill around a call that rarely runs (17.3 Q1).

### 10.6 Exits and the code segment

Code segments are the old allocator's: dual-mapped RW/RX chunks of 64 KiB in
64-byte units, with one change: the RX alias is placed within direct-branch reach
of tcvm's text, as LuaJIT places its machine code (`LuaJIT/src/lj_mcode.c:264-274`).
Every branch a region makes into Rust code is then a direct `b`/`bl`: `b enter`,
`b jit_exit`, `b jit_recompile`/`jit_entry_fail`, the entry-fail and deopt
tail-outs into opcode handlers, and `bl` to helpers. The old `map_dual` lets the
RX alias land wherever the kernel picks; the old JIT never branched into Rust, so
it never needed more.

- **Window.** All branch targets (handlers, `slow` routines, helpers) lie in the
  text of the image that contains tcvm, so the window is ±128 MiB around that
  text minus the text's own size (about 1.3 MiB today), computed once from the
  image's text bounds (dyld on macOS, `dl_iterate_phdr` on Linux).
- **Placement.** `mach_vm_remap`/`mach_vm_map` on macOS, `mmap` on Linux, given a
  target address inside the window and checked on return; a probe loop walks the
  window in 64 KiB steps around the text, both directions, as LuaJIT's
  `mcode_alloc` does. Segments are allocated in one reserved run where the system
  allows it, so region-to-region branches (exits, milestone 3 direct calls) are direct as
  well.
- **Exhaustion.** Code memory is bounded by the window (well over 100 MiB). When
  no segment fits, `jit_hot` refuses new compiles and logs it once; code cannot be
  flushed while frames hold resume points into it (11.2), and a prototype's code
  returns to the free list when it dies.
- **Encoder.** `b`/`bl` check the ±128 MiB range at emission and fail the compile
  (a strike) rather than emit a wrong branch.

`exit_common` is a naked function in tcvm's own text (`backend/aarch64/abi.rs`),
its offsets `const` operands and `jit_exit` a `sym` operand, so segments hold only
regions (after their header and bitmap units) and need no per-segment code. A
segment copy existed in the old JIT, whose segments could land anywhere; with the
window above every segment reaches the text directly, while two segments at its
ends need not reach each other, so one generated copy would not do. A region's
layout is: prologue, body blocks in loop-contiguous order, resume blocks (each a
32-byte-aligned continuation, entered only by a callee's return), the cold
section (exit stubs, the
region's `exit_region` trampoline, the entry-fail stub, out-of-line slow paths
such as the barrier call), then the constant island (8-byte literals reached by
`ldr x, label`).

Exit stubs and the two trampolines:

```
exit_K:        mov  w16, #K                 ; 2 instructions per guard
               b    exit_region
exit_region:   ldr  x17, =region            ; once per region: the Region pointer
               orr  x16, x17, x16, lsl #48  ; x16 = region | K << 48 (K < 65536)
               b    exit_common
exit_common:   ldr  x17, [x23, #EXIT_REGS]  ; State.jit.exit_regs, 64 + 256 words
               stp  x0, x1, [x17] … str x28, [x17, #…]      ; words 0-21: x0-x15, x19-x21, x26-x28
               stp  d0, d1, [x17, #176] …                  ; words 22-53: d0-d31 (64-bit, not q)
               and  x9, x16, #0xFFFF_FFFF_FFFF              ; the Region
               ldr  w10, [x9, #NUM_SPILLS]                  ; this region's spill count
               (copy w10 words from [x29, #16] to [x17, #512])   ; words 64-319
               ldr  w10, [x9, #FRAME_SIZE]  ; pop the region frame, whatever its size
               ldp  x13, x30, [x29]
               add  x11, x29, x10
               mov  sp, x11
               mov  x29, x13
               mov  x20, x16                ; insn = region | K << 48
               b    jit_exit                ; base, rt, closure, thread are the pinned registers
```

x16 and x17 are never allocated (x17 is regalloc2's scratch only inside parallel
moves, which no stub interrupts), so the stubs clobber nothing. The spill copy
reads exactly the region's spill area. Regions needing more than 256 spill slots
are refused by the backend (a strike, 6.2), which never happens for a sane
region.

`jit_exit` (a `slow` handler) unpacks the region and the exit id from `insn`,
walks the snapshot against the register image and the spill copy, writes the
home slots, counts, updates feedback, publishes the frame and dispatches the
resume instruction (5.6). The entry-fail stub of the prologue needs no image: it
pops the frame (its size is known when the stub is emitted), loads the region
into `insn` and tails `jit_entry_fail` (6.4).

GC exits are the collector checks' exits (5.7), with a `Before` snapshot of the
loop header or the entry, whose handler returns `Exit::Gc` after writing the
slots; the executor returns `Pending`, the collection runs as `Lua::enter` exits
(4.6), and the next step resumes at the published pc in the interpreter, which
dispatches the word there: a loop entry's or function entry's own JIT word
re-enters the region.

### 10.7 Encoder extensions

Over the old `aarch64_asm.rs` (`add/sub/mul/sdiv/msub/csel/ccmp/and/orr/eor/
lslv/lsrv/asrv/neg/mvn/cmp/cset/fcmp/fadd…/scvtf/ldr/str/ldp/stp/b/b.cond/cbz/
cbnz/ret`): `adds/subs` (flag-setting with immediate and register), `smull`,
`cmn`, `tst`, `tbz/tbnz`, `ldrsw`, `ldr/str` with register index and extend,
`ldr` literal, `adr`, `adrp`, logical immediates (encoder for the bitmask
immediate), `movz/movk/movn` sequences, `sxtw/uxtw`, `frintm/frintp`, `fsqrt`,
`fabs`, `fmov` immediate, `fcvtzs`, `csinc/csinv`, `bl`, `br`, `blr`, `ccmp`
register form, `lsl/lsr/asr` immediate forms (`ubfm`/`sbfm`), `ubfx/sbfx`.

### 10.8 Code publication

The old `code.rs` path: write through the RW mapping, `sync_icache` over the RX
range (`ic ivau` per 64-byte line plus `dsb ish; isb` on aarch64), then install
the entry (11.3). The Rosetta flake noted in #128 is an x86-64 concern for later.

## 11. Runtime data structures and lifetimes

### 11.1 Per prototype

`Prototype` gains two fields: `feedback: Box<[Cell<u8>]>`, the feedback byte per
instruction (4.8), and `jit: Lock<Option<Gc<'gc, JitState<'gc>>>>`, null until the
first `jit_hot` for the prototype. `LuaClosure` copies nothing new.

`State` gains the entry table, `jit.entries: Box<[Cell<*const u8>]>` behind a
pointer at a fixed `layout` offset, as LuaJIT's `J->trace[]`: slot `d` holds the
entry address that `JIT_ENTRY d`/`JIT_LOOP d` tail into (5.2), or
`jit_recompile` while a recompile is pending (6.4). Slots are allocated from a
free list, grown by doubling (the handlers read the pointer each time, so growth
needs nothing else), and freed when an entry is blacklisted or its prototype dies.
The handler path is two dependent loads and a branch instead of the
closure → proto → `jit` → entry chain.

```rust
pub struct JitState<'gc> {
    entries: [Entry; MAX_ENTRIES],          // MAX_ENTRIES = 4
    shapes_seen: Vec<(u32 /*pc*/, Shape<'gc>)>,
    regions: Vec<Gc<'gc, Region<'gc>>>,      // live and retired, until the prototype dies
    strikes: u8,
    blacklisted: u8,                         // bit per entry
}

pub struct Entry {
    pc: u32,                 // the FUNC, LOOP, FORLOOP or JMP_BACK word
    original: Instruction,   // the word JIT_ENTRY/JIT_LOOP replaced
    back_edges: Box<[u32]>,  // a LOOP entry's back edges, pointed at it (5.2)
    slot: u16,               // index into State.jit.entries, the word's `d`
    region: Option<Gc<Region>>,
    recompiles: u8,
    state: EntryState,       // Empty | Compiled | Recompile | Blacklisted
    nest: Box<[u32]>,        // a loop entry's loop headers, innermost first (eviction, 5.2)
    seen: Vec<(u8, TypeSet)>,// kinds failed entry guards saw, by register (9.1)
    age: u32,
}
```

### 11.2 Regions

```rust
pub struct Region<'gc> {
    code: CodeBlock,                         // the segment allocation; freed on drop
    entry: *const u8,
    pool: ConstPool<'gc>,                    // traced: the only GC references compiled code embeds
    exits: Box<[ExitInfo]>,                  // snapshot offset, tag, counter
    snaps: Box<[u32]>,                       // Appendix C; the snapshot holds the resume pc and kind
    consts: Box<[u64]>,                      // raw words of snapshot constants
    entry_regs: Box<[u8]>,                   // the registers the entry guards test (6.4)
    entry_fails: Cell<u32>,
    calls: Box<[CallSite]>,                  // per call site: a, wanted, kind (for direct-call patching later)
    proto: Gc<'gc, Prototype<'gc>>,
    entry_pc: u32,
    nest: Box<[u32]>,                        // the entry's loop headers (5.2)
    frame_size: u32,                         // the native frame exit_common pops (10.6)
    num_spills: u32,
    retired: Cell<bool>,
}
```

A `Region` is a `Collect` object: tracing it traces the pool and the prototype. A
retired region is kept in `JitState.regions` until the prototype dies, because a
frame somewhere may hold one of its resume points as a continuation (word 1 of a
callee header), and the return must land in valid code. The code memory of
a retired region is therefore never freed before the prototype; a later
improvement can scan all threads' frames at a full collection and free retired
regions no frame references.

### 11.3 Installing

`jit_hot` compiles, then: allocates the code block, publishes it (10.8), creates
the `Region`, stores it in `entries[i].region`, allocates a table slot and
writes the region's entry address into it, and writes the `JIT_ENTRY`/`JIT_LOOP`
word over `Code[pc]` with `d` the slot. The word write is a cell
write of a plain instruction; no barrier is involved (interp spec 14.4). The
`JitState` write into the prototype is a `Lock` write with a backward barrier on
the prototype.

### 11.4 Replacing and retiring

A recompile builds the new region, installs it in the same entry (the opcode word
is unchanged; only the table slot and `entries[i].region` change), and marks the
old one retired. Direct calls into the old region (5.4, milestone 3) are repatched through
the callers' `CallSite` lists: each region records which callee entries it jumps
to directly, and each `Entry` keeps a list of (region, patch offset) pairs to
rewrite when its code changes. Until milestone 3 no such edges exist.

Blacklisting restores `original` into `Code[pc]`, frees the table slot, retires the
region and sets the entry's bit; `jit_hot` checks the bit first.

### 11.5 Global assumptions

One fact the metamethod forms depend on: that no number metatable is set
(`ARITH_MM_R` and `_MMI` with the immediate on the left take the metamethod from
the table only then; `rt.number_metatable()` in the handlers). Compiled code reads
the numbers' metatable cell in `State` at each such site (`NumMtSet`: two loads and
a compare) and exits when it is set. A region-level epoch, compared at the entry
and after every resume point, would cost the same three instructions per call, and
every dependent site is itself a call.

The string metatable is not such a fact. It is an ordinary table, and Lua code
can rewrite its fields without `set_metatable_of`: after
`getmetatable("").__index = {upper = function() return "hijacked" end}`,
`("a"):upper()` returns `hijacked` (lua 5.5.1). A string method site (one whose
feedback saw only strings) is compiled like a `_PROTO` site: compare the word in
the strings' metatable cell in `State` with the metatable's at compile time
(`TypeMtIs`; the region's constant pool keeps that table alive), guard the
metatable's shape and load its `__index` slot, compare that with the string
library table, guard the library's shape and load the method's slot. When the
`CALL` that follows has an intrinsic for the method (`s:sub`, `s:byte`, `s:len`),
its callee guard compares the slot's function. Every step is a load and a compare,
and all of them hoist out of loops that write no metatable and no shape.

### 11.6 Sizes

| Structure | Size |
|---|---|
| feedback byte | 1 byte per instruction of every prototype |
| `ExitInfo` | 16 bytes per exit |
| snapshot entry | 4 bytes; a typical guard snapshot has 2 to 6 entries |
| exit stub | 8 bytes of code per guard |
| `exit_regs` buffer | 2.5 KiB per `State` |
| hot counter table | 128 bytes per `State` |
| entry table | 8 bytes per installed entry, grown by doubling |
| `FUNC` | one instruction per prototype; `LOOP` one per non-numeric loop but `goto` loops |

## 12. Intrinsics

A `CALL` whose callee value is refined to a constant native function (7.5) with a
registered intrinsic compiles inline, with the same argument-shape conditions as
the fast entry in `ff.rs`, and deopts `Before` the `CALL` when a condition fails
(the interpreter then runs the full builtin through `native_call`). The callee
refinement is guarded by a pointer compare, hoisted when loop-invariant.

| Builtin | Inline code | Deopt when |
|---|---|---|
| `math.sqrt`, `math.sin`, `math.cos` | `F64` argument (or `I32` converted); `fsqrt` inline, `sin`/`cos` by helper; result `F64` | not a number, `nargs != 1` |
| `math.floor`, `math.ceil` | `frintm`/`frintp` then `FToIExact` into `I32`; an `I32` argument is the identity | result outside i32 (the fast entry misses too) |
| `math.abs` | `I32` with `MIN` deopt, or `fabs` | |
| `math.huge`, `math.pi`, `math.maxinteger` | constants read through the global/field path, no call | |
| `type(v)` | a constant string per type bit when the type is known, else a helper returning the interned name | |
| `rawget(t, k)`, `rawset(t, k, v)` | helpers | not a table, a nil or NaN key |
| `rawequal`, `rawlen` | helpers: raw equality (numbers by value), the raw length | `rawlen` of another type |
| `select('#', ...)`, `select(n, ...)` | with a constant selector and one result: from the arguments, or from `nv` and the extras when a `VARARG` up to `top` is the instruction before (the builder keeps the callee and the selector across its slot reloads) | |
| `pairs(t)` | `(next, t, nil)` as `ff_pairs` when `jit_mm` finds no `__pairs`; feeds `TFORPREP`, whose `_NEXT` form the region then compiles | `__pairs` present |
| `ipairs(v)` | `(ipairs_iter, v, 0)` | |
| `setmetatable(t, mt)` | helper (`set_metatable_fast`, else `set_metatable` for a table without one), an allocation for 5.7; from milestone 6, `t` a `NEWTABLE` of the region and `mt` a constant: the allocation takes the shape with the metatable (7.6), no call; `mt` a fresh `{__index = K}` per object (havlak's `Vector.new`, json's values: 6-9% of their cycles in `adopt` and `join_by_memo`): a memo from the template and `K` to the class and shape | a metatable already set, a nil one |
| `getmetatable(t)` | helper, the `__metatable` field honoured | |
| `assert(v, ...)` | truthiness test; the arguments are the results | falsy |
| `string.sub`, `string.byte`, `string.len`, `#s` | helpers (allocation for `sub`), small-integer positions, `byte`'s first result only; `#s` is `LEN` (7.6). A one-byte `sub` result comes from a table of the 256 preinterned one-byte strings (two loads; json interns one per character today) | other positions |
| `tostring(v)`, `tonumber(v)` | helpers, one argument | `__tostring` |
| `pcall`, `xpcall`, `error`, `coroutine.*`, `string.format`, `table.*` | no intrinsic: an ordinary call through `enter` | |

A builtin is recognized at compile time by its fast entry (`ff_*`, which each
builtin with one has to itself) or, for a plain native, by its Rust function
(`src/jit/build/intrinsics.rs`). A `CALL` whose shape no intrinsic covers is an
ordinary call. Where a helper declines (the last column), the `CALL` is a call of
that builtin in an arm that joins the inline result, rather than a deopt: such an
exit never widens, and would fire on every call of the site that meets the case.

The callee and the selector are looked up when the `CALL` is built. When an
argument is a field load whose nil arm joins before the `CALL` (any receiver
with `__index`), milestone 2 finds the join's block parameter instead of the
callee and compiles a plain call: json's `self.input:sub(self.index,
self.index)` is a full native call, about 20% of its cycles (17.3 B1). The
lookup must see through parameters whose incoming values are all the same, as
`simplify` would make them, before deciding.

`table.sort` stays a native (17.4 D2 asks whether the JIT should see a sort
written in Lua instead): its comparator calls go through `stage_call`, which
from milestone 3 enters the comparator's region directly (17.3 C5).

## 13. Inlining (milestone 5)

Inlining is the largest remaining factor on the call-heavy benchmarks (17.3 I1):
LuaJIT's traces inline four levels per sphere test in ray (273 instructions
against tcvm's 1266), seven functions in richard's scheduler, every metamethod
and constructor of mandel-metatable, and two levels of fib's recursion. It was
designed with milestone 1's snapshots (J8) and is built after direct calls,
whose callee identity it uses.

A `Call` whose callee is a known Lua closure (5.4's sources: a constant upvalue
or global, a `_PROTO` holder's or own slot, a closure the region created, callee
feedback; 17.4 D1) with a small prototype (at most `INLINE_LIMIT = 60`
instructions, not vararg, no `TBC`, no `CLOSURE` capturing its parameters by
reference, no `TAILCALL` other than to itself) is inlined:

- The callee's register `r` maps to caller slot `a + 4 + r`, the slot the callee
  frame would have had (`base + a + 4 + r`), with the window check for
  `a + 4 + callee.max_stack_size` done once before the inlined body; R1 then
  holds for the inlined frame without any special case, since a tail-out from
  inside the inlined body
  must first write the callee's header (a snapshot with two frames, Appendix C,
  writes it on deopt; a real call from inside the inlined body writes it inline
  before the call, so that the callee of the callee returns into a real frame
  with a resume point).
- The callee's `RETURN` becomes a jump to the continuation block with the result
  values as block parameters; `wanted` results are selected there.
- Upvalue reads of the inlinee go through its closure value, a constant.
- The hot-exit and recompile machinery is unchanged: an exit inside an inlined
  body has a two-frame snapshot and its site feedback is recorded on the callee
  prototype's pc.
- Multiple results: a callee `RETURN` of `n` values jumps to the continuation
  with `n` block parameters, and `wanted` selects among them (ray's `vsub` and
  `mul` return three, qt's `f` four); a callee with a `RETURN` up to `top` is not
  inlined.
- Metamethod callees: an `ARITH_MM` or `__index`-function site whose metamethod
  is known (17.4 D6: from callee feedback, or from the class when the operand's
  shape is known at compile time) guards it with `GuardSame` on the loaded
  metamethod and inlines it like a call with the operands as arguments
  (mandel-metatable's `__mul` and `__add`).
- Recursion: a call of the region's own prototype is inlined up to
  `RECURSION_UNROLL = 2` levels, the innermost level a direct call (5.4). This is
  LuaJIT's up-recursion unrolling, which lets it beat C's per-call cost on fib.
- Calls created in the region: the callee of a call through a closure the
  region allocated has the `CLOSURE`'s prototype, so fixpoint-fact's
  `f(f)(x)` chain inlines, and its closures then sink (9.10).

Budget: depth 4, `INLINE_LIMIT` bytecodes per callee, and at most 1500 IR
instructions per region after inlining and tail duplication (9.11), counted in
IR instructions rather than bytecodes because that is what compile time follows
(17.7). ray's main region is already 1631 IR instructions without inlining, so
the budget picks the hottest call sites first (by the callee's call count, 6.1)
and stops inlining before a region reaches `MAX_INSTS`.

## 14. Tooling and testing

Environment variables, read once at `Lua` creation:

| Variable | Effect |
|---|---|
| `TCVM_JIT=off` | counters never trip |
| `TCVM_JIT_HOT_CALL`, `TCVM_JIT_HOT_LOOP` | thresholds (6.1) |
| `TCVM_JIT_LOG=1` | one line per compile, install, exit over `EXIT_HOT`, recompile, blacklist |
| `TCVM_JIT_DUMP=ir,opt,vcode,asm` | print the stages for every compile |
| `TCVM_JIT_CHECK=1` | verifier after every pass, `regalloc2` checker, snapshot replay check |
| `TCVM_JIT_FASTALLOC=1` | the single-pass allocator |
| `TCVM_JIT_ONLY=proto-name` | compile only prototypes whose source name matches |
| `TCVM_JIT_TIME=1` | one line per compile with its IR size and each stage's time |

The analysis of section 17 used a diagnostic build whose `TCVM_JIT_LOG` also
prints each exit's count at every power of two, each region's code address and
size, and the address of one helper (to find the image's slide), and scripts
that map a cycle-sampled `xctrace` profile onto regions and instruction offsets
through that log and `TCVM_JIT_DUMP=asm`. Milestone 3 makes that logging part of
`TCVM_JIT_LOG`, so the per-benchmark tables of Appendix F can be regenerated at
each milestone.

`Lua::set_jit(bool)` turns compiling on or off for one state, as LuaJIT's
`jit.on`/`jit.off` do; installed regions stay. Tests whose premise is
interpreted execution (counting the collector's runs in boxed arithmetic) use it.

Tests:

- **IR snapshot tests** (`insta`): bytecode of small Lua functions compiled with
  fixed feedback, printed after the builder and after each pass.
- **Exec tests**: hand-built IR lowered, allocated, emitted and run through a
  test region entry for arithmetic, boxing, tables and exits, with the register
  image and the written slots checked.
- **Differential tests**: every file in `test-files/` and the LJR suite run with
  `TCVM_JIT=off` and with `TCVM_JIT_HOT_*=1` (compile at the first opportunity),
  outputs compared; a second run with `TCVM_JIT_CHECK=1`.
- **Deopt stress**: a mode that forces every exit once (`TCVM_JIT_DEOPT_ALL=1`
  makes each guard's first execution deopt), which exercises every snapshot.
- **Protocol tests** in `src/vm/tests/`: a region that calls a Lua function that
  yields, errors, and triggers a collection, and is resumed at its resume point;
  frame walks through JIT frames for error positions, `xpcall` handlers and
  `pcall` catches below a JIT frame.
- **Compile-time bench**: the old `benches/jit_pipeline.rs` pattern, per stage.
- **Runtime bench**: the benchmark files under `cargo run -p tcvm-cli`, measured
  with the methodology in the perf notes (interleaved A/B, cycles).

## 15. Milestones

Each milestone has exit criteria; a milestone is done when all tests of section 14
pass for its coverage and its measurements are recorded.

1. **Protocol and numerics.** Interpreter first: the feedback byte and its
   slow-path writers (4.8), ISA opcodes `FUNC`/`LOOP`/`JMP_BACK` (emitted by the
   compiler) and `JIT_ENTRY`/`JIT_LOOP` (the ISA goes from 204 to 209 opcodes), each
   measured with the JIT off. Then the entry table, counters, `jit_hot`,
   `jit_recompile`, `jit_entry_fail`, `JitState`/`Region`, the builder with
   constants, moves, `_II`/`_FF`/`_IF`/`_FI` arithmetic, `I64` arithmetic and
   loops with `jit_box_i64` and the collector checks of 5.7, immediates,
   compares with the generic compare helpers (7.3), `JT/JF`,
   `FORPREP`/`FORLOOP_I/F`, `JMP`, `RETURN0/1`, type inference, peeling, GVN,
   the constant part of the call-boundary pass (9.8), block layout (9.9), the
   aarch64 backend with `regalloc2`, exits with snapshots,
   `jit_exit`, the never-executed block rule (6.3), use-driven speculation (9.1),
   and the `Call` instruction with resume points for unknown callees
   (so a region can call out). Everything else is `Deopt`. Exit: `primes2`,
   `collatz_bench`, `mandel_bench`, `iterative_fib_bench` run compiled with no
   exits in steady state (for `collatz_bench`, after its overflow recompile, with
   `n` in an x register, 7.2); compile time per stage
   measured on them; the interpreter changes measured with the JIT off against
   main (6.1, 4.8).
2. **Helpers, allocation, tables and metamethods.** Generic arithmetic helpers,
   `_NN` and locked sites, `CONCAT`, `LEN`, `NEWTABLE`, `SETLIST`, `CLOSURE`,
   barriers, `VARARG` forms, `CLOSE`, upvalue reads and writes, field forms with
   ICs, `GETTABLE`/`SETTABLE` array paths, `_PROTO` chains, `ARITH_MM*`, metamethod
   calls, `TFORCALL_NEXT/_IPAIRS`, `MULTRET` forms, the `TAILCALL` fast arm,
   intrinsics (12), LICM (9.5) with the rest of the call-boundary pass and
   snapshot re-pruning (9.8), hot-exit recompilation with polymorphic shape
   chains, blacklisting. Exit: `fft`, `nbody`, `particles_bench` and the LJR suite
   compiled end to end; `TCVM_JIT_DEOPT_ALL` passes on all test files; results
   versus the interpreter and LuaJIT recorded in the perf notes. This was two
   milestones until 2026-10-10: the first one's exit benchmarks, `fft` and `nbody`,
   are array and field code, which only the second compiled.
3. **Calls.** Revised 2026-10-11 after the measurements of section 17, which
   reorder the rest of the plan: calls first, since every benchmark pays them
   and their callee identity is what inlining builds on. Direct calls with
   patching and direct tail calls (5.4, 5.5, 11.4); callee feedback (17.4 D1);
   the builtins' fast entries as header cores, so regions reach them (5.4);
   `stage_call` into compiled entries (12); no collector check for calls (5.7);
   native fast calls with droppable frames (J10, 5.4) and cold-arm stubs
   (10.5); single-result landing and a typed entry for known callees (5.4); `pcall` through `enter`; the callee-lookup fix
   and `b == 0` intrinsics (12, 7.7). Exit: `enter` and `op_jit_entry` under 2%
   of the cycles of every LJR benchmark whose callees are known; coroutine-ring
   and chameneos faster compiled than interpreted and at or ahead of LuaJIT;
   recursive-fib-uv at most 6 cycles per call; json's `s:sub` the intrinsic.
4. **Speculation and code quality.** Compare kinds (7.3), number-only generic
   arms, equality by tags, mixed compares, inline float `%` and `x ^ 2` (7.2),
   constant upvalues of every type and the inline `SETUPVAL` barrier (7.5),
   `__index` chains resolved at compile time and tail duplication for
   polymorphic receivers (7.4, 9.11), precise helper effects and array-site
   reuse (7.6), the lowering fixes (10.3), rematerialization at uses (9.8), jump
   threading (9.9), LICM of guarded arithmetic (9.5), range analysis and bounds
   checks against the loop limit (9.7). Exit: queen's instructions within 10% of
   LuaJIT's; qt, richard and json ahead of LuaJIT; `jit_index` under 3% of
   deltablue; spill instructions per hot region reported for ray, qt and queen.
5. **Inlining** (13), with metamethod callees (17.4 D6) and recursion
   unrolling. Exit: recursive-fib-uv/gv, ray, queen and richard ahead of LuaJIT;
   the LJR geomean at least 1.5× LuaJIT; compile time within 17.1's targets.
6. **Allocation and sinking.** Inline `NEWTABLE` and `CLOSURE`, appends,
   `setmetatable` at allocation and its memo (7.6, 12), allocation sinking
   (9.10). Exit: mandel-metatable at least 2× LuaJIT; binary-trees, havlak and
   fixpoint-fact further ahead; the LJR geomean at least 2× LuaJIT (17.1).
7. **x86-64.** The second target per Appendix E, run under Rosetta for
   correctness. (Milestone 4 until the 2026-10-11 revision.)

Every milestone from 3 on records, besides its exit benchmarks, the whole LJR
suite and the design benchmarks against the interpreter and LuaJIT, compile time
per IR instruction and per benchmark (17.7), and the exits per benchmark
(Appendix F), so a change that helps one benchmark and costs another shows. The
items outside the JIT (17.8) are decided separately; they bound table-sort and
part of json and havlak whatever the JIT does.

Compile-time budgets, measured in milestone 1 on a 200-instruction function on
the M4 Pro: builder 10 µs, passes 15 µs, lowering 10 µs, allocation 25 µs
(`Ion`), emission 5 µs; total under 70 µs. The old pipeline did 76 µs for `mix`
with a weaker output. The allocation figure is a target, not a citation: the
first milestone 1 measurement replaces it.

Measured in milestone 1 (warm: each region compiled 2000 times in one process;
a single cold compile, which `TCVM_JIT_TIME` shows, costs two to five times as
much, and a process's first compile also maps the code segment). `mandel_bench`'s
inner loop, 233 IR instructions after the passes: builder 9.1 µs, passes 65.7 µs,
lowering 16.1 µs, allocation 31.6 µs, emission 3.3 µs, 126 µs in all; the regions
of 35 to 100 instructions take 25 to 60 µs. Only the passes are far over their
budget. The first measurement had them at 137 µs for 188 instructions, 40% of it
in malloc, which the storage of 8.1 removed; what remains is the fixpoints of
inference (all instructions up to 50 rounds), the per-parameter scans of
narrowing, list moves on insertion, and the speculation pass's dataflow.
At the end of milestone 2 the whole pipeline runs at 1.0-2.0 µs per IR
instruction over the LJR suite's regions, passes 0.5-0.9 and allocation 0.4-0.8
(17.7); the budget from here on is section 17.1's 0.5 µs per instruction.

Measured at the end of milestone 2 (2026-10-11 at `e43dc0a`, before the review's
changes to the metamethod forms, `LEN`, string methods and `next`; the LJR suite,
median cycles of 5 interleaved runs on the M4 Pro). Compiled code is 1.41× the
interpreter's speed (geomean of 16) and 2.51× LuaJIT's interpreter (`-joff`);
LuaJIT's compiler is 1.12× faster than it. Ahead of LuaJIT: fixpoint-fact 4.6×,
havlak 2.8×, binary-trees 2.6×, deltablue 2.1×, table-sort-cmp and qt even.
Behind: recursive-fib-uv 3.3×, ray 3.0×, mandel-metatable and recursive-fib-gv
2.2×, queen and chameneos 1.7×, table-sort 1.6× (its printing, which tcvm does not
buffer), coroutine-ring 1.5×, json 1.3×, richard 1.2×. coroutine-ring and
chameneos run 13% and 15% slower compiled than interpreted. Re-measured after the
review's changes at `dc2266a`: compiled code 1.45× the interpreter, LuaJIT 1.14×
faster than it; json and qt now even with LuaJIT, mandel-metatable 2.0× behind
(Appendix F, with the analysis in section 17).

## 16. Risks and open questions

- **Counting and feedback cost in the interpreter** (6.1, 4.8): measured in
  milestone 1 with the JIT off against main (5 interleaved rounds, median
  cycles): the program benchmarks are 1.040× main (geomean of 25; 1.062× when
  `FUNC` and `LOOP` were dispatched), the microbenchmarks 1.131× (19). What
  remains: a numeric loop's count on every iteration (an empty loop 1.75×,
  `iadd` 1.25×), 11 instructions per call (recursive fib 1.055×), a generic
  `for`'s count (1.07×), and the slow paths' feedback records, which compute the
  kinds on every run though they store only on a change (big-integer
  arithmetic 1.65× main's instructions; `fixpoint-fact` 1.09×, `richard` 1.08×).
  Counting a numeric loop by its trip count at `FORPREP` is the open option for
  the first; LuaJIT counts every `FORL` iteration.
- **`regalloc2` on loops**: the in-house spiller was measured to beat a whole-value
  scan on `mix2`'s loops; `Ion` splits live ranges and should do as well, but
  this is checked in milestone 1 on the old `mix`/`mix2` sources. If it loses,
  the fix is in isel (rematerialization of constants and unboxes, which `Ion`
  lacks), not a return to the old allocator. Milestone 2's code confirms the
  rematerialization half (constants and rematerializable block parameters in
  spill slots, 9.8) and adds the cold helper calls that clobber the hot path's
  registers (10.5); `Ion` is also a third of compile time (10.4, 17.7).
- **Retired region memory**: regions live until their prototype dies (11.2). A
  program that recompiles a long-lived prototype many times holds a few KiB per
  recompile, bounded by `MAX_COMPILES` per prototype (6.4).
- **`debug` library**: `debug.getinfo` and `debug.traceback` with a JIT frame in
  the chain see correct frames (5.8). `debug.setlocal` is a call, so the JIT frame
  is at a tail-out with its live registers in their slots (R1) and reloads every
  register after the resume (8.7): the write takes effect. Hooks are a non-goal.
- **Irreducible control flow** from `goto`: refused (9.9), a strike.
- **Very large functions**: the builder caps a region at `MAX_INSTS = 4000` IR
  instructions; beyond that only loop entries are compiled.
- **`I64` boxing at calls**: a big integer live across a call is boxed at every
  call (R1 needs a `Value` in its slot), an allocation per call that the
  interpreter also pays when it produces the value. Loops that keep big integers
  in registers and call nothing allocate nothing; a loop that calls with a big
  integer live costs one box per iteration until inlining (13) removes the call.
- **Snapshot-only values across calls**: a value live only in snapshots after a
  call is rematerialized from its home slot (it was stored by R1), so it is never
  an allocator use after a call.
- **Compile time under inlining and sinking**: inlining grows regions 2-3×,
  tail duplication and sinking add more, and compile time follows IR size at
  1-2 µs per instruction today (17.7). The IR budget of 13 bounds a region, the
  per-pass budgets of 17.3 T1 bound the rate; a benchmark whose compile share
  passes 2% is a regression like any other (15).
- **Native stack use of fast calls** (J10): a chain of compiled calls holds one
  native frame per call, up to `FAST_STACK` below dispatch's entry; past it a
  fast entry takes the slow path, which uses no native stack. Lua's own stack
  limit (65,500 slots) keeps chains of small frames under the budget, so the
  check is a safety net for frames with many spill slots.
- **Dropped chains**: a native call, a deopt or any other way into the
  interpreter from inside a chain of fast calls drops the whole chain, and each
  caller resumes through its header at the slow path's cost. Keeping the
  frames across such a visit (the interpreter running below them, the
  callee-saved registers saved at the visit) is the next step if chains that
  call natives turn out to matter; per-coroutine stacks would remove the drops
  at coroutine switches as well.
- **Global facts** (11.5): the number metatable is checked at each site that
  depends on it, string method sites guard the string metatable per site, and
  userdata and other per-type metatables do not affect compiled fast paths.

## 17. Performance plan (after milestone 2)

Milestone 2 compiles the whole LJR suite and runs it 1.45× faster than the
interpreter, but LuaJIT's compiler is still 1.14× faster than it (geomean of 16,
dc2266a, Appendix F). This section is the analysis of where compiled code loses
that time and the plan that the remaining milestones (15) follow. Each item
says what changes, which section carries the design, what it is expected to
save, and what it costs at compile time. The per-benchmark evidence (profiles,
exit counts, hot-path machine code against LuaJIT's traces and clang -O2 ports
of the kernels) is in Appendix F; figures there are cycles and instructions
retired, never wall-clock time.

### 17.1 Targets

- **Speed.** The LJR suite at least 2× LuaJIT's compiler (geomean), and no
  benchmark behind LuaJIT except where the time is spent outside compiled code
  in a part listed in 17.8. The design benchmarks of section 1
  (`mandel_bench`, `fft`, `nbody`, `particles_bench`, ...) are measured the same
  way at each milestone.
- **Compile time.** At most 0.5 µs per IR instruction after the passes (the
  pipeline measures 1.0-2.0 µs today, 17.7), and at most 2% of each LJR benchmark's
  run (deltablue is at 7.7% today, havlak 4.8%). LuaJIT's trace compiler is the
  ceiling never to reach: it spends 39% of json's run, 47% of qt's, 24% of
  queen's and 25-88% of ray's (17.7), while tcvm compiles each of those in
  under 25 ms.

### 17.2 Where the time goes

Cycle shares of the JIT-on run (`xctrace` CPU Profiler, leaf frames; *calls* is
`enter` plus `op_jit_entry`, the path every call out of a region takes today):

| Benchmark | LJ / tcvm | region code | calls | helpers | interpreter, runtime, natives, libc | GC | main causes (17.3) |
|---|---|---|---|---|---|---|---|
| recursive-fib-uv | 0.30 | 61 | 39 | 0 | 0 | 0 | C1, C2, I1 |
| recursive-fib-gv | 0.48 | 60 | 40 | 0 | 0 | 0 | C1, C2, I1, F5 |
| ray | 0.18 | 85 | 10 | 3 | 1 | 0 | I1, S2, S8, Q1, Q3 |
| mandel-metatable | 0.51 | 40 | 14 | 36 | 7 | 3 | I1, A1, A2, A4, B2 |
| queen | 0.61 | 91 | 7 | 0 | 1 | 0 | Q1, Q2, Q6, Q7, I1 |
| coroutine-ring | 0.71 | 18 | 5 | 3 | 73 | 0 | C3, S4, C2 |
| chameneos | 0.59 | 32 | 5 | 2 | 60 | 0 | C3, F4, C2 |
| table-sort | 0.59 | 4 | 1 | 0 | 85 | 4 | R1, R2 |
| richard | 0.82 | 55 | 14 | 10 | 18 | 0 | S5, S3, I1, Q4, S7 |
| json | 1.03 | 44 | 10 | 4 | 32 | 7 | B1, R4, F2, I1, A2 |
| qt | 1.06 | 74 | 16 | 1 | 4 | 4 | S1, Q1, C1, C2, S6 |
| table-sort-cmp | 1.06 | 37 | 4 | 0 | 57 | 1 | C5, R3, S1 |
| deltablue | 2.13 | 36 | 11 | 5 | 30 | 11 | F1, F2, C1, T3 |
| binary-trees-name | 2.51 | 36 | 13 | 18 | 13 | 18 | A1, C1, C6 |
| havlak | 2.72 | 25 | 7 | 10 | 21 | 33 | A2, A1, A3, R5 |
| fixpoint-fact | 4.58 | 29 | 17 | 46 | 5 | 3 | A3, C1, C4, I1, A4 |

(Percent of cycles; the rest is the compiler and unattributed samples. LJ /
tcvm is LuaJIT's cycles over tcvm's: below 1, LuaJIT is faster. ray's LuaJIT
figure is its good mode, see 17.5.)

Four facts carry most of it:

1. **Calls are the largest single cost.** A call from a region writes a header,
   tails `enter` (31 instructions: callee kind, window, arity), dispatches the
   callee's first word, `JIT_ENTRY`, and `op_jit_entry` (4 more) loads the
   region's address; the region pays about 30 instructions of its own (native
   frame pop and push around the call, four header stores, base and closure
   reloads after the return's indirect branch, a nil default for the result, a
   collector check at every entry). fib: 90.6 instructions and 12.9 cycles per
   call against LuaJIT's 36.9 and 3.9 and C's 16.5 and 3.5. A one-level method
   call in a loop is about 150 instructions and 15 cycles; LuaJIT inlines it to
   nothing. `enter` alone is 5-32% of the cycles of 13 of the 16 benchmarks.
2. **Regions never reach the builtins' fast entries** (5.4 says so), so a
   `coroutine.resume`/`yield` from compiled code takes the general native path
   (`native_enter`, `drive`, `native_act`, `resume_into`, `prepend_true`), about
   70 cycles per switch against the interpreter's 31 through `ff_resume` and
   `ff_yield`. That is why coroutine-ring and chameneos run 13-25% slower
   compiled than interpreted.
3. **Cold arms cost the hot path.** Generic arithmetic and compares carry a
   helper arm, array reads a nil and an out-of-bounds arm, field reads an
   `__index` arm. The arms rarely run, but their helper calls clobber the
   caller-saved registers (so the allocator spills around them: up to 41 spill
   stores and 62 spill loads in one hot region of ray and qt), their declared effects end GVN's
   and LICM's reuse at the join (queen loads `a[i]` three times per iteration),
   and an arm that may allocate keeps a collector check at its loop header.
4. **What LuaJIT inlines and sinks, tcvm calls and allocates.** LuaJIT's ray
   trace inlines four levels of calls per sphere test (273 instructions against
   tcvm's 1266), its mandel-metatable trace sinks three of its four tables per
   iteration and records `setmetatable` as one store. Where LuaJIT cannot
   compile (closure creation, `FNEW`, in fixpoint-fact, deltablue and havlak;
   coroutine switches), tcvm already wins by 2-4.6×, and its allocator and
   collector are 3-10× cheaper per object than LuaJIT's.

### 17.3 Work items

Grouped by kind; the milestone that schedules each is in 15. Estimates are from
the Appendix F measurements and are per benchmark.

#### Calls

- **C1. Direct calls (5.4, 11.4; milestone 3 already).** A call whose callee is
  known at compile time (17.4 D1: a constant upvalue, a global through `_ENV`'s
  IC, a method through a `_PROTO`/own slot, a closure the region created, or
  per-site callee feedback) and has a function-entry region jumps to it: the
  identity guard (`GuardSame`, hoisted when invariant), the window check against
  the callee's `max_stack_size` and the arity fill inline, header word 0 written
  in its final form (`closure | nv << 48`), and `b` to the callee region's entry. A callee without a region is entered at its
  `FUNC` through a per-prototype interpreter entry that skips `enter`'s checks.
  fib: −28 instructions and two dependent indirect branches per call, about 12.9
  → 8.5 cycles; mandel-metatable's `enter_fixup` for `conj(x)` (called with one
  argument for two parameters) disappears with the inline arity fill.
- **C2. A cheaper call and return protocol (new; amends 5.4, 5.5, 5.7, 10.5).**
  - No collector check for a call: callees check their own allocations, as the
    interpreter's allocating handlers do. `prune_gc_checks` counts `Call` and
    `CallMm` as allocating today; an entry or loop header whose only
    "allocation" is a call loses its check (fib, qt: 6.7% of `clip`'s samples;
    richard: 7.8% of its main loop's). LICM borrows the header check's
    snapshot before the checks are pruned, so its hoisting is unaffected.
  - Native calls between compiled functions (J10, 5.4): the caller keeps its
    frame and calls the callee's fast entry with `blr`, the callee returns with
    `ret`; the frame pop and push around every call (about 32% of fib's region
    samples and 16% of deltablue's `Vector:each`) and the unpredicted return
    go. Done 2026-10-11; replaces the frameless regions this item first
    proposed, which J10's kept frames make moot.
  - The resume of a known callee computes the caller's base (`sub x22, x22,
    #(a+4)*8`) instead of loading it, and reloads `closure` only before its
    next use or tail-out.
  - A callee whose every `RETURN` yields exactly one value lands its result with
    one load, without the nil default (5 → 1 instructions).
  - No R1 store of a value its slot already holds, and type facts of slots kept
    across a resume (17.4 D4): today every live register is reloaded and
    re-guarded after each call (coroutine-ring's main loop re-guards three).
  - A second, typed entry per function region (later): a direct caller that
    proves its arguments' kinds enters past the entry guards and unboxing.
  - fib, estimated with C1: about 40 instructions and 6 cycles per call, before
    any inlining.
- **C3. Builtins through their fast entries (new; amends 5.4 and 12).** Each
  `ff.rs` fast entry gets a core that takes a written header the way `enter`
  does (callee in word 0, continuation in word 1, arguments at the header plus
  4, `nargs` in `insn`) besides today's CALL-word adapter. `enter`'s native arm
  tails the core when the native has one (one load and a branch), and a region
  with a constant native callee without an intrinsic tails it directly.
  `resume_switch!` and `ff_yield` already return through header word 1, which
  is the region's resume point, so a coroutine switch from compiled code costs
  what it costs the interpreter, as LuaJIT's traces stitch around its
  assembler `coroutine.resume`. coroutine-ring 96 → about 51 cycles per round
  trip (2.01 → about 1.05 Gcyc; LuaJIT 1.42), chameneos 107 → about 68 per
  meeting (5.26 → about 3.4; LuaJIT 3.10). `pcall`, `xpcall` and
  `coroutine.wrap`'s functions take the same path.
- **C4. Direct tail calls (5.5).** A `TAILCALL` with a known callee that has a
  region rewrites the frame as the fast arm does and branches to the region,
  instead of dispatching the callee's first word through the opcode table and
  `op_jit_entry` (a chain of five dependent loads in fixpoint-fact).
- **C5. Native-to-compiled calls.** `stage_call` (natives calling Lua, the
  sort comparator) jumps to the callee's region entry, skipping `enter` and
  `op_jit_entry`. table-sort-cmp spends 53% of its cycles in the native sort's
  call path; this item takes the `enter` part of it, R3 the rest.
- **C6. Return prediction.** Returns were `br x16` through header word 1,
  which the return-address stack never predicts; 36% of BottomUpTree's region
  samples sat on its two resume points. J10's fast calls return with `ret`
  (C2); returns into the interpreter keep word 1.

#### Inlining

- **I1. Inlining (13; was "designed now, built later").** The largest
  remaining factor on the call-heavy benchmarks: ray 205 → about 90 cycles per
  sphere test, queen −27% instructions (`isplaceok` into `addqueen`), richard's
  scheduler loop (LuaJIT's root trace inlines seven functions), mandel-
  metatable's `__add`/`__mul`/`complex`/`norm2`, json's `self:read()` per
  character, and recursion: LuaJIT unrolls fib's recursion two levels per trace
  (up-recursion), which is how it beats C's per-call cost. Section 13 now
  specifies callee sources, budgets, multiple results, metamethod callees and
  recursion.

#### Speculation and feedback gaps

- **S1. Compare kinds (amends 4.8, 7.3).** Compare sites record kinds only on
  their slow paths, and the fast paths handle floats, so a float compare's byte
  stays 0 and compiles to the three-arm generic compare with `jit_lt`/`jit_le`
  (qt's `clip`: every compare; table-sort-cmp's `lhs.x ~= rhs.x`). Fix: type
  the operands from their definitions (9.1 extended to compare uses, entry
  types seeded from the live frame) and, where nothing types them, record a
  compare's kinds once, on its first execution (the fast path tests the byte
  for zero), with the interpreter cost measured with the JIT off as 6.1
  measured the counters.
- **S2. Number-only generic arms (amends 7.2, 7.3).** A generic site whose byte
  saw only numbers compiles its small and float arms and a deopt for the rest,
  not a helper call; with no helper on the path the arms do not spill. The
  selects the arms leave (`select(c, c, false)`, `select(c, k, k)`) are folded
  in GVN.
- **S3. Equality by tags (amends 7.3).** Values with different type tags,
  other than an integer and a float, are unequal without `__eq` (Lua consults
  `__eq` only for two tables or two full userdata). `x ~= NO_TASK` with a nil
  upvalue is 50 instructions through `jit_eq` today and 6 in LuaJIT (richard
  −8%).
- **S4. Mixed integer and float compares inline (amends 7.3).** `scvtf` of the
  `I32` side and `fcmp`: an i32 converts to f64 exactly (`20000000 == 2e7` is
  true in lua 5.5.1). coroutine-ring's `token == n` (n is the float `2e7`)
  calls `jit_eq` every iteration today.
- **S5. Float `%` inline (amends 7.2).** `MOD_FF` calls libm `fmod` through a
  helper: about 90 cycles, against 7 in LuaJIT, which uses `a - floor(a/b)*b`
  (5.1 semantics, which 5.5 does not have: `-4.0 % 2` is `-0.0` and
  `5.5 % -math.huge` is `-inf` in lua 5.5.1, `0` and `nan` in LuaJIT). Inline:
  `q = frintm(a / b)`, `r = fmsub(q, b, a)`, a fix-up of `q` by one when `r`
  lands outside `[0, b)` (or `(b, 0]`), a zero result signed as the dividend's,
  and the helper for non-finite operands and `|a / b| ≥ 2^52`. To be checked
  against `fmod` on generated edge cases before it ships. richard −12%.
- **S6. `x ^ 2` as `x * x`.** `pow(x, 2)` and `x * x` are both correctly
  rounded; 2M random doubles gave no mismatch under lua 5.5.1. Not `x ^ 0.5`
  (1290 of the same 2M differ from `sqrt`). qt's `pow` share is 4%.
- **S7. Constant upvalues of every type (amends 7.5).** Only functions, strings
  and tables are refined to constants today, and only where a site wants it.
  An upvalue never assigned after its capture (a per-upvalue flag the compiler
  sets, as LuaJIT's parser marks `uv->immutable` for `rec_upvalue_constify`) is
  a constant of the closure for any type, guarded once per entry by the
  closure's identity or the cell's value. richard and qt reload and re-test
  such upvalues after every call (about 15 instructions each time).
- **S8. Mixed integer and float values (study).** ray's vectors mix integers
  (`{0, 0, -4}`, the root sphere) and floats, so its arithmetic is generic with
  three arms everywhere. Two ways: a representation pass that keeps a
  `SMALL | FLOAT` value whose uses are all float arithmetic in an `F64` (exact
  for integers below 2^53, which needs 9.7's ranges), or cloning an inlined
  body per argument-kind combination. This is the step that brings ray level
  with LuaJIT; 17.4 D5.

#### Fields and tables

- **F1. `__index` chains resolved at compile time (amends 7.4).** `_PROTO`
  handles one hop; a deeper chain, and a nil own slot under a metatable with
  `__index`, takes the `jit_index` walk every time: deltablue loses 23% of its
  cycles there (`walk_index_chain` and `raw_get`); a two-level `o:get()` is 354
  instructions and 47 cycles, LuaJIT's 6 and 1. Each hop compiles as `_PROTO`'s
  does: the class's `__index` loaded and compared with the constant found at
  compile time, then that table's shape guarded, the key absent or the holder
  slot loaded. The guards are loads and compares that GVN shares and LICM
  hoists. (V8's alternative, a validity cell per chain, is one load and compare
  per site but needs invalidation on every write to a prototype's shape.)
- **F2. Polymorphic receivers (9.11, new; J4's reserve).** A site that saw two
  shapes dispatches on the shape at every field access, since the arms merge
  again at each join: deltablue's `EqualityConstraint:execute` is 2304 bytes
  after three compiles, json's `Parser:read` 3072. Tail duplication after the
  first shape dispatch on a receiver, up to the next call or loop header,
  keeps one shape per path.
- **F3. Out-of-line slots (runtime).** Objects built by `setmetatable({}, mt)`
  and then filled keep every field in aux storage (one more load per access).
  Allocation-site slack tracking, or sizing `NEWTABLE`'s template from the
  transitions its tables take, puts them inline.
- **F4. Inline `SETUPVAL` barrier test (amends 7.5).** The upvalue store calls
  `jit_upval_barrier` unconditionally; test the cell's (an open upvalue's
  thread's) mark bit inline as 5.7's table barrier does (chameneos 1.5%).
- **F5. Global cells (study).** A global read after a call reloads `_ENV`,
  re-tests its tag and shape and reloads the slot (13 instructions, twice per
  fib-gv node). A cell per global that holds the value and is invalidated on
  write (V8's `PropertyCell`) makes the read a constant with no guard, and a
  direct call through a global free of guards. Needs a design of its own.

#### Allocation

- **A1. Inline table allocation (amends 7.6, 5.7).** `NEWTABLE` from a
  template whose size is known: bump-allocate in the region, write the header,
  the template shape and the initial values; no `from_template` call, no
  `memset_pattern16` of nil slots the following stores overwrite (12.5% of
  binary-trees), no barrier and no shape guard on the fresh table. binary-trees
  74 → about 45 cycles per node (LuaJIT 196).
- **A2. `setmetatable` at allocation (amends 12).** `setmetatable(NEWTABLE,
  constant mt)` allocates with the shape the transition gives, recorded in the
  template; `setmetatable(obj, {__index = K})` per object (havlak's
  `Vector.new`, json's values) is memoized by (template, `__index` value) →
  class and shape instead of running `adopt` and `join_by_memo` each time
  (havlak 9.4%, json about 6%).
- **A3. Inline closure allocation (amends 7.6).** `CLOSURE` with only by-value
  and already-open upvalues allocates inline (prototype, header, upvalue words);
  creating a new open upvalue stays in the helper. fixpoint-fact's
  `jit_closure` is 42% of its cycles (about 17 cycles per closure, a third of
  it the helper's callee-saved registers).
- **A4. Allocation sinking (9.10, new).** After inlining, a table or closure
  whose uses are constant-slot loads and stores, a constant-metatable
  `setmetatable`, calls that were inlined and snapshots is not allocated; its
  slots are SSA values and snapshots describe it, and `jit_exit` allocates it
  when an exit needs it. mandel-metatable 272 → an estimated 30-40 cycles per
  iteration (LuaJIT 135, which allocates the loop-carried `z` only);
  fixpoint-fact's two closures per level.
- **A5. Inline array append.** `t[#t + 1] = v` and `SETTABLE` at `asize` with
  capacity left store and bump the size inline (havlak's `Vector:append` goes
  through `jit_newindex` and `set_int_key`, 1.8%).

#### Code quality

- **Q1. Cold arms out of the way (amends 10.3, 10.4).** A helper call on a cold
  arm goes through an out-of-line stub that saves and restores the
  caller-saved registers the hot path keeps live, so the allocator sees no
  clobber there; or, where feedback says the arm never ran, the arm is an exit.
  Values only the arm uses (the boxed key of queen's nil arm) are computed in
  it (code sinking). ray's `vsub` has 18 spill stores and 38 spill loads in 516
  instructions, qt's `clip` 41 and 55 in 784.
- **Q2. Precise helper effects; array-site reuse (amends 7.6, 8.4).**
  `jit_index` is declared `MAY_ALLOC | W_SHAPE | W_META` but does raw gets and
  the `__index` walk only (no Lua code runs, a function returns `FAIL`); with
  read-only effects its arm keeps no collector check and kills no fact. The
  builder reuses an earlier `GETTABLE` site's result for the same table and key
  with no write between, as `field_loads` does for fields: queen's three
  `a[i]` loads become one (65 → about 25 instructions per iteration with Q1
  and Q4-Q7).
- **Q3. Lowering (10.3).** Branch on a boolean or a select of type tests
  directly (`tbz`, two branches) instead of `cset; cmp; b.eq`; fuse a
  `Guard(SMALL|BIGINT)` with the `Unbox{I64}` after it (one test, not two);
  share wide constants within a call-free span, as 10.3 already says (ray's
  `BOX` bound is materialized 25 times in one region); drop constant `mov`s a
  folded compare left dead.
- **Q4. Constants are rematerialized, not spilled (10.4, risk 2 of 16).**
  queen spills the constant 1; richard's main loop carries the upvalue, three
  booleans derived from it, its unboxed float and the constants 2 and 0 as
  eight block parameters in spill slots (one reload is 9.6% of the region's
  samples). The call-boundary pass (9.8) re-emits rematerializable values at
  each use after a resume rather than merging copies into block parameters.
- **Q5. Jump threading in block layout (9.9).** Empty join blocks leave
  branch-to-branch chains (six per iteration in queen's inner loop).
- **Q6. LICM of guarded arithmetic (9.5).** queen's `c - n` and `c + n`,
  loop-invariant `ISub`/`IAdd` with overflow exits, stay in the loop. LICM moves
  an exit only when the header has a `GcCheck` whose snapshot it borrows, and
  only when every operand is outside the loop or moved; find which fails here.
- **Q7. Bounds checks against the loop limit (9.7).** An array index that is a
  loop counter checks `asize` against the loop's `last` once before the loop
  (LuaJIT's ABC); 9.7 only removes checks a dominating compare proves.
- **Q8. Range analysis (9.7).** Drops fib's `n - 1`/`n - 2` overflow checks on
  the `n >= 2` edge and ray's and queen's induction checks.

#### Compile time

- **T1. Per-pass costs (17.7).** The passes are 40% of compile time (GVN,
  speculation, inference, the call-boundary pass, the verifier's liveness),
  `regalloc2` 35%. Each pass gets a budget per IR instruction and the
  compile-time bench of section 14 tracks them.
- **T2. Register allocation.** `Ion` is 0.4-0.8 µs per IR instruction;
  `Fastalloc` is 4-13× faster but its code retires 10-50% more instructions
  (17.7), so it is no replacement. Options: `Ion` for loop regions and a
  linear scan for straight-line function regions, or fewer virtual registers
  (Q1's stubs and Q4's remats shrink live ranges).
- **T3. Recompile churn.** deltablue makes 50 recompiles out of 122 compiles,
  havlak 22 of 103, json 28 of 84, nearly all from hot shape exits. The first
  compile of a site reads the shapes the receiver's other sites saw (6.4's
  receiver sharing) and the sites' IC history where the IC kept it; F2 makes a
  two-shape site cheaper to compile than a recompile.
- **T4. Inlining budget.** Inlining grows regions 2-3× (ray's main region is
  already 1631 IR instructions): the budget in 13 counts IR instructions, not
  bytecodes, and a region over `MAX_INSTS` stops inlining before it stops
  compiling.

#### Bugs found by this analysis

- **B1.** A `CALL` whose arguments are field loads under `__index` loses its
  intrinsic: `self.input:sub(self.index, self.index)` in json's
  `Parser:read` is a full native call (`native_enter`, `lua_sub`, `intern`,
  `memcmp`: about 20% of json's cycles). The callee and selector lookups see
  the block parameter of the join the field loads' nil arms leave. Repro:
  `t = setmetatable({input = s, index = 2}, {__index = P})` with
  `function P:read() self.c = self.input:sub(self.index, self.index) end` in a
  loop. Every compile-time callee decision after such a join is affected.
- **B2.** `math.sqrt(f(x))` (a `CALL` with `b == 0` fed by a multiple-result
  call) goes through `jit_top_args` and a plain call to `ff_sqrt` instead of the
  intrinsic (mandel-metatable's `abs`). With a known callee of one result, the
  inner call's `c` can be treated as 2.

### 17.4 Decisions needed

- **D1. Callee feedback.** Direct calls and inlining need the callee. Constant
  upvalues, globals and `_PROTO` holder slots give it at compile time; method
  fields copied into each object (ray's `intersect = self.intersect`) and
  closures passed around do not. Options: (a) a per-site callee word the
  interpreter's `CALL` writes when it differs (an estimated 3 instructions per
  call, to be measured with the JIT off as 6.1's counters were), (b) the first region records the callee per call
  site in a region cell (one store per call in compiled code) and a hot call
  site triggers a recompile that uses it, (c) compile-time sources only.
- **D2. `table.sort` visible to the JIT.** The same algorithm written in Lua
  and compiled runs table-sort-cmp's sort 1.6× faster than the native sort
  today (2.3× once the comparator inlines; LuaJIT's Lua-written sort with the
  comparator inlined is 1.15 Gcyc against tcvm's native 2.67). This conflicts
  with the earlier choice of continuation-style Rust natives for `sort`; an
  intrinsic that runs a compiled sort loop calling the comparator's region is
  the alternative.
- **D3. Unbuffered stdout.** All of table-sort's loss is printing: one
  `write(2)` per line (6.2 Gcyc of system time) and an interned string per
  number. Pending from the LJR gap analysis.
- **D4. Slot types across calls.** Only `debug.setlocal` can change a caller's
  registers during a call. A state flag that `debug.setlocal` sets (and that
  retires or deopts regions) would let a resume keep its slots' types.
- **D5. Mixed numbers (S8):** representation pass or per-kind cloning.
- **D6. The metamethod forms' constant compare.** Milestone 2 loads an
  `ARITH_MM` site's metamethod from the class without comparing it with a
  constant (nothing recorded which function a site called). Inlining a
  metamethod needs that identity: from D1's callee feedback, or from the class
  seen at compile time when the operand's shape is known.
- **D7. Targets (17.1).**

### 17.5 What tcvm keeps

LuaJIT's aborts are tcvm's wins: closure creation (`FNEW`, `UCLO`) compiles, so
fixpoint-fact (4.6×), havlak (2.7×) and deltablue (2.1×) are ahead, and
tcvm's allocation and collection are 3-10× cheaper per object (havlak's
three-field object 28.5 cycles against 141). Compile time: tcvm's whole-program
compile is 0.2-25 ms per benchmark; LuaJIT's trace compiler takes 24-47% of
queen's, json's and qt's runs, and ray's bimodal runs (1.6 or 3.6 Gcyc) come
from its trace flushes. Inlining (I1) and sinking (A4) must keep that: a budget
per region, no recompile storms.

### 17.6 Order

Calls first (C1-C5 and the B items): they are in every benchmark, and C1's
callee identity is what inlining and metamethod direct calls build on. Then the
speculation and code-quality items, which are small and independent. Then
inlining, then allocation and sinking, which pays most after inlining exposes
the allocations. Section 15 lists the milestones and their exit criteria.

### 17.7 Compile time measured

`TCVM_JIT_TIME` totals per benchmark (release build, one run):

| Benchmark | compiles | IR insts | build | passes | lower | regalloc | emit | total µs | LuaJIT compiler share |
|---|---|---|---|---|---|---|---|---|---|
| recursive-fib-uv | 1 | 41 | 41 | 81 | 22 | 39 | 175 | 358 | 6% |
| queen | 5 | 581 | 131 | 476 | 105 | 268 | 67 | 1047 | 24% |
| richard | 51 | 4667 | 802 | 2443 | 609 | 2046 | 263 | 6163 | 12% |
| deltablue | 122 | 20053 | 2804 | 9614 | 2264 | 7755 | 969 | 23406 | 4% |
| havlak | 103 | 16266 | 2280 | 11045 | 1974 | 9115 | 889 | 25303 | 3% |
| json | 84 | 13488 | 1569 | 6095 | 1443 | 4079 | 625 | 13811 | 39% |
| ray | 31 | 10619 | 1276 | 9583 | 1182 | 8566 | 475 | 21082 | 25-88% |
| qt | 19 | 3704 | 525 | 1569 | 410 | 1501 | 1483 | 5488 | 47% |

(A process's first compile also maps the code segment, which shows in emit.)
That is 1.0-2.0 µs per IR instruction: passes 0.5-0.9, `regalloc2` 0.4-0.8.
Within the compiler's own samples (in a run that also dumped assembly), `regalloc2` is 31-41%, GVN 3-12%, the IR's liveness
and verifier helpers 9-16%, speculation and inference up to 7% each.
`TCVM_JIT_FASTALLOC=1` cuts allocation 4-13× and total compile time 13-72%, but
instructions retired rise 22% (richard), 15% (deltablue), 10% (havlak), 31%
(ray), 51% (queen).

### 17.8 Outside the JIT

These are not compiled-code work, but they decide benchmarks the JIT is
measured on:

- **R1. Buffered stdout** when it is not a terminal (D3).
- **R2. `print` of numbers** formats into the output buffer without creating an
  interned string (table-sort: interning about 9% of samples).
- **R3. A sort the JIT can see** (D2).
- **R4. One-byte strings preinterned**, so `s:sub(i, i)` is two loads (json:
  interning and `memcmp` about 9.5% even with the intrinsic).
- **R5. Marking** of large live graphs: havlak spends 33% in the collector,
  mostly `trace(TableState)` and `trace(Value)`.

## Appendix A: register assignment (aarch64)

| Register | Role in a region |
|---|---|
| x20 | `insn` at entry and tail-outs; free in the body |
| x21 | `pc` at entry and tail-outs; free in the body |
| x22 | `base`, pinned |
| x23 | `rt`, pinned |
| x24 | `closure`, pinned |
| x25 | `thread`, pinned |
| x16, x17 | exit stubs and trampolines; x17 is also regalloc2's Int scratch for parallel moves; never allocated |
| x18 | platform register, never touched |
| x29, x30 | frame pointer and link, saved by the prologue, restored before every tail-out |
| sp | the region frame |
| x0–x15, x19–x21, x26–x28 | allocatable Int class (22 registers); x0–x7 also helper arguments and results |
| d0–d30 | allocatable Float class; d8–d15 survive helper calls |
| d31 | regalloc2's Float scratch for parallel moves; never allocated |

Helper calls clobber x0–x17, d0–d7, d16–d31.

## Appendix B: header words written by a region

A JIT frame's callee header, as written by 5.4:

| Word | Content |
|---|---|
| 0 | the callee as a raw `Value` (rewritten by `enter` to `closure \| nv << 48`) |
| 1 | the resume point's address (32-byte aligned; flags 0; `enter` ors `NATIVE` for a native, the callee's `CLOSURE`/`TBC` their flags) |
| 2 | the region frame's `base` |
| 3 | the bytecode pc after the CALL, as an interpreter caller writes it |

Every reader of words 2 and 3 sees an interpreter-shaped header. Word 1 is only
called (by a return, a native's landing, the unwinder's catch) or masked for its
flags; the only address comparisons against word 1 are with `ret_pcall` and
`ret_xpcall`.

## Appendix C: snapshot encoding

One `u32` arena per region (`region.rs`); an exit records its tag, its counter
and its snapshot's offset, and exits whose encoded snapshots are equal share one.
A snapshot:

```
header:   kind:1 (Before|After)  frames:2  entries:12  (17 spare)
          pc:32                                                          (pc of the innermost frame)
frame k (outer frames only, for inlined calls; three words each):
          proto_pool:32                                                  (the inlined callee's closure)
          call_a:8  wanted:8  base_delta:16                              (callee base = enclosing frame's base + delta; `wanted` is the CALL's `c`)
          caller_pc:32                                                   (the inlined CALL's pc)
entry:    reg:8  rep:3 (Val|I32|I64|F64|B1)  loc:3  index:18
          loc: Reg (register image index) | Spill (spill word index) | Const (constant-word index)
             | Nil | False | True
```

Entries list only registers that are live at the resume pc and whose home slot
does not hold their current value (the builder never makes an entry for a slot
that already holds its value, so there is no location for one). A `Const` entry
indexes the region's raw constant words, deduplicated; nil and the booleans need
none. Every entry's `reg` counts from the region's own base, inlined frames'
registers at their frame's `base_delta` plus their own index, so an inlining
decision (section 13) keeps a nest's frames within 256 registers. `jit_exit`
writes each inlined callee's header (`func` from the pool, `ret =
rt.ret(wanted)`, `caller`, `pc`) outermost first, then the entries, and resumes
the innermost frame. Milestone 1 emits no frames; the decoder handles them, and a
unit test round-trips a snapshot with two.

Milestone 6 adds object entries for sunk allocations (9.10): an object header
(`kind:2` table or closure, `slots:14`, `template_or_proto:16`, a pool index)
followed by one entry per slot in the entry format above, its `reg` field the
slot index. A register entry may then name an object (`loc` `Object`, the index
of the object in the snapshot). `jit_exit` allocates every object of the
snapshot first, fills their slots (an object slot naming another object gets its
pointer), then writes the frames' entries.

## Appendix D: opcode coverage matrix

Strategy per opcode on main's ISA (204 opcodes); `D` is a `Before` deopt until the
named milestone, `H` a helper, `I` inline code.

| Opcodes | M1 | M2 | Notes |
|---|---|---|---|
| `MOVE`, `LOAD`, `LOADI`, `LOADNIL`, `LFALSESKIP` | I | | SSA only |
| `GETUPVAL` | I | | one load from the closure cell (7.5) |
| `GETUPVAL_REF` | D | I | through the cell |
| `SETUPVAL` | D | H | barrier helper |
| `GETTABUP*`, `GETFIELD*`, `SELF*`, `SETTABUP*`, `SETFIELD*` (27 opcodes: 5 generic, 20 forms, 2 `_REF`) | D | I | 7.4; `_REF` receivers via the cell |
| `GETTABLE`, `SETTABLE` | D | I/H | array fast path inline, else helper |
| `NEWTABLE`, `SETLIST` | D | H | `NEWTABLE` allocates (5.7) |
| `ADD`..`SHR` generic (12), `_NN` (7) | D / I | H | never-executed blocks (6.3) stay D; an integer-kinds byte gives `I64` code from M1 |
| `_II`, `_FF`, `_IF`, `_FI`, bitwise `_II`, `POW_II` (27) | I | | 7.2 |
| `ADDI`..`RSHRI` generic (19) | D / I | H | as the register forms |
| `_I`, `_F`, `_IF` immediate forms (27) | I | |  |
| `ARITH_MM`, `_MM_R`, `_MMI` | D | I + call | |
| `UNM`, `BNOT`, `NOT` | I | | slow cases H in M2 |
| `LEN`, `CONCAT` | D | I/H | a string's `LEN` inline |
| `CLOSE`, `TBC` | D | H / D | TBC stays D |
| `JMP` | I | |  |
| `JEQ`..`JNLE` generic, `JEQI`..`JNGEI`, `JEQS`, `JNEQS` | I + H | | inline numeric cases, then `jit_lt`/`jit_le`/`jit_eq` (strings, big integers, mixed numbers); `__eq`/`__lt`/`__le` calls in M2 |
| `_II` compares (6), `_F` compares (8) | I | |  |
| `JT`, `JF`, `JTSET`, `JFSET` | I | |  |
| `CALL`, `CALL_R0`, `CALL_R1`, `CALLS*` | I (5.4) | | `b == 0` D until M2; direct calls M3 |
| `TAILCALL` | D | I fast arm | other callees: the interpreter's handler (`Dispatch`) |
| `RETURN`, `RETURN0`, `RETURN1` | I | | `b == 0`: D in M1, the interpreter's handler (`Dispatch`) from M2 |
| `FORPREP`, `FORLOOP_I`, `FORLOOP_F`, generic `FORLOOP` | I | | generic `FORLOOP` as an `I64` loop; one with nothing recorded never iterated (D); loops that ran as both kinds D |
| `TFORPREP`, `TFORCALL`, `TFORCALL_NEXT`, `TFORCALL_IPAIRS`, `TFORLOOP` | D | I | generic `TFORCALL` is a call |
| `CLOSURE` | D | H | |
| `VARARG`, `VARARGGET`, `VARARGPREP` | D | I/H | |
| `ERRNNIL` | I | | nil test, deopt to raise |
| `NOP` | I | |  |
| `STOP` | D | |  |
| `FUNC`, `LOOP` | I | | emit nothing; a region never counts |
| `JMP_BACK` | I | | a `JMP` |
| `JIT_ENTRY`, `JIT_LOOP` | resolved to the original word by the builder | |  |

## Appendix E: x86-64 extension points

What the second target implements, and nothing else:

1. `backend/x64/inst.rs`: its `MachInst` enum with x86 forms (two-address ALU with
   `Reuse(0)`, `lea`, `[base + index*8 + disp]`, `cmov`, `setcc`, `imul`, `idiv`
   with `Fixed(rax)`/`Fixed(rdx)` and clobbers, `test`, `bt`, SSE2 scalar ops,
   `cvtsi2sd`, `cvttsd2si`, `roundsd`).
2. `backend/x64/lower.rs`: the pattern table of 10.3 in x86 terms; the IR-level
   expansions are shared, so only primitives are matched.
3. `backend/x64/abi.rs`: the register table (r12 `insn`, r13 `pc`, r14 `base`,
   r15 `rt`, rdi `closure`, thread loaded from `rt`), pinned registers (r14, r15,
   rdi, with the thread reloaded from `rt` after helper calls or pinned in rbx),
   the SysV clobber set, the prologue with rbp as frame pointer, `exit_common` with
   its own register image layout, and the `MachineEnv`. The code window of 10.6
   widens to the ±2 GiB of a `rel32` `jmp`/`call` (LuaJIT's x64 range,
   `lj_arch.h:211`; its arm64 range is ±128 MiB, `lj_arch.h:299`).
4. `backend/x64/emit.rs` over the old `x64_asm.rs`.
5. The `layout` constants are shared; `State.jit.exit_regs` is sized for the larger
   image.
6. Tests pinned to machine-code shape are per target under `backend/x64/`; the
   IR, differential and protocol tests run unchanged.

The IR, the builder, the optimizer, the `regalloc2` adapter, the exit and snapshot
formats, the runtime structures, the resume protocol, `jit_exit`, the counters and the
intrinsic table are target-independent by construction and contain no `cfg`
arms.

## Appendix F: the milestone 2 measurements, per benchmark

Measured 2026-10-11 at dc2266a on the M4 Pro. Speeds: median cycles of 5
interleaved rounds of tcvm with the JIT on and off and LuaJIT 2.1 with its
compiler on and off (`/usr/bin/time -l`). Per-operation figures: instruction and
cycle deltas between two run sizes. Profiles: `xctrace` CPU Profiler (cycle
sampling), leaf frames symbolized against the binary, JIT code attributed to
regions and instruction offsets through `TCVM_JIT_LOG`/`TCVM_JIT_DUMP=asm` of the
same run. Exits: a diagnostic build that logs each exit's count at powers of
two. LuaJIT: `-jp=v` (VM states), `-jp=F`, `-jv`, `-jdump`. C ports of the hot
kernels compiled with clang -O2.

| Benchmark | tcvm JIT | tcvm interp | LuaJIT | LuaJIT -joff | (Gcyc) |
|---|---|---|---|---|---|
| mandel-metatable | 0.485 | 0.946 | 0.247 | 1.961 | |
| table-sort-cmp | 3.066 | 3.580 | 3.258 | 4.936 | |
| table-sort | 12.659 | 13.037 | 7.430 | 8.265 | |
| binary-trees-name | 0.959 | 1.336 | 2.404 | 3.482 | |
| fixpoint-fact | 0.424 | 0.560 | 1.941 | 1.923 | |
| coroutine-ring | 2.008 | 1.608 | 1.416 | 2.140 | |
| richard | 1.261 | 3.635 | 1.036 | 5.979 | |
| deltablue | 1.374 | 1.872 | 2.927 | 3.823 | |
| ray | 9.006 | 13.101 | 1.614 | 17.299 | |
| queen | 0.683 | 1.810 | 0.416 | 5.296 | |
| recursive-fib-uv | 4.277 | 5.323 | 1.297 | 7.561 | |
| recursive-fib-gv | 2.606 | 3.874 | 1.252 | 4.976 | |
| havlak | 2.354 | 3.104 | 6.407 | 7.771 | |
| chameneos | 5.264 | 4.615 | 3.102 | 7.707 | |
| qt | 5.083 | 9.794 | 5.396 | 18.548 | |
| json | 2.523 | 4.340 | 2.595 | 8.152 | |

Geomeans relative to the JIT: interpreter 1.447, LuaJIT 0.878, LuaJIT -joff
2.530. LuaJIT's ray is bimodal: 4 of 11 runs fall into a mode with about 270
trace flushes, 88% of the time in its compiler and 3.6 Gcyc.

No benchmark has a steady-state exit problem: every exit that fired is a
collector exit, a never-executed block (recompiled once), or a shape or type
exit that recompiled within `EXIT_HOT`; entry fails stay under 16 per region
(richard's nil-or-table parameters). No entry is blacklisted.

**recursive-fib-uv** (331M calls). Per call: tcvm 90.6 instructions and 12.9
cycles, the interpreter 120.6 and 16.1, LuaJIT 36.9 and 3.9, C with two real
calls 16.5 and 3.5. One function region of 41 IR instructions. Its internal
node is 92 instructions: prologue 3, argument load and collector check 5
(`ldr [rt, #METRICS]; ldp; cmp; b.hs`, a two-load chain), parameter guard and
`n < 2` 6, each call 16 (upvalue load, `subs/b.vs`, box, argument store, four
header stores, `adr`, a literal load, `mov x20`, frame pop, `add x21`,
`b enter`), each resume 11 (`ldur x22, [x22, #-16]`, the closure reload and
mask, frame push, a five-instruction nil default for the result), the add 18
(re-guards of both results, `adds/b.vs`, box, store, return through
`ldur; and; br`). Each call also runs `enter` (31) and `op_jit_entry` (4). The
first instruction after each resume and the frame `stp`/`ldp` hold 14% and
about 32% of the region's samples. LuaJIT's trace 1 is an up-recursive trace
that inlines two levels and checks the closure once per level (C1, C2, I1, Q8).

**recursive-fib-gv** (205M calls): as fib-uv, plus the global `f` read twice
per node after resumes, 13 instructions each (`_ENV` upvalue, its tag test, its
shape guard, the slot). LuaJIT's traces end in "call unroll limit reached" and
return traces, 6.1 cycles per call (C1, C2, I1, F5, S7).

**queen** (45.4M inner iterations, 10.1M `isplaceok` calls). `isplaceok`'s
region is 79% of cycles. Its inner loop is about 65 instructions per iteration
against LuaJIT's 17 and C's 9: three `a[i]` reads each with a bounds check, an
out-of-bounds and a nil arm that call `jit_index`, so GVN cannot reuse the
first load; a collector check at the header kept by those arms; the key boxed
and spilled each iteration for the arms; `c - n` and `c + n` recomputed with
overflow checks; `n`, `c` and the constant 1 reloaded from spill slots; six
branch-to-branch jumps. Call overhead is about 115 instructions per
`isplaceok` call. LuaJIT's loads are typed integer `ALOAD`s with the bounds
check hoisted (Q1, Q2, Q4-Q7, I1).

**fixpoint-fact** (5.05M recursion levels: 2 closures, 4 calls of which 2 tail
calls, 1 multiply each). tcvm 543 instructions and 84 cycles per level; a C port
with heap closures (bump allocation, four indirect calls) 49 and 21.6. Cycles:
`jit_closure` 42% (17 cycles per closure, a third of it the helper's
callee-saved registers), regions 29%, `enter` and `op_jit_entry` 17%, boxing of
products outside i32 7%. Tail calls dispatch the callee's first word through
the opcode table and then `op_jit_entry` (five dependent loads). LuaJIT aborts
every trace on `FNEW` and runs interpreted, 54% of it in its collector
(A3, C1, C4, I1, A4).

**ray** (44.7M sphere tests). Per test: tcvm 1266 instructions and 205 cycles,
LuaJIT 273 and 36 in its good mode, C 62 and 10.7. Regions: `sphere_distance`
45%, `vsub` 14%, `group:intersect` 13%, `dot` 11%; calls 10%. One call is about
104 instructions and 15.6 cycles, a three-result call (`jit_land`) 152 and 17.
Arithmetic is generic with three arms because the vectors mix integers and
floats; the `BOX` bound constant is materialized 25 times in one region; `vsub`
has 18 spill stores and 38 spill loads in 516 instructions. LuaJIT's root trace
inlines `c:intersect`, `sphere_distance`, `vsub` and `dot` per child (I1, S2,
S8, Q1, Q3, D1).

**mandel-metatable** (1.76M inner iterations). tcvm 272 cycles and 2279
instructions per iteration, LuaJIT 135 and 738 (60% of it collection), C with
value structs about 9 cycles. Cycles: `jit_newtable` 17%, `jit_setmetatable`
16.5%, `enter` 10%, `enter_fixup` 3% (`conj(x)` passes one argument for two
parameters). Four tables per iteration at about 23 cycles each to allocate and
set the metatable. LuaJIT sinks three of its four `TDUP`s, records
`setmetatable` as a store, and inlines `__add`, `__mul`, `complex`, `abs`,
`norm2`, `conj` and `sqrt`. `math.sqrt(Complex.norm2(x))` misses the intrinsic
(B2) (I1, A1, A2, A4, D6).

**qt.** `clip` 42% (65.9M calls, 32 cycles each), `prewhite` 14%, `color` 11%,
calls 16%, `pow` 4%. Every float compare is generic with `jit_lt`/`jit_le`
arms, since compare sites record no kinds (S1); `clip` has 41 spill stores and
55 spill loads in 784 instructions; its entry collector check is 6.7% of its
samples. LuaJIT spends 47% of its run in its trace compiler (1157 side traces
ending in "fallback to interpreter") (S1, Q1, C1, C2, S6, I1).

**binary-trees-name** (12.7M nodes). tcvm 74 cycles and 447 instructions per
node, LuaJIT 196 and 876 (64% collection). Per node: `jit_newtable` 13.5
cycles, `memset_pattern16` 9.2 (the template's nil fill that three stores then
overwrite), collection 13.6, `BottomUpTree` 17, `ItemCheck` 9, calls 9. 36% of
`BottomUpTree`'s samples sit on its two resume points, which a return through
`br x16` would explain (unmeasured, C6). Three barriers and a shape guard on
the fresh table (A1, C1).

**coroutine-ring** (2e7 round trips). Per round trip: tcvm 722 instructions and
96 cycles, the interpreter 540 and 81, LuaJIT 469 and 64. With the JIT on, 78%
of cycles are the native call path into `coroutine.resume`/`yield`
(`enter`, `native_enter`, `lua_resume`, `native_act`, `drive`, `resume_into`,
`native_continue`, `yield_from`, `prepend_true`): about 76 cycles per round
trip against 31 through the interpreter's `ff_resume`/`ff_yield`. `token == n`
compares an integer with the float `2e7` through `jit_eq` (5.5 cycles). The main
loop region keeps a collector check, reloads five registers and re-guards three
after the resume. LuaJIT's traces stitch around its assembler fast functions
(C3, S4, C2).

**chameneos** (5e7 resume/yield pairs). Per pair: tcvm 765 instructions and 107
cycles, the interpreter 605 and 95, LuaJIT 493 and 60. The switch is about 70
cycles compiled against 31 interpreted. `meet` stores to closed upvalues through
an unconditional `jit_upval_barrier` call (C3, F4, S3).

**table-sort** (5e6 floats). Phases: generate 0.37 Gcyc (LuaJIT 0.16; `rand` is
a call), sort and assert 3.30 (LuaJIT 6.39; tcvm's Rust quicksort is about 2×
faster), print 9.70 with 1.51 s of system time (LuaJIT 0.66): one `write(2)` per
line, and each number formatted and interned (R1, R2).

**table-sort-cmp** (22.05M comparator calls). Per comparison: tcvm 529
instructions and 121 cycles, the interpreter 625 and 126, LuaJIT 417 and 130.
53% of cycles are the native sort's call path (`sort_drive`, `sort_call`,
`ret_native`, `sort_cont`, `enter`); the comparator region is 30%, most of it
waiting on the `lhs.x`/`rhs.x` loads. The float `~=` is the generic chain.
`rand() % 10000` on floats calls `fmod` (6.5%). The same sort written in Lua is
1.67 Gcyc against the native 2.67 (LuaJIT's, with the comparator inlined,
1.15) (C5, R3, S1, S5).

**richard.** Calls 14% (about 151 instructions per method call), `fmod` 12.9%
(the idler's `% 65536` and friends on floats: 129 instructions and 35 cycles
per iteration against LuaJIT's 10 and 5), `jit_eq` 9.1% (`x ~= NO_TASK` with a
nil upvalue). The main loop header carries eight block parameters derived from
rematerializable values, kept in spill slots; its collector check is 7.8% of the
region's samples. LuaJIT's root trace inlines seven functions (S5, S3, I1, Q4,
S7, C2).

**deltablue.** `jit_index` → `walk_index_chain` → `raw_get` 23% (two-level
`o:get()` 354 instructions and 47 cycles; a nil own field under `__index` 194
and 26; LuaJIT 6 and 1), calls 11%, collection 11%. `EqualityConstraint:execute`
dispatches on two shapes at every access (2304 bytes after three compiles).
Fields of `setmetatable({}, mt)` objects live in aux storage. 50 of 122
compiles are recompiles; compiling is 7.7% of the run. Two compiles are refused
("no path does work before a deopt"). LuaJIT aborts on `FNEW` (149), `UCLO`
(20) and inner loops, and interprets 57% of its run (F1, F2, F3, C1, T3).

**havlak.** Collection 33% (`trace(TableState)`, `trace(Value)`),
`setmetatable(obj, {__index = K})` 9.4% (`adopt`, `join_by_memo`, transition
lookups), `jit_newtable` 4.8%, `memset_pattern16` 3.9%, `jit_closure` 2.8%,
`set_int_key` through `jit_newindex` 1.8%. A three-field object costs 194
instructions and 28.5 cycles (LuaJIT 798 and 141); with its `setmetatable` 555
and 78 (LuaJIT 1470 and 256). LuaJIT spends 41% in its collector and aborts on
`FNEW` (216) and `UCLO` (110) (A2, A1, A3, A5, R5).

**json.** `string.sub` is a full native call in `Parser:read` (about 20% of
cycles, B1); interning of the one-character results and `memcmp` about 9.5%
even with the intrinsic (R4); `Parser` has two shapes, so `read` took four
compiles and grew to 3072 bytes (F2); `self:read()` per character is a call
(I1); `setmetatable` per value about 6% (A2). LuaJIT spends 39% in its trace
compiler ("NYI: return to lower frame" 5619 times, 136 mcode allocation
failures).
