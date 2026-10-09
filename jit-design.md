# TCVM JIT design

Branch `jit2`, written 2026-10-09 against main at bdd8c75 and revised the same day
after a review against the code. This is the document to implement from; it
supersedes the design notes of the old `jit` branch (PR #127, issues #123 and #128).
File references are to main at that commit.

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
13. Inlining (designed now, built later)
14. Tooling and testing
15. Milestones
16. Risks and open questions
- Appendix A: register assignment
- Appendix B: header words written by a region
- Appendix C: snapshot encoding
- Appendix D: opcode coverage matrix
- Appendix E: x86-64 extension points

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
   direct jumps to compiled entries, inlined builtins.
5. Compile time stays in the tens of microseconds for ordinary functions, so the
   JIT can trigger early and compile often. Budgets in section 15.
6. The second target, x86-64, can be added without touching the IR, the optimizer,
   the runtime protocol or the allocator contract. Appendix E lists the seams.

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
  table of 16-bit counters in `State`, in dedicated instructions: a compiler-emitted
  `FUNC` at pc 0, a compiler-emitted `LOOP` at `while`/`repeat`/generic-`for`/
  `goto` loop headers, and the numeric `FORLOOP` forms. Entries are installed only
  over those words, as LuaJIT patches `FUNCF`/`LOOP`/`FORL`. (Sections 5.2, 6.)
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
from the compare. The interpreter cost of the byte is measured with the
`FUNC`/`LOOP` cost in milestone 1.

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
from `Code` (4.8). Four opcodes are added to the ISA:

- `FUNC`, emitted by the compiler at pc 0 of every prototype (before
  `VARARGPREP`): counts a function entry (6.1), then falls through. Every way into
  a Lua function (`CALL`, `enter`, `TAILCALL`, a coroutine's first resume) passes
  it.
- `LOOP`, emitted at the header of every `while`, `repeat`, generic `for` and
  backward-`goto` loop (LuaJIT emits `BC_LOOP` in `parse_while`, `parse_repeat`
  and for a backward `goto`, `lj_parse.c:2724,2784,2804`): counts an iteration,
  then falls through. A generic `for` gets one, where LuaJIT patches `ITERL`,
  because here its back-jump is taken inside `TFORCALL_*` (`tfor_finish!` reads
  the `TFORLOOP` word, `control.rs:138-140`), so `TFORLOOP` cannot be patched.
- `JIT_ENTRY`, shape `Ad { entry: u16 }`, never emitted: written over a `FUNC`
  whose function entry was compiled.
- `JIT_LOOP`, same shape, never emitted: written over a `LOOP`, or over a numeric
  loop's `FORLOOP`/`FORLOOP_I`/`FORLOOP_F` (the back-edge, as LuaJIT's `JFORL`),
  whose loop entry was compiled. A region entered at a `FORLOOP` starts with the
  loop step.

`FUNC` and `LOOP` have no slow path, no adaptive bits and no continuation that
decodes them, so nothing but dispatch reads them. A `FORLOOP` has two readers,
both taught to recognize a JIT word: `write_forloop_form` (`control.rs:157-169`)
leaves it alone (the `FORLOOP_I` region's entry guards and `FORLOOP_I`'s own
fallback to the generic `FORLOOP`, `control.rs:259-267`, cover a changed loop
kind), and `forloop_slow` decodes the original through the entry. Every other
path that would dispatch the word at a pc (a `jit_exit` resume, 5.6; the entry-fail
stub) dispatches the original instead, which also keeps an exit `Before` a loop
header from re-entering the region it just left.

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
tails `jit_entry_fail` (6.4), which counts and then does
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
   they are the registers live at the entry, typed the same way and intersected
   with the kinds of the live frame's values at compile time (J7).
3. Unbox the guarded values into the representations the body wants.

A `JIT_ENTRY` region compiles the whole function from pc 0; a `JIT_LOOP` region
compiles from its `LOOP` or `FORLOOP` and covers everything reachable from it,
including the code after the loop and the returns, with the enclosing loops
rebuilt as real loops by peeling the rest of their current iterations (8.7). A prototype may hold up to `MAX_ENTRIES = 4`
live regions (one function entry, three loop entries); a fifth request evicts the
innermost loop entry (the one whose header is deepest in the loop forest, ties by
age), since a region entered at an outer header already contains the inner loops
and an inner entry only saves the interpreted remainder of one outer iteration
after a deopt inside the inner body. Both kinds coexist: an interpreted frame reaching a `JIT_LOOP`
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
(a call) or a deopt. Helpers may allocate; an allocating helper is followed by the
GC check of 5.6.

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

Native callees take the same path through `enter` into `native_enter`. The fast
entries of `ff.rs` are not reached; their work is done by intrinsics (section 12)
when the callee is known at compile time, and by the full builtin otherwise.

Direct calls to a compiled callee: when the callee value is a compile-time
constant (section 7.5) whose prototype has a live `JIT_ENTRY` region, the region
jumps to that region's entry after writing the header itself, with the window and
arity checks inlined (the callee's `max_stack_size` and `fixed_arity` are known).
This skips `enter`'s loads and tests and the `JIT_ENTRY` dispatch. The target
address is patched through the callee's `JitState` when the callee region is
retired (section 11.4); until the patch the old region stays valid. This is
milestone 4 work; the design only requires that the header and its resume point
are the same whichever way the callee was entered.

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
instruction in milestone 1 and a helper computing `nret = top - (base + a)` later.

`TAILCALL`'s fast arm (non-vararg Lua callee, window fits, no `HAS_OPEN`/`HAS_TBC`
flags) is compiled: copy the arguments down to `base`, nil-fill missing
parameters, rewrite header word 0 to the callee (keeping `nv`), then `become` the
callee's code with `base` unchanged and `pc = callee.code`, or its `JIT_ENTRY`
region directly when it has one. Every other case is a deopt to the `TAILCALL`
instruction.

### 5.6 Exits

Every guard is `b.cond` to a stub; a stub is two instructions:

```
exit_K:  mov  w16, #K
         b    exit_region        ; the region's three-instruction trampoline into exit_common (10.6)
```

`exit_common` is one routine per code segment (section 10.6). It stores x0–x15,
x19–x21, x26–x28, d0–d31 and the region's spill slots to `State.jit.exit_regs` (a
fixed buffer reached through `rt`), pops the native frame, and tails `jit_exit`, a
`slow` handler, with `insn` packing the region pointer and `K`. `jit_exit`:

1. Finds `exit K` in the region's exit table: its snapshot, its resume pc and
   kind, its counter.
2. Interprets the snapshot (Appendix C): for each entry, materializes the `Value`
   from the recorded location (a register image, a spill slot of the frame just
   popped, read before the pop into the same buffer, a constant from the region's
   pool, or an unboxed i32/f64 to box) and stores it to the home slot. Entries for
   unchanged slots are not in the snapshot; dead registers are not in it.
3. For a snapshot with inlined frames (Appendix C), writes the inlined callees'
   headers from the outermost in, so the interpreter sees ordinary frames.
4. Records what the failing guard saw into the site's feedback on every exit,
   reading the site's operands from the frame it just restored (a `Before` exit's
   operands are all in home slots by then), and counts the exit when it is
   widenable (6.4); requests a recompile or a blacklist when a threshold trips.
5. `sync!`s the frame and `become`s the handler of the instruction at the resume
   pc with `insn` loaded from `Code` (which may by now be a quickened form, which is
   fine), or for an "after" snapshot the instruction following it. A
   `JIT_ENTRY`/`JIT_LOOP` word there is replaced by its entry's original (5.2):
   resuming at a loop entry must run the loop's own instruction, not re-enter the
   region that just exited.

Exit kinds:

| Kind | Resume | Used by |
|---|---|---|
| `Before(pc)` | re-execute the instruction at `pc` in the interpreter | type and shape guards, overflow, generic fallbacks, unsupported ops |
| `After(pc)` | continue at `pc + 1`; the instruction's effects are complete | GC checks after an allocation, a helper that completed but reported "slow path needed afterwards" |
| `Gc(pc)` | as `After`, but the handler exits with `Exit::Gc` first | the collector check |

A deopt never raises an error: an operation that would error in the interpreter
(a nil index, a non-callable, a bad `for` value) deopts `Before` and the
interpreter raises it with its own message and position.

Only widenable exits count toward recompilation: type, shape, overflow and
entry-guard exits, which a recompile can cover. A never-executed exit (6.3) is
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
hold anything. The region must only make sure the collector gets its chance:
after every allocating operation (`NEWTABLE`, `CONCAT`, `CLOSURE`, a helper that
boxed an integer or built a string, a native that returned) it emits the
interpreter's check, `ldp x9, x10, [metrics]; cmp x9, x10; b.hs gc_exit_K`, where
`gc_exit_K` is a `Gc(pc)` exit with an `After` snapshot. Natives returning through
a resume point have already had their check in `native_enter`.

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
  helper. `VARARGPREP` with `needs_vararg_table` is a helper that allocates, then a
  GC check.
- `TBC` deopts (it registers a to-be-closed slot, rare in hot code); `CLOSE` with no
  TBC entries is a helper call to `close_upvalues`, with TBC entries a deopt.
- `FORPREP` and `TFORPREP` hidden slots are ordinary registers with known types
  after the prep (section 7.6).

## 6. Hotness, entries and tiering

### 6.1 Counters

`State` gains `hot: [Cell<u16>; 64]`. Each counting instruction carries its
counter index in a free operand byte (`FUNC` and `LOOP` by construction, `FORLOOP`
in the `b` byte its `AImm` shape leaves free), assigned round-robin by the compiler.
A hash of the pc would waste half the table on `FUNC` sites, since `Code` is
16-byte aligned (4.9). Counters decrement by one; one that reaches zero tails
the cold handler `jit_hot`, which resets it to `HOT_START` and decides.
Collisions between sites only make a site hot earlier.

| Site | Handler | Cost |
|---|---|---|
| function entry | `FUNC`: `ldrh; sub; strh; cbz` on `rt + HOT + idx * 2`, then dispatch | one extra dispatch per call plus 4; `call_body!` and `enter` are unchanged |
| numeric loop | `FORLOOP_I`, `FORLOOP_F`, `FORLOOP` on the taken branch | 4 |
| other loops | `LOOP` at the header | one extra dispatch per iteration plus 4 |

No compare handler, `JMP` or `TFORLOOP` counts, so the sign tests on backward
branches in 22 compare handlers are gone. What the interpreter pays instead is
the `FUNC` and `LOOP` dispatches; milestone 1 measures them on `mandel`, `nbody`,
`fib`-shaped call benchmarks and the LJR suite with the JIT off.

Defaults, all tunable through `TCVM_JIT_HOT_CALL` and `TCVM_JIT_HOT_LOOP`:
`HOT_START = 200` for both kinds. A function called 200 times or a loop iterating
200 times compiles. Compile cost (section 15 budgets) is tens of microseconds; 200
iterations of a small loop is a few microseconds, so the earliest compile is paid
back by a loop of a few thousand iterations, which is the shape every benchmark
and most hot code has.

### 6.2 `jit_hot`

A `slow` handler, entered from `FUNC`, `LOOP` or a `FORLOOP` with the frame at that
instruction. It:

1. Finds or creates the prototype's `JitState`.
2. Refuses quickly when the entry is blacklisted, already compiled, or compiling.
3. Runs the compiler (section 15 budget) synchronously, with the live frame
   available for seeding a loop entry's types; the entry pc is the counting
   instruction itself.
4. On success, installs the region (11.3), writes the `JIT_ENTRY`/`JIT_LOOP` word
   over the counting instruction, and enters the region at once, since the frame
   is exactly at the entry.
5. On a compiler failure (an internal limit, never a semantic decline) counts a
   strike, dispatches the original instruction; three strikes blacklist the entry.

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

A generic `FORLOOP` or `TFORCALL` is not a signal: `FORPREP` keeps a generic
`FORLOOP` when the values do not fit, and `TFORPREP` keeps a generic `TFORCALL` for
any iterator other than `next` and `ipairs`.

### 6.4 Exits, recompilation, blacklisting

Each exit has a 32-bit counter in the region's exit table. Every exit records
what it saw (section 7.1): the kinds of the guarded values and an overflow into
the site's feedback byte, the receiver shape for a shape guard (up to four per
site) into `JitState.shapes_seen`. On the `EXIT_HOT = 10`th taking
of one widenable exit (5.6) `jit_exit` marks the entry for recompilation by
pointing its entry-table slot (11.1) at `jit_recompile`, a `slow` handler. The
next arrival at the `JIT_ENTRY`/`JIT_LOOP` word lands there with the frame
exactly at the entry, compiles with the widened feedback and the live frame's
types (J7), replaces the region (11.4) and enters it. Counting stops while an
entry is installed (its `FUNC`/`LOOP`/`FORLOOP` is overwritten), so no counter
could trigger the recompile, and none is needed. After `MAX_RECOMPILES = 4`
recompiles of one entry the entry is blacklisted: its word is restored, its
regions retired, and `jit_hot` refuses it until the prototype dies. Evictions
(5.2) re-create entries with fresh counts, so a prototype also has a lifetime cap,
`MAX_COMPILES = 16` regions of any kind; past it `jit_hot` refuses the prototype,
which bounds the retired-region memory of 11.2.

Widening is monotone, as the interpreter's own lattice is: a site that saw small
ints and floats compiles its float path; one that saw three shapes compiles a
three-way chain; one that saw more than four goes generic (a helper call). A
`JIT_LOOP` region whose entry guards fail repeatedly (the loop is entered with
other types later in the run) is widened the same way: the entry-fail stub tails
`jit_entry_fail`, a `slow` handler that counts on the region, records the frame's
types at the entry as feedback, and dispatches the original instruction.

### 6.5 Interaction with the adaptive interpreter

Compiled code never rewrites bytecode or inline caches. The interpreter keeps
quickening, so by the time a recompile happens the deopted instruction has been
re-specialized by the interpreter's own rules, and the builder reads the newer
form. Retired entries restore the original word: a `FUNC` or `LOOP`, or a
`FORLOOP` form, which `FORPREP` rewrites again on the loop's next start as it
would have.

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
| `_FF` | both float (`Guard FLOAT`: unsigned compare against `BOX`); hardware op; `MOD_FF` and `POW_FF` by helper with a canonical-NaN fixup; result `F64` |
| `_IF`, `_FI` | one small, one float; `scvtf` the small one; result `F64` |
| `_NN` and locked generic | no speculation: `Val` operands, helper `jit_arith(rt, kind, a, b) -> Value or FAIL`; `FAIL` (a metamethod or an error) deopts `Before`; the helper boxes big integers and is followed by a GC check |
| generic, in a never-executed block (6.3) | `Deopt(Before)` |
| generic, byte with integer kinds only | `I64` code (below) |
| generic, any other nonzero byte | generic code: the handler's inline small/float cases, then `jit_arith` |
| immediate forms `_I`, `_F`, `_IF` | one register operand guarded as the form says; the immediate is a constant (`Imm`: 31-bit int or f32-exact float); `ADDI_I` becomes `adds w, w, #imm` when the immediate fits 12 bits |
| `ARITH_MM`, `_MM_R`, `_MMI` | left (or right) operand `Guard TAB`, its shape guarded, the class's metamethod at `mm_at(idx)` loaded from the class and compared against the compile-time constant (a function); a call through 5.4 with `ret_store_a` semantics (one result into `R[a]`). A mismatch deopts |
| `UNM` | small with overflow check, or float `fneg`; else helper |
| `BNOT`, `NOT`, `LEN` | small `mvn`; falsy test; table length via `raw_len_hint` helper after a live `__len` test on the class bits (nil result deopts) |
| `CONCAT` | helper that builds the string from two values (numbers coerced), `FAIL` deopts; GC check after |
| `FORPREP` | integer prep inline when init and step are small (the limit may be any number: helper `jit_for_limit` computes `last` or skips); `last` outside i32 deopts, as the interpreter then writes generic `FORLOOP` (it writes `FORLOOP_I` only when `last`, step and init all fit, `control.rs:221-224`); float prep inline for floats; other combinations deopt; writes the four hidden slots (`R[a]` last, `R[a+1]` step, `R[a+2]` index, `R[a+3]` the visible copy; a float loop keeps the limit in `R[a]`) as typed SSA values |
| `FORLOOP_I` | `cmp idx, last; b.eq exit_loop; add idx, idx, step` with the step a constant when `FORPREP` saw a constant; no overflow check needed (`FORPREP` chose `last` in range) |
| `FORLOOP_F` | `fadd; fcmp; b.cond`, direction chosen at compile time when the step is a constant |
| `FORLOOP` generic with integer kinds (`forloop_slow` writes `BIGINT`) | an `I64` loop: `last`, step and index unboxed once at `FORPREP` or the entry, `cmp; b.eq; add` on x registers, the visible copy boxed only when it escapes |

**`I64` code.** Lua integers wrap (`math.maxinteger + 1 == math.mininteger`,
`(1 << 62) * 4 == 0` in lua 5.5.1), so `add`, `sub`, `mul`, the bitwise ops and
`lslv`/`lsrv` on x registers are the exact semantics, with no overflow check.
`//` and `%` deopt on a zero divisor (the interpreter raises) and handle
`mininteger // -1` (`mininteger`) and `% -1` (0) with a `csel`, since `sdiv`
returns the dividend there without trapping. Values enter `I64` by
`Unbox{I64}` (a small int's `sxtw`, or a boxed int's load from its box, after a
two-way kind test) and leave by `Box{I64}`, which keeps the interpreter's
invariant that a value in i32 is always a small int (`value.rs:144-149`): a range
test, then the small-int `orr`, or the helper `jit_box_i64` and a GC check when
the value is outside i32. A boxing that allocates happens only where the value
escapes: a store to a table or upvalue, an argument or result, a register live at
a tail-out or a snapshot. A loop that keeps a big integer in a register allocates
nothing.

Overflow policy: an `I32` site's overflow exit records `OVERFLOW`, and the
recompile gives that site, and the values that flow from it, `I64`. The range
analysis of section 9.7 removes checks and narrows `I64` back to `I32` where
bounds prove it.

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
| generic register compare (`JLT`, `JNLT`, `JLE`, `JNLE`) | operands typed by inference or use-driven speculation (9.1): `fcmp`, `cmp w` or `I64` `cmp` with `b.cond`; byte with integer kinds only: `I64` `cmp`; otherwise generic inline in the handler's order: both floats, both small, small/float mixes, then helper `jit_cmp` that returns a bool or `FAIL` for strings and metamethods |
| generic `JLTI` … `JNGEI` | a register typed `I32`: `cmp w, #imm; b.cond`; typed `F64`: as `_F`; otherwise small test, float test, helper |
| `JEQ`/`JNEQ` generic | in the handler's order (`compare.rs:67-71`): two floats by `fcmp` first, since a NaN has its own bits and must not equal itself; then bit identity; then both small; then helper for boxed ints and `__eq` |
| `JEQI`, `JNEQI`, `JEQS`, `JNEQS` | inline: small compare, float compare, or bit identity against the interned constant; no metamethod |
| `JT`, `JF`, `JTSET`, `JFSET` | `sub x, v, x_nil; cmp x, #2; b.lo/hs` with `NIL` hoisted; the `SET` forms assign on one edge (an edge block param, as the old `edge_defs` did) |

A compare whose operands the inference proves `I32` or `F64` loses its guards; the
compare of a loop counter against its `last` is the common case.

### 7.4 Fields

| Site form and IC entry | Code |
|---|---|
| `GETFIELD_INL`, `GETTABUP_INL`, `SELF_INL` with `Own{shape, loc}` | receiver `Guard TAB`; `ldr s, [t + SHAPE]; cmp s, #shape; b.ne exit`; `ldr v, [t + loc]`; a nil result reads the class bits live (`ldr w, [class + BITS]; tbnz w, #INDEX, slow`, with the class address a constant since the shape pins the class pointer, 4.7) and takes the slow helper or a deopt when `INDEX` is set; only the nil path pays |
| `_AUX` with a spilled `loc` | as above through `ldr spill, [t + SPILL]; ldr v, [spill + off]` |
| `_ABSENT` with `Absent{shape}` | shape guard; class bits read live; result nil without `INDEX`; with `INDEX` and the class's `__index` (read live) a function, a metamethod call (5.4); otherwise deopt (an `__index` table that lacks the key, or a stale class) |
| `_PROTO` with `ProtoLoad{recv, holder, holder_shape, loc}` | receiver shape guard; `ldr cls, [shape + MT_CACHE]; ldr it, [cls + INDEX_TABLE]; cmp it, #holder`; `ldr hs, [holder + SHAPE]; cmp hs, #holder_shape`; load `loc` from the holder; nil deopts (the walk continues) |
| `SETFIELD_INL/_AUX`, `SETTABUP_*` with `Own` | shape guard; load the existing value, and when it is nil read the class bits live and deopt on `NEWINDEX`; barrier (5.7) unless the stored value is proven non-heap; store. The SET fill never caches a mirrored key on an adopted shape (4.7), so this store never changes class bits |
| `_TRANS` with `Transition{from, to, loc}` | shape guard on `from`; class bits read live, `NEWINDEX` deopts (the shape pins the class, not its bits); value nil deopts (stores nothing in the interpreter, which the deopt reproduces); `has_room(loc)` inline for a spilled `loc` (`ldr cap, [t + SPILL_CAP]; cmp`), true for an inline one; barrier; store the value; store `to` into `t + SHAPE` (the shape pointer is a GC object the table now references: the barrier above covers it, since the table is the parent) |
| `_ABSENT` store | class bits read live; an `__newindex` function on the class: a metamethod call; else deopt |
| `GETFIELD`/`SETFIELD` generic with `Empty` and a zero byte | a recording site that never ran: its block is never executed (6.3), deopt |
| a field site with an `Empty` IC and a nonzero byte (it ran, never cacheable: a dict shape, a mirrored key) | the generic helper (`jit_index`/`jit_newindex`) |
| generic with a filled IC that no longer matches the form | the IC was refilled after the form was written; the builder uses the IC entry, which is authoritative, and the form only as a hint |
| `GETTABUP_REF`, `SETTABUP_REF` | receiver loaded through the cell (`ldr cell, [closure + UPVAL i]; ldr p, [cell + V]; ldr t, [p]`), then as the `_INL` cases using the IC entry |

A shape compare covers later uses of the same receiver until something may change
a shape (`W_SHAPE`: a transition, a generic store helper, `setmetatable`, a call;
8.4). GVN removes the repeats in between, and LICM hoists it out of a loop whose
receiver is loop-invariant and whose body writes no shape, which a loop of stores
into existing fields does not (section 9.5). The class-bit reads are `R_META` and
hoist the same way.

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

`LOAD`, `LOADI`, `LOADNIL`, `LFALSESKIP`, `MOVE` are constants and copies in SSA
and emit nothing by themselves.

### 7.6 Tables, arrays, iteration

| Site | Code |
|---|---|
| `GETTABLE` with a key proven `I32` | receiver `Guard TAB`; `ldr n, [t + ASIZE]; cmp idx, n; b.hs slow; ldr arr, [t + ARRAY]; ldr v, [arr, idx, lsl #3]` (key `k` is at index `k`, 4.7); a nil result reads the class bits live and goes to the slow helper `jit_index(rt, t, k) -> Value or FAIL` when `INDEX` is set; other keys: the helper |
| `SETTABLE` with an `I32` key | bounds check, load old value, nil old value with live `NEWINDEX` bit deopts, barrier, store; else helper `jit_newindex` which returns `FAIL` for a metamethod (deopt) |
| `NEWTABLE` | helper `Table::from_template`; GC check |
| `SETLIST` | helper; a constant count has the values in home slots (R1 stores them first) |
| `LEN` | `raw_len_hint` helper when the live class bits lack `LEN`; deopt otherwise |
| `TFORPREP` | the form it wrote is the feedback: `_NEXT` means `next` over a table from position 0; `_IPAIRS` the array walk |
| `TFORCALL_NEXT` | the interpreter's walk (`table/mod.rs:1467-1579`): parts in the order array, integer hash, named shape slots, dict strings, misc hash, the position a small int `part << 28 \| index + 1` in `R[a+3]` (`next` walks the same order, so a deopt mid-loop continues correctly); array and named slots inline, the other parts by helper; the position stays in the hidden slot's encoding at every tail-out and snapshot; `TFORCALL_NEXT` itself does the following `TFORLOOP`'s back-jump (`tfor_finish!`, `control.rs:138-140`), so the loop variables are SSA values and the nil test on the first ends the loop |
| `TFORCALL_IPAIRS` | `i + 1`, bounds check against `asize`, load, nil ends the loop |
| `TFORCALL` generic | a call (5.4) through `ret_tfor` semantics landing `count` results |
| `CLOSURE` | helper `jit_closure(rt, thread, base, proto_idx)` that does what `op_closure` does, including open upvalue creation and the `HAS_OPEN` flag; GC check; captured registers are stored before it (5.11) |

### 7.7 Calls

| Site | Code |
|---|---|
| `CALL*`, `CALLS*` with an unknown callee | the sequence of 5.4; the callee value is loaded from its SSA value and stored to `R[a]` |
| callee a compile-time constant native with an intrinsic | section 12 |
| callee a compile-time constant Lua closure with a live function-entry region | direct call (5.4, milestone 4) |
| `b == 0` (arguments up to `top`) | milestone 1 deopts; later the count comes from `thread.top` |
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
`u32` newtype; nothing inside an instruction is a `Vec`.

```rust
pub struct Func<'gc> {
    pub blocks: Vec<BlockData>,        // params: Range<ValIdx>, insts: Range<InstIdx>, preds/succs: Range<EdgeIdx>
    pub insts:  Vec<InstData>,         // op, args: Range<ValIdx>, results: Range<ValIdx>, snap: Option<SnapIdx>, effects
    pub vals:   Vec<ValData>,          // ty: Ty, def: ValDef (Inst(i, k) | Param(b, k)), uses: u32
    pub args:   Vec<Val>,              // the operand pool the ranges index
    pub edges:  Vec<BlockCall>,        // target: Block, args: Range<ValIdx>
    pub snaps:  SnapTable,             // section 8.6
    pub pool:   ConstPool<'gc>,        // values, shapes, prototypes, strings, helpers; the only `'gc` holder
    pub entry:  Block,
    pub meta:   RegionMeta,            // entry kind and pc, max_stack_size, memory-register set, vararg-ness
}
```

`InstData` is 32 bytes: `op: Op` (u16 tag plus a u32 payload for an immediate,
a register index, a slot location or a pool index), `args: (u32, u16)`,
`results: (u32, u8)`, `snap: u32` (`NONE` for most), `effects: Effects` (u16),
`block: Block`. Arguments and results are ranges into `args`, so an instruction
with any arity costs no allocation. Block parameters are the values in
`blocks[b].params`; a branch's `BlockCall` carries the argument range for each
target. Instruction order inside a block is the order of `insts` within its range;
passes that insert build a new `Func` by a single ordered rewrite rather than
splicing (a rewrite over flat arenas is a linear copy, which is what the old
`simplify` already did).

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

Boxing and tests: `Unbox{I32|F64|Ptr}(v)`, `Box{I32|F64|B1|Ptr[tag]}(v)`,
`IsType[set](v) -> B1`, `IsFalsy(v) -> B1`, `Select(c, a, b)`.

Guards (all carry a snapshot, all deopt `Before` unless noted): `Guard[set](v) ->
v'`, `GuardShape[pool](t) -> t'`, `GuardConst[pool](v) -> v'` (pointer or bit
identity), `GuardIndexTable[pool](shape)` (class `__index` identity),
`GuardCond(b1)`, `GuardNotNil(v) -> v'`, `GuardFits(v)` (`stack_end` room for a
staged header), `Deopt` (unconditional, `Before`), `GcCheck` (exit `Gc`, `After`).

Integer (`I32` in, `I32` out): `IAdd`, `ISub`, `IMul` (overflow deopts; carry a
snapshot), `IAddNo`, `ISubNo`, `IMulNo` (proven not to overflow, 9.7), `INeg`,
`IDivFloor`, `IModFloor` (zero divisor deopts; expanded before lowering, 10.3),
`IAnd`, `IOr`, `IXor`, `INot`, `IShl`, `IShr` (Lua count semantics, expanded),
`ICmp[cc] -> B1`, `IToF -> F64`.

Integer (`I64` in, `I64` out, wrapping, no snapshot): `LAdd`, `LSub`, `LMul`,
`LNeg`, `LAnd`, `LOr`, `LXor`, `LNot`, `LShl`, `LShr` (Lua count semantics),
`LDivFloor`, `LModFloor` (zero divisor deopts; `mininteger`/`-1` by `csel`),
`LCmp[cc] -> B1`, `IToL` (`sxtw`), `LToI` (deopts outside i32; used by range
narrowing and array keys), `LToF -> F64`.

Float: `FAdd`, `FSub`, `FMul`, `FDiv`, `FNeg`, `FAbs`, `FSqrt`, `FFloor`, `FCeil`,
`FFloorDiv` (`fdiv` then `frintm`), `FMod` and `FPow` (helpers, canonical-NaN
fixup on the result), `FCmp[cc] -> B1` (unordered is false), `FToIExact -> I32`
(deopts when not an exact i32; used by `floor`/`ceil` intrinsics and array keys).

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

Generic Lua operations as helpers: `LuaArith[kind]`, `LuaCmp[cc]`, `LuaEq`,
`LuaConcat`, `LuaLen`, `LuaIndex`, `LuaNewIndex`, `LuaForPrep`, `LuaNext`,
`LuaVararg`. Each is a `Helper` with a fixed signature.

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
their value. Storage is one `Vec<u32>` arena with each snapshot a header word
followed by entries; identical snapshots are deduplicated by hash while building
(most guards in one instruction share one). Appendix C has the encoding. Before
lowering, each snapshot's value references are ordinary IR uses (they keep values
alive and are the allocator's `Any` uses on the guard instruction); after
allocation they are replaced by locations. A value that is a constant or a
rematerializable unbox of an unchanged slot is recorded as such and is not an
allocator use, which is what avoids the old design's pinned constants.

### 8.7 The builder

One pass over the bytecode from the entry pc:

1. **CFG**: block boundaries from branch targets and fallthroughs (the ported
   `cfg.rs`), with `JIT_ENTRY`/`JIT_LOOP` words resolved to their originals, and
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
   `OSR_PEEL_LIMIT = 400` duplicated IR instructions, past which the outermost
   remaining levels stay merged.
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
   earlier. Its snapshot is `After` the defining instruction.
5. **Entries are the exception.** Parameters and the live registers at a loop
   entry are guarded in the prologue even where that point is not down-safe: a
   failed entry guard costs one interpreted run of the original instruction
   (`jit_entry_fail`, 6.4), and the prologue needs the types. A loop entry
   intersects the expectation with the kind of the live frame's value (J7); if
   they disagree, the register is not speculated.
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
different transformation and runs in the builder.

### 9.4 Global value numbering with guard elimination

Dominator-order hashing of pure instructions by op and arguments. Guards hash by
op, argument and payload, and a guard is redundant when an equal guard dominates
it with no intervening write to the class it reads (8.4). Loads (`TabLoad`,
`Load`, `UpvalValue`, `CellLoad`, `ClassMm`) are value-numbered with the last
write to their class as an extra key, which gives load forwarding for free within
a block and across blocks along the dominator tree.

### 9.5 Loop-invariant code motion

For each loop, instructions whose arguments are defined outside the loop and
whose effects read only classes the loop does not write are moved to the
preheader. Guards are moved under the same rule, and only from blocks that
dominate every latch of the loop: a guard in a conditional arm would deopt
iterations that never reach it. A hoisted guard takes a new snapshot, `Before`
the loop header with the preheader's state, and deopts at the loop entry, which is
correct because the loop's first iteration would have deopted there too (peeling
guarantees the first iteration's guards are present before the loop). Allocating
instructions and calls are never hoisted.

### 9.6 Dead code elimination and narrowing

Unused pure instructions are deleted; `Box` of an `Unbox` and `Unbox` of a `Box`
fold; a `Val` that is only ever unboxed to `I32` is kept unboxed from its
definition when the definition is a boxed `I32` producer (narrowing of block
parameters: a parameter whose every incoming argument is `Box(I32)` becomes an
`I32` parameter with the boxes moved to the uses that need `Val`).

### 9.7 Range analysis (after milestone 3)

Interval analysis over `I32` values: constants, `FORPREP` bounds (`last` and the
step give the counter's range), compares on dominating edges, `asize` bounds for
array indices. An `IAdd`/`ISub`/`IMul` whose operand ranges keep the result in
i32 becomes the `No` form without the overflow branch; an `I64` value whose range
fits i32 is narrowed to `I32`, which boxes with one `orr` instead of a range test;
an array index proven in `[0, asize)` by a dominating compare loses its bounds
check. This is V8's
`maglev-range-analysis` idea scoped to what Lua loops need.

### 9.8 Call boundaries, snapshot pruning and critical edges

The call-boundary pass enforces R6. For each tail-out that resumes (`Call`,
`CallMm`) it computes the values live across it; each must be a constant, a pool
load or an unbox of a slot reloaded after the resume, and is re-emitted after the
resume point with its uses below rewritten. Anything
else live across a tail-out is a builder bug, and the verifier rejects it. This
is what lets GVN and LICM share constants freely and lets the backend keep
`NIL`, `BOX` and shape literals in registers between calls (10.3).

Snapshots are then re-pruned against the final liveness (a value only referenced
by snapshots and not otherwise used is a "snapshot-only" value, which the
allocator may place anywhere and which is rematerialized when it is a constant or
an unbox of an unchanged slot). Critical edges are split by inserting empty blocks
so that `regalloc2`'s requirement holds and edge-block parameters have a home.

### 9.9 Block layout

`order.rs` lays out blocks loop-contiguously with the exit stubs at the end; it
rejects irreducible graphs, which Lua's `goto` can produce; such a region fails
compilation with an internal limit and is counted as a strike (6.2), never
mis-compiled.

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
  allows it, so region-to-region branches (exits, M4 direct calls) are direct as
  well.
- **Exhaustion.** Code memory is bounded by the window (well over 100 MiB). When
  no segment fits, `jit_hot` refuses new compiles and logs it once; code cannot be
  flushed while frames hold resume points into it (11.2), and a prototype's code
  returns to the free list when it dies.
- **Encoder.** `b`/`bl` check the ±128 MiB range at emission and fail the compile
  (a strike) rather than emit a wrong branch.

Each segment starts with one copy of `exit_common`; regions follow. A region's
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
exit_common:   ldr  x17, [x23, #EXIT_REGS]  ; State.jit.exit_regs, 64 + 64 words
               stp  x0, x1, [x17] … str x28, [x17, #…]      ; words 0-21: x0-x15, x19-x21, x26-x28
               stp  d0, d1, [x17, #176] …                  ; words 22-53: d0-d31 (64-bit, not q)
               and  x9, x16, #0xFFFF_FFFF_FFFF              ; the Region
               ldr  w10, [x9, #NUM_SPILLS]                  ; this region's spill count
               (copy w10 words from [x29, #16] to [x17, #512])   ; words 64-127
               mov  sp, x29                 ; pop the region frame whatever its size
               ldp  x29, x30, [sp], #16
               mov  x20, x16                ; insn = region | K << 48
               b    jit_exit                ; base, rt, closure, thread are the pinned registers
```

x16 and x17 are never allocated (x17 is regalloc2's scratch only inside parallel
moves, which no stub interrupts), so the stubs clobber nothing. The spill copy
reads exactly the region's spill area. Regions needing more than 64 spill slots
are refused by the backend (a strike, 6.2), which never happens for a sane
region.

`jit_exit` (a `slow` handler) unpacks the region and the exit id from `insn`,
walks the snapshot against the register image and the spill copy, writes the
home slots, counts, updates feedback, publishes the frame and dispatches the
resume instruction (5.6). The entry-fail stub of the prologue needs no image: it
pops the frame and tails `jit_entry_fail` with the region in `insn` (6.4).

GC exits are exits with an `After` snapshot whose handler returns `Exit::Gc` after
writing the slots; the executor returns `Pending`, the collection runs as
`Lua::enter` exits (4.6), and the next step resumes at the published pc, the
instruction after the allocating one, in the interpreter. The next loop entry or
call re-enters compiled code (J9).

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
    pc: u32,                 // the FUNC, LOOP or FORLOOP word
    original: Instruction,   // the word JIT_ENTRY/JIT_LOOP replaced
    slot: u16,               // index into State.jit.entries, the word's `d`
    region: Option<Gc<Region>>,
    recompiles: u8,
    state: EntryState,       // Empty | Compiled | Recompile | Blacklisted
}
```

### 11.2 Regions

```rust
pub struct Region<'gc> {
    code: CodeBlock,                         // the segment allocation; freed on drop
    entry: *const u8,
    pool: ConstPool<'gc>,                    // traced: the only GC references compiled code embeds
    exits: Box<[ExitInfo]>,                  // snapshot offset, resume pc, kind, site pc, guard kind, counter
    snaps: Box<[u32]>,                       // Appendix C
    calls: Box<[CallSite]>,                  // per call site: a, wanted, kind (for direct-call patching later)
    proto: Gc<'gc, Prototype<'gc>>,
    entry_pc: u32,
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
old one retired. Direct calls into the old region (5.4, milestone 4) are repatched through
the callers' `CallSite` lists: each region records which callee entries it jumps
to directly, and each `Entry` keeps a list of (region, patch offset) pairs to
rewrite when its code changes. Until milestone 4 no such edges exist.

Blacklisting restores `original` into `Code[pc]`, frees the table slot, retires the
region and sets the entry's bit; `jit_hot` checks the bit first.

### 11.5 Global assumptions

One fact is compiled in as a constant and checked by a region-level watch rather
than per use: that no number metatable is set (`ARITH_MM_R` and `_MMI` with the
immediate on the left depend on it; `rt.number_metatable()` in the handlers).
Only `set_metatable_of` for a non-table type can change it, and it bumps
`State.jit.epoch: Cell<u32>`. A region that depends on the fact records the epoch
it was compiled under and compares `[rt + EPOCH]` against it at its entry (tailing
the original instruction on mismatch) and at every resume point after a call
(deopting `After` the call on mismatch), since only a call can change the epoch
from inside a region. Three instructions per check; regions without the
dependency emit none.

The string metatable is not such a fact. It is an ordinary table, and Lua code
can rewrite its fields without `set_metatable_of`: after
`getmetatable("").__index = {upper = function() return "hijacked" end}`,
`("a"):upper()` returns `hijacked` (lua 5.5.1). A string method site is compiled
like a `_PROTO` site: compare the string metatable pointer loaded from `State`
with its compile-time constant, read the class's `__index` table live
(`ClassIndexTable`, `R_META`) and compare it with the string library table,
guard that table's shape, load the method's slot and compare it with the
intrinsic's function (`GuardConst`). Every step is a load and a compare, and all of
them hoist out of loops that write no metatable and no shape.

### 11.6 Sizes

| Structure | Size |
|---|---|
| feedback byte | 1 byte per instruction of every prototype |
| `ExitInfo` | 16 bytes per exit |
| snapshot entry | 4 bytes; a typical guard snapshot has 2 to 6 entries |
| exit stub | 8 bytes of code per guard |
| `exit_regs` buffer | 1 KiB per `State` |
| hot counter table | 128 bytes per `State` |
| entry table | 8 bytes per installed entry, grown by doubling |
| `FUNC` | one instruction per prototype; `LOOP` one per non-numeric loop |

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
| `rawget(t, k)`, `rawset(t, k, v)` | the raw paths of 7.6 without the metamethod tests | |
| `rawequal`, `rawlen` | bit identity; `raw_len_hint` helper | |
| `select('#', ...)`, `select(n, ...)` | from `nv` | |
| `pairs(t)` | `(next, t, nil)` as `ff_pairs` when the class lacks `PAIRS`; feeds `TFORPREP`, whose `_NEXT` form the region then compiles | `__pairs` present |
| `ipairs(v)` | `(ipairs_iter, v, 0)` | |
| `setmetatable(t, mt)` | helper `set_metatable_fast`; `Some(true)` is followed by a GC check | the helper returns `None` |
| `getmetatable(t)` | class owner lookup helper | `__metatable` field present |
| `assert(v, ...)` | truthiness test; the arguments are the results | falsy |
| `string.sub`, `string.byte`, `string.len`, `#s` | helpers (allocation for `sub`), `I32` positions | non-small positions |
| `tostring(n)`, `tonumber(s)` | helpers | metamethods |
| `pcall`, `xpcall`, `error`, `coroutine.*`, `string.format`, `table.*` | no intrinsic: an ordinary call through `enter` | |

Intrinsics are a table in `src/jit/intrinsics.rs` keyed by the native function
pointer, filled when the standard library is installed.

## 13. Inlining (designed now, built later)

A `Call` whose callee is a constant Lua closure (7.5) with a small prototype (at
most `INLINE_LIMIT = 40` instructions, not vararg, no `TBC`, no `CLOSURE` capturing
its parameters by reference, no `TAILCALL`) can be inlined at milestone 4:

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

Budget: inlining depth 2, total inlined instructions per region 120.

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
   slow-path writers (4.8), ISA opcodes `FUNC`/`LOOP` (emitted by the compiler)
   and `JIT_ENTRY`/`JIT_LOOP` (the ISA goes from 204 to 208 opcodes), each
   measured with the JIT off. Then the entry table, counters, `jit_hot`,
   `jit_recompile`, `jit_entry_fail`, `JitState`/`Region`, the builder with
   constants, moves, `_II`/`_FF`/`_IF`/`_FI` arithmetic, `I64` arithmetic and
   loops with `jit_box_i64` and its GC-check exit, immediates, compares,
   `JT/JF`, `FORPREP`/`FORLOOP_I/F`, `JMP`, `RETURN0/1`, type inference,
   peeling, GVN, the aarch64 backend with `regalloc2`, exits with snapshots,
   `jit_exit`, the never-executed block rule (6.3), use-driven speculation (9.1),
   and the `Call` instruction with resume points for unknown callees
   (so a region can call out). Everything else is `Deopt`. Exit: `primes2`,
   `collatz_bench`, `mandel_bench`, `iterative_fib_bench` run compiled with no
   exits in steady state (for `collatz_bench`, after its overflow recompile, with
   `n` in an x register, 7.2); compile time per stage
   measured on them; the interpreter changes measured with the JIT off against
   main (6.1, 4.8).
2. **Helpers and allocation.** Generic arithmetic and compare helpers, `_NN` and
   locked sites, `CONCAT`, `NEWTABLE`, `SETLIST`, `CLOSURE`, GC exits, barriers,
   `VARARG` forms, `CLOSE`, upvalue reads and writes, intrinsics for `math.*` and
   `type`. Exit: `fft`, `nbody` compiled end to end; `TCVM_JIT_DEOPT_ALL` passes on
   all test files.
3. **Tables and metamethods.** Field forms with ICs, `GETTABLE`/`SETTABLE` array
   paths, `_PROTO` chains, `ARITH_MM*`, metamethod calls, `TFORCALL_NEXT/_IPAIRS`,
   `LEN`, `MULTRET` forms, `TAILCALL` fast arm, hot-exit recompilation with
   polymorphic shape chains, blacklisting. Exit: `particles_bench` and the LJR
   suite compiled; results versus the interpreter and LuaJIT recorded in the
   perf notes.
4. **Calls.** Direct calls to compiled entries with patching, inlining (13),
   `pcall` through `enter`, range analysis (9.7). Exit: the LJR call-heavy
   benchmarks (`fib`, `richards`, `deltablue`, `havlak`) measured against the
   interpreter and LuaJIT.
5. **x86-64.** The second target per Appendix E, run under Rosetta for
   correctness.

Compile-time budgets, measured in milestone 1 on a 200-instruction function on
the M4 Pro: builder 10 µs, passes 15 µs, lowering 10 µs, allocation 25 µs
(`Ion`), emission 5 µs; total under 70 µs. The old pipeline did 76 µs for `mix`
with a weaker output. The allocation figure is a target, not a citation: the
first milestone 1 measurement replaces it.

## 16. Risks and open questions

- **`FUNC` and `LOOP` dispatch cost** (6.1): one extra dispatch per call and per
  non-numeric loop iteration in the interpreter, measured in milestone 1 with the
  JIT off. If a call-heavy benchmark shows it, `FUNC`'s counter can move into
  `call_body!`/`enter` with `FUNC` kept only as the patch site, which saves the
  decrement but not the dispatch.
- **`regalloc2` on loops**: the in-house spiller was measured to beat a whole-value
  scan on `mix2`'s loops; `Ion` splits live ranges and should do as well, but
  this is checked in milestone 1 on the old `mix`/`mix2` sources. If it loses,
  the fix is in isel (rematerialization of constants and unboxes, which `Ion`
  lacks), not a return to the old allocator.
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
- **The epoch watch** (11.5) covers the number metatable only; string method
  sites guard the string metatable per site, and userdata and other per-type
  metatables do not affect compiled fast paths.

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

One `u32` arena per region. A snapshot:

```
header:   kind:2 (Before|After|Gc)  frames:2  entries:12  (16 spare)
          pc:32                                                          (pc of the innermost frame)
frame k (outer frames only, for inlined calls; three words each):
          proto_pool:32                                                  (the inlined callee's closure)
          call_a:8  wanted:8  base_delta:16                              (callee base = outer base + delta)
          caller_pc:32                                                   (the inlined CALL's pc)
entry:    reg:8  rep:3 (Val|I32|F64|B1|Ptr+tag)  loc:3  index:18
          loc: Reg (register image index) | Spill (spill word index) | Const (pool index)
             | SlotUnbox (the slot already holds the boxed value: no write)
             | Nil | False | True
```

Entries list only registers that are live at the resume pc and whose home slot
does not hold their current value. `jit_exit` writes innermost-frame entries
relative to the innermost base and, for inlined frames, writes each inlined
callee's header (`func` from the pool, `ret = rt.ret(wanted)`, `caller`, `pc`)
before its entries, outermost first.

## Appendix D: opcode coverage matrix

Strategy per opcode on main's ISA (204 opcodes); `D` is a `Before` deopt until the
named milestone, `H` a helper, `I` inline code.

| Opcodes | M1 | M2 | M3 | Notes |
|---|---|---|---|---|
| `MOVE`, `LOAD`, `LOADI`, `LOADNIL`, `LFALSESKIP` | I | | | SSA only |
| `GETUPVAL` | I | | | one load from the closure cell (7.5) |
| `GETUPVAL_REF` | D | I | | through the cell |
| `SETUPVAL` | D | H | | barrier helper |
| `GETTABUP*`, `GETFIELD*`, `SELF*`, `SETTABUP*`, `SETFIELD*` (27 opcodes: 5 generic, 20 forms, 2 `_REF`) | D | D | I | 7.4; `_REF` receivers via the cell |
| `GETTABLE`, `SETTABLE` | D | D | I/H | array fast path inline, else helper |
| `NEWTABLE`, `SETLIST` | D | H | | GC check after `NEWTABLE` |
| `ADD`..`SHR` generic (12), `_NN` (7) | D / I | H | | never-executed blocks (6.3) stay D; an integer-kinds byte gives `I64` code from M1 |
| `_II`, `_FF`, `_IF`, `_FI`, bitwise `_II`, `POW_II` (27) | I | | | 7.2 |
| `ADDI`..`RSHRI` generic (19) | D / I | H | | as the register forms |
| `_I`, `_F`, `_IF` immediate forms (27) | I | | | |
| `ARITH_MM`, `_MM_R`, `_MMI` | D | D | I + call | |
| `UNM`, `BNOT`, `NOT` | I | | | slow cases H in M2 |
| `LEN`, `CONCAT` | D | H | | |
| `CLOSE`, `TBC` | D | H / D | | TBC stays D |
| `JMP` | I | | | |
| `JEQ`..`JNLE` generic, `JEQI`..`JNGEI`, `JEQS`, `JNEQS` | I/D | H | | inline numeric cases; strings and `__eq`/`__lt` via H |
| `_II` compares (6), `_F` compares (8) | I | | | |
| `JT`, `JF`, `JTSET`, `JFSET` | I | | | |
| `CALL`, `CALL_R0`, `CALL_R1`, `CALLS*` | I (5.4) | | direct calls M4 | `b == 0` D until M3 |
| `TAILCALL` | D | | I fast arm | |
| `RETURN`, `RETURN0`, `RETURN1` | I | | | `b == 0` and flags set: D until M3 |
| `FORPREP`, `FORLOOP_I`, `FORLOOP_F`, generic `FORLOOP` | I | | | generic `FORLOOP` as an `I64` loop |
| `TFORPREP`, `TFORCALL`, `TFORCALL_NEXT`, `TFORCALL_IPAIRS`, `TFORLOOP` | D | | I | generic `TFORCALL` is a call |
| `CLOSURE` | D | H | | |
| `VARARG`, `VARARGGET`, `VARARGPREP` | D | I/H | | |
| `ERRNNIL` | I | | | nil test, deopt to raise |
| `NOP` | I | | | |
| `STOP` | D | | | |
| `FUNC`, `LOOP` | I | | | emit nothing; a region never counts |
| `JIT_ENTRY`, `JIT_LOOP` | resolved to the original word by the builder | | | |

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
