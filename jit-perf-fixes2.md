# JIT performance fixes, part 2

This plan follows `jit-perf-fixes-plan.md` (part 1), implemented on `jit2`
through `1113b83`. Part 1's milestone F (per-pass timing in `jit/compile.rs`,
the runners in `tools/`) is in the working tree, uncommitted by decision.

Measured on the LJR suite against the pre-plan JIT (`b2a382b`), part 1 cut
cycles by 1.275× (geomean of 43 programs). But eleven programs got slower,
four of them slower than the interpreter, and particles runs no faster
compiled than interpreted. The investigation is in the regression report
(https://claude.ai/artifact/U7PL4iaUoGsGdsUd3m4m6d).

This plan does three things:

- It removes the causes of those regressions.
- It replaces the way compiled code leaves and re-enters (milestone H, thread
  stacks).
- It carries over the part-1 items that matter for performance and were not
  done.

Paths are relative to `src/` unless they start with `tests/`, `tools/` or
`test-files/`. Line references were read at `1113b83`.

## Decisions taken with the user (2026-10-12)

1. **Every thread runs on a native stack tcvm owns,** the main thread included.
   - `resume` and `yield` are stack switches, made by a direct call to a naked
     switch routine.
   - Going to the host is a switch too: a collection, an async wait, or a
     main-thread yield.
2. **Compiled code never leaves to run Lua.** When it needs the interpreter,
   the interpreter runs on top of the compiled frames that called it.
3. **Only deopt exits write values out to the Lua stack,** and only the
   exiting frame's values.
4. **Heavy Rust work with no depth bound runs on the host stack.** That is
   `load`'s parser and compiler, JIT compilation, and error rendering.
   - Thread stacks reserve 1 MB plus a guard page.
   - A coroutine gets its stack at its first resume; the main thread gets its
     stack at `Executor::start`.
   - Stacks come from a per-`Lua` pool. A stack goes back to the pool when its
     thread dies or the collector frees the thread.
5. **Idle stacks are not reclaimed.** A coroutine suspended mid-run holds at
   least one 16 KB page. This cost is accepted.
6. **Past the native stack budget, calls run in the interpreter,** on top of
   the compiled frames, which stay where they are.
7. **Errors unwind through a chain of entry records.** A Rust panic on a Lua
   stack aborts.
8. **The collector traces parked stacks read-only.** It uses the same frame
   walk the unwinder uses, and traces each compiled frame's whole window.
9. **The value stack stops moving.** It is reserved in the same mapping as the
   native stack.

Contents

1. What part 1 got wrong
2. Milestone G: the cost gate
3. Milestone H: thread stacks
4. Milestone I: calls
5. Milestone J: tables, metatables and metamethods
6. Milestone K: cold arms from feedback
7. Milestone L: allocation
8. Milestone M: the rest of part 1
9. Testing
10. Order, dependencies and targets
11. Open points

---

## 1. What part 1 got wrong

Ratios are the pre-plan JIT's cycles over HEAD's, so a ratio below 1 is a
regression. particles' numbers come from the per-commit runs; the rest come
from the suite run.

| benchmark | HEAD vs pre-plan | cause | fixed by |
|---|---|---|---|
| chameneos | 0.39 | C1: every `resume`/`yield` runs `jit_leave` over all compiled frames | H5 |
| series | 0.53 | C1: native calls (58% of runtime in the walk) | H6, I1 |
| coroutine-ring | 0.62 | C1: `resume`/`yield` | H5 |
| mandel-metatable | 0.66 | C1, an arity miss on every `Complex.conj` call, allocation (36%) | I3, H6, L1, L3 |
| revcomp | 0.71 | A1 made `writerev` compile; its `io.write` calls then pay C1's native call | H6, I1 |
| list | 0.82 | absent keys on objects with `__index` call the `Index` helper in a cold arm (77%) | J1, K1 |
| havlak | 0.87 | C1; index lookups | H6, J1 |
| fixpoint-fact | 0.88 | A5: big-integer boxing through the save-all trampoline; closure creation (55%) | K1, L1 |
| recursive-fib-uv | 0.94 | C1 doubled frames to 32 bytes, so the deep-call path starts at depth 16 instead of 32 | I6 |
| binary-trees-num | 0.94 | A5: `tree[2]` on leaves misses the array part on half the calls, into a cold arm | K1, J4, L1 |
| particles | 0.76 (≈1.0 against its interpreter) | C1: `string.format`/`table.insert` calls and collector exits | H4, H6, I1 |

The cost of a native call: instructions per iteration of a loop calling
`math.ult`.

| build | 0 extra live values | 4 | 8 |
|---|---|---|---|
| before C1 (`9b1e05e`) | 201 | 227 | 273 |
| HEAD | 461 | 602 | 739 |
| HEAD, JIT off | – | 313 | – |

Three lessons bind every milestone below:

1. **Compiled code is never slower than the interpreter at any single
   operation.**
   - Real code spends most of its time at boundaries: native calls,
     metamethods, index misses, allocation and coroutine switches.
   - Part 1 measured the kernels it targeted and let these boundaries become
     twice as slow as the interpreter's.
   - Milestone G makes the rule a check that runs before every commit.
2. **Leaving compiled code must cost what the operation costs, not what the
   stack depth costs.**
   - C1, as part 1 specified it (`jit-perf-fixes-plan.md` C1.5), did two things
     on every exit that was not a fast Lua call: it wrote out every compiled
     frame, and it discarded the native stack.
   - Milestone H removes leaving altogether.
3. **Whether an arm is hot or cold comes from feedback, not from the kind of
   code.**
   - A5 marked every helper and metamethod arm cold (`jit/build/emit.rs:1290`),
     including arms the feedback had already recorded as taken.
   - Milestone K fixes this.

---

## 2. Milestone G: the cost gate

### G1. The operation table

*Files:* new `tools/opcost.py`, cases in new `tools/opcost/*.lua`.

Each case is a loop around one operation. It runs at two sizes so that
start-up and compilation cancel out, the method of `tools/perop.sh`.

- **Reported:** instructions and cycles per operation, for the JIT and for
  `TCVM_JIT=off`.
- **For reference:** LuaJIT, with and without `-joff`.
- **Failure:** a row where the compiled operation costs more than the
  interpreted one fails the run.

Cases that involve a call run with 0, 4 and 8 extra live values. The cases:

- **Native calls:** `math.ult`, `string.format("%d", i)`, `table.insert(t, i)`,
  `select('#', ...)`, and `io.write` to `/dev/null`.
- **Lua calls:**
  - a known callee and an unknown callee;
  - arity short by one and long by one;
  - a vararg callee, a tail call, and an all-results call (`c = 0`);
  - a callee forced to the interpreter.
- **`pcall(f)`:** returning normally, and erroring.
- **`resume`/`yield` round trips,** with each side compiled or interpreted.
- **Field reads:**
  - an own field;
  - through one and through two `__index` tables;
  - an absent key through `__index`;
  - an `__index` function;
  - an integer key past the array part.
- **Metamethods:** `__add` on tables, `__eq`, `__lt`, `__len`, `__concat` and
  `__call`.
- **Allocation:**
  - `{}`, `{x = 1, y = 2}` and `{1, 2, 3}`;
  - a closure with one by-value and one by-reference upvalue;
  - a big integer.
- **Concatenation:** `..` of two strings, and of a string and a number.
- **A collector exit:** a loop that allocates enough to trigger collections.

### G2. A real-code corpus

LJR already carries most of Are We Fast Yet: bounce, cd, deltablue, havlak,
json, list, nbody, permute, queen, richard, storage and towers. Added beside
the suite:

- `test-files/particles_bench.lua`;
- rxi's `json.lua`, encoding and decoding a generated document;
- a Lua lexer and parser written in Lua, parsing a large file;
- a string-building workload: `string.format`, `..`, `table.concat`, and
  `gsub` with a function.

The pass bar is the JIT ahead of its own interpreter on every program, not
only on the geomean.

### G3. Every commit

This applies to any commit that changes `jit/`, the call protocol,
`vm/coro.rs`, `vm/native.rs` or the thread stacks.

- It runs G1, G2 and the regressions of section 1, in two interleaved rounds.
- Its message carries the deltas.
- A failed G1 row blocks the commit, unless the message says why the row is
  accepted.

*Exit criteria:* the table runs in under five minutes, and HEAD's failing rows
are exactly the boundaries section 1 names.

---

## 3. Milestone H: thread stacks

Goal: control never leaves compiled code to do something; it calls it.

- A native, an interpreted function, a metamethod or a collection runs on top
  of the compiled frames.
- A coroutine switch swaps stacks.
- The only writes to the Lua stack are a deopting frame's writes of its own
  values.

### H1. The mapping and the pool

*Files:* new `vm/stack.rs`, `env/thread.rs`, `lua/mod.rs` (`Lua`),
`lua/executor.rs` (`start`).

**One `mmap` per thread,** in this order:

1. a guard page;
2. the native stack, 1 MB, growing down;
3. the value stack, 512 KB, growing up;
4. a guard page.

The two stacks meet 8 KB into a page, so a thread using under 8 KB of each
touches a single 16 KB page. Pages are committed when they are first touched.

**512 KB covers the value stack's whole range:**

- `MAX_STACK` is 65,500 slots (`env/thread.rs:68`);
- natives may push `NATIVE_SLACK` slots past the limit (`:72`);
- frame indices already have to fit in 16 bits (`vm/frame.rs:178`).

**The pool.** `StackPool` lives in the runtime state, not the JIT's, because
the interpreter runs on these stacks with the JIT off too. It keeps a free
list.

- When a stack comes back, its pages past the meeting page are released
  (`madvise(MADV_FREE)`).
- The pool keeps at most 64 stacks, and unmaps any stack returned beyond that.

**When a thread gets its stack:**

- **The main thread:** at `Executor::start` (`lua/executor.rs`), which writes
  the seeded call into the value stack.
- **A coroutine:** at its first resume, in `resume_into`'s `!started` branch
  (`vm/coro.rs:29`). `coroutine.create` stores only the function.

**When a stack goes back to the pool:**

- the coroutine's body returns or errors out;
- `coroutine.close` closes it;
- the main thread finishes;
- the collector frees a suspended thread. `ThreadState` is `no_drop`
  (`env/thread.rs:14, 36`), so it gets a drop that hands the mapping back.

A stack is never freed while code runs on it. A dead coroutine's stack is
released by its resumer, after the switch back.

The stack size is a per-`Lua` setting, 1 MB by default.

### H2. The value stack does not move

*Files:* `env/thread.rs:294-350` (`ensure_slots`, `grow_slots`,
`ensure_frame_slots`), `vm/frame.rs:407` (`rebase`), `env/function.rs:314`
(`UpValue::rebase`), `vm/native.rs` (`native_call`, `native_enter`).

- **`ThreadState::stack` stops being a `Vec`.** It becomes the mapping's value
  region plus a committed length.
- **`grow_slots` stops moving the stack** (`env/thread.rs:307`). It keeps its
  nil fill and its limit check, and `frame::rebase` and `UpValue::rebase` go
  away.
- **Pointers survive natives.** `native_call` and `native_enter` re-read
  `ts.stack.as_mut_ptr()` after a native because "the native may have grown
  the stack"; they now keep their pointers.
- **`BASE` never goes stale.** Compiled code's `BASE` (`x22`), and the bases
  callers derive from it after a fast call (`jit/backend/aarch64/emit.rs:1689`),
  stay valid across anything. That includes a resume that writes arguments
  into a parked coroutine's stack (`resume_into`, `cs.ensure_slots`).

### H3. Stable addresses, and what a parked frame may hold

*Files:* `dmm/arena.rs:118-127, 219-228`, `lua/context.rs:27-33`.

**`State` must not move.**

- The arena keeps its root, `State`, inline (`dmm/arena.rs:125`), inside `Lua`
  (`lua/mod.rs:105`).
- So a host that moves its `Lua` also moves `State`. Every `RT` (`x23`) and
  `Context` saved on a parked stack would then dangle.
- The root moves into a `Box`. `Mutation` is boxed already (`:126`).

**What a parked frame may hold.** The Rust frames that can sit on a parked
stack are:

- the thread's start routine;
- dispatch's own frames (`run` and `call_handler` in `vm/dispatch.rs`);
- the handler that called the switch: `ff_resume`, `ff_yield`, a collector
  check, or an async wait.

Natives never park. They run to completion, or return an action (`NativeOut`).

While such a frame is parked, other code writes the objects it points at:

- **The resumer** writes the thread's `ThreadState`: arguments, `top`,
  `status`, `resumer` (`resume_into`, `vm/coro.rs:29`).
- **The collector** runs, and `ThreadState::trace` nil-fills through `&self`
  (`env/thread.rs:196`).

So three rules hold across a switch:

1. **No `&`/`&mut` to GC-managed memory.**
   - With a `&mut ThreadState` live across `stack_switch`, the compiler may keep
     `ts.top` or `ts.status` in a callee-saved register, which the switch
     restores. The handler would then continue with the values from before the
     park.
   - Even without a miscompile, a live `&mut` while the resumer or the
     collector writes the same object is undefined behaviour.
   - So handlers keep the raw `*mut ThreadState` they receive, end the
     reference before the switch, and take `thread!()` again after it.
2. **No `RefLock` borrow.** A borrow guard on a parked stack leaves the borrow
   flag set, and the next borrow from the resumer or the host panics.
3. **No object kept alive only by a Rust local.** The collector does not scan
   Rust frames, only compiled frames through their site maps (H11). Everything
   a parked Rust frame uses must be reachable from its thread.

None of these frames owns anything that needs dropping, so a parked stack can
be discarded without running them: by `coroutine.close`, or when its thread is
collected.

**Why continuing a parked frame is sound.** A parked frame's `'gc` lifetime
came from an earlier `mutate`, and continuing it in a later one is sound
because:

- the heap does not move;
- everything the frame refers to is either traced (H11) or reachable from its
  thread;
- `State` and `Mutation` keep their addresses.

This is the switch routine's safety contract, stated at the routine.

### H4. The host boundary

*Files:* `lua/executor.rs` (`step_inner`), `vm/dispatch.rs:291` (`enter`),
`vm/abi.rs:439`, `jit/mod.rs:134`, `vm/native.rs:395` (`gc_exit`), `:906`,
`vm/coro.rs:92` (`yield_from`), `jit/mod.rs:324` (`try_compile`), `load` in
`builtin/`, the parser.

**`Executor::step` switches into the current thread's stack** instead of
calling `dispatch::enter`.

- It gets back the reason the thread parked: `Gc`, `Pending`, `Yield` (the main
  thread yielded to the host) or `End`.
- `enter`'s body becomes the thread's start routine, which the first switch
  into a fresh stack runs.

**A collector check parks with `Gc`.** The checks are at `vm/abi.rs:439` in the
handlers, `jit/mod.rs:134` for compiled code, and `vm/native.rs:408` after a
native.

1. The host returns `Pending`.
2. `Lua::enter` collects on the way out (`lua/mod.rs:216`).
3. The next `step` switches back.
4. The check returns as if nothing happened.

Two pieces go away: `gc_exit`'s rewrite of the native header to `ok::RETURN`,
and `enter`'s `land_pending`/`repoll` paths.

**An async native waiting on the host** (`vm/native.rs:906`) parks with
`Pending`, and re-polls when the park returns.

**A main-thread yield with no resumer** (in `yield_from`) parks with `Yield`.
The host's `resume` values are in place when the park returns.

**Host-stack calls.** `on_host(f)` switches to the host stack, runs `f` there,
and switches back.

- Users: `load`'s parse and compile, `try_compile`, and ariadne's error
  rendering.
- The host is parked inside `Executor::step`, inside `Lua::enter`, so `f` runs
  in the same mutation context.

**The parser gets a nesting limit** like Lua's (`LUAI_MAXCCALLS`, 200 levels).
Today, deep enough source overflows whatever stack the parser runs on.

### H5. The switch routine and the coroutine operations

*Files:* new naked routines in `vm/stack.rs`, and in
`jit/backend/aarch64/abi.rs` for the compiled call sites; `vm/coro.rs` (all of
it); `vm/native.rs` (`cont::RESUME`, `cont::WRAP`); `env/thread.rs`.

1. **`stack_switch(save: *mut usize, load: usize, v: u64) -> u64`** is naked,
   about 25 instructions. It:
   - stores x19–x28, x29, x30 and d8–d15 under `sp`;
   - writes `sp` to `*save`;
   - loads `sp` from `load`;
   - restores the same registers from the new stack;
   - returns `v` in `x0`.

   A fresh stack is prepared so that the first switch into it returns into its
   start routine.
2. **`resume` from compiled code.** A call whose callee is guarded to be
   `coroutine.resume` or a `wrap` function (I1's native identity) becomes
   `bl jit_resume`, passing the coroutine, the argument window and the count.
   - **Fast path checks:** the coroutine is suspended and has a stack, and
     `resume_depth + 1 < MAX_RESUME_DEPTH` (`vm/coro.rs:17`).
   - **Fast path work:**
     - copies the arguments to where the coroutine waits;
     - sets both threads' `status`, `resumer` and `resume_depth`;
     - sets the running thread (`ctx.set_thread`);
     - switches.
   - **Slow path:** anything else (a first resume, a dead coroutine, a missing
     argument) takes a Rust path that ends in the same switch, or raises.
   - **On return:** the coroutine's values are in the caller's window, with
     `true` before them for `resume` and nothing before them for `wrap`. `x0`
     holds the count. The call lands like a native call (I1).
3. **`yield` from compiled code** is `bl jit_yield`, passing the window and the
   count.
   - **Fast path checks:** there is a resumer, and the thread may yield
     (`no_yield`, `main`).
   - **Fast path work:** copies the values into the resumer's window, sets the
     statuses, and switches to the resumer's parked stack.
   - **On return:** when the coroutine is resumed again, the switch returns
     with the resume arguments in its window and their count in `x0`.
4. **From the interpreter,** `ff_resume`, `ff_wrap` and `ff_yield`
   (`vm/coro.rs:239-333`) call the same routines. The handler calls the switch,
   and when the switch returns it lands the values through the `CALL`'s
   continuation.

   A resumer now always waits inside a switch call, compiled or not. So these
   reduce to the same routines:
   - `resume_switch!`, `waits_in_dispatch` and `deliver`;
   - `native_continue`, as used for coroutines;
   - the copying in `yield_from` and `end_into_resumer`.
5. **The end of a body** (`ret_coroutine_end`, `vm/coro.rs:337`):
   - the results go to the resumer's window;
   - the coroutine becomes dead;
   - it switches back for the last time, and the resumer releases its stack.
6. **An error the coroutine does not catch** (H8): the coroutine becomes dead
   with the error and switches back. `resume` returns `false, err`, and `wrap`
   raises the error in the resumer as today (`cont::WRAP`).
7. **`coroutine.close` of a suspended coroutine:**
   - runs its pending `__close` variables from its value stack, as today;
   - drops the parked native stack without running it;
   - releases the stack.
8. **Unchanged:** `status`, `running` and `isyieldable`; yields refused inside
   `__close` and message handlers (`no_yield`); and the 200-deep resume limit
   ("C stack overflow").

The interpreter runs on thread stacks on x86-64 too, so `stack_switch` and the
start routine need x86-64 versions. The compiled call sites are aarch64 only.

### H6. Lua on top of compiled frames

*Files:*
- `jit/backend/aarch64/abi.rs:119-210` (`exit_common`, `dispatch_entry`);
- `jit/state.rs:133-144` (`NativeBase`);
- `jit/backend/aarch64/emit.rs`: `fast_call` (`:587`), `known_call` (`:606`),
  the slow call path and `fast_miss` (`:453`), `CallMm`, `TailEnter`,
  `Dispatch`;
- `jit/build/emit.rs:1093-1130` (`mm_call`, `staged_call`);
- `vm/native.rs` (`invoke`, `native_act`);
- `vm/ops/call.rs:297` (`enter`).

1. **Levels and entry records.** A *level* is a run of dispatch started from
   compiled code, on the same native stack.
   - **The entry record.** Starting a level pushes an entry record onto the
     native stack. It holds:
     - the previous record;
     - `sp` and `fp`;
     - the caller's return address;
     - x19–x28 and d8–d15. Regions use these registers without saving them,
       and the code running in the level may enter regions;
     - the `NativeBase` fields of the level below (`jit/state.rs:133`);
     - the value-stack base of the frame it runs.
   - `dispatch_entry` (`abi.rs:203`) already records `NativeBase` for a fresh
     entry; a level saves and restores it.
   - The thread's current record is a field of `ThreadState`.
2. **Calling something that is not a compiled fast entry.** That means a
   function without a region, a fast entry whose checks fail, or a callee past
   the native budget.
   1. The caller writes the callee's header, with word 1 set to
      `ret_to_native` (a new continuation handler).
   2. It calls into a new level with `enter`'s arguments (`vm/ops/call.rs:297`).
   3. The callee runs in the interpreter. It can enter compiled regions, which
      fast-call further down the stack.
   4. The callee's `RETURN` reaches `ret_to_native`, which:
      - pops the record;
      - restores `sp`, `fp`, x19–x28, d8–d15 and `NativeBase`;
      - returns to the caller's fast resume point with the results, laid out
        as a fast callee leaves them (`x0` first, the count in `INSN`).

   The caller's frame was never left, so nothing is reloaded.
3. **Natives** are called directly (I1):
   1. The caller writes the native's header: word 0 is the `NativeHdr`, word 1
      the flags.
   2. It publishes `top_base` and `top`.
   3. It calls the native through a C-ABI shim on the current stack.
   - A plain return lands the results.
   - Any other outcome starts a level that runs `native_act` (`vm/native.rs`)
     on the native's frame, and ends through `ret_to_native` like a Lua callee.
     Such outcomes are a `NativeOut` asking to call Lua, a continuation, or a
     yield.
   - **The call's clobbers are the C ABI's** (`caller_saved()`, `abi.rs:73`),
     not every allocatable register. A Rust native preserves x19–x28 and
     d8–d15, and a level it starts saves them in its entry record (H6.1).
     - Values regalloc2 keeps in x19, x20, x21, x26–x28 and d8–d15 stay there
       across the call, at no cost.
     - Only values beyond those are spilled: a store before the call and a
       load after it.
4. **Metamethod calls** (`CallMm`, `staged_call`) are calls like any other. A
   compiled metamethod is fast-called; anything else goes through a level.
5. **Tail calls** from compiled code to a callee without a fast entry close the
   frame, then start a level whose `ret_to_native` returns to the tail-caller's
   caller.
6. **What goes away:**
   - the `bl leave_stub; ...; b enter` sequences in `fast_miss` and the slow
     call path, and the stub itself;
   - the reset of `sp` to `NATIVE_SP` on the way out (`abi.rs:171-175`);
   - the slow-resume entries and their reload stubs (part 1 C1.6). Nothing
     ever resumes a compiled frame from the interpreter now.

### H7. One frame walk

*Files:* `vm/frame.rs:320-400` (`Frame`, `frames`), `vm/unwind.rs:101, 279`,
`builtin/debug.rs`, `env/thread.rs:194`.

**The walkers today** follow header words 2 and 3:

- `frame::frames` (`vm/frame.rs:368`);
- the unwinder (`vm/unwind.rs:101`);
- `debug.traceback` and `getinfo`;
- error positions;
- the collector's trace (`env/thread.rs:194`).

A compiled frame's header has only word 0 written (part 1 D4), so they can't
see compiled frames. `frames` becomes the one walk everything uses:

1. It follows headers from the published top, as now.
2. At a header whose word 1 is `ret_to_native`, it reads that level's entry
   record and walks the compiled frames above it, starting from the record's
   `fp`. For each compiled frame:
   - its region comes from `[fp + 16]`;
   - its site comes from the return address (`Region::site`,
     `jit/region.rs:248`);
   - its base is derived as `jit_leave` derives it today (`jit/mod.rs:213`);
   - its pc comes from the site's `pc_after`;
   - the frames inlined into it come from the site's inline frames, innermost
     first.
3. The lowest compiled frame of a level was called by the interpreter, so its
   header is complete. The header walk continues from it.

`Region::site`'s binary search now runs only on this walk: for errors, debug
calls and collections, never for calls.

### H8. Errors

*Files:* `vm/unwind.rs`, `vm/native.rs:620-690` (`ret_pcall`, `ff_pcall`,
`ff_xpcall`).

**`unwind`** works in three steps:

1. It finds the catching frame through H7's walk: a protected native frame, or
   the coroutine's bottom.
2. It closes upvalues and runs `__close` handlers from the value stack, as now.
3. If the catching frame is in a lower level, it resets `sp`, `fp` and
   `NativeBase` from that level's entry record. It then continues in that
   level's dispatch with the catch continuation (`ret_pcall`, `ret_xpcall`).

The compiled frames above are dropped, along with their values.

**A `pcall` called from compiled code** is a native that calls Lua, so its
callee always runs in a level (H6.3). A caught error returns `false, err` to
the compiled caller through `ret_to_native`. A later fast path could land the
catch directly at the call site.

**No catch:**

- in a coroutine: see H5.6;
- in the main thread: park with `End` and the error, as `uncaught` does now.

**Rust panics.** A panic must not unwind into dispatch or compiled frames,
which have no unwind information. The native shim and the start routine abort
on a panic.

### H9. Deep recursion past the budget

*Files:* `jit/backend/aarch64/abi.rs:189-195` (`FAST_STACK`, `RAS_STACK`),
`emit.rs` (the fast entry's stack check, and the deep path's `NATIVE_LIMIT`
check from `8192ad5`).

- **The budget.** The native budget becomes the thread stack's size minus
  256 KB of headroom for natives. It is measured from the stack's end, not
  from dispatch's entry.
- **Refusing.** Three kinds of entry refuse below the budget: a fast entry, a
  known entry's deep path, and a region entered from the interpreter. The call
  then goes through a level (H6.2).
- **Why this works.** The interpreter's Lua-to-Lua calls take no native stack,
  so deep recursion continues interpreted on top of the compiled frames. It
  continues until `MAX_STACK` raises "stack overflow", as it does today.

### H10. Deopt writes one frame

*Files:* `jit/backend/aarch64/abi.rs:119-185` (`exit_common`), `jit/mod.rs:84`
(`jit_exit`), `jit/mod.rs:213` (`jit_leave`).

**An exit:**

1. writes its own frame from its snapshot, inline frames included;
2. closes the region frame;
3. continues in the interpreter:
   - If the frame was fast-called, its header gets word 1 = `ret_to_native`.
     The interpreter runs as a level on top of the caller's frame, which stays
     compiled.
   - If the frame was entered from dispatch, nothing changes.

`exit_common`'s call that walks the callers goes away, and so does
`jit_leave`'s loop. The snapshot decode (`entry_value`, `const_word`) stays,
for the one frame.

### H11. Tracing parked stacks

*Files:* `env/thread.rs:185-215` (`ThreadState::trace`), `jit/region.rs` (site
maps), `jit/backend/aarch64/abi.rs:93` (`image_index`).

**Why today's trace can't be reused.** `ThreadState::trace`
(`env/thread.rs:194`) walks headers, then nil-fills everything above the live
top. Both steps assume complete headers. For compiled frames it uses H7's
walk instead, read-only.

**What gets traced, per compiled frame:**

- **Word 0,** the closure.
- **The whole window,** `base .. base + max_stack`. Words 1–3 of headers inside
  the window are skipped, since they hold continuations and pointers.
- **Spill slots** that the site's map types as values.
- **Registers,** when the frame parked from a cold arm through the
  register-saving trampoline (a collector check, an async wait). The site's map
  then names registers too. They are read from the register image the
  trampoline stored (`image_index`, `abi.rs:93`).
- **Registers across a native call that started a level.** If that level parks
  (a collection or a yield inside the Lua the native called), the frame can
  have values in callee-saved registers. The native call site's map names
  them, and they are read from the level's entry record (H6.1).

**The live top** is the end of the highest compiled window; the nil fill starts
above it.

**Registers at a switch need no tracing:**

- **Pinned registers** are saved but never traced:
  - `CLOSURE` is word 0;
  - `THREAD` is the thread itself;
  - `RT` is the root;
  - `BASE` and `PC` point into the value stack and a prototype's code.
- **Allocatable registers** are dead at a switch. A call to the switch routine
  clobbers every allocatable register (`allocatable()`, `abi.rs:61`), as a
  call into another region does. So nothing live is in a register at a
  `resume` or `yield` site.

**Regions need nothing new.** Word 0 keeps the prototype alive, and
`JitState::regions` keeps every region compiled for it, retired ones included
(`jit/state.rs:322`).

**Minor collections.** A parked stack is written only by its own thread, so it
follows the value stack's rule: a thread that hasn't run since the last
collection needs no rescan.

### H12. What goes away

- `jit_leave`'s walk, and the leave stub;
- `Region::site` on hot paths;
- the slow-resume and reload stubs;
- header words 1–3 written by walks (now only a deopting frame and calls into a
  level write them);
- the `NATIVE_SP` resets;
- `gc_exit`, `land_pending` and `repoll`;
- `frame::rebase` and `UpValue::rebase`;
- dispatch's coroutine delivery machinery.

### Verification

**Protocol tests** in `src/vm/tests/protocol.rs`:

- `resume`/`yield` with each side compiled or interpreted, with values of every
  representation live across the switch;
- a yield inside a `pcall` inside a coroutine, from compiled code;
- an error in a coroutine, reaching `resume` and `wrap`;
- a collection while several coroutines are parked in compiled frames, with the
  collector forced on every allocation;
- `coroutine.close` of a parked coroutine with pending `__close` variables;
- recursion past the native budget: compiled, then interpreted, then compiled
  again;
- a deopt in a callee whose callers stay compiled;
- `debug.traceback` from a native called through three compiled frames, with an
  inlined frame between them;
- `os.exit` with parked coroutines;
- dropping `Lua` with parked stacks.

**Suite runs** with `TCVM_JIT_DEOPT_ALL=1` and `TCVM_JIT_CHECK=1`.

**Memory:** 100,000 suspended coroutines, with a resident size per coroutine of
at most 16 KB plus its value-stack use.

### Exit criteria

- **G1 native-call rows:** at most 120 instructions per call at 4 live values
  (HEAD 602, interpreter 313). No added cost per live value while the values
  fit in the callee-saved registers, and 2 per value beyond them.
- **Coroutines:** coroutine-ring at most 68 cycles per round trip (LuaJIT
  68.6). chameneos ahead of both the pre-plan JIT and the interpreter.
- **Others:** series, revcomp, particles and havlak ahead of both the pre-plan
  JIT and their interpreter.

---

## 4. Milestone I: calls

### I1. Natives called directly (replaces part 1 C2)

*Files:* `vm/native.rs` (`invoke` `:367`, `native_call` `:429`, `native_enter`
`:469`), `vm/ff.rs`, `jit/helpers.rs`, `jit/build/intrinsics.rs`,
`jit/build/emit.rs`, `jit/backend/aarch64/emit.rs`.

**Why compiled code can't call a native today.** A native is reachable only
through dispatch handlers: `native_call`, `native_enter` and the fast entries
of `vm/ff.rs`. A handler finishes by tail-calling the next handler, never by
returning, so compiled code cannot `bl` one and get control back.

**What already exists.** Every native has a plain Rust function
(`NativeKind::Plain`, `Cont` or `Async`), and `invoke` calls it on an argument
window.

**The plan:**

- **One shim for every native.** A C-ABI helper,
  `jit_call_native(rt, ts, nc, win, nargs) -> NativeOut`:
  1. writes the native's frame header and publishes `top_base` and `top`, as
     `native_call` does;
  2. calls `invoke`;
  3. returns its outcome to compiled code.

  This is the call H6.3 makes. A plain return lands the results from the
  window; any other outcome starts a level. No native needs an entry of its
  own.
- **Known native callees** (from D1's feedback or a constant, guarded by
  identity) call the shim directly, with no Lua-callee checks. Today a constant
  native callee still runs the nine-instruction Lua checks, and then leaves
  (section 1).
- **Unknown callees.** The fast call tests for a native before the Lua checks,
  and calls the shim.
- **Results.** A `c = 2` call (one result) takes its result from `x0`.
- **Intrinsics fall back to the shim.** `jit/build/intrinsics.rs` already
  compiles `select`, `type`, `rawget`, `rawset`, `rawequal`, `rawlen`,
  `tonumber`, `tostring`, `getmetatable`, `setmetatable`, `pairs`, `ipairs`,
  `assert`, `string.len`, `string.byte`, `string.sub` and the math functions
  inline or as helpers. A case their inline code does not cover exits today;
  it calls the shim instead.
- **New intrinsics** for builtins the corpus calls on hot paths that have none
  today: `table.insert(t, v)` (append), `table.remove(t)` (pop), `math.min`
  and `math.max`. More are added as G1 and profiles point at them.
  `string.format` and `io.write` go through the shim.
- **`pcall` and `xpcall`** are natives that call Lua, so they go through the
  shim and a level (H6.3, H8). Part 1 C2's dedicated entries for them, and for
  `select`, are not needed.

### I2. Callee feedback (part 1 D1)

Unchanged from part 1:

- a callee slot per call site, written by the interpreter, costing three
  instructions on a hit;
- measured with the JIT off before merging, with a 3% budget on
  recursive-fib-uv and fixpoint-fact.

It feeds I1, I4, J2, J3 and inlining (M1).

### I3. Arity mismatches (part 1 D2.2)

*Files:* `jit/backend/aarch64/emit.rs` (`known_call` `:606`, the fast entry),
`jit/build/emit.rs` (`call`), `jit/build/inline.rs`.

- **Known callee:**
  - The caller fills missing arguments with nil and calls the entry past the
    arity check (`KNOWN_ENTRY`).
  - Extra arguments stay where they are; a fixed-parameter callee never reads
    them.
  - `TailEnter`'s nil fill becomes the shared routine.
- **Unknown callee:**
  - When `nargs < nparams`, the fast entry's arity check branches to a fix-up
    in the callee instead of missing. The fix-up fills the missing slots with
    nil and continues at the known entry.
  - `nargs > nparams` passes.
- **mandel-metatable** declares `Complex.conj` as `(x, y)` and calls it as
  `Complex.conj(x)`. That call then never misses.

### I4. Typed entries and results in registers (rest of part 1 D2, D3)

As part 1 specifies them:

- D2.1's `checked` and `typed` entries;
- D2.3's direct `bl` for self-recursion;
- D3's float results in `d0`, and result types from the callee's region
  (`ret_ty`).

### I5. All-results calls

A call that takes all results (`c = 0`) lands them through the `jit_land`
helper (`jit/helpers.rs:507`). Compiled callees already return a count in
`INSN`. The call site copies up to 3 results inline, and uses the helper only
for more.

### I6. The deep-call threshold

*Files:* `jit/backend/aarch64/abi.rs:191-195` (`RAS_STACK`), `dispatch_entry`,
H6's levels.

**The problem.**

- `RAS_STACK = 512` bytes was sized for 16-byte frames, to stay under the
  return address stack's ~48 entries (the comment at `abi.rs:191`).
- C1 made frames 32 bytes or more, so the indirect path now starts at depth 16.
- recursive-fib-uv runs 18–23% more cycles at depths 24–40, and no more at
  depth 15.

**The fix:**

1. Measure the return address stack's depth on the M4, with a call-depth
   sweep comparing `blr`/`ret` against `br` with a tagged return address.
2. Set the threshold in bytes from that depth and the median frame size of the
   suite's regions.
3. Set `NATIVE_DEEP` per level, from the level's entry `sp`.
4. Re-measure `8192ad5`'s `NATIVE_LIMIT` check on the deep path (+8% cycles on
   recursive-fib-uv) once the deep path is rare.

*Exit:* recursive-fib-uv at depths 24, 30 and 40 at or below the pre-C1 cycles
(6.01, 5.91 and 6.58 G).

### Exit criteria

- G1 Lua-call rows at or below the interpreter's, and the arity rows no slower
  than the matching calls.
- mandel-metatable: no `fast_miss` and no `jit_land` in its profile.
- recursive-fib-uv and mandel-metatable ahead of the pre-plan JIT.

---

## 5. Milestone J: tables, metatables and metamethods

### J1. `__index` chains inline

*Files:* `jit/build/tables.rs`: `index` (`:82`), `nil_walks` (`:142`),
`emit_get` (`:173`), `proto_load` (`:386`), `holder_of` (`:423`), `get_chain`
(`:439`).

**What is inline today:** one level, a key found in the class's `__index` table
(`proto_load`, `holder_of`).

**Everything else calls the `Index` helper in a cold arm:**

- an absent key or a nil slot, on a table whose class has `__index`
  (`nil_walks`);
- a key held two levels up;
- a nil found in the holder.

**Who pays for it:**

- list's tail elements have no `next`, so every `x.next` on them walks
  `Element` through the helper;
- havlak's classes inherit two levels deep.

**The fix: resolve the walk at compile time** from the receiver's shape, as
`holder_of` does for one level.

- For each `__index` table on the way, the code has:
  - a `ClassIndex` guard that it is the same table;
  - a shape guard on it;
  - then either the load from its slot, or the next level.
- A chain that ends in a table without `__index` ends in the constant nil.
- Chains go up to 4 levels. A dict-mode table on the way keeps the helper.
- The guards are on classes and shapes, which GVN and LICM already hoist (part
  1 A6), so a loop pays them once.

Absent keys on a table without a metatable already fold to nil.

### J2. `__index` and `__newindex` functions

When the class's `__index` (or `__newindex`) is a function known at compile
time (the class pins it, and `ClassMm` guards it), the site calls it on the hot
path. It calls it as a known callee (I3, I4), or inlines it (M1).

### J3. Metamethod calls

This covers sites whose feedback recorded a metamethod: arithmetic,
comparison, `__len`, `__concat`, `__call` and `__eq`. On the hot path:

1. the receiver's class is guarded;
2. its metamethod is loaded and guarded by identity (`ClassMm`);
3. then comes a known call, or the inlined body (part 1 E1.5).

Examples: mandel-metatable's `__add`/`__mul`, and the vector classes of ray-
and nbody-style code.

### J4. Integer keys past the array part

binary-trees' leaves are `{item}` (array size 1), so `tree[2]` on them misses.
A site whose feedback has `ARR_MISS` (`jit/feedback.rs:24`) gets an inline miss
path:

- no hash part and no `__index`: nil;
- a hash part: the integer-key hash lookup, inline;
- otherwise: the helper.

### Exit criteria

- G1 field rows at or below the interpreter's, and no `jit_index` on the hot
  path. That covers one and two `__index` levels, absent through `__index`, and
  an `__index` function.
- list, havlak, binary-trees-num and mandel-metatable ahead of the pre-plan JIT.
  deltablue, json and richard no worse.

---

## 6. Milestone K: cold arms from feedback

*Files:* `jit/build/emit.rs:1153-1300` (`helper_arms`, `calls_mm`, `cold`),
`jit/build/tables.rs`, `jit/build/mod.rs` (`begin_cold`),
`jit/backend/aarch64/lower.rs` (`Helper` in cold blocks), `jit/feedback.rs`.

### K1. The rule

An arm is cold only when its site's feedback says the arm's case never
happened:

- no `ARR_MISS` for an array miss;
- no `MM` for a metamethod arm;
- no `BIGINT` for a boxing arm;
- never-ran for a block.

Otherwise the arm is warm:

- it is laid out in line;
- its helper is an ordinary call whose clobbers regalloc2 sees, so only values
  live across it are spilled;
- where J or L gives it an inline form, it gets that form instead.

### K2. The trampolines

Part 1 A5's save-all trampolines stay, for cold arms only. That is where their
cost is paid only when something unexpected happened.

### K3. An arm that turns hot

A cold arm in a compiled region counts its takings, with a counter per arm like
an exit's. At the exit threshold it asks for a recompile, in which its site's
feedback has the bit set. This extends A1's recording for exits to cold arms.

*Exit:* binary-trees-num, fixpoint-fact and list ahead of the pre-plan JIT, and
no G1 row runs a trampoline.

---

## 7. Milestone L: allocation

### L1. Inline allocation

*Files:* `dmm/heap.rs:210-219, 420-475` (each space's `cursor`/`limit`,
`alloc`, `alloc_slow`), `dmm/types.rs:121` (`GcBoxHeader`),
`env/table/mod.rs:49-95` (`from_template`, `alloc`), `jit/helpers.rs:195, 291,
500` (`jit_closure`, `jit_newtable`, `jit_box_i64`), `jit/layout.rs`.

**Why inline allocation is cheap.** The heap allocates by bumping a per-space
`cursor` toward `limit` (`dmm/heap.rs:439-441`). It charges the collector a
hole at a time, so the fast path is a compare and an add.

**Compiled code does it inline:**

1. Load the space's `cursor` and `limit`, through `RT`, at new `layout.rs`
   offsets.
2. Compare them, and bump `cursor`.
3. Store the header word: `GcBoxHeader`'s tagged vtable, with the birth flags.
4. Store the object's fields.
5. Past the limit, call the helper.

**What gets allocated inline:**

- **Tables** from a constructor template without spilled slots: the
  `TableState` fields and the template's inline slots, then nils
  (`Table::alloc`).
- **Closures** whose upvalues are all by value or the enclosing closure's own.
  A closure that opens a new upvalue keeps the helper.
- **Big integers** (`jit_box_i64`): a one-word leaf object.

**Collector checks** stay where part 1 put them; only the helper's hole refill
can raise the debt.

### L2. `SETLIST`

A constructor with a known count stores into the array part inline.
`jit_setlist` (`jit/helpers.rs:304`) stays for constructors that use `top`.

### L3. Allocation sinking (part 1 E5)

After L1 and M1, as part 1 specifies it.

*Exit:*

- G1 allocation rows at or below the interpreter's.
- fixpoint-fact's closure creation under 15% of its profile (55% now).
- mandel-metatable's allocation share halved.

---

## 8. Milestone M: the rest of part 1

### M1. Inlining leftovers (part 1 E1)

- Calls inside inlined bodies (part 1 E1.7). With H7 such a call needs no
  header writes, only the site's inline frames.
- Recursion unrolled two levels.
- Metamethod callees (E1.5), through J3.
- Operations that `needs_frame` excludes today (`jit/build/inline.rs:152`),
  where L1 or J makes them frame-free: `CLOSURE`, `SETLIST`, `CONCAT`,
  `SETUPVAL`, and the `TFOR*` loops.

### M2. Shape exits that learn one site per recompile (part 1 A9, deltablue)

Settling records the shapes of every inline cache the exiting region guards,
not only the one that exited.

### M3. Smaller leftovers

- Jump threading of constant block parameters (part 1 E6).
- Float `%`'s inline quotient fix-up, and negative divisors (E7).

### M4. `CONCAT`

A two- or three-operand concatenation of strings and numbers calls a helper
with an ordinary call's clobbers (the C ABI's), instead of `helper_or_mm`'s full
set. With L1 the result string is allocated inline.

### M5. Rematerialization (part 1 E4) stays deferred

regalloc2 has no rematerialization, and per-block materialization added
instructions in loops when it was tried. Revisit it if G1 or the spill counts
point at it.

### M6. Part 1 B5 is dropped

It would flap.

---

## 9. Testing

Part 1's matrix (its section 9) stays, with these additions:

| Check | Command | What it proves |
|---|---|---|
| G1 operation table | `tools/opcost.py` | no operation is slower compiled than interpreted |
| G2 real-code corpus | `tools/ljr.py --suite` on the corpus | the JIT is ahead of its interpreter on every program |
| thread stacks | `src/vm/tests/protocol.rs` (H's verification), the official suite's coroutine and error files | switches, levels, unwinding and tracing |
| collector stress | `TCVM_GC_STRESS=1` (new: a collection at every hole refill) with `TCVM_JIT_HOT_CALL=1 TCVM_JIT_HOT_LOOP=1` | parked and running compiled frames are traced correctly |

---

## 10. Order, dependencies and targets

| Step | Items | Depends on | Measured by | Target |
|---|---|---|---|---|
| 1 | G1–G3 | – | – | the table reproduces section 1 |
| 2 | K1–K3, I6 | G | binary-trees-num, fixpoint-fact, list, recursive-fib-uv | ahead of the pre-plan JIT |
| 3 | H1–H4 | G | G1 collector rows, the suite under stress | no regression; collector exits write no frames |
| 4 | H5 | H1–H4 | coroutine-ring, chameneos | ring ≤ 68 cycles per round trip |
| 5 | H6–H12, I1 | H1–H5 | G1 native rows, series, revcomp, particles, havlak | ≤ 120 instructions per native call at 4 live values |
| 6 | I2–I5 | H6 | fib, mandel-metatable, spectral | part 1's D targets (fib ≤ 24 instructions per call) |
| 7 | J1–J4 | I2 for J2/J3 | list, havlak, deltablue, json, mandel-metatable | field rows ≤ interpreter |
| 8 | L1, L2 | – | fixpoint-fact, binary-trees-num, particles | allocation rows ≤ interpreter |
| 9 | M1, L3 | I, J, L1 | mandel-metatable, ray, richard | part 1's E targets |
| 10 | M2–M4 | – | deltablue, the string corpus | – |

Step 2 can go before H because it touches neither the call protocol nor the
stacks. H is the largest milestone, and it rewrites the call paths that I1, J2
and J3 build on.

---

## 11. Open points

1. **Moving `Lua` (H3).** Boxing the arena root costs nothing on the hot path.
   The alternative is a pinned `Lua` API.
2. **JIT compilation on the host stack (H4).** Measure regalloc2's stack use.
   If it is small and bounded, compile on the thread stack and save two
   switches per compile.
3. **Suspended coroutines** cost at least 16 KB each (decision 5).
4. **Stack size and headroom (H1, H9).** The 1 MB size and the 256 KB headroom
   are first guesses. G1's deep rows and the official suite's recursion tests
   should set them.
5. **x86-64** needs its own switch routine and start routine (H5). The JIT
   stays aarch64-only.
6. **`debug.sethook` and `debug.setlocal`** stay unimplemented (part 1's scope).
   H7 makes `debug.getlocal` and `getinfo` work on compiled frames, read-only.
7. **`pcall` from compiled code** goes through a level (H8). A direct catch
   site is a later step, if G1's `pcall` rows need it.
