//! Tracing, optimizing JIT compiler for TCVM.
//!
//! A hot region is lowered from Lua bytecode into an SSA IR, optimized, and
//! emitted as native code (aarch64 / x86-64). This module doc records the
//! design decisions and — more usefully — the properties of the surrounding
//! runtime that force them. See `docs/jit-ir.md` for the long form.
//!
//! # Decisions
//!
//! - **CFG SSA with basic-block versioning**, not a linear trace. Blocks are
//!   specialized on their entry type context, so a type check is emitted once
//!   and every downstream block is compiled knowing the answer.
//! - **A region is one `Prototype`.** Calls stay real calls. `FrameState`
//!   carries a parent chain and call sites are marked inlinable, so adding
//!   inlining later is a frontend pass, not an IR rewrite.
//! - **One optimizing IR, lowered in place** (`lua.getindex` -> `guard.shape`
//!   + `slot.get`), then a dumb machine IR for regalloc and encoding.
//! - **Watchpoints, not guards, for metamethod presence.** Compiled code
//!   declares dependencies; metatable mutation invalidates dependents.
//! - **Guard failure deopts to the interpreter.** Exit descriptors carry
//!   `(pc, framestate, entry context)`, so lazy compile-on-exit stubs are an
//!   additive change if we want them.
//!
//! Because exits deopt, block versioning is *eager*: versions are discovered at
//! compile time by propagating the entry context through the CFG, not minted at
//! runtime on guard failure. A genuinely polymorphic site deopts and
//! re-profiles rather than converging on two versions. Exit counters plus
//! recompile-with-widened-context are the mitigation.
//!
//! Also because exits deopt, "refuse to compile this" is always a legal answer.
//! v1 aborts on MULTRET, varargs functions, and TBC slots.
//!
//! # What the runtime forces on the IR
//!
//! Each of these shows up as a concrete IR feature; none is incidental.
//!
//! **`Value` is a 16-byte tagged pair, not NaN-boxed, and Integer/Float are
//! distinct Lua types.** Type tests are byte compares, but a raw `i64` in a
//! register needs an explicit `pack`/`unpack` to become a `Value`, and
//! arithmetic bifurcates into `i64` and `f64` families that specialization has
//! to resolve. Nothing in this IR *boxes* in the allocating sense — Lua numbers
//! never heap-allocate — which is why `pack.int` is pure and why the ops are
//! not called `box`/`unbox`.
//!
//! **Shape guards are self-invalidating, but prove less than they look like
//! they do.** A new string key transitions the table to a fresh `Shape`, and
//! deletion migrates it to dict mode (also a fresh shape), so a shape pointer
//! compare proves the property slot layout. Two traps:
//!   - Never specialize on a dict-mode shape: the dict sentinel is *shared* by
//!     every dict table with the same `MtCache`, so a match there proves
//!     nothing about layout.
//!   - A shape guard does **not** prove metamethod absence. `MtCacheData::bits`
//!     is a `Cell` mutated in place by `maybe_update_mt_bit`, so `mt.__index = f`
//!     changes the meaning of `t.x` without changing any shape pointer. Hence
//!     `assume.no_mm` and the watchpoint machinery.
//!
//! **The GC is incremental mark-and-sweep and non-moving, with explicit
//! barriers.** Non-moving means a derived interior pointer (a `properties` Vec
//! data pointer, say) stays valid while its owning table is reachable, so only
//! the *base* object needs rooting. But `ThreadState::trace` traces
//! `stack[..live_top]`, so a live boxed value held only in a machine register
//! across a GC point is invisible to the collector — see the rooting rule on
//! `Effects`. Stores of `Gc` pointers from compiled code must emit barriers, so
//! barriers are explicit IR nodes.
//!
//! **Open upvalues address stack slots by index** (`UpvalueState::Open`), so a
//! local captured by a `CLOSURE` cannot live purely in SSA. The frontend
//! pre-scans for `ParentLocal` upvalue descriptors and pins those registers to
//! memory.
//!
//! **Compiled code runs on the native stack, and a Lua callee can yield.** We
//! cannot suspend a native frame, so calls return a status: on `Suspended` the
//! FrameState is written back as a real `LuaFrame` with `pc` past the `CALL`
//! and the region returns, letting the executor resume us *in the interpreter*.
//! This is nearly free — `op_call` already leaves the caller's pc past the
//! `CALL` and `op_return` lands results at `func_idx`.
//!
//! # Type feedback
//!
//! Compilation is triggered from a *running* frame (a hot back-edge, via OSR, or
//! a hot call), so the concrete types of every live register are on the stack at
//! compile time. That is the seed context; versioning propagates it.
//!
//! For the heap, the per-site inline caches supply two things:
//!   - `Prototype::ic_table` — the receiver's `Shape`, which turns a field read
//!     into a constant-offset `slot.get`.
//!   - `Prototype::ic_types` — the value kinds the site has actually *loaded*.
//!
//! The second is not optional. A shape proves *where* a field lives, never
//! *what* it holds, so without it every field read yields `any` and every
//! arithmetic op consuming one stays a metamethod-capable call — the load gets
//! fast and the math stays slow. With it, lowering emits a guarded load
//! (`slot.get` + `guard.type`) and the arithmetic specializes.
//!
//! This is the same shape as LuaJIT's guarded load — `lj_record_idx` types an
//! `HLOAD` from the value it just observed (`IRType t = itype2irt(oldv)`) and
//! tags it `IRTG` — with one difference forced by us *not* being a trace
//! recorder: we never execute the ops we compile, so there is no observed value
//! to read a type from. The IC is our substitute for the recorder, and a better
//! evidence base than a single recorded instant, since it accumulates across
//! every execution of the site. (V8 solves the same problem in the other layer,
//! by tracking a field's representation in the Map itself, which costs map
//! deprecation and object migration; we don't.)

pub mod backend;
pub mod frontend;
pub mod ir;
pub mod region;
#[cfg(test)]
mod runtime_tests;
