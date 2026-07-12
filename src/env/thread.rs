use crate::dmm::{Collect, Gc, Mutation, Ref, RefLock, RefMut, Trace};
use crate::env::error::Error;
use crate::env::function::{Function, LuaClosure, Upvalue};
use crate::env::value::Value;
use crate::vm::interp::Continuation;
use crate::vm::sequence::{BoxSequence, CallbackAction};

/// Copy wrapper stored in Value.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Thread<'gc>(Gc<'gc, RefLock<ThreadState<'gc>>>);

/// Authoritative thread state. Read by the executor's driver loop and by
/// `coroutine.status` / `coroutine.isyieldable` to decide control flow.
///
/// "Running" is no longer represented here — a thread is running iff its
/// `RefLock` is currently mutably borrowed (the executor holds the borrow
/// for the duration of `Executor::step`). This avoids a redundant flag
/// drifting from reality.
#[derive(Clone, Copy, PartialEq, Eq, Collect)]
#[collect(internal, require_static)]
pub enum ThreadStatus {
    /// Newly created or freshly cleared; no seeded call. Not resumable.
    Stopped,
    /// Suspended (yielded, or freshly created via `coroutine.create`).
    /// Resumable.
    Suspended,
    /// Currently somewhere on the executor's thread stack but not the top
    /// — i.e. a coroutine that resumed another coroutine.
    Normal,
    /// Has finished. Stack values `[bottom..]` are the return values, ready
    /// for `take_result` or for the resumer's `WaitThread` to consume.
    Result { bottom: usize },
}

/// A Lua bytecode frame on a thread's frame stack. Only Lua function
/// execution pushes one of these; native callbacks that don't suspend run
/// inline within the calling Lua frame.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct LuaFrame<'gc> {
    pub closure: Gc<'gc, LuaClosure<'gc>>,
    pub base: usize,
    pub pc: usize,
    pub num_results: u8,
    /// Caller-supplied args beyond `num_params`; the below-base region is
    /// `stack[base - num_extras .. base]`. Set by `VARARGPREP`, else 0.
    pub num_extras: u32,
    /// Fixup dispatched by `op_return` when this frame unwinds. `None` for
    /// normal calls; set by metamethod/iterator helpers that need
    /// post-return processing.
    #[collect(require_static)]
    pub continuation: Option<Continuation>,
}

/// Stack-window metadata threaded through every suspension point.
///
/// - `bottom` is the callback's `args_base` — `stack[bottom..]` is the
///   active window (yielded values, sequence args, etc.).
/// - `func_idx == bottom - 1` (typically) is where the original Lua CALL
///   expects results to land.
/// - `returns` is the CALL instruction's `returns` field (0 = "all").
///
/// Stored on `Frame::Sequence`, `Frame::WaitThread`, `PendingAction`,
/// and as the yielded-state stash on `ThreadState`.
#[derive(Clone, Copy, Debug)]
pub struct CallSite {
    pub bottom: usize,
    pub func_idx: usize,
    pub returns: u8,
    /// Set only when this call site backs a continuation-driven native
    /// metamethod/iterator (`schedule_meta_call`) that suspended. On result
    /// delivery the executor applies this `Continuation`'s payload against the
    /// caller frame (`apply_native_continuation`) instead of the plain
    /// `func_idx`/`returns` landing — this is how a suspended native target
    /// replays `StoreResult`/`CondJump`/`TForCall`/`IgnoreResult`, since it
    /// has no Lua frame to park the continuation on. `None` for ordinary
    /// calls, where `func_idx`/`returns` drive the landing.
    pub cont: Option<Continuation>,
}

/// A frame on a thread's frame stack. The interpreter only pushes
/// `Frame::Lua`; the executor driver pushes the others when a callback
/// suspends, an error unwinds, a coroutine waits, etc.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub enum Frame<'gc> {
    /// Running Lua bytecode.
    Lua(LuaFrame<'gc>),
    /// A pinned multi-step native callback awaiting (re-)poll. The
    /// `call_site` mirrors the original Lua CALL so terminal
    /// `SequencePoll::Return` lands results in the right place.
    Sequence {
        seq: BoxSequence<'gc>,
        #[collect(require_static)]
        call_site: CallSite,
        pending_error: Option<Error<'gc>>,
    },
    /// A coroutine that hasn't been resumed yet. Replaced on first resume by
    /// a real call frame.
    Start(Function<'gc>),
    /// Current thread is waiting on an inner thread it resumed; on the
    /// inner thread reaching a terminal/yielded state, the executor pops
    /// this frame and lands the inner thread's values into the original
    /// CALL's expected slot.
    WaitThread {
        #[collect(require_static)]
        call_site: CallSite,
    },
    /// Unwinding marker. The driver pops Lua/Wait frames (closing upvalues)
    /// until a `Sequence` frame is found and stamped with `pending_error`,
    /// or until the thread terminates with the error.
    Error(Error<'gc>),
}

/// The mutable state of a thread/coroutine.
///
/// # The stack invariant
///
/// `stack` is *physical storage*; `top` is the *logical* stack top. They are
/// not the same thing and `Vec::len()` must never be read as the logical end.
/// Three rules, enforced by the `stack_*` accessors below — go through them
/// rather than touching `stack`/`top` directly:
///
///  1. `top <= stack.len()` at all times.
///  2. The vec is **grown, never shrunk**, while any frame is live. A native
///     callback runs on the *same* vec as its caller, so a shrink would pull
///     the backing store out from under an outer frame's register window (and
///     the interpreter's raw `registers` pointer). The only sanctioned shrink
///     is [`ThreadState::discard_above`], whose caller must guarantee nothing
///     above the cut is live.
///  3. Because of (2), removing values means **lowering `top` and nil-filling
///     what it vacates** — the nil-fill is what actually releases them, since
///     the slots stay physically present and everything below `live_top` is
///     traced (see the `Collect` impl).
///
/// The *live region* is `[0 .. max(top, top_lua.base + max_stack_size))`: a
/// Lua frame's register window is live regardless of `top` (the interpreter
/// addresses registers off `base`, not `top`), while `top` covers the
/// value-passing window — args and results in flight between frames — which
/// can sit above the frames during a native call.
pub struct ThreadState<'gc> {
    /// Backing store. May hold dead slots above the live region; they are
    /// kept nil so the `Collect` impl below cannot retain them.
    pub stack: Vec<Value<'gc>>,
    pub frames: Vec<Frame<'gc>>,
    pub open_upvalues: Vec<Upvalue<'gc>>,
    pub tbc_slots: Vec<usize>,
    pub status: ThreadStatus,
    /// Logical stack top — the end of the value-passing window. Always valid:
    /// it is the sole signal of "how many values are here" across every
    /// hand-off (native args/results, yields, sequence polls, thread resume,
    /// `Result`). Multires producers (`VARARG`/`CALL`/`TAILCALL` with the `0`
    /// sentinel, native multi-return) publish through it; their consumers read
    /// it back.
    pub top: usize,
    /// Back-reference to the owning Thread handle, needed for creating open upvalues.
    pub thread_handle: Option<Thread<'gc>>,
    /// A native callback's non-`Return` `CallbackAction` deposited by
    /// `op_call`/`op_tailcall` and consumed by the executor driver loop on
    /// the next pump. `None` between pumps. The interpreter never observes
    /// this (it bails out via `return Ok(())` immediately after setting it).
    pub pending_action: Option<PendingAction<'gc>>,
    /// Where the thread's yielded values currently live. Set when the
    /// thread suspends via a `Yield` action (or a sequence's
    /// `SequencePoll::Yield`/`TailYield`). Consumed on resume to recover
    /// where the call's results should land. `None` outside of yielded
    /// state.
    pub yield_bottom: Option<CallSite>,
}

// SAFETY: traces every field that can own a `Gc` pointer. The only subtlety is
// `stack` (issue #43): it is grown-not-shrunk, so slots above the live region
// are dead scratch and must NOT be traced — tracing the whole vec would retain
// whatever a since-returned callee happened to leave in its registers.
// `live_top` is the sound live high-water:
//   * Lua registers live in `[base, base + max_stack)`; callee bases strictly
//     increase up the stack, so the topmost Lua frame's `base + max_stack`
//     bounds every frame's live registers (lower frames' live regs sit below
//     their callee's base, which is <= the top frame's base). Varargs/extras
//     sit below a base, so `[0..live_top]` covers them too.
//   * `top` adds the value-passing window that can sit above the Lua frames
//     (native args/results, a paused multires window). It is `max`ed with the
//     frame floor, so it can only raise the bound, never lower it, and thus
//     can never cause an under-trace.
// Correctness therefore rests on rule (3) of the stack invariant: anything
// dropped from the logical stack is nil-filled, so a dead slot that happens to
// fall below `live_top` still retains nothing.
// `tbc_slots`/`status`/`top`/`yield_bottom` hold no `Gc` pointers.
unsafe impl<'gc> Collect<'gc> for ThreadState<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        let live_top = self
            .frames
            .iter()
            .filter_map(|f| match f {
                Frame::Lua(lf) => Some(lf.base + lf.closure.proto.max_stack_size as usize),
                _ => None,
            })
            .max()
            .unwrap_or(0)
            .max(self.top)
            .min(self.stack.len());
        cc.trace(&self.stack[..live_top]);
        cc.trace(&self.frames);
        cc.trace(&self.open_upvalues);
        cc.trace(&self.thread_handle);
        cc.trace(&self.pending_action);
    }
}

/// A native callback wants to suspend / call / yield / resume; the executor
/// driver translates this into frame-stack operations on the next pump.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct PendingAction<'gc> {
    pub action: CallbackAction<'gc>,
    #[collect(require_static)]
    pub call_site: CallSite,
}

impl<'gc> ThreadState<'gc> {
    /// View the top frame as a Lua frame. Returns `None` if the stack is
    /// empty *or* the top is non-Lua (Sequence/Start/WaitThread/Error).
    /// Most interpreter sites can `.unwrap()` this — the dispatch loop only
    /// runs when a Lua frame is on top — but the executor driver loop must
    /// match all variants.
    #[inline]
    pub fn top_lua(&self) -> Option<&LuaFrame<'gc>> {
        match self.frames.last()? {
            Frame::Lua(lf) => Some(lf),
            _ => None,
        }
    }

    #[inline]
    pub fn top_lua_mut(&mut self) -> Option<&mut LuaFrame<'gc>> {
        match self.frames.last_mut()? {
            Frame::Lua(lf) => Some(lf),
            _ => None,
        }
    }

    /// Hot-path accessor used by interpreter handlers that have already
    /// guaranteed (statically) that the top frame is Lua. UB in release if
    /// it isn't; debug builds panic.
    ///
    /// # Safety
    /// Caller must ensure `self.frames.last()` is `Some(Frame::Lua(_))`.
    #[inline]
    pub unsafe fn top_lua_unchecked(&self) -> &LuaFrame<'gc> {
        match unsafe { self.frames.last().unwrap_unchecked() } {
            Frame::Lua(lf) => lf,
            _ => {
                debug_assert!(false, "top_lua_unchecked: non-Lua frame on top");
                unsafe { std::hint::unreachable_unchecked() }
            }
        }
    }

    /// # Safety
    /// Caller must ensure `self.frames.last_mut()` is `Some(Frame::Lua(_))`.
    #[inline]
    pub unsafe fn top_lua_unchecked_mut(&mut self) -> &mut LuaFrame<'gc> {
        match unsafe { self.frames.last_mut().unwrap_unchecked() } {
            Frame::Lua(lf) => lf,
            _ => {
                debug_assert!(false, "top_lua_unchecked_mut: non-Lua frame on top");
                unsafe { std::hint::unreachable_unchecked() }
            }
        }
    }

    /// Push a new Lua frame.
    #[inline]
    pub fn push_lua(&mut self, lf: LuaFrame<'gc>) {
        self.frames.push(Frame::Lua(lf));
    }

    // --- Stack accessors -------------------------------------------------
    //
    // These enforce the stack invariant documented on `ThreadState`. Prefer
    // them over touching `stack` / `top` directly; the only code that should
    // index `stack` raw is register access off a frame's `base`.

    /// Make `stack[..n]` physically addressable. Grow-only, per rule (2).
    #[inline]
    pub fn ensure_slots(&mut self, n: usize) {
        if self.stack.len() < n {
            self.stack.resize(n, Value::nil());
        }
    }

    /// The logical window `stack[bottom..top]`.
    #[inline]
    pub fn window(&self, bottom: usize) -> &[Value<'gc>] {
        &self.stack[bottom..self.top]
    }

    /// Publish a new logical top, nil-filling any slots it vacates (rule 3).
    /// Raising the top only exposes slots the caller has already written.
    #[inline]
    pub fn set_top(&mut self, n: usize) {
        self.ensure_slots(n);
        if n < self.top {
            self.stack[n..self.top].fill(Value::nil());
        }
        self.top = n;
    }

    /// Replace the window at `bottom` with `values` and publish the new top.
    pub fn set_window<I>(&mut self, bottom: usize, values: I)
    where
        I: IntoIterator<Item = Value<'gc>>,
        I::IntoIter: ExactSizeIterator,
    {
        let values = values.into_iter();
        let end = bottom + values.len();
        self.ensure_slots(end);
        for (i, v) in values.enumerate() {
            self.stack[bottom + i] = v;
        }
        self.set_top(end);
    }

    /// Move the window at `bottom` out, leaving `top == bottom`.
    pub fn take_window(&mut self, bottom: usize) -> Vec<Value<'gc>> {
        let values = self.window(bottom).to_vec();
        self.set_top(bottom);
        values
    }

    /// Insert `v` at `at`, shifting `stack[at..top]` up one slot. The call
    /// convention wants the function immediately below its args, but a
    /// suspended callback leaves only args behind — this splices the function
    /// back in.
    pub fn insert_at(&mut self, at: usize, v: Value<'gc>) {
        debug_assert!(at <= self.top);
        self.ensure_slots(self.top + 1);
        self.stack.copy_within(at..self.top, at + 1);
        self.stack[at] = v;
        self.top += 1;
    }

    /// Cut the stack down to exactly `n` slots, releasing the storage above it.
    /// The **only** sanctioned shrink (rule 2): the caller must guarantee no
    /// live frame window and no in-flight native call sits above `n` — i.e. a
    /// thread being seeded, unwound, or terminated, never one with a native
    /// callback on the stack.
    pub fn discard_above(&mut self, n: usize) {
        debug_assert!(n <= self.stack.len());
        self.stack.truncate(n);
        self.top = n;
    }
}

impl<'gc> Thread<'gc> {
    pub fn new(mc: &Mutation<'gc>) -> Self {
        let state = ThreadState {
            stack: Vec::new(),
            frames: Vec::new(),
            open_upvalues: Vec::new(),
            tbc_slots: Vec::new(),
            status: ThreadStatus::Stopped,
            top: 0,
            thread_handle: None,
            pending_action: None,
            yield_bottom: None,
        };
        let thread = Thread(Gc::new(mc, RefLock::new(state)));
        // Store the back-reference
        thread.borrow_mut(mc).thread_handle = Some(thread);
        thread
    }

    pub fn borrow(self) -> Ref<'gc, ThreadState<'gc>> {
        self.0.borrow()
    }

    pub fn borrow_mut(self, mc: &Mutation<'gc>) -> RefMut<'gc, ThreadState<'gc>> {
        self.0.borrow_mut(mc)
    }

    pub fn status(self) -> ThreadStatus {
        self.0.borrow().status
    }

    /// Pointer equality between two thread handles.
    pub fn ptr_eq(self, other: Thread<'gc>) -> bool {
        Gc::ptr_eq(self.0, other.0)
    }

    pub fn inner(self) -> Gc<'gc, RefLock<ThreadState<'gc>>> {
        self.0
    }

    pub(crate) fn from_inner(g: Gc<'gc, RefLock<ThreadState<'gc>>>) -> Self {
        Thread(g)
    }
}
