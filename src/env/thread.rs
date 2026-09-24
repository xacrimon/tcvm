use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};

use crate::dmm::{Collect, Gc, Mutation, Ref, RefLock, RefMut, Trace};
use crate::env::error::Error;
use crate::env::function::{Function, LuaFn, Upvalue};
use crate::env::value::Value;
use crate::lua::Context;
use crate::vm::interp::Continuation;
use crate::vm::sequence::{BoxSequence, Suspend};

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
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct LuaFrame<'gc> {
    pub(crate) closure: LuaFn<'gc>,
    /// Read through `base()`. 32 bits: a frame sits inside the stack, and
    /// every growth of the stack goes through `grow_slots`, which keeps it
    /// below 2^32 slots.
    pub(crate) base: u32,
    /// Resume address: points *past* the instruction being executed, into
    /// `closure.proto.code` (which the frame keeps alive). A raw pointer
    /// rather than an index so CALL saves it with one store.
    #[collect(require_static)]
    pub(crate) pc: *const crate::instruction::Instruction,
    pub(crate) num_results: u8,
    /// `frame_flags` bits. Zero means RETURN can take its fast path: fixed
    /// results land in a Lua caller with nothing to close and no continuation.
    /// Mostly left set once the hazard is gone (a stale bit just costs the slow
    /// path); a slow TAILCALL clears `OPEN_UPVALUES` after closing them.
    pub(crate) flags: u8,
    /// Caller-supplied args beyond `num_params`; the below-base region is
    /// `stack[base - num_extras .. base]`. Set by `VARARGPREP`, else 0.
    pub(crate) num_extras: u32,
    /// Fixup dispatched by `op_return` when this frame unwinds. `None` for
    /// normal calls; set by metamethod/iterator helpers that need
    /// post-return processing.
    #[collect(require_static)]
    pub(crate) continuation: Option<Continuation>,
}

/// Stack-window metadata threaded through every suspension point.
///
/// - `bottom` is the callback's `args_base` — `stack[bottom..]` is the
///   active window (yielded values, sequence args, etc.).
/// - `func_idx == bottom - 1` (typically) is where the original Lua CALL
///   expects results to land.
/// - `returns` is the CALL instruction's `returns` field (0 = "all").
///
/// Stored on `ExecKind::Sequence`, `ExecKind::WaitThread`, `PendingAction`,
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

/// Bits of `LuaFrame::flags`.
pub mod frame_flags {
    /// `continuation` is `Some`: RETURN must apply it (`op_return_cont` or
    /// `cont_resume`).
    pub const HAS_CONT: u8 = 1;
    /// A CLOSURE in this frame captured one of its locals; RETURN must close.
    pub const OPEN_UPVALUES: u8 = 2;
    /// A TBC in this frame registered a to-be-closed slot.
    pub const TBC: u8 = 4;
    /// The frame below is not a Lua frame, or there is none.
    pub const PARENT_NON_LUA: u8 = 8;
}

// Two records per cache line.
const _: () = assert!(std::mem::size_of::<LuaFrame<'static>>() == 32);

impl<'gc> LuaFrame<'gc> {
    #[inline(always)]
    pub fn base(&self) -> usize {
        self.base as usize
    }

    #[inline(always)]
    pub fn set_base(&mut self, base: usize) {
        debug_assert!(base <= u32::MAX as usize);
        self.base = base as u32;
    }

    /// `pc` as an index into `closure.proto.code`.
    pub fn pc_index(&self) -> usize {
        unsafe {
            self.pc
                .offset_from_unsigned(self.closure.proto.code.as_ptr())
        }
    }
}

/// A frame the executor pushes when a callback suspends, an error unwinds, a
/// coroutine waits, etc. It sits above the first `depth` Lua frames and
/// below the rest.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct ExecFrame<'gc> {
    pub depth: usize,
    pub kind: ExecKind<'gc>,
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub enum ExecKind<'gc> {
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
    /// Unwinding marker. The driver pops Lua/Wait/pass-through frames
    /// (closing upvalues) until a catching `Sequence` frame is found and
    /// stamped with `pending_error`, or until the thread terminates with
    /// the error.
    Error(Error<'gc>),
}

/// A frame seen while walking every frame of a thread, see
/// [`ThreadState::frames_rev`].
pub enum FrameRef<'a, 'gc> {
    Lua(&'a LuaFrame<'gc>),
    Exec(&'a ExecKind<'gc>),
}

/// Most stack slots a thread's Lua frames may use (LuaJIT's `LUAI_MAXSTACK`).
/// Calls fail with "stack overflow" past [`STACK_LIMIT`]; the slots above it
/// are headroom for a message handler to report that (PUC's `STACKERRSPACE`).
pub(crate) const MAX_STACK: usize = 65_500;
pub(crate) const STACK_LIMIT: usize = MAX_STACK - 200;

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
///  3. Removing values means **lowering `top`**, nothing more. The vacated
///     slots keep their stale contents until the collector traces the thread,
///     which nil-fills everything above the live region (see the `Collect`
///     impl); the mutator never reads a slot above `top` or outside a live
///     frame's window, so it never observes them.
///
/// The *live region* is `[0 .. max(top, top_lua.base + max_stack_size))`: a
/// Lua frame's register window is live regardless of `top` (the interpreter
/// addresses registers off `base`, not `top`), while `top` covers the
/// value-passing window — args and results in flight between frames — which
/// can sit above the frames during a native call.
pub struct ThreadState<'gc> {
    /// Backing store. May hold dead slots above the live region; the
    /// collector clears them when it traces the thread.
    pub(crate) stack: ValueStack<'gc>,
    /// Lua frames, innermost last.
    pub(crate) frames: Vec<LuaFrame<'gc>>,
    /// Executor frames, innermost last, each placed among `frames` by its
    /// `depth`. Kept apart so the interpreter's frames are plain records.
    pub(crate) exec_frames: Vec<ExecFrame<'gc>>,
    pub(crate) open_upvalues: Vec<Upvalue<'gc>>,
    pub(crate) tbc_slots: Vec<usize>,
    pub(crate) status: ThreadStatus,
    /// Logical stack top — the end of the value-passing window. Always valid:
    /// it is the sole signal of "how many values are here" across every
    /// hand-off (native args/results, yields, sequence polls, thread resume,
    /// `Result`). Multires producers (`VARARG`/`CALL`/`TAILCALL` with the `0`
    /// sentinel, native multi-return) publish through it; their consumers read
    /// it back.
    pub(crate) top: usize,
    /// Back-reference to the owning Thread handle, needed for creating open upvalues.
    pub(crate) thread_handle: Option<Thread<'gc>>,
    /// A native callback's `Suspend` request, deposited by the interpreter's
    /// native-call paths and consumed by the executor driver loop on the next
    /// pump. `None` between pumps. The interpreter never observes this (it
    /// leaves dispatch right after setting it).
    pub(crate) pending_action: Option<PendingAction<'gc>>,
    /// Where the thread's yielded values currently live. Set when the
    /// thread suspends via a `Yield` action (or a sequence's
    /// `SequencePoll::Yield`/`TailYield`). Consumed on resume to recover
    /// where the call's results should land. `None` outside of yielded
    /// state.
    pub(crate) yield_bottom: Option<CallSite>,
    /// The error value that killed this coroutine, set when an uncaught error
    /// unwinds out of it (status -> `Stopped`). `coroutine.close` surfaces it
    /// as `(false, err)` and clears it; `None` for a coroutine that died by
    /// normal return or was never run.
    pub(crate) death_error: Option<Value<'gc>>,
    /// End a Lua frame's register window may not cross: [`STACK_LIMIT`], or
    /// [`MAX_STACK`] while a message handler runs.
    pub(crate) stack_limit: usize,
}

/// The value stack's storage: a `Vec` the mutator uses as such, that the
/// collector may also clear from `trace(&self)` (hence the cell).
#[derive(Default)]
pub struct ValueStack<'gc>(UnsafeCell<Vec<Value<'gc>>>);

impl<'gc> ValueStack<'gc> {
    pub fn new() -> Self {
        ValueStack(UnsafeCell::new(Vec::new()))
    }
}

impl<'gc> Deref for ValueStack<'gc> {
    type Target = Vec<Value<'gc>>;
    #[inline(always)]
    fn deref(&self) -> &Vec<Value<'gc>> {
        unsafe { &*self.0.get() }
    }
}

impl<'gc> DerefMut for ValueStack<'gc> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Vec<Value<'gc>> {
        self.0.get_mut()
    }
}

// SAFETY: traces every field that can own a `Gc` pointer. The only subtlety is
// `stack` (issue #43): it is grown-not-shrunk, so slots above the live region
// are dead scratch and must NOT be traced — tracing the whole vec would retain
// whatever a since-returned callee happened to leave in its registers.
// `live_top` is the sound live high-water (see its doc for why the innermost
// frame's window bounds every frame's registers).
// The dead region is nil-filled here, as in LuaJIT and PUC Lua, rather than by
// the mutator as it vacates slots. This keeps every slot valid: after a
// trace, each slot below `live_top` holds a marked object or a primitive and
// each slot above holds nil; until the next trace the mutator writes only
// values it could reach; and a `borrow_mut` since the last trace re-grays the
// thread, so a slot can re-enter the live region only after the trace that
// cleared it. Writing through `&self` is sound because collection runs
// outside `mutate`, with no borrow of the thread outstanding.
// `tbc_slots`/`status`/`top`/`yield_bottom` hold no `Gc` pointers.
unsafe impl<'gc> Collect<'gc> for ThreadState<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        let live = self.live_top().min(self.stack.len());
        let stack = unsafe { &mut *self.stack.0.get() };
        stack[live..].fill(Value::nil());
        cc.trace(&stack[..live]);
        cc.trace(&self.frames);
        cc.trace(&self.exec_frames);
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
    pub action: Box<Suspend<'gc>>,
    #[collect(require_static)]
    pub call_site: CallSite,
}

impl<'gc> ThreadState<'gc> {
    /// The owning `Thread`. `Thread::new` stores it before the state is
    /// reachable, so it is only `None` inside that constructor.
    #[inline(always)]
    pub(crate) fn handle(&self) -> Thread<'gc> {
        debug_assert!(self.thread_handle.is_some());
        unsafe { self.thread_handle.unwrap_unchecked() }
    }

    /// Lua frames below the innermost executor frame.
    #[inline]
    pub(crate) fn exec_depth(&self) -> usize {
        self.exec_frames.last().map_or(0, |e| e.depth)
    }

    /// Whether the innermost frame is a Lua frame.
    #[inline]
    pub(crate) fn top_is_lua(&self) -> bool {
        self.frames.len() > self.exec_depth()
    }

    /// Whether the thread has no frames at all.
    pub(crate) fn frames_empty(&self) -> bool {
        self.frames.is_empty() && self.exec_frames.is_empty()
    }

    /// The innermost frame if it is a Lua frame. Most interpreter sites can
    /// `.unwrap()` this — the dispatch loop only runs when a Lua frame is on
    /// top — but the executor driver loop must also handle `top_exec`.
    #[inline]
    pub(crate) fn top_lua(&self) -> Option<&LuaFrame<'gc>> {
        if self.top_is_lua() {
            self.frames.last()
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn top_lua_mut(&mut self) -> Option<&mut LuaFrame<'gc>> {
        if self.top_is_lua() {
            self.frames.last_mut()
        } else {
            None
        }
    }

    /// # Safety
    /// The innermost frame must be a Lua frame.
    #[inline]
    pub(crate) unsafe fn top_lua_unchecked(&self) -> &LuaFrame<'gc> {
        debug_assert!(self.top_is_lua(), "non-Lua frame on top");
        unsafe { self.frames.last().unwrap_unchecked() }
    }

    /// Raw pointer to the innermost frame. Derived from the buffer rather than
    /// a `&mut` to the element, so it stays usable across later borrows of
    /// `frames` and may be offset to neighbouring frames.
    ///
    /// # Safety
    /// The innermost frame must be a Lua frame.
    #[inline]
    pub(crate) unsafe fn top_lua_ptr(&mut self) -> *mut LuaFrame<'gc> {
        debug_assert!(self.top_is_lua(), "non-Lua frame on top");
        unsafe { self.frames.as_mut_ptr().add(self.frames.len() - 1) }
    }

    /// The innermost frame if it is an executor frame.
    pub(crate) fn top_exec(&self) -> Option<&ExecKind<'gc>> {
        if self.top_is_lua() {
            None
        } else {
            self.exec_frames.last().map(|e| &e.kind)
        }
    }

    pub(crate) fn top_exec_mut(&mut self) -> Option<&mut ExecKind<'gc>> {
        if self.top_is_lua() {
            None
        } else {
            self.exec_frames.last_mut().map(|e| &mut e.kind)
        }
    }

    /// Push an executor frame above every current frame.
    pub(crate) fn push_exec(&mut self, kind: ExecKind<'gc>) {
        let depth = self.frames.len();
        self.exec_frames.push(ExecFrame { depth, kind });
    }

    /// Pop the innermost frame if it is an executor frame.
    pub(crate) fn pop_exec(&mut self) -> Option<ExecKind<'gc>> {
        if self.top_is_lua() {
            None
        } else {
            self.exec_frames.pop().map(|e| e.kind)
        }
    }

    /// Pop the innermost frame, which must be a Lua frame.
    #[inline]
    pub(crate) fn pop_lua(&mut self) {
        debug_assert!(self.top_is_lua(), "non-Lua frame on top");
        self.frames.pop();
    }

    /// Drop everything a previous run left behind; `status` is the caller's.
    pub(crate) fn reset(&mut self) {
        self.discard_above(0);
        self.frames.clear();
        self.exec_frames.clear();
        self.open_upvalues.clear();
        self.tbc_slots.clear();
        self.pending_action = None;
        self.yield_bottom = None;
        self.death_error = None;
        self.stack_limit = STACK_LIMIT;
    }

    /// Every frame, innermost first.
    pub(crate) fn frames_rev(&self) -> impl Iterator<Item = FrameRef<'_, 'gc>> {
        let (mut lua, mut exec) = (self.frames.len(), self.exec_frames.len());
        std::iter::from_fn(move || {
            if exec > 0 && self.exec_frames[exec - 1].depth == lua {
                exec -= 1;
                Some(FrameRef::Exec(&self.exec_frames[exec].kind))
            } else if lua > 0 {
                lua -= 1;
                Some(FrameRef::Lua(&self.frames[lua]))
            } else {
                None
            }
        })
    }

    /// One past the highest slot anything on this thread can still read.
    /// Lua registers live in `[base, base + max_stack)`, and a call's
    /// function slot is the caller's first free register, so the innermost
    /// Lua frame's window bounds every frame's live registers (extras sit
    /// below a base). `top` adds the value-passing window a native or a
    /// paused multires producer may have pushed above it.
    pub(crate) fn live_top(&self) -> usize {
        self.frames
            .last()
            .map_or(0, |lf| lf.base() + lf.closure.proto.max_stack_size as usize)
            .max(self.top)
    }

    /// Raise `err` on this thread: resolve its position prefix against the
    /// current frames, then install the unwinding marker for the executor.
    pub(crate) fn raise(&mut self, ctx: Context<'gc>, err: Error<'gc>) {
        let err = crate::vm::debug::locate(ctx, self, err);
        self.push_exec(ExecKind::Error(err));
    }

    /// Push a new Lua frame. `base` must sit directly above the function
    /// slot: `op_return` locates it as `base - 1 - num_extras` (VARARGPREP
    /// later shifts `base` up by `num_extras`), which wraps for `base == 0`.
    #[inline]
    pub(crate) fn push_lua(&mut self, mut lf: LuaFrame<'gc>) {
        debug_assert!(
            lf.base() >= 1,
            "Lua frame base must leave room for the function slot"
        );
        if !self.top_is_lua() {
            lf.flags |= frame_flags::PARENT_NON_LUA;
        }
        if self.frames.len() == self.frames.capacity() {
            self.grow_frames();
        }
        self.frames.push(lf);
    }

    #[cold]
    #[inline(never)]
    fn grow_frames(&mut self) {
        self.frames.reserve(1);
    }

    /// `push_lua` for the interpreter: room already made (`reserve_frames`) and
    /// the parent is the running Lua frame, so `flags` is stored as given.
    ///
    /// # Safety
    /// `frames.len() < frames.capacity()`, and the innermost frame is a Lua frame.
    #[inline(always)]
    pub(crate) unsafe fn push_lua_unchecked(&mut self, lf: LuaFrame<'gc>) {
        debug_assert!(lf.base() >= 1);
        debug_assert!(self.frames.len() < self.frames.capacity());
        debug_assert!(self.top_is_lua());
        let len = self.frames.len();
        unsafe {
            self.frames.as_mut_ptr().add(len).write(lf);
            self.frames.set_len(len + 1);
        }
    }

    #[inline(always)]
    pub(crate) fn frames_full(&self) -> bool {
        self.frames.len() == self.frames.capacity()
    }

    pub(crate) fn reserve_frames(&mut self, n: usize) {
        self.frames.reserve(n);
    }

    // --- Stack accessors -------------------------------------------------
    //
    // These enforce the stack invariant documented on `ThreadState`. Prefer
    // them over touching `stack` / `top` directly; the only code that should
    // index `stack` raw is register access off a frame's `base`.

    /// Make `stack[..n]` physically addressable. Grow-only, per rule (2).
    #[inline]
    pub(crate) fn ensure_slots(&mut self, n: usize) {
        if self.stack.len() < n {
            self.grow_slots(n);
        }
    }

    /// Out of line so the growth path (a `resize` loop) never sits in a
    /// handler's fast path or forces it to set up a stack frame.
    #[cold]
    #[inline(never)]
    fn grow_slots(&mut self, n: usize) {
        // Frame bases are stored in 32 bits (`LuaFrame::base`).
        assert!(n <= u32::MAX as usize, "Lua stack exceeds 2^32 slots");
        self.stack.resize(n, Value::nil());
    }

    /// `ensure_slots` for a Lua frame's register window ending at `n`; `false`
    /// when that would cross `stack_limit`, which the caller raises as a stack
    /// overflow. Only growth is checked, so a window inside slots a native
    /// already pushed past the limit is allowed.
    #[inline]
    #[must_use]
    pub(crate) fn ensure_frame_slots(&mut self, n: usize) -> bool {
        self.stack.len() >= n || self.grow_frame_slots(n)
    }

    #[cold]
    #[inline(never)]
    fn grow_frame_slots(&mut self, n: usize) -> bool {
        if n > self.stack_limit {
            return false;
        }
        self.grow_slots(n);
        true
    }

    /// A message handler is running in the headroom above [`STACK_LIMIT`].
    pub(crate) fn in_error_headroom(&self) -> bool {
        self.stack_limit > STACK_LIMIT
    }

    /// The logical window `stack[bottom..top]`.
    #[inline]
    pub(crate) fn window(&self, bottom: usize) -> &[Value<'gc>] {
        &self.stack[bottom..self.top]
    }

    /// `set_top` for a caller that knows `n` is already physically covered
    /// (the interpreter, whose CALL sized the window).
    #[inline]
    pub(crate) fn set_top_unchecked(&mut self, n: usize) {
        debug_assert!(n <= self.stack.len());
        self.top = n;
    }

    /// Publish a new logical top. Raising it only exposes slots the caller
    /// has already written; lowering it leaves the vacated slots to the
    /// collector (rule 3).
    #[inline]
    pub(crate) fn set_top(&mut self, n: usize) {
        self.ensure_slots(n);
        self.top = n;
    }

    /// Replace the window at `bottom` with `values` and publish the new top.
    pub(crate) fn set_window<I>(&mut self, bottom: usize, values: I)
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
    pub(crate) fn take_window(&mut self, bottom: usize) -> Vec<Value<'gc>> {
        let values = self.window(bottom).to_vec();
        self.set_top(bottom);
        values
    }

    /// Insert `v` at `at`, shifting `stack[at..top]` up one slot. The call
    /// convention wants the function immediately below its args, but a
    /// suspended callback leaves only args behind — this splices the function
    /// back in.
    pub(crate) fn insert_at(&mut self, at: usize, v: Value<'gc>) {
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
    pub(crate) fn discard_above(&mut self, n: usize) {
        debug_assert!(n <= self.stack.len());
        self.stack.truncate(n);
        self.top = n;
    }
}

impl<'gc> Thread<'gc> {
    pub fn new(mc: &Mutation<'gc>) -> Self {
        let state = ThreadState {
            stack: ValueStack::new(),
            frames: Vec::new(),
            exec_frames: Vec::new(),
            open_upvalues: Vec::new(),
            tbc_slots: Vec::new(),
            status: ThreadStatus::Stopped,
            top: 0,
            thread_handle: None,
            pending_action: None,
            yield_bottom: None,
            death_error: None,
            stack_limit: STACK_LIMIT,
        };
        let thread = Thread(Gc::new(mc, RefLock::new(state)));
        // Store the back-reference
        thread.borrow_mut(mc).thread_handle = Some(thread);
        thread
    }

    pub(crate) fn borrow(self) -> Ref<'gc, ThreadState<'gc>> {
        self.0.borrow()
    }

    pub(crate) fn borrow_mut(self, mc: &Mutation<'gc>) -> RefMut<'gc, ThreadState<'gc>> {
        self.0.borrow_mut(mc)
    }

    pub fn status(self) -> ThreadStatus {
        self.0.borrow().status
    }

    /// Pointer equality between two thread handles.
    pub fn ptr_eq(self, other: Thread<'gc>) -> bool {
        Gc::ptr_eq(self.0, other.0)
    }

    pub(crate) fn inner(self) -> Gc<'gc, RefLock<ThreadState<'gc>>> {
        self.0
    }

    pub(crate) fn from_inner(g: Gc<'gc, RefLock<ThreadState<'gc>>>) -> Self {
        Thread(g)
    }
}
