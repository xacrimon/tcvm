use crate::dmm::{Collect, Gc, Mutation, Ref, RefLock, RefMut};
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
    /// Number of caller-supplied arguments beyond `num_params`. `VARARG` and
    /// `VARARGGET` read from `stack[base - num_extras .. base]`. Set by
    /// `VARARGPREP` for vararg functions; stays 0 for non-vararg frames.
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
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct ThreadState<'gc> {
    // See #43: custom Collect impl to avoid unused stack slots keeping values alive.
    pub stack: Vec<Value<'gc>>,
    pub frames: Vec<Frame<'gc>>,
    pub open_upvalues: Vec<Upvalue<'gc>>,
    pub tbc_slots: Vec<usize>,
    pub status: ThreadStatus,
    /// Dynamic top register. Written by multires producers (`VARARG`
    /// `count=0`, `CALL`/`TAILCALL` `returns=0`, native multi-return) and
    /// read by multires consumers (`CALL` `args=0`, `RETURN` `count=0`,
    /// `SETLIST` `count=0`). Outside those windows the value is undefined
    /// and must not be relied upon.
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
    #[collect(require_static)]
    pub yield_bottom: Option<CallSite>,
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
