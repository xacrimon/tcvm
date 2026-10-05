use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};

use crate::dmm::allocator_api::MetricsAlloc;
use crate::dmm::{Collect, Gc, Mutation, Ref, RefLock, RefMut, Trace};
use crate::env::error::Error;
use crate::env::function::{LuaFn, Upvalue};
use crate::env::value::Value;
use crate::lua::Context;
use crate::vm::interp::Handler;

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

/// A frame on a thread's frame stack: a Lua function's, or a native's that
/// called back into Lua and waits for the results (`NATIVE`).
///
/// Every frame says where its results go: `ret`, run when it returns, with the
/// frame below on top again. A call site picks it (CALL, a metamethod or
/// iterator, the executor), and it decodes what it needs from its own
/// instruction, LuaJIT Remake style.
// `repr(C)`, as `ret` before `pc` matters: a continuation reloads the
// caller's `closure` and `pc`, and a load pair over the `pc` CALL just
// stored stalls store forwarding.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
#[repr(C)]
pub struct LuaFrame<'gc> {
    /// A `NATIVE` frame's holds the native function, and is never
    /// dereferenced as a Lua closure.
    pub(crate) closure: LuaFn<'gc>,
    /// Where this frame's results go; see [`ret_args`](crate::vm::interp).
    #[collect(require_static)]
    pub(crate) ret: Handler,
    /// Resume address: points *past* the instruction being executed, into
    /// `closure.proto.code` (which the frame keeps alive). A raw pointer
    /// rather than an index so CALL saves it with one store. A `NATIVE`
    /// frame's is its continuation (`NativeCont`).
    #[collect(require_static)]
    pub(crate) pc: *const crate::instruction::Instruction,
    /// Read through `base()`. 32 bits: a frame sits inside the stack, and
    /// every growth of the stack goes through `grow_slots`, which keeps it
    /// below 2^32 slots.
    pub(crate) base: u32,
    /// Caller-supplied args beyond `num_params`; the below-base region is
    /// `stack[base - num_extras .. base]`. Set by `VARARGPREP`, else 0. Fits:
    /// a frame's window ends below `MAX_STACK`. A `NATIVE` frame's is where
    /// in its window the call it waits for sits, and its results land.
    pub(crate) num_extras: u16,
    /// `frame_flags` bits. Zero means RETURN can jump straight to `ret`.
    /// Mostly left set once the hazard is gone (a stale bit just costs the slow
    /// path); a slow TAILCALL clears `OPEN_UPVALUES` after closing them.
    pub(crate) flags: u8,
}

/// Stack-window metadata threaded through every suspension point.
///
/// - `bottom` is the callback's `args_base` — `stack[bottom..]` is the
///   active window (yielded values, a native's arguments, etc.).
/// - `func_idx` is the call's function slot, below `bottom`.
/// - `ret` takes the results once they are in: the continuation of the call
///   site, as a returning frame's `ret` would.
///
/// Stored on `ExecKind::WaitThread`, `PendingAction`,
/// and as the yielded-state stash on `ThreadState`.
#[derive(Clone, Copy)]
pub struct CallSite {
    pub bottom: usize,
    pub func_idx: usize,
    pub(crate) ret: Handler,
}

/// A call site's results, waiting for the next `run_thread` to hand them to
/// `ret` (see `CallSite`): `stack[values..top]`, the call's function at
/// `func_slot`.
#[derive(Clone, Copy)]
pub(crate) struct PendingRet {
    pub(crate) ret: Handler,
    pub(crate) func_slot: usize,
    pub(crate) values: usize,
}

/// Bits of `LuaFrame::flags`.
pub mod frame_flags {
    /// A CLOSURE in this frame captured one of its locals; RETURN must close.
    pub const OPEN_UPVALUES: u8 = 1;
    /// A TBC in this frame registered a to-be-closed slot.
    pub const TBC: u8 = 2;
    /// A native's frame: its window starts at `base`, its function slot is
    /// `base - 1`.
    pub const NATIVE: u8 = 4;
    /// The native's call catches errors (`pcall`).
    pub const PROTECTED: u8 = 8;
    /// ... after running the message handler in its window's first slot
    /// (`xpcall`).
    pub const HANDLER: u8 = 16;
    /// The continuation returns a successful call's results as the
    /// native's (`OnOk::Return`), so the VM may do that instead of running it.
    pub const PASS: u8 = 32;
    /// ... after `true` (`OnOk::ReturnTrue`).
    pub const PASS_TRUE: u8 = 64;
    /// With `PROTECTED`: catches an exit too (`Protect::Base`).
    pub const BASE: u8 = 128;
}

// Two records per cache line.
const _: () = assert!(std::mem::size_of::<LuaFrame<'static>>() == 32);

impl<'gc> LuaFrame<'gc> {
    /// Write a flagless Lua frame to `dst`.
    ///
    /// # Safety
    /// `dst` is valid for writes.
    #[inline(always)]
    pub(crate) unsafe fn write_lua(
        dst: *mut Self,
        closure: LuaFn<'gc>,
        pc: *const crate::instruction::Instruction,
        ret: Handler,
        base: u32,
        num_extras: u16,
    ) {
        unsafe {
            (&raw mut (*dst).closure).write(closure);
            (&raw mut (*dst).pc).write(pc);
            (&raw mut (*dst).ret).write(ret);
            (&raw mut (*dst).base).write(base);
            (&raw mut (*dst).num_extras).write(num_extras);
            (&raw mut (*dst).flags).write(0);
        }
    }

    #[inline(always)]
    pub fn base(&self) -> usize {
        self.base as usize
    }

    /// A native's frame (see `frame_flags::NATIVE`).
    #[inline(always)]
    pub fn is_native(&self) -> bool {
        self.flags & frame_flags::NATIVE != 0
    }

    /// For a Lua frame a `pcall` (1) or `xpcall` (2) entry called without
    /// pushing its own frame, how far below this frame's function slot that
    /// call's slot is. Its `ret` marks it, as LuaJIT Remake's
    /// `OnProtectedCallSuccessReturn` does.
    #[inline]
    pub(crate) fn elided_protect(&self) -> Option<usize> {
        use crate::vm::interp::{ret_pcall, ret_xpcall};
        if std::ptr::fn_addr_eq(self.ret, ret_pcall as Handler) {
            Some(1)
        } else if std::ptr::fn_addr_eq(self.ret, ret_xpcall as Handler) {
            Some(2)
        } else {
            None
        }
    }

    /// The slot of the call that pushed this Lua frame.
    #[inline]
    pub(crate) fn func_slot(&self) -> usize {
        self.base() - 1 - self.num_extras as usize
    }

    #[inline(always)]
    pub fn set_base(&mut self, base: usize) {
        debug_assert!(base <= u32::MAX as usize);
        self.base = base as u32;
    }

    /// `pc` as an index into `closure.proto.code`.
    pub fn pc_index(&self) -> usize {
        unsafe { self.pc.offset_from_unsigned(self.closure.code) }
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

/// A to-be-closed variable, in a live frame or detached from an unwound one.
#[derive(Collect, Clone, Copy)]
#[collect(internal, no_drop)]
pub(crate) enum TbcEntry<'gc> {
    Slot(usize),
    /// Its frame was unwound by an error; `level` is where it closes from: the
    /// bottom of the frame that closes them (`vm::unwind`), or 0 on a dead
    /// coroutine.
    Detached {
        level: usize,
        value: Value<'gc>,
    },
}

impl<'gc> TbcEntry<'gc> {
    #[inline]
    pub(crate) fn pos(self) -> usize {
        match self {
            TbcEntry::Slot(slot) => slot,
            TbcEntry::Detached { level, .. } => level,
        }
    }

    pub(crate) fn value(self, stack: &[Value<'gc>]) -> Value<'gc> {
        match self {
            TbcEntry::Slot(slot) => stack[slot],
            TbcEntry::Detached { value, .. } => value,
        }
    }
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub enum ExecKind<'gc> {
    /// A coroutine that hasn't been resumed yet. Replaced on first resume by
    /// a real call frame.
    Start(Value<'gc>),
    /// Current thread is waiting on an inner thread it resumed; on the
    /// inner thread reaching a terminal/yielded state, the executor pops
    /// this frame and lands the inner thread's values into the original
    /// CALL's expected slot.
    WaitThread {
        #[collect(require_static)]
        call_site: CallSite,
    },
    /// An error for the executor: raised by the executor itself, or one
    /// nothing on the thread catches anymore (see `vm::unwind`).
    Error(Error<'gc>),
}

/// A frame seen while walking every frame of a thread, see
/// [`ThreadState::frames_rev`].
pub enum FrameRef<'a, 'gc> {
    Lua(&'a LuaFrame<'gc>),
    Native(&'a LuaFrame<'gc>),
    /// The `pcall` or `xpcall` that called Lua frame `.0` from its entry
    /// without a frame of its own; see [`LuaFrame::elided_protect`].
    Elided(&'a LuaFrame<'gc>),
    Exec(&'a ExecKind<'gc>),
}

/// Most stack slots a thread's Lua frames may use (LuaJIT's `LUAI_MAXSTACK`).
/// Calls fail with "stack overflow" past [`STACK_LIMIT`]; the slots above it
/// are headroom for a message handler to report that (PUC's `STACKERRSPACE`).
pub(crate) const MAX_STACK: usize = 65_500;
pub(crate) const STACK_LIMIT: usize = MAX_STACK - 200;
/// Slots a native may fill past `stack_limit` without `Stack::check_stack`,
/// for fixed-count results (`LUA_MINSTACK`).
pub(crate) const NATIVE_SLACK: usize = 20;

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
    pub(crate) frames: Vec<LuaFrame<'gc>, MetricsAlloc<'gc>>,
    /// Executor frames, innermost last, each placed among `frames` by its
    /// `depth`. Kept apart so the interpreter's frames are plain records.
    pub(crate) exec_frames: Vec<ExecFrame<'gc>, MetricsAlloc<'gc>>,
    /// Upvalues still pointing into `stack`, sorted by slot, so a `CLOSE` or
    /// return closes a tail of the list. `grow_slots` rebases them.
    pub(crate) open_upvalues: Vec<Upvalue<'gc>>,
    /// Open to-be-closed variables by stack position, innermost last
    /// (`L->tbclist`). Each leaves the list just before its `__close` runs.
    pub(crate) tbc_list: Vec<TbcEntry<'gc>>,
    /// Closing variables for `coroutine.close`/`wrap`, where yields are errors.
    pub(crate) no_yield: bool,
    /// Entry thread of an executor, PUC's main thread: never resumable,
    /// yieldable or closable from Lua.
    pub(crate) main: bool,
    pub(crate) status: ThreadStatus,
    /// Logical stack top — the end of the value-passing window. Always valid:
    /// it is the sole signal of "how many values are here" across every
    /// hand-off (native args/results, yields, thread resume,
    /// `Result`). Multires producers (`VARARG`/`CALL`/`TAILCALL` with the `0`
    /// sentinel, native multi-return) publish through it; their consumers read
    /// it back.
    pub(crate) top: usize,
    /// Back-reference to the owning Thread handle, needed for creating open upvalues.
    pub(crate) thread_handle: Option<Thread<'gc>>,
    /// A yield or resume left to the executor, consumed on its next pump.
    /// The interpreter never observes this (it leaves dispatch right after
    /// setting it).
    pub(crate) pending_action: Option<PendingAction<'gc>>,
    /// Results the executor delivered to a call site; the next `run_thread`
    /// starts by running its continuation.
    pub(crate) pending_ret: Option<PendingRet>,
    /// Where the thread's yielded values currently live. Set when the
    /// thread suspends by a yield. Consumed on resume to recover
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
    /// The thread that resumed this one, while it runs (status `Normal`).
    pub(crate) resumer: Option<Thread<'gc>>,
    /// Threads below this one in the resume chain.
    pub(crate) resume_depth: u16,
    /// The futures of the async natives with a frame here, innermost last.
    /// Declared before `task_arena`, which holds them, so dropped first.
    pub(crate) tasks: Vec<crate::vm::async_native::Task>,
    pub(crate) task_arena: crate::vm::async_native::TaskArena,
    /// Values async natives keep across awaits (`Local`), each with the
    /// epoch of the native it belongs to.
    pub(crate) locals: Vec<(Value<'gc>, u32)>,
    /// The epoch of the innermost async native, 0 outside any.
    pub(crate) local_epoch: u32,
    /// Set while an `AsyncFn` runs and has yet to spawn its future.
    pub(crate) spawning: Option<crate::vm::async_native::TaskHeader>,
    /// What this thread's async natives' `Cx`s see their polls through.
    pub(crate) async_env: Option<std::rc::Rc<crate::vm::async_native::EnvCell>>,
}

/// The value stack's storage: a `Vec` the mutator uses as such, that the
/// collector may also clear from `trace(&self)` (hence the cell).
pub struct ValueStack<'gc>(UnsafeCell<Vec<Value<'gc>, MetricsAlloc<'gc>>>);

impl<'gc> ValueStack<'gc> {
    pub fn new(mc: &Mutation<'gc>) -> Self {
        ValueStack(UnsafeCell::new(Vec::new_in(MetricsAlloc::new(mc))))
    }
}

impl<'gc> Deref for ValueStack<'gc> {
    type Target = Vec<Value<'gc>, MetricsAlloc<'gc>>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0.get() }
    }
}

impl<'gc> DerefMut for ValueStack<'gc> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
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
// `status`/`top`/`yield_bottom` hold no `Gc` pointers.
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
        cc.trace(&self.resumer);
        cc.trace(&self.tbc_list);
        for (v, _) in &self.locals {
            cc.trace(v);
        }
    }
}

/// A yield or resume a native asked for that dispatch couldn't make (no
/// resumer waiting in dispatch, or natives run by the executor), for the
/// executor to make on its next pump, the values at `call_site.bottom..top`.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct PendingAction<'gc> {
    pub(crate) kind: PendingKind<'gc>,
    #[collect(require_static)]
    pub(crate) call_site: CallSite,
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) enum PendingKind<'gc> {
    /// To the resumer, or the host; the values it resumes with are the
    /// call's results.
    Yield,
    /// The coroutine's values when it yields or returns are the call's
    /// results.
    Resume(Thread<'gc>),
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
    pub(crate) fn pop_lua(&mut self) -> LuaFrame<'gc> {
        debug_assert!(self.top_is_lua(), "non-Lua frame on top");
        unsafe { self.frames.pop().unwrap_unchecked() }
    }

    /// Drop everything a previous run left behind; `status` is the caller's.
    pub(crate) fn reset(&mut self) {
        // An open upvalue left behind would keep pointing into `stack`.
        debug_assert!(self.open_upvalues.is_empty());
        self.discard_above(0);
        self.frames.clear();
        self.exec_frames.clear();
        self.open_upvalues.clear();
        self.tbc_list.clear();
        self.no_yield = false;
        self.pending_action = None;
        self.pending_ret = None;
        self.yield_bottom = None;
        self.death_error = None;
        self.stack_limit = STACK_LIMIT;
        while !self.tasks.is_empty() {
            self.drop_task();
        }
        self.locals.clear();
        self.local_epoch = 0;
    }

    /// Every frame, innermost first.
    pub(crate) fn frames_rev(&self) -> impl Iterator<Item = FrameRef<'_, 'gc>> {
        let (mut lua, mut exec) = (self.frames.len(), self.exec_frames.len());
        let mut elided = None;
        std::iter::from_fn(move || {
            if let Some(f) = elided.take() {
                Some(FrameRef::Elided(f))
            } else if exec > 0 && self.exec_frames[exec - 1].depth == lua {
                exec -= 1;
                Some(FrameRef::Exec(&self.exec_frames[exec].kind))
            } else if lua > 0 {
                lua -= 1;
                let f = &self.frames[lua];
                Some(if f.is_native() {
                    FrameRef::Native(f)
                } else {
                    if f.elided_protect().is_some() {
                        elided = Some(f);
                    }
                    FrameRef::Lua(f)
                })
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
            .map_or(0, |lf| {
                // A native's window is all below `top`.
                if lf.is_native() {
                    lf.base()
                } else {
                    lf.base() + lf.closure.max_stack_size as usize
                }
            })
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
    pub(crate) fn push_lua(&mut self, lf: LuaFrame<'gc>) {
        debug_assert!(
            lf.base() >= 1,
            "Lua frame base must leave room for the function slot"
        );
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

    /// `push_lua` for the interpreter, room already made (`reserve_frames`):
    /// write a flagless Lua frame above `top` and return it.
    ///
    /// # Safety
    /// `top` is the innermost frame, a Lua frame, and
    /// `frames.len() < frames.capacity()`.
    #[inline(always)]
    pub(crate) unsafe fn push_lua_above(
        &mut self,
        top: *mut LuaFrame<'gc>,
        closure: LuaFn<'gc>,
        pc: *const crate::instruction::Instruction,
        ret: Handler,
        base: u32,
        num_extras: u16,
    ) -> *mut LuaFrame<'gc> {
        debug_assert!(base >= 1);
        debug_assert!(self.frames.len() < self.frames.capacity());
        debug_assert!(self.top_is_lua());
        debug_assert!(std::ptr::eq(top, unsafe { self.top_lua_ptr() }));
        unsafe {
            let new = top.add(1);
            LuaFrame::write_lua(new, closure, pc, ret, base, num_extras);
            self.frames.set_len(self.frames.len() + 1);
            new
        }
    }

    /// `push_lua` with room already made, for a frame of any kind.
    ///
    /// # Safety
    /// `frames.len() < frames.capacity()`.
    #[inline(always)]
    pub(crate) unsafe fn push_unchecked(&mut self, lf: LuaFrame<'gc>) {
        debug_assert!(self.frames.len() < self.frames.capacity());
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
        let old = self.stack.as_ptr().addr();
        self.stack.resize(n, Value::nil());
        let new = self.stack.as_mut_ptr();
        if new.addr() != old {
            for uv in &self.open_upvalues {
                uv.rebase(old, new);
            }
        }
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

    /// A native left more values than [`NATIVE_SLACK`] past `stack_limit`.
    #[inline]
    pub(crate) fn native_overflowed(&self) -> bool {
        self.top > self.stack_limit + NATIVE_SLACK
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
    /// thread being seeded or one with no frames left, never one with a native
    /// callback on the stack. Unwinding a single frame doesn't qualify: an
    /// outer frame's window can extend past its base.
    pub(crate) fn discard_above(&mut self, n: usize) {
        debug_assert!(n <= self.stack.len());
        self.stack.truncate(n);
        self.top = n;
    }
}

impl<'gc> Thread<'gc> {
    pub fn new(mc: &Mutation<'gc>) -> Self {
        let state = ThreadState {
            stack: ValueStack::new(mc),
            frames: Vec::new_in(MetricsAlloc::new(mc)),
            exec_frames: Vec::new_in(MetricsAlloc::new(mc)),
            open_upvalues: Vec::new(),
            tbc_list: Vec::new(),
            no_yield: false,
            main: false,
            status: ThreadStatus::Stopped,
            top: 0,
            thread_handle: None,
            pending_action: None,
            pending_ret: None,
            yield_bottom: None,
            death_error: None,
            stack_limit: STACK_LIMIT,
            resumer: None,
            resume_depth: 0,
            tasks: Vec::new(),
            task_arena: crate::vm::async_native::TaskArena::new(),
            locals: Vec::new(),
            local_epoch: 0,
            spawning: None,
            async_env: None,
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

    /// The state, for the interpreter to run the thread on, without a borrow
    /// guard: dispatch switches between coroutines' states without returning
    /// to release one. Emits the write barrier `borrow_mut` would.
    ///
    /// # Safety
    /// No other reference to the state may be in use while the result is.
    #[inline]
    pub(crate) unsafe fn state_mut(self, mc: &Mutation<'gc>) -> &'gc mut ThreadState<'gc> {
        unsafe { &mut *self.0.unlock(mc).as_ptr() }
    }

    /// [`Thread::state_mut`] when its barrier would do nothing, else `None`:
    /// for fast paths that would rather bail out than call the collector.
    ///
    /// # Safety
    /// As [`Thread::state_mut`].
    #[inline]
    pub(crate) unsafe fn state_mut_if_clean(
        self,
        mc: &Mutation<'gc>,
    ) -> Option<&'gc mut ThreadState<'gc>> {
        Gc::write_if_clean(mc, self.0).map(|w| unsafe { &mut *w.unlock().as_ptr() })
    }

    pub fn status(self) -> ThreadStatus {
        self.0.borrow().status
    }

    /// Status as another thread sees it. A main thread is always `Normal`,
    /// as PUC's is from inside a coroutine, which also keeps one executor
    /// from resuming or closing another's entry thread.
    pub(crate) fn peer_status(self) -> ThreadStatus {
        let ts = self.0.borrow();
        if ts.main {
            ThreadStatus::Normal
        } else {
            ts.status
        }
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
