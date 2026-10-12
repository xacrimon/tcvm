use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use crate::dmm::{Collect, Gc, Mutation, Ref, RefLock, RefMut, Trace};
use crate::env::error::Error;
use crate::env::function::Upvalue;
use crate::env::value::Value;
use crate::instruction::Instruction;
use crate::vm::frame;

/// Copy wrapper stored in Value.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Thread<'gc>(Gc<'gc, RefLock<ThreadState<'gc>>>);

/// Authoritative thread state, read by the executor and by
/// `coroutine.status` / `coroutine.isyieldable`.
///
/// "Running" is not represented: a thread runs iff dispatch holds its state.
#[derive(Clone, Copy, PartialEq, Eq, Collect)]
#[collect(internal, require_static)]
pub enum ThreadStatus {
    /// Newly created or cleared, or dead by an error; no seeded call.
    Stopped,
    /// Suspended (yielded, or created and not yet resumed). Resumable.
    Suspended,
    /// Resumed another coroutine and waits for it.
    Normal,
    /// Finished: `stack[bottom..top]` are the return values.
    Result { bottom: usize },
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
/// `stack` is the value region of the thread's mapping (`vm::stack`), which
/// never moves; `stack.len()` is how far it is committed (nil-filled) and
/// `top` is the *logical* stack top. Three rules, enforced by the `stack_*`
/// accessors below:
///
///  1. `top <= stack.len()` at all times.
///  2. The committed length only grows while any frame is live. The only
///     sanctioned shrink is [`ThreadState::discard_above`], for a thread
///     with no frames.
///  3. Removing values means **lowering `top`**. The vacated slots keep stale
///     contents until the collector traces the thread, which nil-fills
///     everything above the live region; the mutator never reads a slot above
///     `top` or outside a live frame's window.
///
/// A thread gets its mapping when it first runs: a coroutine at its first
/// resume, a main thread at `Executor::start`. Frames are headers inside
/// `stack` (`vm::frame`), chained from `top_base` through their caller
/// words; `top_base` and `top_pc` are published by dispatch at every exit,
/// so outside dispatch they are current.
pub struct ThreadState<'gc> {
    /// The value stack. May hold dead slots above the live region; the
    /// collector clears them when it traces the thread.
    pub(crate) stack: ValueStack<'gc>,
    /// One past `stack`'s last committed slot: what CALL compares a window
    /// against. Kept current by every path that commits or discards slots.
    pub(crate) stack_end: *const Value<'gc>,
    /// The mapping `stack` lies in, none before the thread first runs.
    pub(crate) mem: Option<crate::vm::stack::ThreadStack>,
    /// Where this thread continues when switched to (`vm::stack`); `cont`
    /// is 0 before it first runs.
    pub(crate) sw: crate::vm::stack::SwitchPoint,
    /// The entry record of the level the thread runs in (`vm::stack`), 0
    /// before it first runs and on x86-64.
    pub(crate) record: usize,
    /// Compiled entries refuse with the stack pointer below this: the native
    /// stack's end plus headroom for natives.
    pub(crate) native_limit: usize,
    /// Fast calls with the stack pointer below this branch with a tagged
    /// return address instead of `blr` (5.4): this level's return address
    /// stack is full from here.
    pub(crate) native_deep: usize,
    /// The frame pointer and return address of the compiled code a native
    /// runs for through `jit_call_native` (I1): its frame's caller while
    /// the frame returns through `ret_shim`.
    pub(crate) shim: [usize; 2],
    /// The function a seeded thread calls, until it gets its stack.
    pub(crate) entry: Option<Value<'gc>>,
    /// Logical stack top: the end of the value-passing window. Multires
    /// producers publish through it; their consumers read it back.
    pub(crate) top: usize,
    /// The top frame's base, null with no frames.
    pub(crate) top_base: *mut Value<'gc>,
    /// The top frame's pc, when it is a Lua frame.
    pub(crate) top_pc: *const Instruction,
    /// Whether the seeded call at slot 0 has been entered (header 0 written).
    pub(crate) started: bool,
    /// Debug: a handler published this frame (`sync!`) since the last
    /// dispatch, which then checks the published state.
    #[cfg(debug_assertions)]
    pub(crate) synced: bool,
    /// Debug: the source location of that `sync!`.
    #[cfg(debug_assertions)]
    pub(crate) sync_site: (&'static str, u32),
    /// An error nothing on the thread caught; the executor reports it.
    pub(crate) uncaught: Option<Error<'gc>>,
    /// Upvalues still pointing into `stack`, sorted by slot, so a `CLOSE` or
    /// return closes a tail of the list.
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
    /// Back-reference to the owning Thread handle, for creating open upvalues.
    pub(crate) thread_handle: Option<Thread<'gc>>,
    /// Where the values a resume delivers land: the yield frame's base. Set
    /// while suspended by a yield.
    pub(crate) yield_bottom: Option<usize>,
    /// The error value that killed this coroutine, for `coroutine.close` to
    /// surface as `(false, err)`.
    pub(crate) death_error: Option<Value<'gc>>,
    /// End a Lua frame's register window may not cross: [`STACK_LIMIT`], or
    /// [`MAX_STACK`] while a message handler runs.
    pub(crate) stack_limit: usize,
    /// The thread that resumed this one, while it runs (status `Normal`).
    pub(crate) resumer: Option<Thread<'gc>>,
    /// While this thread waits on a coroutine it resumed: where the values
    /// that coroutine yields or returns land.
    pub(crate) resume_wait: usize,
    /// The error that killed a coroutine this thread resumed, handed over
    /// with the switch back.
    pub(crate) transfer: Option<Error<'gc>>,
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

/// The value stack: the committed part of the mapping's value region,
/// which the collector may also clear from `trace(&self)` (hence the cell).
pub struct ValueStack<'gc>(UnsafeCell<StackView<'gc>>);

struct StackView<'gc> {
    ptr: NonNull<Value<'gc>>,
    len: usize,
}

impl<'gc> ValueStack<'gc> {
    fn empty() -> Self {
        ValueStack(UnsafeCell::new(StackView {
            ptr: NonNull::dangling(),
            len: 0,
        }))
    }
}

impl<'gc> Deref for ValueStack<'gc> {
    type Target = [Value<'gc>];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        let v = unsafe { &*self.0.get() };
        unsafe { std::slice::from_raw_parts(v.ptr.as_ptr(), v.len) }
    }
}

impl<'gc> DerefMut for ValueStack<'gc> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let v = self.0.get_mut();
        unsafe { std::slice::from_raw_parts_mut(v.ptr.as_ptr(), v.len) }
    }
}

// SAFETY: traces every field that can own a `Gc` pointer. Slots above the
// live region are dead scratch and must NOT be traced; they are nil-filled
// here instead, as in LuaJIT and PUC, so every slot below the live region
// holds a marked object or a primitive and every slot above holds nil. Header
// words are skipped except word 0, traced as the frame's function, and dead
// words 2 and 3, cleared (`vm::frame`). A compiled frame below a level (H11)
// keeps values in its native frame and in the home slots of the registers
// its call site's snapshot names; the rest of its window is cleared.
// Writing through `&self` is sound because collection runs outside `mutate`,
// with no borrow of the thread outstanding and every thread parked.
unsafe impl<'gc> Collect<'gc> for ThreadState<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        let live = self.live_top().min(self.stack.len());
        let stack = unsafe {
            let v = &*self.stack.0.get();
            std::slice::from_raw_parts_mut(v.ptr.as_ptr(), v.len)
        };
        stack[live..].fill(Value::nil());
        let sp = stack.as_ptr();
        let mut upper = live;
        for f in frame::frames(self) {
            if f.is_elided() {
                continue;
            }
            let bi = unsafe { f.base.offset_from_unsigned(sp) };
            debug_assert!(bi >= frame::HDR && bi <= upper);
            // A frame of nothing (`enter_raise`) has no function.
            if f.func & frame::PTR_MASK != 0 {
                cc.trace(&unsafe { frame::function(f.base) });
            }
            if f.caller_words_dead() {
                stack[bi - 2..bi].fill(Value::nil());
            }
            match f.compiled {
                None => cc.trace(&stack[bi..upper]),
                // SAFETY: the site's snapshot names the frame's spill slots
                // that hold values across the call and the registers whose
                // home slots hold theirs. Its other registers below the
                // callee's header slot are dead until written, and may hold
                // what an earlier frame left there: they are cleared, as the
                // slots above the live region are. A to-be-closed variable's
                // slot is the exception, read by a close the region exits
                // for. From the header slot up the slots are the callee's (a
                // vararg callee's extras sit below its header).
                Some((fp, site, region)) => unsafe {
                    use crate::jit::region::{Loc, SRep, SnapView};
                    let snap = SnapView::decode(&(&*region).snaps[(*site).snap as usize..]);
                    let mut homes = crate::jit::build::cfg::RegSet::EMPTY;
                    for e in snap.entries() {
                        match (e.loc, e.rep) {
                            (Loc::Spill(i), SRep::Val) => {
                                let p = (fp + crate::jit::FRAME_SPILLS + 8 * i as usize)
                                    as *const Value<'gc>;
                                cc.trace(&*p);
                            }
                            (Loc::Home, _) => homes.insert(e.reg as usize),
                            _ => {}
                        }
                    }
                    let hdr = (*site).hdr as usize;
                    let tbc = |s| {
                        self.tbc_list
                            .iter()
                            .any(|e| matches!(e, TbcEntry::Slot(i) if *i == s))
                    };
                    for (r, slot) in stack[bi..upper].iter_mut().enumerate() {
                        if r >= hdr || homes.contains(r) || tbc(bi + r) {
                            cc.trace(slot);
                        } else {
                            *slot = Value::nil();
                        }
                    }
                },
            }
            upper = bi - frame::HDR;
        }
        cc.trace(&stack[..upper]);
        cc.trace(&self.open_upvalues);
        cc.trace(&self.thread_handle);
        cc.trace(&self.resumer);
        cc.trace(&self.tbc_list);
        cc.trace(&self.uncaught);
        cc.trace(&self.death_error);
        cc.trace(&self.entry);
        cc.trace(&self.transfer);
        for (v, _) in &self.locals {
            cc.trace(v);
        }
    }
}

impl<'gc> ThreadState<'gc> {
    /// The owning `Thread`. `Thread::new` stores it before the state is
    /// reachable, so it is only `None` inside that constructor.
    #[inline(always)]
    pub(crate) fn handle(&self) -> Thread<'gc> {
        debug_assert!(self.thread_handle.is_some());
        unsafe { self.thread_handle.unwrap_unchecked() }
    }

    /// Seed a thread with the call of `f`, entered on its first resume.
    pub(crate) fn seed(&mut self, f: Value<'gc>) {
        debug_assert!(self.top_base.is_null());
        self.entry = Some(f);
        self.started = false;
        self.status = ThreadStatus::Suspended;
    }

    /// Give the thread its mapping if it has none, and lay out the seeded
    /// call `[f, _, _, _]` at slot 0, its arguments to follow at slot 4.
    pub(crate) fn ensure_stack(&mut self, pool: &std::rc::Rc<crate::vm::stack::StackPool>) {
        if self.mem.is_none() {
            let mem = pool.acquire();
            self.native_limit = mem.native_end() + crate::vm::stack::NATIVE_HEADROOM;
            let v = self.stack.0.get_mut();
            v.ptr = unsafe { NonNull::new_unchecked(mem.values()) };
            v.len = 0;
            self.mem = Some(mem);
            self.top = 0;
            self.update_stack_end();
        }
        if let Some(f) = self.entry.take() {
            self.set_window(0, [f, Value::nil(), Value::nil(), Value::nil()]);
        }
    }

    /// Hand the mapping back to its pool. The thread runs no more: it has no
    /// frames, and nothing reads its stack.
    pub(crate) fn release_stack(&mut self) {
        drop(self.take_stack());
    }

    /// Take the mapping, leaving the thread without a stack: [`release_stack`]
    /// for a thread still running on it, which hands it to the next one.
    ///
    /// [`release_stack`]: Self::release_stack
    pub(crate) fn take_stack(&mut self) -> Option<crate::vm::stack::ThreadStack> {
        debug_assert!(self.top_base.is_null());
        let mut mem = self.mem.take()?;
        mem.value_used = self.stack.len() * size_of::<Value<'gc>>();
        *self.stack.0.get_mut() = StackView {
            ptr: NonNull::dangling(),
            len: 0,
        };
        self.sw = Default::default();
        self.record = 0;
        self.top = 0;
        self.update_stack_end();
        Some(mem)
    }

    /// The top frame's base as an index.
    #[inline]
    pub(crate) fn top_base_index(&self) -> usize {
        debug_assert!(!self.top_base.is_null());
        unsafe { self.top_base.offset_from_unsigned(self.stack.as_ptr()) }
    }

    /// Whether the top frame is a native's.
    #[inline]
    pub(crate) fn top_is_native(&self) -> bool {
        !self.top_base.is_null() && unsafe { frame::is_native(self.top_base) }
    }

    /// Drop everything a previous run left behind, its stack too; `status`
    /// is the caller's.
    pub(crate) fn reset(&mut self) {
        // An open upvalue left behind would keep pointing into `stack`.
        debug_assert!(self.open_upvalues.is_empty());
        self.top_base = std::ptr::null_mut();
        self.top_pc = std::ptr::null();
        self.release_stack();
        self.entry = None;
        self.started = false;
        self.uncaught = None;
        self.open_upvalues.clear();
        self.tbc_list.clear();
        self.no_yield = false;
        self.yield_bottom = None;
        self.death_error = None;
        self.stack_limit = STACK_LIMIT;
        while !self.tasks.is_empty() {
            self.drop_task();
        }
        self.locals.clear();
        self.local_epoch = 0;
    }

    /// One past the highest slot anything on this thread can still read:
    /// the top Lua frame's register window, or everything up to `top`,
    /// whichever reaches higher (a native's window or a paused multires
    /// producer may sit above the registers).
    pub(crate) fn live_top(&self) -> usize {
        if self.top_base.is_null() || unsafe { frame::is_native(self.top_base) } {
            return self.top;
        }
        let base = self.top_base_index();
        let max_stack = unsafe { frame::closure(self.top_base) }.max_stack_size as usize;
        (base + max_stack).max(self.top)
    }

    // --- Stack accessors -------------------------------------------------
    //
    // These enforce the stack invariant documented on `ThreadState`. Prefer
    // them over touching `stack` / `top` directly; the only code that should
    // index `stack` raw is register access off a frame's `base`.

    /// Make `stack[..n]` addressable. Grow-only, per rule (2).
    #[inline]
    pub(crate) fn ensure_slots(&mut self, n: usize) {
        if self.stack.len() < n {
            self.grow_slots(n);
        }
    }

    /// Commit up to `n`, nil-filling. Out of line so the growth path never
    /// sits in a handler's fast path.
    #[cold]
    #[inline(never)]
    fn grow_slots(&mut self, n: usize) {
        assert!(
            self.mem.is_some() && n <= crate::vm::stack::VALUE_SLOTS,
            "value stack overflow"
        );
        let v = self.stack.0.get_mut();
        for i in v.len..n {
            unsafe { v.ptr.as_ptr().add(i).write(Value::nil()) };
        }
        v.len = n;
        self.update_stack_end();
    }

    #[inline(always)]
    fn update_stack_end(&mut self) {
        self.stack_end = self.stack.as_ptr().wrapping_add(self.stack.len());
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

    /// `set_top` for a caller that knows `n` is already physically covered.
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

    /// Cut the committed stack down to `n` slots. The **only** sanctioned
    /// shrink (rule 2): the caller must guarantee no live frame window and no
    /// in-flight native call sits above `n`.
    pub(crate) fn discard_above(&mut self, n: usize) {
        debug_assert!(n <= self.stack.len());
        debug_assert!(self.top_base.is_null() || self.top_base_index() <= n);
        self.stack.0.get_mut().len = n;
        self.top = n;
        self.update_stack_end();
    }

    /// Copy `n` values from `src` (another thread's stack) to `at` and make
    /// them the top. Those past the stack's end are dropped: `top` is then
    /// past `stack_limit`, which the receiving continuation reports as too
    /// many results.
    pub(crate) fn copy_in(&mut self, at: usize, src: *const Value<'gc>, n: usize) {
        let n = n.min(crate::vm::stack::VALUE_SLOTS.saturating_sub(at));
        self.ensure_slots(at + n);
        unsafe { frame::copy_values(self.slot_ptr(at), src, n) };
        self.top = at + n;
    }

    /// The stack slot at index `i` as a pointer.
    #[inline(always)]
    pub(crate) fn slot_ptr(&mut self, i: usize) -> *mut Value<'gc> {
        debug_assert!(i <= self.stack.len());
        unsafe { self.stack.as_mut_ptr().add(i) }
    }

    /// The index of a pointer into the stack.
    #[inline(always)]
    pub(crate) fn slot_index(&self, p: *const Value<'gc>) -> usize {
        debug_assert!(
            p >= self.stack.as_ptr() && p <= self.stack_end,
            "slot {p:p} outside the stack {:p}..{:p} (len {})",
            self.stack.as_ptr(),
            self.stack_end,
            self.stack.len()
        );
        unsafe { p.offset_from_unsigned(self.stack.as_ptr()) }
    }
}

impl<'gc> Thread<'gc> {
    pub fn new(mc: &Mutation<'gc>) -> Self {
        let state = ThreadState {
            stack: ValueStack::empty(),
            stack_end: std::ptr::null(),
            mem: None,
            sw: Default::default(),
            record: 0,
            native_limit: 0,
            native_deep: 0,
            shim: [0; 2],
            entry: None,
            top: 0,
            top_base: std::ptr::null_mut(),
            top_pc: std::ptr::null(),
            started: false,
            #[cfg(debug_assertions)]
            synced: false,
            #[cfg(debug_assertions)]
            sync_site: ("", 0),
            uncaught: None,
            open_upvalues: Vec::new(),
            tbc_list: Vec::new(),
            no_yield: false,
            main: false,
            status: ThreadStatus::Stopped,
            thread_handle: None,
            yield_bottom: None,
            death_error: None,
            stack_limit: STACK_LIMIT,
            resumer: None,
            resume_wait: 0,
            transfer: None,
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
        let mut ts = thread.borrow_mut(mc);
        ts.thread_handle = Some(thread);
        ts.update_stack_end();
        drop(ts);
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
