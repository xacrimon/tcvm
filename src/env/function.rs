use std::cell::Cell;

use crate::Context;
use crate::dmm::{Collect, Gc, GcWeak, Lock, Mutation, RefLock, Trace};
use crate::env::error::Error;
use crate::env::shape::Shape;
use crate::env::string::LuaString;
use crate::env::table::{SlotLoc, TableState};
use crate::env::thread::{Thread, ThreadState};
use crate::env::value::Value;
use crate::instruction::UpValueDescriptor;
use crate::vm::abi::Handler;
use crate::vm::native::Execution;
use crate::vm::native::native_call;

/// Debug record for a local register: active for `start_pc <= pc < end_pc`.
/// Hidden loop-control slots appear as `(for state)` like luac's, so every
/// register a frame keeps live has a record.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct LocVar<'gc> {
    pub name: LuaString<'gc>,
    pub start_pc: u32,
    pub end_pc: u32,
}

/// A compiled Lua function. Immutable once created.
/// Shared by all closures created from the same function definition.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct Prototype<'gc> {
    #[collect(require_static)]
    pub code: Code,
    pub constants: Box<[Value<'gc>]>,
    pub prototypes: Box<[Gc<'gc, Prototype<'gc>>]>,
    #[collect(require_static)]
    pub upvalue_desc: Box<[UpValueDescriptor]>,
    pub num_params: u8,
    pub is_vararg: bool,
    /// Lua 5.5 `PF_VATAB`: a named vararg parameter that escaped the optimized
    /// below-base form, so `VARARGPREP` materializes a real table in
    /// `R[num_params]`. When `false`, varargs stay below-base.
    pub needs_vararg_table: bool,
    pub max_stack_size: u8,
    pub num_upvalues: u8,
    /// Chunk name as given to `load` (`@file`, `=name`, or the source text
    /// itself, which is `load`'s default); shared by nested prototypes.
    pub source: LuaString<'gc>,
    /// Lines of the `function` keyword and its `end`; both 0 for a main chunk.
    pub line_defined: u32,
    pub last_line_defined: u32,
    /// Source line per instruction, parallel to `code`.
    pub(crate) lineinfo: Box<[u32]>,
    /// Named locals in declaration order with their live pc ranges.
    pub locvars: Box<[LocVar<'gc>]>,
    /// Parallel to `upvalue_desc`.
    pub upvalue_names: Box<[LuaString<'gc>]>,
    /// Inline-cache table indexed by `ic_idx` embedded in
    /// GETTABUP/SETTABUP/GETFIELD/SETFIELD/SELF instructions. One entry
    /// per cache site (call site, not instruction count). The slice
    /// lives inline in the prototype (no separate `Gc` allocation,
    /// no `RefLock`); per-slot `Lock<InlineCache>` exposes
    /// counter-free reads via `get()` and barrier-aware writes via
    /// the parent `Prototype`'s `Gc`. Entries are [`InlineCache`].
    pub ic_table: IcTable<'gc>,
    /// Per distinct constructor template, indexed by `NEWTABLE`.
    pub templates: Box<[Template<'gc>]>,
}

/// A prototype's bytecode, in cells: the interpreter rewrites an
/// instruction in place to the variant specialized for what its inline cache
/// holds (quickening).
pub struct Code(Box<[std::cell::Cell<crate::instruction::Instruction>]>);

impl Code {
    pub fn new(code: Box<[crate::instruction::Instruction]>) -> Self {
        Code(code.into_iter().map(std::cell::Cell::new).collect())
    }

    /// The first instruction, for the interpreter's raw reads (and writes,
    /// which the cells permit).
    #[inline(always)]
    pub fn as_ptr(&self) -> *const crate::instruction::Instruction {
        self.0.as_ptr().cast()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = crate::instruction::Instruction> + '_ {
        self.0.iter().map(std::cell::Cell::get)
    }
}

/// What `NEWTABLE` starts a constructor's table with: the shape of its
/// constant field names in order, per slot the value of a constant field
/// nothing else in the constructor writes, else nil, and room in the array
/// part for its positional items.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct Template<'gc> {
    pub shape: Shape<'gc>,
    pub values: Box<[Value<'gc>]>,
    /// Positional items, not counting a trailing call or `...`.
    pub items: u32,
}

impl<'gc> Prototype<'gc> {
    pub fn line_for_pc(&self, pc: usize) -> Option<u32> {
        self.lineinfo.get(pc).copied()
    }
}

/// Per-call-site inline cache for one constant string key, keyed on the
/// receiver's shape. `Empty` until a slow path fills it.
///
/// Metamethods are read live through `Shape::has_mm`, so a cached shape
/// stays a valid identity as its metatable's metamethods change; only
/// `ProtoLoad`, which caches what `__index` names, also checks
/// `MtCache::index_table`.
// A power-of-two size keeps indexing `ic_table` to a shift.
#[derive(Clone, Copy, Collect, Default)]
#[collect(internal, no_drop)]
#[repr(align(32))]
pub enum InlineCache<'gc> {
    // First, so the hottest hit tests a zero discriminant.
    /// Tables of `shape` hold the key at `loc`.
    Own {
        shape: Shape<'gc>,
        #[collect(require_static)]
        loc: SlotLoc,
    },
    /// Tables of `shape` lack the key: a load is nil unless `__index` fires.
    Absent { shape: Shape<'gc> },
    /// Tables of `from` lack the key; adding it moves them to `to`, which
    /// holds it at `loc`, the slot after `from`'s last.
    Transition {
        from: Shape<'gc>,
        to: Shape<'gc>,
        #[collect(require_static)]
        loc: SlotLoc,
    },
    /// Tables of `recv` lack the key, and while their metatable's `__index`
    /// is `holder`, it holds the key at `loc` if its shape is `holder_shape`.
    /// `holder` is weak, as a metatable's `__index` may be.
    ProtoLoad {
        recv: Shape<'gc>,
        holder: GcWeak<'gc, RefLock<TableState<'gc>>>,
        holder_shape: Shape<'gc>,
        #[collect(require_static)]
        loc: SlotLoc,
    },
    #[default]
    Empty,
}

const _: () = assert!(size_of::<InlineCache<'static>>() == 32);

/// [`Prototype::ic_table`]. Its trace empties the `ProtoLoad` entries whose
/// holder was dropped, so they stop reserving its allocation.
pub struct IcTable<'gc>(Box<[Lock<InlineCache<'gc>>]>);

impl<'gc> IcTable<'gc> {
    pub fn new(len: usize) -> Self {
        IcTable(vec![Lock::new(InlineCache::Empty); len].into_boxed_slice())
    }
}

impl<'gc> std::ops::Deref for IcTable<'gc> {
    type Target = [Lock<InlineCache<'gc>>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// SAFETY: traces every entry it keeps. Emptying one from `trace(&self)` adopts
// nothing, and the collector never runs while the mutator reads an entry.
unsafe impl<'gc> Collect<'gc> for IcTable<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        for slot in self.iter() {
            let entry = slot.get();
            if let InlineCache::ProtoLoad { holder, .. } = entry
                && holder.is_dropped()
            {
                unsafe { slot.as_cell() }.set(InlineCache::Empty);
            } else {
                cc.trace(&entry);
            }
        }
    }
}

/// A captured local: open while its frame lives, `v` pointing at its slot in
/// `thread`'s stack (rebased when the stack moves), then closed, `v` pointing
/// at `closed`. Reads are one load through `v`, whichever state.
pub struct UpvalueCell<'gc> {
    v: Cell<*mut Value<'gc>>,
    closed: Cell<Value<'gc>>,
    /// The thread whose stack `v` points into, while open.
    thread: Cell<Option<Thread<'gc>>>,
}

pub type Upvalue<'gc> = Gc<'gc, UpvalueCell<'gc>>;

/// A Lua closure's upvalue: the value itself if its descriptor is
/// `by_value`, else the cell it shares.
#[derive(Clone, Copy)]
pub union UpvalueSlot<'gc> {
    pub(crate) value: Value<'gc>,
    pub(crate) cell: Upvalue<'gc>,
}

// SAFETY: `closed` and `thread` are the only Gc pointers, and every write that
// may make either hold a new one goes through a barrier on the cell.
unsafe impl<'gc> Collect<'gc> for UpvalueCell<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        cc.trace(&self.closed.get());
        cc.trace(&self.thread.get());
    }
}

impl<'gc> UpvalueCell<'gc> {
    /// An open upvalue for `slot` of `thread`'s stack.
    pub(crate) fn new_open(
        mc: &Mutation<'gc>,
        thread: Thread<'gc>,
        slot: *mut Value<'gc>,
    ) -> Upvalue<'gc> {
        Gc::new(
            mc,
            UpvalueCell {
                v: Cell::new(slot),
                closed: Cell::new(Value::nil()),
                thread: Cell::new(Some(thread)),
            },
        )
    }

    pub(crate) fn new_closed(mc: &Mutation<'gc>, value: Value<'gc>) -> Upvalue<'gc> {
        let uv = Gc::new(
            mc,
            UpvalueCell {
                v: Cell::new(std::ptr::null_mut()),
                closed: Cell::new(value),
                thread: Cell::new(None),
            },
        );
        uv.v.set(uv.closed.as_ptr());
        uv
    }

    #[inline(always)]
    pub(crate) fn get(&self) -> Value<'gc> {
        // SAFETY: `v` is a live slot of `thread`'s stack while open (closed
        // before the frame goes, rebased when the stack moves), else `closed`.
        unsafe { *self.v.get() }
    }

    /// The object a store into this cell must barrier: the cell when closed,
    /// the owning thread when open on another thread's stack, none when open
    /// on `running`, whose state is borrowed for dispatch.
    #[inline(always)]
    pub(crate) fn barrier_target(this: Upvalue<'gc>, running: Thread<'gc>) -> Option<Gc<'gc, ()>> {
        match this.thread.get() {
            None => Some(Gc::erase(this)),
            Some(t) if !t.ptr_eq(running) => Some(Gc::erase(t.inner())),
            Some(_) => None,
        }
    }

    /// Store `value` after the caller ran the barrier of [`Self::barrier_target`].
    ///
    /// # Safety
    /// The barrier was handled.
    #[inline(always)]
    pub(crate) unsafe fn set_barriered(this: Upvalue<'gc>, value: Value<'gc>) {
        // SAFETY: as in `get`.
        unsafe { *this.v.get() = value };
    }

    /// The stack slot of an open upvalue.
    #[inline(always)]
    pub(crate) fn slot(&self) -> *mut Value<'gc> {
        debug_assert!(self.thread.get().is_some());
        self.v.get()
    }

    /// Point an open upvalue into its stack's new buffer, which moved from
    /// address `old`.
    #[inline]
    pub(crate) fn rebase(&self, old: usize, new: *mut Value<'gc>) {
        let index = (self.v.get().addr() - old) / size_of::<Value>();
        // SAFETY: the slot moved with the rest of the stack.
        self.v.set(unsafe { new.add(index) });
    }

    #[inline]
    pub(crate) fn close(this: Upvalue<'gc>, mc: &Mutation<'gc>) {
        mc.backward_barrier(Gc::erase(this), None);
        this.closed.set(this.get());
        this.v.set(this.closed.as_ptr());
        this.thread.set(None);
    }
}

/// A Lua closure: bytecode, with its upvalues in the cell after it.
pub struct LuaClosure<'gc> {
    pub proto: Gc<'gc, Prototype<'gc>>,
    // Copies of the `proto` fields CALL needs, so entering a function is one
    // dependent load shorter (closure -> code, not closure -> proto -> code).
    // The pointers stay valid because `proto` is immutable and kept alive by
    // this closure; they are not traced (the `Gc` above is).
    pub code: *const crate::instruction::Instruction,
    pub constants: *const Value<'gc>,
    pub ic_table: *const Lock<InlineCache<'gc>>,
    pub max_stack_size: u8,
    pub num_params: u8,
    pub is_vararg: bool,
    /// `num_params`, or `u16::MAX` for a vararg function: a call passing
    /// fewer arguments than this needs the fixups (`nargs < fixed_arity`;
    /// a CALL compares `nargs + 1 <= fixed_arity`). One compare for both
    /// tests, and no argument count reaches the vararg value.
    pub fixed_arity: u16,
    /// Here rather than read from `proto`, which the collector may have freed
    /// by the time it sizes this cell.
    num_upvalues: u8,
}

impl<'gc> LuaClosure<'gc> {
    pub fn upvalues(&self) -> &[UpvalueSlot<'gc>] {
        // SAFETY: a `LuaClosure` only exists as a `FunctionKind::Lua`, allocated
        // with its upvalues after it (`Function::new_lua`).
        unsafe {
            let fk = (self as *const Self).byte_sub(std::mem::offset_of!(FunctionKind, Lua.0));
            let p = Gc::<FunctionKind>::trailing_ptr_of(&*fk.cast::<FunctionKind>());
            std::slice::from_raw_parts(p.cast().as_ptr(), self.num_upvalues as usize)
        }
    }
}

/// A native closure: a Rust function, with its upvalues in the cell after it.
pub struct NativeClosure<'gc> {
    pub(crate) function: NativeKind,
    /// What CALL and TAILCALL jump to with this closure in the `closure` slot:
    /// `native_call`, or for a builtin with a fast path its own entry
    /// (LuaJIT's `ff_*`), which handles the common argument shape inline,
    /// never errors, and leaves every other shape to `native_call`, so
    /// `function` stays the complete implementation. An entry must check the
    /// opcode: after a TAILCALL it returns its results from the frame instead
    /// of dispatching the next instruction.
    pub(crate) entry: Handler,
    num_upvalues: u32,
    _upvalues: std::marker::PhantomData<Value<'gc>>,
}

impl<'gc> NativeClosure<'gc> {
    pub fn upvalues(&self) -> &[Value<'gc>] {
        // SAFETY: as in `LuaClosure::upvalues`.
        unsafe {
            let fk = (self as *const Self).byte_sub(std::mem::offset_of!(FunctionKind, Native.0));
            let p = Gc::<FunctionKind>::trailing_ptr_of(&*fk.cast::<FunctionKind>());
            std::slice::from_raw_parts(p.cast().as_ptr(), self.num_upvalues as usize)
        }
    }
}

/// A native function: reads its arguments from `stack` and leaves its results
/// there in their place. `closure` is its own closure, which carries its
/// upvalues; the running thread is `stack.exec()`. Three arguments rather than
/// one context struct, because a struct wider than two words is passed through
/// memory.
///
/// An error (any Lua value) unwinds to the nearest `pcall` or to the host as
/// `RuntimeError::Lua`.
pub type NativeFn = for<'gc, 'a> fn(
    ctx: Context<'gc>,
    closure: &'a NativeClosure<'gc>,
    stack: Stack<'gc, 'a>,
) -> Result<(), Error<'gc>>;

// One register: a plain native's return never goes through memory.
const _: () = assert!(std::mem::size_of::<Result<(), Error<'static>>>() == 8);

#[derive(Clone, Copy)]
pub(crate) enum NativeKind {
    Plain(NativeFn),
    Cont(crate::vm::native::ContFn),
    Async(crate::vm::async_native::AsyncFn),
}

/// A mutable view into the running thread's value stack, spanning
/// `stack[bottom..*top]`. The callback sees `stack[0..len()]` as its
/// arguments on entry; any values it leaves in the window (via `push`,
/// `extend`, or `replace`) become the callback's return values.
///
/// The logical window length is carried by `*top` (an alias of
/// `thread.top`), NOT by `Vec::len()`: the backing vec is treated as
/// storage kept at its high-water length and is never shrunk by a
/// callback. The result count is therefore signalled through the logical
/// top, which the invoker reads back as `*top - bottom` after the call —
/// decoupling the window from the shared vec so a native call can never
/// truncate it below an outer frame's register window.
pub struct Stack<'gc, 'a> {
    /// The owning thread: `thread.stack` is the storage and `thread.top` the
    /// authoritative logical top. One reference rather than two so the view
    /// is two words and is passed to natives in registers.
    thread: &'a mut ThreadState<'gc>,
    bottom: usize,
}

impl<'gc, 'a> Stack<'gc, 'a> {
    #[inline]
    pub(crate) fn new(thread: &'a mut ThreadState<'gc>, bottom: usize) -> Self {
        debug_assert!(bottom <= thread.top && thread.top <= thread.stack.len());
        Stack { thread, bottom }
    }

    /// Give the thread back.
    #[inline]
    pub(crate) fn into_parts(self) -> (&'a mut ThreadState<'gc>, usize) {
        (self.thread, self.bottom)
    }

    /// The thread this stack belongs to.
    #[inline]
    pub(crate) fn thread_mut(&mut self) -> &mut ThreadState<'gc> {
        self.thread
    }

    /// The executor state of the thread this stack belongs to.
    #[inline]
    pub fn exec(&self) -> Execution<'gc> {
        Execution::new(self.thread.handle(), self.thread.main)
    }

    /// Stack-bottom index relative to the underlying vec.
    #[inline]
    pub fn bottom(&self) -> usize {
        self.bottom
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.thread.top - self.bottom
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.thread.top == self.bottom
    }

    /// The argument at `i`, or `None` past the logical top (`lua_isnone`).
    #[inline]
    pub fn arg(&self, i: usize) -> Option<Value<'gc>> {
        (i < self.len()).then(|| self.get(i))
    }

    /// Read the value at index `i` within the callback's window, or `Nil`
    /// if `i` is past the logical top. Mirrors Lua's "missing args are
    /// nil" rule.
    #[inline]
    pub fn get(&self, i: usize) -> Value<'gc> {
        let idx = self.bottom + i;
        if idx < self.thread.top {
            // `top <= values.len()` is rule 1 of the stack invariant.
            debug_assert!(idx < self.thread.stack.len());
            unsafe { *self.thread.stack.get_unchecked(idx) }
        } else {
            Value::nil()
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[Value<'gc>] {
        &self.thread.stack[self.bottom..self.thread.top]
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Value<'gc>] {
        &mut self.thread.stack[self.bottom..self.thread.top]
    }

    /// Discard everything in the window (args included). Lowers the logical
    /// top without shrinking the backing vec.
    #[inline]
    pub fn clear(&mut self) {
        self.thread.top = self.bottom;
    }

    #[inline]
    pub fn push(&mut self, v: Value<'gc>) {
        let top = self.thread.top;
        self.thread.ensure_slots(top + 1);
        self.thread.stack[top] = v;
        self.thread.top = top + 1;
    }

    /// Remove and return the top value, or `Nil` if the window is empty.
    #[inline]
    pub fn pop(&mut self) -> Value<'gc> {
        if self.is_empty() {
            return Value::nil();
        }
        self.thread.top -= 1;
        self.thread.stack[self.thread.top]
    }

    #[inline]
    pub fn extend<I: IntoIterator<Item = Value<'gc>>>(&mut self, iter: I) {
        for v in iter {
            self.push(v);
        }
    }

    /// Shift `stack[i..]` up one and put `v` at `i`.
    #[inline]
    pub fn insert(&mut self, i: usize, v: Value<'gc>) {
        let at = self.bottom + i;
        let top = self.thread.top;
        debug_assert!(at <= top);
        self.thread.ensure_slots(top + 1);
        let stack = &mut self.thread.stack;
        // A few values, the usual case, move faster than a `memmove` call.
        if top - at <= 8 {
            for j in (at..top).rev() {
                stack[j + 1] = stack[j];
            }
        } else {
            stack.copy_within(at..top, at + 1);
        }
        stack[at] = v;
        self.thread.top += 1;
    }

    /// Remove the value at `i`, shifting `stack[i + 1..]` down one.
    #[inline]
    pub fn remove(&mut self, i: usize) {
        let at = self.bottom + i;
        let top = self.thread.top;
        debug_assert!(at < top);
        self.thread.stack.copy_within(at + 1..top, at);
        self.thread.top -= 1;
    }

    /// Drop everything above the first `n` values (no-op if shorter).
    #[inline]
    pub fn truncate(&mut self, n: usize) {
        let at = self.bottom + n;
        if at < self.thread.top {
            self.thread.top = at;
        }
    }

    /// How many Lua frames the running thread has. Hidden: a hook for tests.
    #[doc(hidden)]
    pub fn lua_frame_count(&self) -> usize {
        crate::vm::frame::frames(self.thread)
            .filter(|f| !f.is_native())
            .count()
    }

    /// Replace the whole window (args included) with `values`: the common
    /// "return these" shape. Results overwrite the args in place.
    #[inline]
    pub fn replace(&mut self, values: &[Value<'gc>]) {
        let end = self.bottom + values.len();
        self.thread.ensure_slots(end);
        for (i, v) in values.iter().enumerate() {
            self.thread.stack[self.bottom + i] = *v;
        }
        self.thread.top = end;
    }

    /// Whether `n` more values fit above the top within the thread's stack
    /// limit (`lua_checkstack`). A native whose result count depends on its
    /// arguments checks before pushing.
    #[inline]
    pub fn check_stack(&self, n: usize) -> bool {
        n <= self.thread.stack_limit.saturating_sub(self.thread.top)
    }

    /// `replace` with `n` slots, or `None` unless `check_stack(n)`. The slots
    /// hold stale values inside the window: write every one or `truncate`.
    #[inline]
    pub fn replace_slots(&mut self, n: usize) -> Option<&mut [Value<'gc>]> {
        if !self.check_stack(n) {
            return None;
        }
        let end = self.bottom + n;
        self.thread.ensure_slots(end);
        self.thread.top = end;
        Some(&mut self.thread.stack[self.bottom..end])
    }

    /// `replace(&[v])` without going through memory: a by-value `Value` stays
    /// in a register, whereas a one-element slice is spilled to the stack and
    /// read back through a pointer, with a copy loop around it.
    #[inline(always)]
    pub fn ret1(&mut self, v: Value<'gc>) {
        let end = self.bottom + 1;
        self.thread.ensure_slots(end);
        self.thread.stack[self.bottom] = v;
        self.thread.top = end;
    }
}

impl<'gc, 'a> std::ops::Index<usize> for Stack<'gc, 'a> {
    type Output = Value<'gc>;
    #[inline]
    fn index(&self, i: usize) -> &Value<'gc> {
        let idx = self.bottom + i;
        debug_assert!(idx < self.thread.top);
        &self.thread.stack[idx]
    }
}

/// Copy wrapper stored in Value. Single Gc pointer for size efficiency.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Function<'gc>(Gc<'gc, FunctionKind<'gc>>);

/// Followed in its cell by the closure's upvalues: `UpvalueSlot`s for a Lua
/// closure, `Value`s for a native.
pub enum FunctionKind<'gc> {
    /// Inline, not behind another `Gc`: CALL reaches the closure's `code`
    /// with one load fewer.
    Lua(LuaClosure<'gc>),
    Native(NativeClosure<'gc>),
}

// SAFETY: traces `proto` and the upvalues; a `LuaClosure`'s raw pointers alias
// data `proto` owns.
unsafe impl<'gc> Collect<'gc> for FunctionKind<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        match self {
            FunctionKind::Lua(c) => {
                cc.trace(&c.proto);
                for (slot, desc) in c.upvalues().iter().zip(&c.proto.upvalue_desc) {
                    // SAFETY: `by_value` says which field CLOSURE wrote.
                    unsafe {
                        if desc.by_value {
                            cc.trace(&slot.value);
                        } else {
                            cc.trace(&slot.cell);
                        }
                    }
                }
            }
            FunctionKind::Native(c) => cc.trace(c.upvalues()),
        }
    }
}

// SAFETY: only `Function::new_lua` and `new_native_with_entry` allocate one,
// with this many bytes; no drop glue.
unsafe impl<'gc> crate::dmm::TrailingBytes for FunctionKind<'gc> {
    #[inline(always)]
    fn trailing_len(&self) -> usize {
        8 * match self {
            FunctionKind::Lua(c) => c.num_upvalues as usize,
            FunctionKind::Native(c) => c.num_upvalues as usize,
        }
    }
}

const _: () = assert!(size_of::<UpvalueSlot<'static>>() == 8 && size_of::<Value<'static>>() == 8);

/// A `Function` known to hold a Lua closure. Derefs to the closure without
/// re-checking the kind, so frames can keep one pointer and still reach
/// `proto`, `upvalues` and the CALL-path copies directly.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct LuaFn<'gc>(Gc<'gc, FunctionKind<'gc>>);

impl<'gc> LuaFn<'gc> {
    /// # Safety
    /// `f` must hold `FunctionKind::Lua`.
    #[inline(always)]
    pub unsafe fn from_function_unchecked(f: Function<'gc>) -> Self {
        debug_assert!(matches!(&*f.0, FunctionKind::Lua(_)));
        LuaFn(f.0)
    }

    pub fn function(self) -> Function<'gc> {
        Function(self.0)
    }

    /// The first upvalue: a constant offset from the closure.
    #[inline(always)]
    pub(crate) fn upvalue_ptr(self) -> *mut UpvalueSlot<'gc> {
        Gc::trailing_ptr(self.0).cast().as_ptr()
    }

    /// The closure's cell, for a frame header word (`vm::frame`).
    #[inline(always)]
    pub(crate) fn as_ptr(self) -> *const FunctionKind<'gc> {
        Gc::as_ptr(self.0)
    }

    /// # Safety
    /// `p` came from [`as_ptr`](Self::as_ptr) on a live closure.
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(p: *const FunctionKind<'gc>) -> Self {
        LuaFn(unsafe { Gc::from_ptr(p) })
    }
}

impl<'gc> std::ops::Deref for LuaFn<'gc> {
    type Target = LuaClosure<'gc>;
    #[inline(always)]
    fn deref(&self) -> &LuaClosure<'gc> {
        match &*self.0 {
            FunctionKind::Lua(c) => c,
            // SAFETY: the constructor's contract.
            FunctionKind::Native(_) => unsafe { std::hint::unreachable_unchecked() },
        }
    }
}

impl<'gc> Function<'gc> {
    /// A Lua closure of `proto`, `init` writing its `proto.num_upvalues`
    /// upvalues, each the field `proto.upvalue_desc` says.
    #[inline]
    pub fn new_lua(
        mc: &Mutation<'gc>,
        proto: Gc<'gc, Prototype<'gc>>,
        init: impl FnOnce(*mut UpvalueSlot<'gc>),
    ) -> Self {
        let closure = LuaClosure {
            proto,
            code: proto.code.as_ptr(),
            constants: proto.constants.as_ptr(),
            ic_table: proto.ic_table.as_ptr(),
            max_stack_size: proto.max_stack_size,
            num_params: proto.num_params,
            is_vararg: proto.is_vararg,
            fixed_arity: if proto.is_vararg {
                u16::MAX
            } else {
                proto.num_params as u16
            },
            num_upvalues: proto.num_upvalues,
        };
        // SAFETY: `init` writes every upvalue.
        Function(unsafe {
            Gc::new_with_trailing(mc, FunctionKind::Lua(closure), |p| init(p.cast().as_ptr()))
        })
    }

    pub fn new_native(mc: &Mutation<'gc>, function: NativeFn, upvalues: &[Value<'gc>]) -> Self {
        Self::new_native_with_entry(mc, NativeKind::Plain(function), upvalues, native_call)
    }

    /// A continuation native: crate-internal, since it names its
    /// continuations by [`CONT_TABLE`](crate::vm::native::CONT_TABLE) index.
    pub(crate) fn new_cont(
        mc: &Mutation<'gc>,
        function: crate::vm::native::ContFn,
        upvalues: &[Value<'gc>],
    ) -> Self {
        Self::new_native_with_entry(mc, NativeKind::Cont(function), upvalues, native_call)
    }

    pub fn new_async(
        mc: &Mutation<'gc>,
        function: crate::vm::async_native::AsyncFn,
        upvalues: &[Value<'gc>],
    ) -> Self {
        Self::new_native_with_entry(mc, NativeKind::Async(function), upvalues, native_call)
    }

    /// A native whose CALLs go to `entry` (see `NativeClosure::entry`).
    pub(crate) fn new_native_with_entry(
        mc: &Mutation<'gc>,
        function: NativeKind,
        upvalues: &[Value<'gc>],
        entry: Handler,
    ) -> Self {
        let closure = NativeClosure {
            function,
            entry,
            num_upvalues: u32::try_from(upvalues.len()).expect("too many upvalues"),
            _upvalues: std::marker::PhantomData,
        };
        // SAFETY: copies every upvalue.
        Function(unsafe {
            Gc::new_with_trailing(mc, FunctionKind::Native(closure), |p| {
                std::ptr::copy_nonoverlapping(upvalues.as_ptr(), p.cast().as_ptr(), upvalues.len())
            })
        })
    }

    pub fn as_lua(self) -> Option<LuaFn<'gc>> {
        match &*self.0 {
            FunctionKind::Lua(_) => Some(LuaFn(self.0)),
            _ => None,
        }
    }

    pub fn as_native(self) -> Option<&'gc NativeClosure<'gc>> {
        match self.0.as_ref() {
            FunctionKind::Native(nc) => Some(nc),
            _ => None,
        }
    }

    pub fn inner(self) -> Gc<'gc, FunctionKind<'gc>> {
        self.0
    }

    pub(crate) fn from_inner(g: Gc<'gc, FunctionKind<'gc>>) -> Self {
        Function(g)
    }
}
