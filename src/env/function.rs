use crate::Context;
use crate::dmm::{Collect, Gc, Lock, Mutation, RefLock, Trace};
use crate::env::error::Error;
use crate::env::shape::Shape;
use crate::env::string::LuaString;
use crate::env::value::Value;
use crate::instruction::UpValueDescriptor;
use crate::vm::sequence::{CallbackAction, Execution};

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
    pub code: Box<[crate::instruction::Instruction]>,
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
    /// GETTABUP/SETTABUP/GETFIELD/SETFIELD instructions. One entry
    /// per cache site (call site, not instruction count). The slice
    /// lives inline in the prototype (no separate `Gc` allocation,
    /// no `RefLock`); per-slot `Lock<InlineCache>` exposes
    /// counter-free reads via `get()` and barrier-aware writes via
    /// the parent `Prototype`'s `Gc`. See `src/env/shape/mod.rs` for
    /// the IC payload.
    pub ic_table: Box<[Lock<InlineCache<'gc>>]>,
}

impl<'gc> Prototype<'gc> {
    pub fn line_for_pc(&self, pc: usize) -> Option<u32> {
        self.lineinfo.get(pc).copied()
    }
}

/// Per-call-site monomorphic inline cache. `Empty` initially; a slow
/// path fills it on first miss with the observed shape and slot. Future
/// hits skip the metatable lookup entirely.
///
/// Metatable-mutation tracking is handled by `Shape::has_mm`, which
/// reads the live `MtCache` bitset on the metatable. Bits are updated
/// in place by every metamethod-named write to the metatable, so a
/// `Shape` pointer cached here remains a valid identity even as the
/// metatable's metamethod set evolves.
#[derive(Clone, Copy, Collect, Default)]
#[collect(internal, no_drop)]
pub enum InlineCache<'gc> {
    #[default]
    Empty,
    Mono {
        /// Shape pointer the cache was filled against.
        shape: Shape<'gc>,
        /// Slot index in `TableState::properties`. `u32::MAX` =
        /// "key absent in shape" (so a get returns the metamethod
        /// chain on this branch and a set must transition).
        #[collect(require_static)]
        slot: u32,
    },
}

impl<'gc> InlineCache<'gc> {
    pub const ABSENT_SLOT: u32 = u32::MAX;
}

/// An upvalue — open (references a stack slot) or closed (owns the value).
#[derive(Collect)]
#[collect(internal, no_drop)]
pub enum UpvalueState<'gc> {
    Open {
        thread: crate::env::thread::Thread<'gc>,
        index: usize,
    },
    Closed(Value<'gc>),
}

pub type Upvalue<'gc> = Gc<'gc, RefLock<UpvalueState<'gc>>>;

/// A Lua closure (bytecode + upvalues).
pub struct LuaClosure<'gc> {
    pub proto: Gc<'gc, Prototype<'gc>>,
    pub upvalues: Box<[Upvalue<'gc>]>,
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
}

// SAFETY: `proto` and `upvalues` are the only owned Gc pointers; the raw
// pointers alias data owned by `proto`.
unsafe impl<'gc> Collect<'gc> for LuaClosure<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        cc.trace(&self.proto);
        cc.trace(&self.upvalues);
    }
}

/// A native closure (Rust function + optional upvalues).
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct NativeClosure<'gc> {
    #[collect(require_static)]
    pub function: NativeFn,
    pub upvalues: Box<[Value<'gc>]>,
    /// Interpreter fast path, if the builtin has one (see `op_call_fast`).
    #[collect(require_static)]
    pub fast: FastCall,
}

/// Builtins the interpreter can run inline for their common argument shape
/// (LuaJIT's `ff_*` fast functions). `op_call_fast` handles exactly that
/// shape and never errors; anything else falls back to `NativeClosure::function`,
/// which is the complete implementation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum FastCall {
    None = 0,
    /// `math.sqrt(float)`
    Sqrt,
    /// `math.abs(number)`
    Abs,
    /// `math.floor(number)`
    Floor,
    /// `math.ceil(number)`
    Ceil,
}

/// Signature of a native callback invoked by the VM on `CALL` / `TAILCALL`.
///
/// Arguments are read from the `Stack` view; return values are produced by
/// leaving them on the stack above `bottom`. The returned [`CallbackAction`]
/// tells the executor what to do next:
///   - `Return` keeps the hot path (sync return, results in the stack window).
///   - `Sequence`, `Call`, `Yield`, `Resume` are the suspension paths handled
///     by the executor's driver loop.
///
/// On error, return [`Error`] (any Lua value); the executor unwinds Lua
/// frames until a `Sequence` catcher (e.g. `pcall`) handles it, or surfaces
/// it to the host as `RuntimeError::Lua`.
pub type NativeFn = for<'gc, 'a> fn(
    ctx: NativeContext<'gc, 'a>,
    stack: Stack<'gc, 'a>,
) -> Result<CallbackAction<'gc>, Error<'gc>>;

/// Contextual handles passed to a native callback alongside its `Stack`.
pub struct NativeContext<'gc, 'a> {
    pub ctx: Context<'gc>,
    pub upvalues: &'a [Value<'gc>],
    pub exec: Execution<'gc, 'a>,
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
    values: &'a mut Vec<Value<'gc>>,
    /// Authoritative logical top (an alias of `thread.top`). Mutators
    /// update it; read accessors bound by it.
    top: &'a mut usize,
    bottom: usize,
}

impl<'gc, 'a> Stack<'gc, 'a> {
    #[inline]
    pub(crate) fn new(values: &'a mut Vec<Value<'gc>>, top: &'a mut usize, bottom: usize) -> Self {
        debug_assert!(bottom <= *top && *top <= values.len());
        Stack {
            values,
            top,
            bottom,
        }
    }

    /// Destructure the borrowed view back into its underlying parts. Used
    /// by `async_sequence` to ferry the live stack (and logical top)
    /// through a `SharedSlot`.
    #[inline]
    pub(crate) fn into_parts(self) -> (&'a mut Vec<Value<'gc>>, &'a mut usize, usize) {
        (self.values, self.top, self.bottom)
    }

    /// Stack-bottom index relative to the underlying vec.
    #[inline]
    pub fn bottom(&self) -> usize {
        self.bottom
    }

    #[inline]
    pub fn len(&self) -> usize {
        *self.top - self.bottom
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        *self.top == self.bottom
    }

    /// Read the value at index `i` within the callback's window, or `Nil`
    /// if `i` is past the logical top. Mirrors Lua's "missing args are
    /// nil" rule.
    #[inline]
    pub fn get(&self, i: usize) -> Value<'gc> {
        let idx = self.bottom + i;
        if idx < *self.top {
            // `top <= values.len()` is rule 1 of the stack invariant.
            debug_assert!(idx < self.values.len());
            unsafe { *self.values.get_unchecked(idx) }
        } else {
            Value::nil()
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[Value<'gc>] {
        &self.values[self.bottom..*self.top]
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Value<'gc>] {
        &mut self.values[self.bottom..*self.top]
    }

    /// Discard everything in the window (args included). Lowers the logical top
    /// without shrinking the backing vec; the discarded slots are nil-filled,
    /// since leaving them set would let the GC trace still reach them.
    #[inline]
    pub fn clear(&mut self) {
        self.values[self.bottom..*self.top].fill(Value::nil());
        *self.top = self.bottom;
    }

    #[inline]
    pub fn push(&mut self, v: Value<'gc>) {
        if *self.top == self.values.len() {
            self.values.push(v);
        } else {
            self.values[*self.top] = v;
        }
        *self.top += 1;
    }

    #[inline]
    pub fn extend<I: IntoIterator<Item = Value<'gc>>>(&mut self, iter: I) {
        for v in iter {
            self.push(v);
        }
    }

    /// Replace the whole window (args included) with `values`: the common
    /// "return these" shape. Results overwrite the args in place; only slots
    /// the window loses are nil-filled (rule 3), so returning as many values as
    /// there were arguments fills nothing.
    #[inline]
    pub fn replace(&mut self, values: &[Value<'gc>]) {
        let end = self.bottom + values.len();
        if end > self.values.len() {
            self.values.resize(end, Value::nil());
        }
        for (i, v) in values.iter().enumerate() {
            self.values[self.bottom + i] = *v;
        }
        self.truncate_to(end);
    }

    /// `replace(&[v])` without going through memory: a by-value `Value` stays
    /// in registers, whereas a one-element slice is built on the stack and
    /// read back as one 16-byte load, which defeats store forwarding.
    #[inline(always)]
    pub fn ret1(&mut self, v: Value<'gc>) {
        let end = self.bottom + 1;
        if end > self.values.len() {
            self.values.resize(end, Value::nil());
        }
        self.values[self.bottom] = v;
        self.truncate_to(end);
    }

    /// Lower the window end to `end`, nil-filling what it vacates (rule 3).
    #[inline(always)]
    fn truncate_to(&mut self, end: usize) {
        if end < *self.top {
            self.values[end..*self.top].fill(Value::nil());
        }
        *self.top = end;
    }
}

impl<'gc, 'a> std::ops::Index<usize> for Stack<'gc, 'a> {
    type Output = Value<'gc>;
    #[inline]
    fn index(&self, i: usize) -> &Value<'gc> {
        let idx = self.bottom + i;
        debug_assert!(idx < *self.top);
        &self.values[idx]
    }
}

/// Copy wrapper stored in Value. Single Gc pointer for size efficiency.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Function<'gc>(Gc<'gc, FunctionKind<'gc>>);

#[derive(Collect)]
#[collect(internal, no_drop)]
pub enum FunctionKind<'gc> {
    /// Inline, not behind another `Gc`: CALL reaches the closure's `code`
    /// with one load fewer.
    Lua(LuaClosure<'gc>),
    Native(NativeClosure<'gc>),
}

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
    pub fn new_lua(
        mc: &Mutation<'gc>,
        proto: Gc<'gc, Prototype<'gc>>,
        upvalues: Box<[Upvalue<'gc>]>,
    ) -> Self {
        let closure = LuaClosure {
            proto,
            upvalues,
            code: proto.code.as_ptr(),
            constants: proto.constants.as_ptr(),
            ic_table: proto.ic_table.as_ptr(),
            max_stack_size: proto.max_stack_size,
            num_params: proto.num_params,
            is_vararg: proto.is_vararg,
        };
        Function(Gc::new(mc, FunctionKind::Lua(closure)))
    }

    pub fn new_native(mc: &Mutation<'gc>, function: NativeFn, upvalues: Box<[Value<'gc>]>) -> Self {
        Self::new_native_fast(mc, function, upvalues, FastCall::None)
    }

    pub fn new_native_fast(
        mc: &Mutation<'gc>,
        function: NativeFn,
        upvalues: Box<[Value<'gc>]>,
        fast: FastCall,
    ) -> Self {
        Function(Gc::new(
            mc,
            FunctionKind::Native(NativeClosure {
                function,
                upvalues,
                fast,
            }),
        ))
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
