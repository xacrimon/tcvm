use crate::Context;
use crate::dmm::{Collect, Gc, Lock, Mutation, RefLock};
use crate::env::error::Error;
use crate::env::shape::Shape;
use crate::env::string::LuaString;
use crate::env::value::Value;
use crate::instruction::UpValueDescriptor;
use crate::vm::sequence::{CallbackAction, Execution};

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
    pub source: Option<LuaString<'gc>>,
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
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct LuaClosure<'gc> {
    pub proto: Gc<'gc, Prototype<'gc>>,
    pub upvalues: Box<[Upvalue<'gc>]>,
}

/// A native closure (Rust function + optional upvalues).
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct NativeClosure<'gc> {
    #[collect(require_static)]
    pub function: NativeFn,
    pub upvalues: Box<[Value<'gc>]>,
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
            self.values[idx]
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

    /// Convenience for the common "clear args, push N results" pattern.
    #[inline]
    pub fn replace(&mut self, values: &[Value<'gc>]) {
        self.clear();
        self.extend(values.iter().copied());
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
    Lua(Gc<'gc, LuaClosure<'gc>>),
    Native(NativeClosure<'gc>),
}

impl<'gc> Function<'gc> {
    pub fn new_lua(
        mc: &Mutation<'gc>,
        proto: Gc<'gc, Prototype<'gc>>,
        upvalues: Box<[Upvalue<'gc>]>,
    ) -> Self {
        let closure = Gc::new(mc, LuaClosure { proto, upvalues });
        Function(Gc::new(mc, FunctionKind::Lua(closure)))
    }

    pub fn new_native(mc: &Mutation<'gc>, function: NativeFn, upvalues: Box<[Value<'gc>]>) -> Self {
        Function(Gc::new(
            mc,
            FunctionKind::Native(NativeClosure { function, upvalues }),
        ))
    }

    pub fn as_lua(self) -> Option<Gc<'gc, LuaClosure<'gc>>> {
        match &*self.0 {
            FunctionKind::Lua(cl) => Some(*cl),
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
