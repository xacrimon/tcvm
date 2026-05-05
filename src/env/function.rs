use crate::Context;
use crate::dmm::{Collect, Gc, Lock, Mutation, RefLock};
use crate::env::shape::Shape;
use crate::env::string::LuaString;
use crate::env::value::Value;
use crate::instruction::UpValueDescriptor;

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
/// Metatable-mutation staleness is handled by `Shape::maybe_has_mm`
/// (called from the fast path when the slot value is nil) — its
/// `mm_cache` snapshot is filled by the same slow path that fills this
/// IC, so they always carry the same generation. Storing a separate
/// `mt_gen` here would only force fast-path misses for non-nil-value
/// accesses, where Lua semantics already bypass the metatable.
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
/// leaving them on the stack above `bottom`. Return `Ok(())` for a normal
/// return, `Err(NativeError)` to trigger a VM error (the message is dropped
/// at the boundary for now — see the `native_call` plan for future
/// plumbing).
pub type NativeFn =
    for<'gc, 'a> fn(ctx: NativeContext<'gc, 'a>, stack: Stack<'gc, 'a>) -> Result<(), NativeError>;

/// Contextual handles passed to a native callback alongside its `Stack`.
pub struct NativeContext<'gc, 'a> {
    pub ctx: Context<'gc>,
    pub upvalues: &'a [Value<'gc>],
}

/// A mutable view into the running thread's value stack, starting at
/// `bottom`. The callback sees `stack[0..len()]` as its arguments on entry;
/// any values it leaves on the stack (via `push`, `extend`, or `replace`)
/// become the callback's return values.
pub struct Stack<'gc, 'a> {
    values: &'a mut Vec<Value<'gc>>,
    bottom: usize,
}

impl<'gc, 'a> Stack<'gc, 'a> {
    #[inline]
    pub(crate) fn new(values: &'a mut Vec<Value<'gc>>, bottom: usize) -> Self {
        debug_assert!(bottom <= values.len());
        Stack { values, bottom }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len() - self.bottom
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.len() == self.bottom
    }

    /// Read the value at index `i` within the callback's window, or `Nil`
    /// if `i` is past the end. Mirrors Lua's "missing args are nil" rule.
    #[inline]
    pub fn get(&self, i: usize) -> Value<'gc> {
        self.values
            .get(self.bottom + i)
            .copied()
            .unwrap_or(Value::nil())
    }

    #[inline]
    pub fn as_slice(&self) -> &[Value<'gc>] {
        &self.values[self.bottom..]
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Value<'gc>] {
        &mut self.values[self.bottom..]
    }

    /// Discard everything in the window (args included).
    #[inline]
    pub fn clear(&mut self) {
        self.values.truncate(self.bottom);
    }

    #[inline]
    pub fn push(&mut self, v: Value<'gc>) {
        self.values.push(v);
    }

    #[inline]
    pub fn extend<I: IntoIterator<Item = Value<'gc>>>(&mut self, iter: I) {
        self.values.extend(iter);
    }

    /// Convenience for the common "clear args, push N results" pattern.
    #[inline]
    pub fn replace(&mut self, values: &[Value<'gc>]) {
        self.values.truncate(self.bottom);
        self.values.extend_from_slice(values);
    }
}

impl<'gc, 'a> std::ops::Index<usize> for Stack<'gc, 'a> {
    type Output = Value<'gc>;
    #[inline]
    fn index(&self, i: usize) -> &Value<'gc> {
        &self.values[self.bottom + i]
    }
}

/// Error returned by a [`NativeFn`] to abort execution.
#[derive(Debug, Clone)]
pub struct NativeError {
    pub message: String,
}

impl NativeError {
    pub fn new(message: impl Into<String>) -> Self {
        NativeError {
            message: message.into(),
        }
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
