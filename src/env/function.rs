use crate::Context;
use crate::dmm::{Collect, Gc, Mutation, RefLock};
use crate::env::string::LuaString;
use crate::env::value::Value;
use crate::instruction::UpValueDescriptor;

/// A field-name string constant paired with its precomputed table hash.
///
/// Populated by the compiler for keys consumed by GETFIELD / SETFIELD /
/// GETTABUP / SETTABUP / ERRNNIL. The hash is produced with
/// `foldhash::fast::FixedState::default()` over a `Value::string(name)`
/// and matches what the table's BuildHasher will compute at lookup
/// time, so the hot-path lookup can skip rehashing.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct FieldConstant<'gc> {
    pub name: LuaString<'gc>,
    #[collect(require_static)]
    pub hash: u64,
}

/// A compiled Lua function. Immutable once created.
/// Shared by all closures created from the same function definition.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct Prototype<'gc> {
    #[collect(require_static)]
    pub code: Box<[crate::instruction::Instruction]>,
    pub constants: Box<[Value<'gc>]>,
    pub field_constants: Box<[FieldConstant<'gc>]>,
    pub prototypes: Box<[Gc<'gc, Prototype<'gc>>]>,
    #[collect(require_static)]
    pub upvalue_desc: Box<[UpValueDescriptor]>,
    pub num_params: u8,
    pub is_vararg: bool,
    pub max_stack_size: u8,
    pub num_upvalues: u8,
    pub source: Option<LuaString<'gc>>,
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
