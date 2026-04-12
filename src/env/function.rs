use crate::dmm::{Collect, Gc, Mutation, RefLock};
use crate::env::string::LuaString;
use crate::env::value::Value;

/// A compiled Lua function. Immutable once created.
/// Shared by all closures created from the same function definition.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct Prototype<'gc> {
    #[collect(require_static)]
    pub code: Box<[crate::instruction::Instruction]>,
    pub constants: Box<[Value<'gc>]>,
    pub prototypes: Box<[Gc<'gc, Prototype<'gc>>]>,
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
    pub function: fn(),
    pub upvalues: Box<[Value<'gc>]>,
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

    pub fn new_native(mc: &Mutation<'gc>, function: fn(), upvalues: Box<[Value<'gc>]>) -> Self {
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

    pub fn inner(self) -> Gc<'gc, FunctionKind<'gc>> {
        self.0
    }
}
