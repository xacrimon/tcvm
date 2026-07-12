//! The constant pool — the only part of the IR that touches `'gc`.
//!
//! Everything else in the IR is plain data: hashable (so GVN can key on an
//! instruction), printable, and constructible in a unit test with no arena. GC
//! references live here behind `u32` indices, which also confines rooting to a
//! single structure if compilation ever moves off the mutator thread.

use std::collections::HashMap;

use crate::dmm::{Collect, Gc};
use crate::env::function::Prototype;
use crate::env::shape::Shape;
use crate::env::string::LuaString;
use crate::env::value::Value;

macro_rules! pool_ref {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
        pub struct $name(pub u32);

        impl $name {
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

pool_ref!(
    /// A boxed Lua constant.
    ConstRef
);
pool_ref!(
    /// A concrete table shape. Never a dict sentinel — see `Refine::Shape`.
    ShapeRef
);
pool_ref!(
    /// A sub-prototype, for `CLOSURE`.
    ProtoRef
);
pool_ref!(
    /// An interned key string.
    StrRef
);

#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct ConstPool<'gc> {
    values: Vec<Value<'gc>>,
    shapes: Vec<Shape<'gc>>,
    protos: Vec<Gc<'gc, Prototype<'gc>>>,
    strings: Vec<LuaString<'gc>>,

    // Interning tables. Keyed by raw bits / pointer identity, so they hold no
    // `Gc` pointers of their own and need no tracing.
    #[collect(require_static)]
    value_dedup: HashMap<(u8, u64), ConstRef>,
    #[collect(require_static)]
    shape_dedup: HashMap<usize, ShapeRef>,
    #[collect(require_static)]
    proto_dedup: HashMap<usize, ProtoRef>,
    #[collect(require_static)]
    string_dedup: HashMap<usize, StrRef>,
}

impl<'gc> ConstPool<'gc> {
    pub fn new() -> Self {
        ConstPool {
            values: Vec::new(),
            shapes: Vec::new(),
            protos: Vec::new(),
            strings: Vec::new(),
            value_dedup: HashMap::new(),
            shape_dedup: HashMap::new(),
            proto_dedup: HashMap::new(),
            string_dedup: HashMap::new(),
        }
    }

    pub fn intern_value(&mut self, v: Value<'gc>) -> ConstRef {
        // `Value`'s own `Hash` folds only the payload, so two values with the
        // same bits but different tags would collide. Key on both.
        let key = (v.kind() as u8, value_bits(v));
        if let Some(&r) = self.value_dedup.get(&key) {
            return r;
        }
        let r = ConstRef(self.values.len() as u32);
        self.values.push(v);
        self.value_dedup.insert(key, r);
        r
    }

    /// Intern a shape. Rejects dict sentinels: they are shared by every dict
    /// table carrying the same metatable, so guarding one proves nothing about
    /// property layout. Callers treat `None` as "don't specialize here".
    pub fn intern_shape(&mut self, s: Shape<'gc>) -> Option<ShapeRef> {
        if s.is_dict() {
            return None;
        }
        let key = Gc::as_ptr(s.inner()) as usize;
        if let Some(&r) = self.shape_dedup.get(&key) {
            return Some(r);
        }
        let r = ShapeRef(self.shapes.len() as u32);
        self.shapes.push(s);
        self.shape_dedup.insert(key, r);
        Some(r)
    }

    pub fn intern_proto(&mut self, p: Gc<'gc, Prototype<'gc>>) -> ProtoRef {
        let key = Gc::as_ptr(p) as usize;
        if let Some(&r) = self.proto_dedup.get(&key) {
            return r;
        }
        let r = ProtoRef(self.protos.len() as u32);
        self.protos.push(p);
        self.proto_dedup.insert(key, r);
        r
    }

    pub fn intern_string(&mut self, s: LuaString<'gc>) -> StrRef {
        let key = Gc::as_ptr(s.inner()) as usize;
        if let Some(&r) = self.string_dedup.get(&key) {
            return r;
        }
        let r = StrRef(self.strings.len() as u32);
        self.strings.push(s);
        self.string_dedup.insert(key, r);
        r
    }

    pub fn value(&self, r: ConstRef) -> Value<'gc> {
        self.values[r.index()]
    }

    pub fn shape(&self, r: ShapeRef) -> Shape<'gc> {
        self.shapes[r.index()]
    }

    pub fn proto(&self, r: ProtoRef) -> Gc<'gc, Prototype<'gc>> {
        self.protos[r.index()]
    }

    pub fn string(&self, r: StrRef) -> LuaString<'gc> {
        self.strings[r.index()]
    }
}

impl<'gc> Default for ConstPool<'gc> {
    fn default() -> Self {
        Self::new()
    }
}

/// The raw payload bits of a value, for interning. Mirrors `Value`'s internal
/// representation without exposing it.
fn value_bits(v: Value<'_>) -> u64 {
    use crate::env::value::ValueKind;
    match v.kind() {
        ValueKind::Nil => 0,
        ValueKind::Boolean => v.get_boolean().unwrap() as u64,
        ValueKind::Integer => v.get_integer().unwrap() as u64,
        ValueKind::Float => v.get_float().unwrap().to_bits(),
        ValueKind::String => Gc::as_ptr(v.get_string().unwrap().inner()) as usize as u64,
        ValueKind::Table => Gc::as_ptr(v.get_table().unwrap().inner()) as usize as u64,
        ValueKind::Function => Gc::as_ptr(v.get_function().unwrap().inner()) as usize as u64,
        ValueKind::Thread => Gc::as_ptr(v.get_thread().unwrap().inner()) as usize as u64,
        ValueKind::Userdata => Gc::as_ptr(v.get_userdata().unwrap().inner()) as usize as u64,
    }
}
