use core::hash::{Hash, Hasher};

use crate::dmm::{Collect, Gc};
use crate::env::function::Function;
use crate::env::string::LuaString;
use crate::env::table::Table;
use crate::env::thread::Thread;
use crate::env::userdata::Userdata;

#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
#[repr(align(8))]
pub enum Value<'gc> {
    Nil,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(LuaString<'gc>),
    Table(Table<'gc>),
    Function(Function<'gc>),
    Thread(Thread<'gc>),
    Userdata(Userdata<'gc>),
}

macro_rules! type_methods {
    ($variant:ident, $lowercase:ident, $type:ty) => {
        paste::paste! {
            pub fn [<get_ $lowercase>](self) -> Option<$type> {
                match self {
                    Self::$variant(v) => Some(v),
                    _ => None,
                }
            }
        }
    };
}

impl<'gc> Value<'gc> {
    type_methods!(Boolean, boolean, bool);
    type_methods!(Integer, integer, i64);
    type_methods!(Float, float, f64);
    type_methods!(String, string, LuaString<'gc>);
    type_methods!(Table, table, Table<'gc>);
    type_methods!(Function, function, Function<'gc>);
    type_methods!(Thread, thread, Thread<'gc>);
    type_methods!(Userdata, userdata, Userdata<'gc>);

    pub fn is_nil(&self) -> bool {
        matches!(self, Value::Nil)
    }

    pub fn is_falsy(&self) -> bool {
        matches!(self, Value::Nil | Value::Boolean(false))
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Nil => "nil",
            Value::Boolean(_) => "boolean",
            Value::Integer(_) | Value::Float(_) => "number",
            Value::String(_) => "string",
            Value::Table(_) => "table",
            Value::Function(_) => "function",
            Value::Thread(_) => "thread",
            Value::Userdata(_) => "userdata",
        }
    }
}

impl<'gc> PartialEq for Value<'gc> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Nil, Value::Nil) => true,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Integer(a), Value::Float(b)) => (*a as f64) == *b && (*b as i64) == *a,
            (Value::Float(a), Value::Integer(b)) => *a == (*b as f64) && (*a as i64) == *b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Table(a), Value::Table(b)) => Gc::ptr_eq(a.inner(), b.inner()),
            (Value::Function(a), Value::Function(b)) => Gc::ptr_eq(a.inner(), b.inner()),
            (Value::Thread(a), Value::Thread(b)) => Gc::ptr_eq(a.inner(), b.inner()),
            (Value::Userdata(a), Value::Userdata(b)) => Gc::ptr_eq(a.inner(), b.inner()),
            _ => false,
        }
    }
}

impl<'gc> Eq for Value<'gc> {}

impl<'gc> Hash for Value<'gc> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Value::Nil => 0u8.hash(state),
            Value::Boolean(b) => {
                1u8.hash(state);
                b.hash(state);
            }
            Value::Integer(i) => {
                2u8.hash(state);
                i.hash(state);
            }
            Value::Float(f) => {
                let i = *f as i64;
                if f.fract() == 0.0 && f.is_finite() && (i as f64) == *f {
                    // Must hash identically to the equivalent integer
                    2u8.hash(state);
                    i.hash(state);
                } else {
                    3u8.hash(state);
                    f.to_bits().hash(state);
                }
            }
            Value::String(s) => {
                4u8.hash(state);
                s.hash(state);
            }
            Value::Table(t) => {
                5u8.hash(state);
                Gc::as_ptr(t.inner()).hash(state);
            }
            Value::Function(f) => {
                6u8.hash(state);
                Gc::as_ptr(f.inner()).hash(state);
            }
            Value::Thread(t) => {
                7u8.hash(state);
                Gc::as_ptr(t.inner()).hash(state);
            }
            Value::Userdata(u) => {
                8u8.hash(state);
                Gc::as_ptr(u.inner()).hash(state);
            }
        }
    }
}
