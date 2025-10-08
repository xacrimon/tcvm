// https://codeberg.org/playXE/CapyScheme/src/branch/main/src/runtime/value.rs
// https://codeberg.org/playXE/CapyScheme/src/branch/main/tests/z3_value_encoding_proof.py
// https://github.com/kyren/piccolo/tree/master/src

use std::mem;

#[derive(Clone, Copy)]
#[repr(u8)]
#[repr(align(8))]
pub enum Value {
    Nil,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(*const ()),
    Userdata(*const ()),
    Function(*const ()),
    Thread(*const ()),
    Table(*const ()),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ValueType {
    Nil = 0,
    Boolean = 1,
    Integer = 2,
    Float = 3,
    String = 4,
    Userdata = 5,
    Function = 6,
    Thread = 7,
    Table = 8,
}

macro_rules! type_methods {
    ($variant:ident, $lowercase:ident, $type:ty) => {
        paste::paste! {
            pub fn [<new_ $lowercase>](v: $type) -> Self {
                Self::$variant(v)
            }

            pub fn [<as_$lowercase>](self) -> $type {
                match self {
                    Self::$variant(v) => v,
                    _ => unsafe { std::hint::unreachable_unchecked() },
                }
            }

            pub fn [<get_ $lowercase>](self) -> Option<$type> {
                match self {
                    Self::$variant(v) => Some(v),
                    _ => None,
                }
            }

            pub fn [<try_ $lowercase>](self) -> Result<$type, ()> {
                match self {
                    Self::$variant(v) => Ok(v),
                    _ => Err(()),
                }
            }
        }
    };
}

impl Value {
    pub fn ty(self) -> ValueType {
        unsafe {
            let discriminant = *<*const _>::from(&self).cast::<u8>();
            mem::transmute::<u8, ValueType>(discriminant)
        }
    }

    type_methods!(Boolean, boolean, bool);
    type_methods!(Integer, integer, i64);
    type_methods!(Float, float, f64);
    type_methods!(String, string, *const ());
    type_methods!(Userdata, userdata, *const ());
    type_methods!(Function, function, *const ());
    type_methods!(Thread, thread, *const ());
    type_methods!(Table, table, *const ());
}
