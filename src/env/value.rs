// https://codeberg.org/playXE/CapyScheme/src/branch/main/src/runtime/value.rs
// https://codeberg.org/playXE/CapyScheme/src/branch/main/tests/z3_value_encoding_proof.py
// https://github.com/kyren/piccolo/tree/master/src

use crate::env::Table;

#[derive(Clone, Copy)]
#[repr(align(8))]
pub enum Value<'gc> {
    Nil,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(*const ()),
    Table(Table<'gc>),
    Function(*const ()),
    Thread(*const ()),
    Userdata(*const ()),
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
    type_methods!(String, string, *const ());
    type_methods!(Table, table, Table<'gc>);
    type_methods!(Function, function, *const ());
    type_methods!(Thread, thread, *const ());
    type_methods!(Userdata, userdata, *const ());
}
