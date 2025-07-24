use std::mem;

#[derive(Clone, Copy)]
#[repr(u8)]
enum Repr {
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
                Self(Repr::$variant(v))
            }

            pub fn [<as_$lowercase>](self) -> $type {
                match self.0 {
                    Repr::$variant(v) => v,
                    #[cfg(debug_assertions)]
                    _ => unreachable!(),
                    #[cfg(not(debug_assertions))]
                    _ => unsafe { std::hint::unreachable_unchecked() },
                }
            }

            pub fn [<get_ $lowercase>](self) -> Option<$type> {
                match self.0 {
                    Repr::$variant(v) => Some(v),
                    _ => None,
                }
            }

            pub fn [<try_ $lowercase>](self) -> Result<$type, ()> {
                match self.0 {
                    Repr::$variant(v) => Ok(v),
                    _ => Err(()),
                }
            }
        }
    };
}

#[derive(Clone, Copy)]
pub struct Value(Repr);

impl Value {
    pub fn ty(self) -> ValueType {
        unsafe {
            let discriminant = *<*const _>::from(&self.0).cast::<u8>();
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
