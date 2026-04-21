//! Conversion traits for passing Rust values into and out of Lua calls.

use crate::env::function::Function;
use crate::env::{LuaString, Table, Thread, Value};
use crate::lua::TypeError;

/// A Rust value that lowers to a single `Value<'gc>`.
pub trait IntoValue<'gc> {
    fn into_value(self) -> Value<'gc>;
}

/// A Rust value built from a single `Value<'gc>`.
pub trait FromValue<'gc>: Sized {
    fn from_value(v: Value<'gc>) -> Result<Self, TypeError>;
}

/// An argument list pushed onto a Lua call's stack.
pub trait IntoMultiValue<'gc> {
    fn push_into(self, stack: &mut Vec<Value<'gc>>);
}

/// A Rust type constructed from the return-value sequence of a Lua call.
pub trait FromMultiValue<'gc>: Sized {
    fn from_multi_value(values: &[Value<'gc>]) -> Result<Self, TypeError>;
}

// ---------------------------------------------------------------------------
// IntoValue / FromValue
// ---------------------------------------------------------------------------

impl<'gc> IntoValue<'gc> for Value<'gc> {
    fn into_value(self) -> Value<'gc> {
        self
    }
}

impl<'gc> FromValue<'gc> for Value<'gc> {
    fn from_value(v: Value<'gc>) -> Result<Self, TypeError> {
        Ok(v)
    }
}

impl<'gc> IntoValue<'gc> for bool {
    fn into_value(self) -> Value<'gc> {
        Value::Boolean(self)
    }
}

impl<'gc> FromValue<'gc> for bool {
    fn from_value(v: Value<'gc>) -> Result<Self, TypeError> {
        v.get_boolean().ok_or(TypeError::Mismatch {
            expected: "boolean",
            got: v.type_name(),
        })
    }
}

impl<'gc> IntoValue<'gc> for i64 {
    fn into_value(self) -> Value<'gc> {
        Value::Integer(self)
    }
}

impl<'gc> FromValue<'gc> for i64 {
    fn from_value(v: Value<'gc>) -> Result<Self, TypeError> {
        match v {
            Value::Integer(i) => Ok(i),
            Value::Float(f) if f.fract() == 0.0 && f.is_finite() => Ok(f as i64),
            other => Err(TypeError::Mismatch {
                expected: "integer",
                got: other.type_name(),
            }),
        }
    }
}

impl<'gc> IntoValue<'gc> for f64 {
    fn into_value(self) -> Value<'gc> {
        Value::Float(self)
    }
}

impl<'gc> FromValue<'gc> for f64 {
    fn from_value(v: Value<'gc>) -> Result<Self, TypeError> {
        match v {
            Value::Integer(i) => Ok(i as f64),
            Value::Float(f) => Ok(f),
            other => Err(TypeError::Mismatch {
                expected: "number",
                got: other.type_name(),
            }),
        }
    }
}

impl<'gc> IntoValue<'gc> for LuaString<'gc> {
    fn into_value(self) -> Value<'gc> {
        Value::String(self)
    }
}

impl<'gc> FromValue<'gc> for LuaString<'gc> {
    fn from_value(v: Value<'gc>) -> Result<Self, TypeError> {
        v.get_string().ok_or(TypeError::Mismatch {
            expected: "string",
            got: v.type_name(),
        })
    }
}

impl<'gc> IntoValue<'gc> for Table<'gc> {
    fn into_value(self) -> Value<'gc> {
        Value::Table(self)
    }
}

impl<'gc> FromValue<'gc> for Table<'gc> {
    fn from_value(v: Value<'gc>) -> Result<Self, TypeError> {
        v.get_table().ok_or(TypeError::Mismatch {
            expected: "table",
            got: v.type_name(),
        })
    }
}

impl<'gc> IntoValue<'gc> for Function<'gc> {
    fn into_value(self) -> Value<'gc> {
        Value::Function(self)
    }
}

impl<'gc> FromValue<'gc> for Function<'gc> {
    fn from_value(v: Value<'gc>) -> Result<Self, TypeError> {
        v.get_function().ok_or(TypeError::Mismatch {
            expected: "function",
            got: v.type_name(),
        })
    }
}

impl<'gc> IntoValue<'gc> for Thread<'gc> {
    fn into_value(self) -> Value<'gc> {
        Value::Thread(self)
    }
}

impl<'gc> FromValue<'gc> for Thread<'gc> {
    fn from_value(v: Value<'gc>) -> Result<Self, TypeError> {
        v.get_thread().ok_or(TypeError::Mismatch {
            expected: "thread",
            got: v.type_name(),
        })
    }
}

impl<'gc, T: FromValue<'gc>> FromValue<'gc> for Option<T> {
    fn from_value(v: Value<'gc>) -> Result<Self, TypeError> {
        if v.is_nil() {
            Ok(None)
        } else {
            T::from_value(v).map(Some)
        }
    }
}

impl<'gc, T: IntoValue<'gc>> IntoValue<'gc> for Option<T> {
    fn into_value(self) -> Value<'gc> {
        match self {
            None => Value::Nil,
            Some(t) => t.into_value(),
        }
    }
}

// ---------------------------------------------------------------------------
// IntoMultiValue
// ---------------------------------------------------------------------------

impl<'gc> IntoMultiValue<'gc> for () {
    fn push_into(self, _stack: &mut Vec<Value<'gc>>) {}
}

impl<'gc> IntoMultiValue<'gc> for &[Value<'gc>] {
    fn push_into(self, stack: &mut Vec<Value<'gc>>) {
        stack.extend_from_slice(self);
    }
}

macro_rules! into_multi_tuple {
    ($($t:ident),+) => {
        impl<'gc, $($t),+> IntoMultiValue<'gc> for ($($t,)+)
        where
            $($t: IntoValue<'gc>,)+
        {
            #[allow(non_snake_case)]
            fn push_into(self, stack: &mut Vec<Value<'gc>>) {
                let ($($t,)+) = self;
                $(stack.push($t.into_value());)+
            }
        }
    };
}

into_multi_tuple!(A);
into_multi_tuple!(A, B);
into_multi_tuple!(A, B, C);
into_multi_tuple!(A, B, C, D);
into_multi_tuple!(A, B, C, D, E);
into_multi_tuple!(A, B, C, D, E, F);

// ---------------------------------------------------------------------------
// FromMultiValue
// ---------------------------------------------------------------------------

impl<'gc> FromMultiValue<'gc> for () {
    fn from_multi_value(_values: &[Value<'gc>]) -> Result<Self, TypeError> {
        Ok(())
    }
}

impl<'gc> FromMultiValue<'gc> for Vec<Value<'gc>> {
    fn from_multi_value(values: &[Value<'gc>]) -> Result<Self, TypeError> {
        Ok(values.to_vec())
    }
}

macro_rules! from_multi_single {
    ($t:ty) => {
        impl<'gc> FromMultiValue<'gc> for $t {
            fn from_multi_value(values: &[Value<'gc>]) -> Result<Self, TypeError> {
                let v = values.first().copied().unwrap_or(Value::Nil);
                <$t as FromValue<'gc>>::from_value(v)
            }
        }
    };
}

from_multi_single!(bool);
from_multi_single!(i64);
from_multi_single!(f64);
from_multi_single!(Value<'gc>);
from_multi_single!(LuaString<'gc>);
from_multi_single!(Table<'gc>);
from_multi_single!(Function<'gc>);
from_multi_single!(Thread<'gc>);

impl<'gc, T: FromValue<'gc>> FromMultiValue<'gc> for Option<T> {
    fn from_multi_value(values: &[Value<'gc>]) -> Result<Self, TypeError> {
        let v = values.first().copied().unwrap_or(Value::Nil);
        <Option<T> as FromValue<'gc>>::from_value(v)
    }
}

macro_rules! from_multi_tuple {
    ($($idx:tt => $t:ident),+) => {
        impl<'gc, $($t),+> FromMultiValue<'gc> for ($($t,)+)
        where
            $($t: FromValue<'gc>,)+
        {
            fn from_multi_value(values: &[Value<'gc>]) -> Result<Self, TypeError> {
                Ok(($(
                    <$t as FromValue<'gc>>::from_value(
                        values.get($idx).copied().unwrap_or(Value::Nil),
                    )?,
                )+))
            }
        }
    };
}

from_multi_tuple!(0 => A, 1 => B);
from_multi_tuple!(0 => A, 1 => B, 2 => C);
from_multi_tuple!(0 => A, 1 => B, 2 => C, 3 => D);
from_multi_tuple!(0 => A, 1 => B, 2 => C, 3 => D, 4 => E);
from_multi_tuple!(0 => A, 1 => B, 2 => C, 3 => D, 4 => E, 5 => F);
