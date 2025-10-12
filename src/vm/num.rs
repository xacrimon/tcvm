use crate::env::Value;

fn exact_float_to_int(f: f64) -> Option<i64> {
    if !f.is_finite() {
        return None;
    }

    const MIN: i64 = -(2<<53 - 1);
    const MAX: i64 = 2<<53 - 1;

    if f < MIN as f64 || f > MAX as f64 {
        return None;
    }

    if f.trunc() != f {
        return None;
    }

    let i = unsafe { f.to_int_unchecked() };
    Some(i)
}

#[inline(always)]
pub fn op_arith<'gc, Op: ArithOp>(lhs: Value, rhs: Value) -> Option<Value<'gc>> {
    if let (Value::Integer(lhs), Value::Integer(rhs)) = (lhs, rhs) {
        return Some(Op::int(lhs, rhs));
    }

    let lhs = match lhs {
        Value::Integer(v) => v as f64,
        Value::Float(v) => v,
        _ => return None,
    };

    let rhs = match rhs {
        Value::Integer(v) => v as f64,
        Value::Float(v) => v,
        _ => return None,
    };

    Some(Op::float(lhs, rhs))
}

pub trait ArithOp {
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc>;
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc>;
}

pub struct Add;

impl ArithOp for Add {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::Integer(lhs.wrapping_add(rhs))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::Float(lhs + rhs)
    }
}

pub struct Sub;

impl ArithOp for Sub {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::Integer(lhs.wrapping_sub(rhs))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::Float(lhs - rhs)
    }
}

pub struct Mul;

impl ArithOp for Mul {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::Integer(lhs.wrapping_mul(rhs))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::Float(lhs * rhs)
    }
}

pub struct Mod;

impl ArithOp for Mod {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::Integer(lhs.wrapping_rem(rhs))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::Float(lhs % rhs)
    }
}

pub struct Pow;

impl ArithOp for Pow {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::Float((lhs as f64).powf(rhs as f64))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::Float(lhs.powf(rhs))
    }
}

pub struct Div;

impl ArithOp for Div {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::Float((lhs as f64) / (rhs as f64))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::Float(lhs / rhs)
    }
}

pub struct IDiv;

impl ArithOp for IDiv {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::Integer(lhs.wrapping_div(rhs))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::Integer((lhs / rhs).floor() as i64)
    }
}

#[inline(always)]
pub fn op_bit<'gc, Op: BitOp>(lhs: Value, rhs: Value) -> Option<Value<'gc>> {
    let lhs = match lhs {
        Value::Integer(v) => v,
        Value::Float(v) => exact_float_to_int(v).unwrap(),
        _ => return None,
    };

    let rhs = match rhs {
        Value::Integer(v) => v,
        Value::Float(v) => exact_float_to_int(v).unwrap(),
        _ => return None,
    };

    Some(Op::int(lhs, rhs))
}

pub trait BitOp {
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc>;
}

pub struct BAnd;

impl BitOp for BAnd {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::Integer(lhs & rhs)
    }
}

pub struct BOr;

impl BitOp for BOr {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::Integer(lhs | rhs)
    }
}

pub struct BXor;

impl BitOp for BXor {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::Integer(lhs ^ rhs)
    }
}

pub struct Shl;

impl BitOp for Shl {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::Integer(lhs.wrapping_shl(rhs as u32))
    }
}

pub struct Shr;

impl BitOp for Shr {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::Integer(lhs.wrapping_shr(rhs as u32))
    }
}
