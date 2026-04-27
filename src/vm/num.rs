use crate::env::Value;

fn exact_float_to_int(f: f64) -> Option<i64> {
    if !f.is_finite() {
        return None;
    }

    const MIN: i64 = -(2 << 53 - 1);
    const MAX: i64 = 2 << 53 - 1;

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
    if let (Some(li), Some(ri)) = (lhs.get_integer(), rhs.get_integer()) {
        return Some(Op::int(li, ri));
    }

    let lhs = if let Some(v) = lhs.get_integer() {
        v as f64
    } else if let Some(v) = lhs.get_float() {
        v
    } else {
        return None;
    };

    let rhs = if let Some(v) = rhs.get_integer() {
        v as f64
    } else if let Some(v) = rhs.get_float() {
        v
    } else {
        return None;
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
        Value::integer(lhs.wrapping_add(rhs))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::float(lhs + rhs)
    }
}

pub struct Sub;

impl ArithOp for Sub {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(lhs.wrapping_sub(rhs))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::float(lhs - rhs)
    }
}

pub struct Mul;

impl ArithOp for Mul {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(lhs.wrapping_mul(rhs))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::float(lhs * rhs)
    }
}

pub struct Mod;

impl ArithOp for Mod {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(lhs.wrapping_rem(rhs))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::float(lhs % rhs)
    }
}

pub struct Pow;

impl ArithOp for Pow {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::float((lhs as f64).powf(rhs as f64))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::float(lhs.powf(rhs))
    }
}

pub struct Div;

impl ArithOp for Div {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::float((lhs as f64) / (rhs as f64))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::float(lhs / rhs)
    }
}

pub struct IDiv;

impl ArithOp for IDiv {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(lhs.wrapping_div(rhs))
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::integer((lhs / rhs).floor() as i64)
    }
}

#[inline(always)]
pub fn op_bit<'gc, Op: BitOp>(lhs: Value, rhs: Value) -> Option<Value<'gc>> {
    let lhs = if let Some(v) = lhs.get_integer() {
        v
    } else if let Some(v) = lhs.get_float() {
        exact_float_to_int(v).unwrap()
    } else {
        return None;
    };

    let rhs = if let Some(v) = rhs.get_integer() {
        v
    } else if let Some(v) = rhs.get_float() {
        exact_float_to_int(v).unwrap()
    } else {
        return None;
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
        Value::integer(lhs & rhs)
    }
}

pub struct BOr;

impl BitOp for BOr {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(lhs | rhs)
    }
}

pub struct BXor;

impl BitOp for BXor {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(lhs ^ rhs)
    }
}

pub struct Shl;

impl BitOp for Shl {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(lhs.wrapping_shl(rhs as u32))
    }
}

pub struct Shr;

impl BitOp for Shr {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(lhs.wrapping_shr(rhs as u32))
    }
}

pub fn write_float(dst: &mut Vec<u8>, f: f64) {
    let mut buf = zmij::Buffer::new();
    let s = buf.format(f);
    dst.extend_from_slice(s.as_bytes());
}

pub fn coerce_to_str(buf: &mut Vec<u8>, val: Value) -> bool {
    if let Some(s) = val.get_string() {
        buf.extend_from_slice(s.as_bytes());
        true
    } else if let Some(n) = val.get_integer() {
        buf.extend_from_slice(n.to_string().as_bytes());
        true
    } else if let Some(f) = val.get_float() {
        write_float(buf, f);
        true
    } else {
        false
    }
}
