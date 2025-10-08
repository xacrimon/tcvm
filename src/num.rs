use crate::value::Value;

pub trait ArithOp {
    fn int(lhs: i64, rhs: i64) -> Value;
    fn float(lhs: f64, rhs: f64) -> Value;
}

#[inline(always)]
pub fn op_arith<Op: ArithOp>(lhs: Value, rhs: Value) -> Option<Value> {
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

pub struct Add;

impl ArithOp for Add {
    #[inline(always)]
    fn int(lhs: i64, rhs: i64) -> Value {
        Value::Integer(lhs.wrapping_add(rhs))
    }

    #[inline(always)]
    fn float(lhs: f64, rhs: f64) -> Value {
        Value::Float(lhs + rhs)
    }
}

pub struct Sub;

impl ArithOp for Sub {
    #[inline(always)]
    fn int(lhs: i64, rhs: i64) -> Value {
        Value::Integer(lhs.wrapping_sub(rhs))
    }

    #[inline(always)]
    fn float(lhs: f64, rhs: f64) -> Value {
        Value::Float(lhs - rhs)
    }
}

pub struct Mul;

impl ArithOp for Mul {
    #[inline(always)]
    fn int(lhs: i64, rhs: i64) -> Value {
        Value::Integer(lhs.wrapping_mul(rhs))
    }

    #[inline(always)]
    fn float(lhs: f64, rhs: f64) -> Value {
        Value::Float(lhs * rhs)
    }
}

pub struct Mod;

impl ArithOp for Mod {
    #[inline(always)]
    fn int(lhs: i64, rhs: i64) -> Value {
        Value::Integer(lhs.wrapping_rem(rhs))
    }

    #[inline(always)]
    fn float(lhs: f64, rhs: f64) -> Value {
        Value::Float(lhs % rhs)
    }
}

pub struct Pow;

impl ArithOp for Pow {
    #[inline(always)]
    fn int(lhs: i64, rhs: i64) -> Value {
        Value::Float((lhs as f64).powf(rhs as f64))
    }

    #[inline(always)]
    fn float(lhs: f64, rhs: f64) -> Value {
        Value::Float(lhs.powf(rhs))
    }
}

pub struct Div;

impl ArithOp for Div {
    #[inline(always)]
    fn int(lhs: i64, rhs: i64) -> Value {
        Value::Float((lhs as f64) / (rhs as f64))
    }

    #[inline(always)]
    fn float(lhs: f64, rhs: f64) -> Value {
        Value::Float(lhs / rhs)
    }
}

pub struct IDiv;

impl ArithOp for IDiv {
    #[inline(always)]
    fn int(lhs: i64, rhs: i64) -> Value {
        Value::Integer(lhs.wrapping_div(rhs))
    }

    #[inline(always)]
    fn float(lhs: f64, rhs: f64) -> Value {
        Value::Integer((lhs / rhs).floor() as i64)
    }
}
