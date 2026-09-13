use crate::env::Value;

pub fn exact_float_to_int(f: f64) -> Option<i64> {
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
pub fn op_arith_int<'gc, Op: ArithOp>(lhs: i64, rhs: i64) -> Option<Value<'gc>> {
    if Op::INT_ZERO_DIVISOR_INVALID && rhs == 0 {
        return None;
    }

    Some(Op::int(lhs, rhs))
}

#[inline(always)]
pub fn op_arith_float<'gc, Op: ArithOp>(lhs: f64, rhs: f64) -> Value<'gc> {
    Op::float(lhs, rhs)
}

/// Coerces a numeric operand to `f64`; `None` if it is not a number.
#[inline(always)]
fn to_float(v: &Value) -> Option<f64> {
    if let Some(i) = v.get_integer() {
        Some(i as f64)
    } else {
        v.get_float()
    }
}

/// The mixed int/float arm. Callers must have excluded same-type operands
/// first, so this never sees int-int and needs no zero-divisor guard.
#[inline(always)]
pub fn op_arith_mixed<'gc, Op: ArithOp>(lhs: &Value, rhs: &Value) -> Option<Value<'gc>> {
    let lhs = to_float(lhs)?;
    let rhs = to_float(rhs)?;

    Some(op_arith_float::<Op>(lhs, rhs))
}

#[inline(always)]
pub fn op_arith<'gc, Op: ArithOp>(lhs: Value, rhs: Value) -> Option<Value<'gc>> {
    if let (Some(li), Some(ri)) = (lhs.get_integer(), rhs.get_integer()) {
        return op_arith_int::<Op>(li, ri);
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

    Some(op_arith_float::<Op>(lhs, rhs))
}

pub trait ArithOp {
    const INT_ZERO_DIVISOR_INVALID: bool = false;

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
    const INT_ZERO_DIVISOR_INVALID: bool = true;

    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        let r = lhs.wrapping_rem(rhs);
        let adjusted = if r != 0 && (r ^ rhs) < 0 {
            r.wrapping_add(rhs)
        } else {
            r
        };

        Value::integer(adjusted)
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        let r = lhs % rhs;
        let adjusted = if (r > 0.0 && rhs < 0.0) || (r < 0.0 && rhs > 0.0) {
            r + rhs
        } else {
            r
        };

        Value::float(adjusted)
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
    const INT_ZERO_DIVISOR_INVALID: bool = true;

    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        let q = lhs.wrapping_div(rhs);
        let r = lhs.wrapping_rem(rhs);
        let adjusted = if r != 0 && (lhs ^ rhs) < 0 {
            q.wrapping_sub(1)
        } else {
            q
        };

        Value::integer(adjusted)
    }

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        // Lua's `//` on floats is `floor(a/b)` and stays a float — keep the
        // float type (and inf/nan; `as i64` would saturate large quotients).
        Value::float((lhs / rhs).floor())
    }
}

#[inline(always)]
pub fn op_bit_int<'gc, Op: BitOp>(lhs: i64, rhs: i64) -> Value<'gc> {
    Op::int(lhs, rhs)
}

#[inline(always)]
fn bitwise_coerce_int(v: &Value) -> Option<i64> {
    if let Some(i) = v.get_float() {
        Some(i as i64)
    } else {
        v.get_integer()
    }
}

#[inline(always)]
pub fn op_bit_mixed<'gc, Op: BitOp>(lhs: &Value, rhs: &Value) -> Option<Value<'gc>> {
    let lhs: i64 = bitwise_coerce_int(lhs)?;
    let rhs = bitwise_coerce_int(rhs)?;

    Some(op_bit_int::<Op>(lhs, rhs))
}

#[inline(always)]
pub fn op_bit<'gc, Op: BitOp>(lhs: Value, rhs: Value) -> Option<Value<'gc>> {
    let lhs = if let Some(v) = lhs.get_integer() {
        v
    } else if let Some(v) = lhs.get_float() {
        exact_float_to_int(v)?
    } else {
        return None;
    };

    let rhs = if let Some(v) = rhs.get_integer() {
        v
    } else if let Some(v) = rhs.get_float() {
        exact_float_to_int(v)?
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

/// Lua's `luaV_shiftl`: a logical left shift by `y` bits, zero-filling the vacant
/// bits. A negative `y` shifts right instead, and any displacement with |y| >= 64
/// shifts every bit out and yields 0 — so this is *not* Rust's `<<`, which both
/// sign-extends on the right and masks the count mod 64. Right shift is this with
/// `y` negated. (manual: "Both right and left shifts fill the vacant bits with
/// zeros. Negative displacements shift to the other direction; displacements with
/// absolute values equal to or higher than the number of bits ... result in zero".)
#[inline(always)]
fn shift_left(x: i64, y: i64) -> i64 {
    const NBITS: i64 = i64::BITS as i64;
    if y <= -NBITS || y >= NBITS {
        0
    } else if y >= 0 {
        ((x as u64) << y) as i64
    } else {
        ((x as u64) >> -y) as i64
    }
}

pub struct Shl;

impl BitOp for Shl {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(shift_left(lhs, rhs))
    }
}

pub struct Shr;

impl BitOp for Shr {
    #[inline(always)]
    fn int<'gc>(lhs: i64, rhs: i64) -> Value<'gc> {
        // `wrapping_neg` so `rhs == i64::MIN` (a right shift by 2^63) doesn't
        // overflow. `shift_left` maps the resulting huge magnitude to 0.
        Value::integer(shift_left(lhs, rhs.wrapping_neg()))
    }
}

/// Append a float in Lua's canonical textual form (the `..` concat path). This
/// must match `tostring`/`print` exactly — Lua uses the same `tostringbuff` for
/// both — so it delegates to `push_float` (`%.15g`→`%.17g`, integer-looking
/// floats tagged `.0`, lowercase `inf`/`nan`) rather than a shortest-round-trip
/// formatter, which would print e.g. `0.3333333333333333` for `1/3`.
pub fn write_float(dst: &mut Vec<u8>, f: f64) {
    crate::builtin::util::push_float(dst, f);
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
