use crate::dmm::Mutation;
use crate::env::Value;

/// Lua's `luaV_flttointns` with mode F2Ieq: the float must be integral and in
/// [-2^63, 2^63).
pub fn exact_float_to_int(f: f64) -> Option<i64> {
    const MIN: f64 = i64::MIN as f64;
    const MAX: f64 = -MIN;

    if !(MIN..MAX).contains(&f) || f.trunc() != f {
        return None;
    }

    Some(f as i64)
}

/// Lua raw equality (`==` without metamethods).
#[inline(always)]
pub fn raw_eq(a: Value, b: Value) -> bool {
    match (a.get_float(), b.get_float()) {
        (Some(x), Some(y)) => x == y,
        (Some(x), None) => b
            .get_integer()
            .is_some_and(|i| exact_float_to_int(x) == Some(i)),
        (None, Some(y)) => a
            .get_integer()
            .is_some_and(|i| exact_float_to_int(y) == Some(i)),
        // Same bits, or equal heap-boxed integers.
        (None, None) => a == b,
    }
}

// Mixed int/float ordering, following Lua's `LTintfloat` & co. When |i| <= 2^53
// the `i as f64` cast is exact and a float compare decides. Beyond that the cast
// rounds (`2^53+1 <= 2^53` would hold), so the float is rounded towards the
// integer side of the inequality and compared as an integer; a float outside the
// i64 range (or NaN) is decided by sign.

/// Lua's `l_intfitsf`: `i` is exactly representable as an f64.
#[inline(always)]
fn int_fits_float(i: i64) -> bool {
    (i as u64).wrapping_add(1 << 53) <= 1 << 54
}

/// `i < f`, `i < ceil(f)`
#[inline(always)]
pub fn lt_int_float(i: i64, f: f64) -> bool {
    if int_fits_float(i) {
        return (i as f64) < f;
    }
    match exact_float_to_int(f.ceil()) {
        Some(fi) => i < fi,
        None => f > 0.0,
    }
}

/// `i <= f`, `i <= floor(f)`
#[inline(always)]
pub fn le_int_float(i: i64, f: f64) -> bool {
    if int_fits_float(i) {
        return (i as f64) <= f;
    }
    match exact_float_to_int(f.floor()) {
        Some(fi) => i <= fi,
        None => f > 0.0,
    }
}

/// `f < i`, `floor(f) < i`
#[inline(always)]
pub fn lt_float_int(f: f64, i: i64) -> bool {
    if int_fits_float(i) {
        return f < (i as f64);
    }
    match exact_float_to_int(f.floor()) {
        Some(fi) => fi < i,
        None => f < 0.0,
    }
}

/// `f <= i`, `ceil(f) <= i`
#[inline(always)]
pub fn le_float_int(f: f64, i: i64) -> bool {
    if int_fits_float(i) {
        return f <= (i as f64);
    }
    match exact_float_to_int(f.ceil()) {
        Some(fi) => fi <= i,
        None => f < 0.0,
    }
}

#[inline(always)]
pub fn op_arith_int<'gc, Op: ArithOp>(
    mc: &Mutation<'gc>,
    lhs: i64,
    rhs: i64,
) -> Option<Value<'gc>> {
    if Op::ZERO_DIVISOR.is_some() && rhs == 0 {
        return None;
    }

    Some(Op::int(mc, lhs, rhs))
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

/// Float arithmetic after coercing both operands. Callers must have excluded int-int, which
/// needs integer semantics and a zero-divisor guard.
#[inline(always)]
pub fn op_arith_mixed<'gc, Op: ArithOp>(lhs: &Value, rhs: &Value) -> Option<Value<'gc>> {
    let lhs = to_float(lhs)?;
    let rhs = to_float(rhs)?;

    Some(op_arith_float::<Op>(lhs, rhs))
}

#[inline(always)]
pub fn op_arith<'gc, Op: ArithOp>(
    mc: &Mutation<'gc>,
    lhs: Value,
    rhs: Value,
) -> Option<Value<'gc>> {
    if let (Some(li), Some(ri)) = (lhs.get_integer(), rhs.get_integer()) {
        return op_arith_int::<Op>(mc, li, ri);
    }

    let lhs = if let Some(v) = lhs.get_integer() {
        v as f64
    } else {
        lhs.get_float()?
    };

    let rhs = if let Some(v) = rhs.get_integer() {
        v as f64
    } else {
        rhs.get_float()?
    };

    Some(op_arith_float::<Op>(lhs, rhs))
}

/// Outcome of the out-of-line numeric path shared by the arithmetic and bitwise slow handlers.
pub enum SlowNum<'gc> {
    Value(Value<'gc>),
    ModByZero,
    DivByZero,
    NotNumbers,
}

/// Everything the arithmetic fast paths leave behind: heap-boxed integers, i32 overflow,
/// zero divisors and int/float mixes.
#[inline]
pub fn op_arith_slow<'gc, Op: ArithOp>(
    mc: &Mutation<'gc>,
    lhs: Value<'gc>,
    rhs: Value<'gc>,
) -> SlowNum<'gc> {
    if let (Some(li), Some(ri)) = (lhs.get_integer(), rhs.get_integer()) {
        return match op_arith_int::<Op>(mc, li, ri) {
            Some(v) => SlowNum::Value(v),
            // `op_arith_int` only returns `None` when `ZERO_DIVISOR` gated it.
            None => match Op::ZERO_DIVISOR.expect("None only when ZERO_DIVISOR is Some") {
                ZeroDivisor::Mod => SlowNum::ModByZero,
                ZeroDivisor::Div => SlowNum::DivByZero,
            },
        };
    }
    match op_arith_mixed::<Op>(&lhs, &rhs) {
        Some(v) => SlowNum::Value(v),
        None => SlowNum::NotNumbers,
    }
}

#[inline]
pub fn op_bit_slow<'gc, Op: BitOp>(
    mc: &Mutation<'gc>,
    lhs: Value<'gc>,
    rhs: Value<'gc>,
) -> SlowNum<'gc> {
    match op_bit::<Op>(mc, lhs, rhs) {
        Some(v) => SlowNum::Value(v),
        None => SlowNum::NotNumbers,
    }
}

/// Which error a zero integer divisor raises.
pub enum ZeroDivisor {
    Mod,
    Div,
}

pub trait ArithOp {
    /// `None` when this op's `int` never divides by its second operand (Add/Sub/Mul),
    /// or always falls through to float division regardless of the divisor (Div/Pow) —
    /// either way, no zero check is needed before calling `int`.
    const ZERO_DIVISOR: Option<ZeroDivisor> = None;

    fn int<'gc>(mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc>;
    fn float_raw(lhs: f64, rhs: f64) -> f64;

    /// The inline-integer fast path: `None` when the result does not fit an i32 (or the
    /// divisor is zero), in which case the caller falls back to `int`, which agrees with
    /// this on every `Some`.
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>>;

    #[inline(always)]
    fn float<'gc>(lhs: f64, rhs: f64) -> Value<'gc> {
        Value::float(Self::float_raw(lhs, rhs))
    }
}

pub struct Add;

impl ArithOp for Add {
    #[inline(always)]
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>> {
        lhs.checked_add(rhs).map(Value::small)
    }

    #[inline(always)]
    fn int<'gc>(mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(mc, lhs.wrapping_add(rhs))
    }

    #[inline(always)]
    fn float_raw(lhs: f64, rhs: f64) -> f64 {
        lhs + rhs
    }
}

pub struct Sub;

impl ArithOp for Sub {
    #[inline(always)]
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>> {
        lhs.checked_sub(rhs).map(Value::small)
    }

    #[inline(always)]
    fn int<'gc>(mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(mc, lhs.wrapping_sub(rhs))
    }

    #[inline(always)]
    fn float_raw(lhs: f64, rhs: f64) -> f64 {
        lhs - rhs
    }
}

pub struct Mul;

impl ArithOp for Mul {
    #[inline(always)]
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>> {
        lhs.checked_mul(rhs).map(Value::small)
    }

    #[inline(always)]
    fn int<'gc>(mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(mc, lhs.wrapping_mul(rhs))
    }

    #[inline(always)]
    fn float_raw(lhs: f64, rhs: f64) -> f64 {
        lhs * rhs
    }
}

pub struct Mod;

impl ArithOp for Mod {
    const ZERO_DIVISOR: Option<ZeroDivisor> = Some(ZeroDivisor::Mod);

    #[inline(always)]
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>> {
        // `checked_rem` is `None` for a zero divisor and for MIN % -1, both handled by `int`.
        let r = lhs.checked_rem(rhs)?;
        let adjusted = if r != 0 && (r ^ rhs) < 0 { r + rhs } else { r };
        Some(Value::small(adjusted))
    }

    #[inline(always)]
    fn int<'gc>(mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc> {
        let r = lhs.wrapping_rem(rhs);
        let adjusted = if r != 0 && (r ^ rhs) < 0 {
            r.wrapping_add(rhs)
        } else {
            r
        };

        Value::integer(mc, adjusted)
    }

    #[inline(always)]
    fn float_raw(lhs: f64, rhs: f64) -> f64 {
        let r = lhs % rhs;
        if (r > 0.0 && rhs < 0.0) || (r < 0.0 && rhs > 0.0) {
            r + rhs
        } else {
            r
        }
    }
}

pub struct Pow;

impl ArithOp for Pow {
    #[inline(always)]
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>> {
        Some(Value::float((lhs as f64).powf(rhs as f64)))
    }

    #[inline(always)]
    fn int<'gc>(_mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc> {
        Value::float((lhs as f64).powf(rhs as f64))
    }

    #[inline(always)]
    fn float_raw(lhs: f64, rhs: f64) -> f64 {
        lhs.powf(rhs)
    }
}

pub struct Div;

impl ArithOp for Div {
    #[inline(always)]
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>> {
        Some(Value::float(lhs as f64 / rhs as f64))
    }

    #[inline(always)]
    fn int<'gc>(_mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc> {
        Value::float((lhs as f64) / (rhs as f64))
    }

    #[inline(always)]
    fn float_raw(lhs: f64, rhs: f64) -> f64 {
        lhs / rhs
    }
}

pub struct IDiv;

impl ArithOp for IDiv {
    const ZERO_DIVISOR: Option<ZeroDivisor> = Some(ZeroDivisor::Div);

    #[inline(always)]
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>> {
        let q = lhs.checked_div(rhs)?;
        let r = lhs.wrapping_rem(rhs);
        // `q - 1` only when the signs differ, so q <= 0 and it cannot overflow.
        let adjusted = if r != 0 && (lhs ^ rhs) < 0 { q - 1 } else { q };
        Some(Value::small(adjusted))
    }

    #[inline(always)]
    fn int<'gc>(mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc> {
        let q = lhs.wrapping_div(rhs);
        let r = lhs.wrapping_rem(rhs);
        let adjusted = if r != 0 && (lhs ^ rhs) < 0 {
            q.wrapping_sub(1)
        } else {
            q
        };

        Value::integer(mc, adjusted)
    }

    #[inline(always)]
    fn float_raw(lhs: f64, rhs: f64) -> f64 {
        // Lua's `//` on floats is `floor(a/b)` and stays a float — keep the
        // float type (and inf/nan; `as i64` would saturate large quotients).
        (lhs / rhs).floor()
    }
}

#[inline(always)]
pub fn op_bit<'gc, Op: BitOp>(mc: &Mutation<'gc>, lhs: Value, rhs: Value) -> Option<Value<'gc>> {
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

    Some(Op::int(mc, lhs, rhs))
}

pub trait BitOp {
    fn int<'gc>(mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc>;

    /// Inline-integer fast path; `None` defers to `int`, which agrees with this on
    /// every `Some`.
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>>;
}

pub struct BAnd;

impl BitOp for BAnd {
    #[inline(always)]
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>> {
        Some(Value::small(lhs & rhs))
    }

    #[inline(always)]
    fn int<'gc>(mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(mc, lhs & rhs)
    }
}

pub struct BOr;

impl BitOp for BOr {
    #[inline(always)]
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>> {
        Some(Value::small(lhs | rhs))
    }

    #[inline(always)]
    fn int<'gc>(mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(mc, lhs | rhs)
    }
}

pub struct BXor;

impl BitOp for BXor {
    #[inline(always)]
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>> {
        Some(Value::small(lhs ^ rhs))
    }

    #[inline(always)]
    fn int<'gc>(mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(mc, lhs ^ rhs)
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
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>> {
        i32::try_from(shift_left(lhs as i64, rhs as i64))
            .ok()
            .map(Value::small)
    }

    #[inline(always)]
    fn int<'gc>(mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc> {
        Value::integer(mc, shift_left(lhs, rhs))
    }
}

pub struct Shr;

impl BitOp for Shr {
    #[inline(always)]
    fn small<'gc>(lhs: i32, rhs: i32) -> Option<Value<'gc>> {
        // Widen to i64 before negating: unlike `int` below, `rhs` is `i32` here, so
        // even `i32::MIN` negates without overflow once it's an `i64`.
        i32::try_from(shift_left(lhs as i64, -(rhs as i64)))
            .ok()
            .map(Value::small)
    }

    #[inline(always)]
    fn int<'gc>(mc: &Mutation<'gc>, lhs: i64, rhs: i64) -> Value<'gc> {
        // `wrapping_neg` so `rhs == i64::MIN` (a right shift by 2^63) doesn't
        // overflow. `shift_left` maps the resulting huge magnitude to 0.
        Value::integer(mc, shift_left(lhs, rhs.wrapping_neg()))
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
