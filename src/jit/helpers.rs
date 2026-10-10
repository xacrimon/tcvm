//! Rust routines regions call with `bl` (the C ABI). None grows the stack,
//! switches threads, runs Lua or raises (R5).

use crate::env::MetamethodBits;
use crate::env::string::LuaString;
use crate::env::thread::ThreadState;
use crate::env::value::{Value, ValueKind};
use crate::instruction::ArithKind;
use crate::jit::ir::ops::{HELPER_FAIL, HELPER_FAIL_VAL, HelperId};
use crate::lua::{Context, State};
use crate::vm::frame::land_results;
use crate::vm::num::{self, ArithOp, SlowNum};
use crate::vm::ops::meta::binop_metamethod;

/// The address a region calls for `h`.
pub(crate) fn address(h: HelperId) -> usize {
    match h {
        HelperId::FMod => jit_fmod as *const () as usize,
        HelperId::FPow => jit_pow as *const () as usize,
        HelperId::Lt => jit_lt as *const () as usize,
        HelperId::Le => jit_le as *const () as usize,
        HelperId::Eq => jit_eq as *const () as usize,
        HelperId::Arith(_) => jit_arith as *const () as usize,
        HelperId::Unm => jit_unm as *const () as usize,
        HelperId::BNot => jit_bnot as *const () as usize,
        HelperId::Len => jit_len as *const () as usize,
        HelperId::Concat => jit_concat as *const () as usize,
    }
}

/// A float result as a `Value` holds it: NaNs canonical.
#[inline(always)]
fn canonical(x: f64) -> f64 {
    if x.is_nan() {
        f64::from_bits(0x7FF8_0000_0000_0000)
    } else {
        x
    }
}

pub(crate) extern "C" fn jit_fmod(a: f64, b: f64) -> f64 {
    canonical(<num::Mod as ArithOp>::float_raw(a, b))
}

pub(crate) extern "C" fn jit_pow(a: f64, b: f64) -> f64 {
    canonical(<num::Pow as ArithOp>::float_raw(a, b))
}

/// The interpreter's primitive `<` (`le`: `<=`), as `cmp_slow` has it.
fn compare(a: Value<'_>, b: Value<'_>, le: bool) -> i32 {
    let r = if let (Some(x), Some(y)) = (a.get_integer(), b.get_integer()) {
        if le { x <= y } else { x < y }
    } else if let (Some(x), Some(y)) = (a.get_integer(), b.get_float()) {
        if le {
            num::le_int_float(x, y)
        } else {
            num::lt_int_float(x, y)
        }
    } else if let (Some(x), Some(y)) = (a.get_float(), b.get_integer()) {
        if le {
            num::le_float_int(x, y)
        } else {
            num::lt_float_int(x, y)
        }
    } else if let (Some(x), Some(y)) = (a.get_float(), b.get_float()) {
        if le { x <= y } else { x < y }
    } else if let (Some(x), Some(y)) = (a.get_string(), b.get_string()) {
        if le { x <= y } else { x < y }
    } else {
        return HELPER_FAIL;
    };
    r as i32
}

pub(crate) extern "C" fn jit_lt(_rt: *const State<'static>, a: u64, b: u64) -> i32 {
    unsafe { compare(Value::from_raw(a), Value::from_raw(b), false) }
}

pub(crate) extern "C" fn jit_le(_rt: *const State<'static>, a: u64, b: u64) -> i32 {
    unsafe { compare(Value::from_raw(a), Value::from_raw(b), true) }
}

/// The interpreter's `==` without running `__eq`, which fails instead.
pub(crate) extern "C" fn jit_eq(rt: *const State<'static>, a: u64, b: u64) -> i32 {
    let (a, b) = unsafe { (Value::from_raw(a), Value::from_raw(b)) };
    if num::raw_eq(a, b) {
        return 1;
    }
    // Lua 5.5: `__eq` only between two tables or two userdata.
    let both = |k| a.kind() == k && b.kind() == k;
    if both(ValueKind::Table) || both(ValueKind::Userdata) {
        let ctx = unsafe { Context::from_state(rt) };
        if !binop_metamethod(ctx, a, b, crate::env::MetamethodBits::EQ).is_nil() {
            return HELPER_FAIL;
        }
    }
    0
}

/// `arith_slow`'s numeric cases; everything it would raise or hand to a
/// metamethod fails.
pub(crate) extern "C" fn jit_arith(rt: *const State<'static>, kind: u32, a: u64, b: u64) -> u64 {
    use ArithKind::*;
    let ctx = unsafe { Context::from_state(rt) };
    let mc = ctx.mutation();
    let (a, b) = unsafe { (Value::from_raw(a), Value::from_raw(b)) };
    const KINDS: [ArithKind; 13] = [
        None, Add, Sub, Mul, Mod, Pow, Div, IDiv, BAnd, BOr, BXor, Shl, Shr,
    ];
    let r = match KINDS[kind as usize] {
        Add => num::op_arith_slow::<num::Add>(mc, a, b),
        Sub => num::op_arith_slow::<num::Sub>(mc, a, b),
        Mul => num::op_arith_slow::<num::Mul>(mc, a, b),
        Mod => num::op_arith_slow::<num::Mod>(mc, a, b),
        Pow => num::op_arith_slow::<num::Pow>(mc, a, b),
        Div => num::op_arith_slow::<num::Div>(mc, a, b),
        IDiv => num::op_arith_slow::<num::IDiv>(mc, a, b),
        BAnd => num::op_bit_slow::<num::BAnd>(mc, a, b),
        BOr => num::op_bit_slow::<num::BOr>(mc, a, b),
        BXor => num::op_bit_slow::<num::BXor>(mc, a, b),
        Shl => num::op_bit_slow::<num::Shl>(mc, a, b),
        Shr => num::op_bit_slow::<num::Shr>(mc, a, b),
        None => SlowNum::NotNumbers,
    };
    match r {
        SlowNum::Value(v) => v.to_raw(),
        _ => HELPER_FAIL_VAL,
    }
}

/// `-v` as `op_unm` and `unm_slow` compute it without a metamethod.
pub(crate) extern "C" fn jit_unm(rt: *const State<'static>, v: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let v = unsafe { Value::from_raw(v) };
    if v.is_float() {
        Value::float(-v.read_float()).to_raw()
    } else if let Some(i) = v.get_integer() {
        Value::integer(ctx.mutation(), i.wrapping_neg()).to_raw()
    } else {
        HELPER_FAIL_VAL
    }
}

/// `~v` as `op_bnot` and `bnot_slow` compute it without a metamethod.
pub(crate) extern "C" fn jit_bnot(rt: *const State<'static>, v: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    match unsafe { Value::from_raw(v) }.get_integer() {
        Some(i) => Value::integer(ctx.mutation(), !i).to_raw(),
        None => HELPER_FAIL_VAL,
    }
}

/// `#v` of a string, or of a table without `__len`.
pub(crate) extern "C" fn jit_len(rt: *const State<'static>, v: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let v = unsafe { Value::from_raw(v) };
    let n = if let Some(s) = v.get_string() {
        s.len()
    } else if let Some(t) = v.get_table()
        && ctx.mm_of(v, MetamethodBits::LEN).is_nil()
    {
        t.raw_len()
    } else {
        return HELPER_FAIL_VAL;
    };
    Value::integer(ctx.mutation(), n as i64).to_raw()
}

/// `a .. b` of strings and numbers, as `op_concat` builds it.
pub(crate) extern "C" fn jit_concat(rt: *const State<'static>, a: u64, b: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let (a, b) = unsafe { (Value::from_raw(a), Value::from_raw(b)) };
    let s = ctx.with_buf(|buf| {
        buf.reserve(num::concat_len(a) + num::concat_len(b));
        (num::coerce_to_str(buf, a) && num::coerce_to_str(buf, b)).then(|| LuaString::new(ctx, buf))
    });
    match s {
        Some(s) => Value::string(s).to_raw(),
        None => HELPER_FAIL_VAL,
    }
}

/// Box an integer outside i32.
pub(crate) extern "C" fn jit_box_i64(rt: *const State<'static>, v: i64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    Value::integer(ctx.mutation(), v).to_raw()
}

/// Land a call's `nret` results at `values` into `R[a..]`: `c - 1` of them,
/// or all with `top` set for MULTRET (`c == 0`).
pub(crate) extern "C" fn jit_land(
    _rt: *const State<'static>,
    thread: *mut ThreadState<'static>,
    base: *mut Value<'static>,
    a: usize,
    values: *const Value<'static>,
    nret: usize,
    c: usize,
) {
    unsafe {
        let dst = base.add(a);
        let wanted = if c == 0 { nret } else { c - 1 };
        land_results(dst, values, nret, wanted);
        if c == 0 {
            let ts = &mut *thread;
            ts.top = ts.slot_index(dst) + wanted;
        }
    }
}
