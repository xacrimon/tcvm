//! Rust routines regions call with `bl` (the C ABI). None grows the stack,
//! switches threads, runs Lua or raises (R5).

use crate::env::thread::ThreadState;
use crate::env::value::{Value, ValueKind};
use crate::jit::ir::ops::HELPER_FAIL;
use crate::lua::{Context, State};
use crate::vm::frame::land_results;
use crate::vm::num::{self, ArithOp};
use crate::vm::ops::meta::binop_metamethod;

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
