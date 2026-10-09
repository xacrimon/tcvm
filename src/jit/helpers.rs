//! Rust routines regions call with `bl` (the C ABI). None grows the stack,
//! switches threads, runs Lua or raises (R5).

use crate::env::thread::ThreadState;
use crate::env::value::Value;
use crate::lua::{Context, State};
use crate::vm::frame::land_results;
use crate::vm::num::{self, ArithOp};

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
