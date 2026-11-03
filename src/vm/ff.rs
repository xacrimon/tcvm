//! Fast entries of builtins: handlers that do the common argument
//! shape inline from the CALL's registers and leave every other shape to
//! `native_call`, so the builtin stays the complete implementation.

use crate::env::LuaString;
use crate::env::feedback::FeedbackVector;
use crate::env::value::Value;
use crate::instruction::{Instruction, Op, Reg};
use crate::jit::feedback::{self as fb, Bits};
use crate::vm::abi::{Slot, handler};
use crate::vm::frame::{self, HDR, copy_values, fill_nil};
use crate::vm::native::native_call;

/// OR `bits` into the feedback of the call `call` being made from the frame at `base`: the
/// argument kinds of a builtin whose result type follows them (D1 (a)), which the JIT's arms
/// for it come from.
#[inline(always)]
fn record(base: *mut Value<'_>, call: Instruction, bits: Bits) {
    // SAFETY: the caller's frame is a Lua frame, `call` one of its calls.
    unsafe { FeedbackVector::add_call_bits(frame::closure(base).feedback, call.d(), bits) };
}

/// Declare a fast entry. The body names `call` (the CALL or TAILCALL word),
/// `a`, `nargs`, `c` (wanted + 1, 0 for all), `base` and `rt`, and uses
/// `miss!()` to leave the call to the builtin and `land1!()` to dispatch
/// once the one result is in `R[a]` (`land1!(gc)` after an allocation: the
/// collector's exit must see the landed call, since it resumes after it).
/// Nothing before the last `miss!()` may have a visible effect.
macro_rules! fast_entry {
    ($(#[$m:meta])* $name:ident => |$call:ident, $a:ident, $nargs:ident, $c:ident, $base:ident, $rt:ident| $body:block) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            $(#[$m])*
            entry fn $name {
                let $call = insn.as_insn();
                let ($a, b, $c) = ($call.a(), $call.b(), $call.c());
                let $nargs = if b == 0 {
                    let ts = thread!();
                    ts.top - ts.slot_index(base) - $a as usize - HDR
                } else {
                    b as usize - 1
                };
                let $base = base;
                let $rt = rt;
                macro_rules! miss {
                    () => {
                        tail!(native_call)
                    };
                }
                macro_rules! land1 {
                    (gc) => {{
                        land1!(@land);
                        gc_check!();
                        next!()
                    }};
                    () => {{
                        land1!(@land);
                        next!()
                    }};
                    (@land) => {{
                        set_closure!(unsafe { frame::closure(base) });
                        if $call.op() == Op::TAILCALL {
                            // Not RETURN1's handler: the frame may have something to close.
                            tail!(
                                crate::vm::ops::call::op_return,
                                insn = Slot::insn(Instruction::ret(Reg($a), 2))
                            )
                        }
                        if $c == 0 {
                            let ts = thread!();
                            ts.top = ts.slot_index(base) + $a as usize + 1;
                        } else {
                            unsafe { fill_nil(base.add($a as usize + 1), ($c as usize).saturating_sub(2)) };
                        }
                    }};
                }
                $body
            }
        }
    };
}

/// What a one-argument math entry made of its argument.
enum Math1 {
    Float(f64),
    Small(i32),
    /// Not the common shape: leave the call to the full builtin, whose
    /// result these feedback bits describe (`OVERFLOW`, `OTHER_RESULT`).
    Miss(Bits),
}

/// A one-argument math builtin: `$float`/`$small` map a float or inline
/// integer argument to a `Math1`. `$record`: the builtin's result type
/// follows its argument's, whose kind it records.
macro_rules! math1 {
    ($name:ident, $record:literal, |$x:ident| $float:expr, |$i:ident| $small:expr) => {
        fast_entry! {
            $name => |call, a, nargs, c, base, _rt| {
                if nargs != 1 {
                    miss!()
                }
                let arg = &reg![a + 4];
                let (r, kind) = if arg.is_float() {
                    let $x = arg.read_float();
                    ($float, fb::FLOAT)
                } else if let Some($i) = arg.get_small() {
                    ($small, fb::SMALL)
                } else {
                    (Math1::Miss(0), fb::kind(*arg))
                };
                if $record {
                    let result = if let Math1::Miss(bits) = r { bits } else { 0 };
                    record(base, call, kind | result);
                }
                match r {
                    Math1::Float(f) => reg![a].write_float(f),
                    Math1::Small(i) => reg![a] = Value::small(i),
                    Math1::Miss(_) => miss!(),
                }
                land1!()
            }
        }
    };
}

/// `floor`/`ceil`'s rounded `r` as an inline integer, else the miss with what
/// the builtin makes of it: a big integer, or a float without one.
#[inline(always)]
fn rounded(r: f64) -> Math1 {
    // `as` saturates and maps NaN to 0, so the round trip only holds for an
    // integral value in range (-0.0 becomes 0, as in Lua).
    if r as i32 as f64 == r {
        Math1::Small(r as i32)
    } else if (i64::MIN as f64..-(i64::MIN as f64)).contains(&r) {
        Math1::Miss(fb::OVERFLOW)
    } else {
        Math1::Miss(fb::OTHER_RESULT)
    }
}

math1!(ff_sqrt, false, |x| Math1::Float(x.sqrt()), |i| Math1::Float(
    f64::from(i).sqrt()
));
math1!(ff_sin, false, |x| Math1::Float(x.sin()), |i| Math1::Float(
    f64::from(i).sin()
));
math1!(ff_cos, false, |x| Math1::Float(x.cos()), |i| Math1::Float(
    f64::from(i).cos()
));
// `abs(i32::MIN)` leaves the inline range.
math1!(ff_abs, true, |x| Math1::Float(x.abs()), |i| i
    .checked_abs()
    .map_or(Math1::Miss(fb::OVERFLOW), Math1::Small));
math1!(ff_floor, true, |x| rounded(x.floor()), |i| Math1::Small(i));
math1!(ff_ceil, true, |x| rounded(x.ceil()), |i| Math1::Small(i));
math1!(ff_exp, false, |x| Math1::Float(x.exp()), |i| Math1::Float(
    f64::from(i).exp()
));
math1!(ff_tan, false, |x| Math1::Float(x.tan()), |i| Math1::Float(
    f64::from(i).tan()
));
math1!(ff_asin, false, |x| Math1::Float(x.asin()), |i| Math1::Float(
    f64::from(i).asin()
));
math1!(ff_acos, false, |x| Math1::Float(x.acos()), |i| Math1::Float(
    f64::from(i).acos()
));

/// A float or inline integer argument as a float.
#[inline(always)]
fn num_arg(v: &Value<'_>) -> Option<f64> {
    if v.is_float() {
        Some(v.read_float())
    } else {
        v.get_small().map(f64::from)
    }
}

/// [`num_arg`] with the argument's kind; a fast path computes `fb::kind`, a
/// select chain, only for what it leaves to the builtin.
#[inline(always)]
fn num_kind(v: &Value<'_>) -> (Option<f64>, Bits) {
    if v.is_float() {
        (Some(v.read_float()), fb::FLOAT)
    } else if let Some(i) = v.get_small() {
        (Some(f64::from(i)), fb::SMALL)
    } else {
        (None, fb::kind(*v))
    }
}

fast_entry! {
    /// `atan(y [, x])`: `atan2`, as Lua computes even the one-argument form.
    ff_atan => |call, a, nargs, c, _base, _rt| {
        let Some(y) = num_arg(&reg![a + 4]) else { miss!() };
        let x = match nargs {
            1 => 1.0,
            2 => match num_arg(&reg![a + 5]) {
                Some(x) => x,
                None => miss!(),
            },
            _ => miss!(),
        };
        reg![a].write_float(y.atan2(x));
        land1!()
    }
}

fast_entry! {
    /// `log(x [, base])`, bases 2 and 10 by their own functions as in Lua.
    ff_log => |call, a, nargs, c, _base, _rt| {
        let Some(x) = num_arg(&reg![a + 4]) else { miss!() };
        let r = match nargs {
            1 => x.ln(),
            2 => match num_arg(&reg![a + 5]) {
                Some(2.0) => x.log2(),
                Some(10.0) => x.log10(),
                Some(b) => x.ln() / b.ln(),
                None => miss!(),
            },
            _ => miss!(),
        };
        reg![a].write_float(r);
        land1!()
    }
}

fast_entry! {
    /// `fmod(x, y)`: C's `%` of two inline integers (a zero divisor raises
    /// in the builtin), else `fmod` of the numbers. Records the kinds.
    ff_fmod => |call, a, nargs, c, base, _rt| {
        if nargs != 2 {
            miss!()
        }
        let (x, y) = (&reg![a + 4], &reg![a + 5]);
        if let (Some(i), Some(j)) = (x.get_small(), y.get_small()) {
            record(base, call, fb::SMALL | fb::SMALL << fb::ARGS_SHIFT);
            // `i32::MIN % -1` overflows; any `% -1` is 0.
            let r = match j {
                0 => miss!(),
                -1 => 0,
                _ => i % j,
            };
            reg![a] = Value::small(r);
            land1!()
        }
        let ((x, kx), (y, ky)) = (num_kind(x), num_kind(y));
        record(base, call, kx | ky << fb::ARGS_SHIFT);
        let (Some(x), Some(y)) = (x, y) else { miss!() };
        reg![a].write_float(x % y);
        land1!()
    }
}

/// `min(...)`/`max(...)` of inline integers and floats: the builtin's pick
/// by `<`, the first of equals, compared as floats (exact for inline
/// integers). Records the kinds, the first argument's and the others'.
macro_rules! min_max {
    ($name:ident, $max:literal) => {
        fast_entry! {
            $name => |call, a, nargs, c, base, _rt| {
                if nargs == 0 {
                    miss!()
                }
                let (mut at, first) = num_kind(&reg![a + 4]);
                let (mut pick, mut rest) = (a as usize + 4, 0);
                for i in a as usize + 5..a as usize + 4 + nargs {
                    let (x, k) = num_kind(&reg![i]);
                    rest |= k;
                    if let (Some(p), Some(x)) = (at, x)
                        && if $max { p < x } else { x < p }
                    {
                        (pick, at) = (i, Some(x));
                    }
                }
                record(base, call, first | rest << fb::ARGS_SHIFT);
                if (first | rest) & !(fb::SMALL | fb::FLOAT) != 0 {
                    miss!()
                }
                reg![a] = reg![pick];
                land1!()
            }
        }
    };
}

min_max!(ff_min, false);
min_max!(ff_max, true);

fast_entry! {
    /// `assert(v, ...)` with a truthy `v`: the arguments are the results.
    ff_assert => |call, a, nargs, c, base, _rt| {
        if nargs < 1 || reg![a + 4].is_falsy() {
            miss!()
        }
        set_closure!(unsafe { frame::closure(base) });
        if call.op() == Op::TAILCALL {
            // `b` is the argument count as RETURN wants it (0: up to `top`).
            tail!(
                crate::vm::ops::call::op_return,
                insn = Slot::insn(Instruction::ret(Reg(a + HDR as u8), call.b()))
            )
        }
        let n = if c == 0 { nargs } else { (c as usize - 1).min(nargs) };
        unsafe { copy_values(base.add(a as usize), base.add(a as usize + HDR), n) };
        if c == 0 {
            let ts = thread!();
            ts.top = ts.slot_index(base) + a as usize + n;
        } else {
            unsafe { fill_nil(base.add(a as usize + n), c as usize - 1 - n) };
        }
        next!()
    }
}

fast_entry! {
    /// `setmetatable(t, mt)` on a table with no metatable yet: nothing to
    /// protect, so it cannot fail. The moves the shapes remember are stores.
    ff_setmetatable => |call, a, nargs, c, _base, rt| {
        if nargs != 2 {
            miss!()
        }
        let Some(t) = reg![a + 4].get_table() else {
            miss!()
        };
        let Some(mt) = reg![a + 5].get_table() else {
            miss!()
        };
        match t.set_metatable_fast(rt.mutation(), mt, rt.symbols()) {
            Some(false) => {
                reg![a] = Value::table(t);
                land1!()
            }
            Some(true) => {
                reg![a] = Value::table(t);
                land1!(gc)
            }
            None => {}
        }
        if t.metatable().is_some() {
            miss!()
        }
        t.set_metatable(rt, Some(mt));
        reg![a] = Value::table(t);
        land1!(gc)
    }
}

fast_entry! {
    /// `string.sub(s, i [, j])` with inline-integer positions.
    ff_sub => |call, a, nargs, c, _base, rt| {
        if nargs != 2 && nargs != 3 {
            miss!()
        }
        let Some(s) = reg![a + 4].get_string() else {
            miss!()
        };
        let Some(i) = reg![a + 5].get_small() else {
            miss!()
        };
        let j = if nargs == 3 {
            let Some(j) = reg![a + 6].get_small() else {
                miss!()
            };
            i64::from(j)
        } else {
            -1
        };
        let bytes = s.as_bytes();
        let len = bytes.len();
        let start = crate::builtin::posrelat(i64::from(i), len).max(1);
        let end = crate::builtin::posrelat(j, len).min(len as i64);
        let r = if start <= end {
            LuaString::new(rt, &bytes[(start - 1) as usize..end as usize])
        } else {
            LuaString::new(rt, b"")
        };
        reg![a] = Value::string(r);
        land1!(gc)
    }
}
