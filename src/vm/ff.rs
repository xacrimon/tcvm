//! Fast entries of builtins: handlers that do the common argument
//! shape inline from the CALL's registers and leave every other shape to
//! `native_call`, so the builtin stays the complete implementation.

use crate::env::LuaString;
use crate::env::value::Value;
use crate::instruction::{Instruction, Op, Reg};
use crate::vm::abi::{Slot, handler};
use crate::vm::frame::{self, HDR, copy_values, fill_nil};
use crate::vm::native::native_call;

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
    /// Not the common shape: leave the call to the full builtin.
    Miss,
}

/// A one-argument math builtin: `$float`/`$small` map a float or inline
/// integer argument to a `Math1`.
macro_rules! math1 {
    ($name:ident, |$x:ident| $float:expr, |$i:ident| $small:expr) => {
        fast_entry! {
            $name => |call, a, nargs, c, _base, _rt| {
                if nargs != 1 {
                    miss!()
                }
                let arg = &reg![a + 4];
                let r = if arg.is_float() {
                    let $x = arg.read_float();
                    $float
                } else if let Some($i) = arg.get_small() {
                    $small
                } else {
                    Math1::Miss
                };
                match r {
                    Math1::Float(f) => reg![a].write_float(f),
                    Math1::Small(i) => reg![a] = Value::small(i),
                    Math1::Miss => miss!(),
                }
                land1!()
            }
        }
    };
}

math1!(ff_sqrt, |x| Math1::Float(x.sqrt()), |i| Math1::Float(
    f64::from(i).sqrt()
));
math1!(ff_sin, |x| Math1::Float(x.sin()), |i| Math1::Float(
    f64::from(i).sin()
));
math1!(ff_cos, |x| Math1::Float(x.cos()), |i| Math1::Float(
    f64::from(i).cos()
));
// `abs(i32::MIN)` leaves the inline range.
math1!(ff_abs, |x| Math1::Float(x.abs()), |i| i
    .checked_abs()
    .map_or(Math1::Miss, Math1::Small));
// A rounded float becomes an integer when it fits inline; the boxed range is
// left to the builtin. `as` saturates and maps NaN to 0, so the round trip
// only holds for an integral value in i32 range (-0.0 becomes 0, as in Lua).
math1!(
    ff_floor,
    |x| {
        let r = x.floor();
        if r as i32 as f64 == r {
            Math1::Small(r as i32)
        } else {
            Math1::Miss
        }
    },
    |i| Math1::Small(i)
);
math1!(
    ff_ceil,
    |x| {
        let r = x.ceil();
        if r as i32 as f64 == r {
            Math1::Small(r as i32)
        } else {
            Math1::Miss
        }
    },
    |i| Math1::Small(i)
);

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
    /// protect, so it cannot fail.
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
