use std::cell::RefCell;

use rand_core::Rng;
use rand_pcg::Pcg64;

use crate::Context;
use crate::builtin::util::{
    arg_error, check_any, check_integer, check_number, compare_error_msg, float_to_integer,
    num_to_value,
};
use crate::env::function::NativeKind;
use crate::env::{
    Error, Function, LuaString, NativeClosure, NativeFn, Stack, Table, Userdata, Value,
};
use crate::vm::interp::{self, Handler};
use crate::vm::num;

pub fn load<'gc>(ctx: Context<'gc>) {
    // Third column: the CALL entry, `call` unless the builtin has a fast path.
    let call: Handler = interp::op_call_native;
    let fns: &[(&str, NativeFn, Handler)] = &[
        ("abs", lua_abs, interp::ff_abs),
        ("acos", lua_acos, call),
        ("asin", lua_asin, call),
        ("atan", lua_atan, call),
        ("ceil", lua_ceil, interp::ff_ceil),
        ("cos", lua_cos, interp::ff_cos),
        ("deg", lua_deg, call),
        ("exp", lua_exp, call),
        ("floor", lua_floor, interp::ff_floor),
        ("fmod", lua_fmod, call),
        ("frexp", lua_frexp, call),
        ("ldexp", lua_ldexp, call),
        ("log", lua_log, call),
        ("max", lua_max, call),
        ("min", lua_min, call),
        ("modf", lua_modf, call),
        ("rad", lua_rad, call),
        ("random", lua_random, call),
        ("randomseed", lua_randomseed, call),
        ("sin", lua_sin, interp::ff_sin),
        ("sqrt", lua_sqrt, interp::ff_sqrt),
        ("tan", lua_tan, call),
        ("tointeger", lua_tointeger, call),
        ("type", lua_type, call),
        ("ult", lua_ult, call),
    ];

    // Shared PRNG state for `random`/`randomseed`, mirroring Lua's per-closure
    // `RanState` userdata held as upvalue 0 of both functions.
    let rng = Userdata::new(ctx.mutation(), RefCell::new(RngState::from_entropy()), 0);

    let rng_upvalues = [Value::userdata(rng)];
    let lib = Table::new(ctx);
    for &(name, handler, entry) in fns {
        let upvalues: &[Value<'gc>] = if name == "random" || name == "randomseed" {
            &rng_upvalues
        } else {
            &[]
        };
        let handler = Function::new_native_with_entry(
            ctx.mutation(),
            NativeKind::Plain(handler),
            upvalues,
            entry,
        );
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(handler));
    }

    let set = |name: &str, v: Value<'gc>| {
        lib.raw_set(ctx, Value::string(LuaString::new(ctx, name.as_bytes())), v);
    };
    set("pi", Value::float(std::f64::consts::PI));
    set("huge", Value::float(f64::INFINITY));
    set("maxinteger", Value::integer(ctx.mutation(), i64::MAX));
    set("mininteger", Value::integer(ctx.mutation(), i64::MIN));

    let lib_name = Value::string(LuaString::new(ctx, b"math"));
    ctx.globals().raw_set(ctx, lib_name, Value::table(lib));
}

// ---------------------------------------------------------------------------
// Functions returning a float of a single numeric argument
// ---------------------------------------------------------------------------

macro_rules! float_unary {
    ($name:ident, $fname:literal, $op:expr) => {
        fn $name<'gc>(
            ctx: Context<'gc>,
            _closure: &NativeClosure<'gc>,
            mut stack: Stack<'gc, '_>,
        ) -> Result<(), Error<'gc>> {
            let x = check_number(ctx, stack.get(0), $fname, 1)?;
            let f: fn(f64) -> f64 = $op;
            stack.ret1(Value::float(f(x)));
            Ok(())
        }
    };
}

float_unary!(lua_acos, "acos", f64::acos);
float_unary!(lua_asin, "asin", f64::asin);
float_unary!(lua_cos, "cos", f64::cos);
float_unary!(lua_exp, "exp", f64::exp);
float_unary!(lua_sin, "sin", f64::sin);
float_unary!(lua_sqrt, "sqrt", f64::sqrt);
float_unary!(lua_tan, "tan", f64::tan);
float_unary!(lua_deg, "deg", f64::to_degrees);
float_unary!(lua_rad, "rad", f64::to_radians);

fn lua_abs<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let v = stack.get(0);
    let result = if let Some(i) = v.get_integer() {
        // Wrapping matches Lua: abs(mininteger) == mininteger.
        Value::integer(ctx.mutation(), i.wrapping_abs())
    } else {
        Value::float(check_number(ctx, v, "abs", 1)?.abs())
    };
    stack.ret1(result);
    Ok(())
}

/// `atan(y [, x])` — two-argument form is `atan2`; `x` defaults to 1.
fn lua_atan<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let y = check_number(ctx, stack.get(0), "atan", 1)?;
    let x_arg = stack.get(1);
    let x = if x_arg.is_nil() {
        1.0
    } else {
        check_number(ctx, x_arg, "atan", 2)?
    };
    stack.ret1(Value::float(y.atan2(x)));
    Ok(())
}

/// `log(x [, base])`. Special-cases bases 2 and 10 to their dedicated libm
/// routines, matching PUC-Lua's accuracy.
fn lua_log<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let x = check_number(ctx, stack.get(0), "log", 1)?;
    let base_arg = stack.get(1);
    let result = if base_arg.is_nil() {
        x.ln()
    } else {
        let base = check_number(ctx, base_arg, "log", 2)?;
        if base == 2.0 {
            x.log2()
        } else if base == 10.0 {
            x.log10()
        } else {
            x.ln() / base.ln()
        }
    };
    stack.ret1(Value::float(result));
    Ok(())
}

/// `fmod(x, y)` — C `fmod` for floats; for two integers, the C `%` remainder
/// (sign of the dividend), with `y == 0` an error and `y == -1` short-circuited
/// to avoid overflow on `mininteger % -1`.
fn lua_fmod<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let a = stack.get(0);
    let b = stack.get(1);
    let result = if let (Some(x), Some(y)) = (a.get_integer(), b.get_integer()) {
        if y == 0 {
            return Err(arg_error(ctx, "fmod", 2, "zero"));
        } else if y == -1 {
            Value::integer(ctx.mutation(), 0)
        } else {
            Value::integer(ctx.mutation(), x % y)
        }
    } else {
        let x = check_number(ctx, a, "fmod", 1)?;
        let y = check_number(ctx, b, "fmod", 2)?;
        Value::float(x % y)
    };
    stack.ret1(result);
    Ok(())
}

/// `modf(x)` — `(integral_part, fractional_part)`, both floats.
fn lua_modf<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let x = check_number(ctx, stack.get(0), "modf", 1)?;
    // Lua 5.5 returns the integral part as an integer when it fits (pushnumint).
    let (ip, fp) = if x.is_infinite() {
        // Lua's modf returns the integral part (±inf) and a +0.0 fractional
        // part (it special-cases `n == ip`), regardless of sign.
        (Value::float(x), 0.0_f64)
    } else {
        (num_to_value(ctx.mutation(), x.trunc()), x.fract())
    };
    stack.replace(&[ip, Value::float(fp)]);
    Ok(())
}

/// `x = m * 2^e` with `0.5 <= |m| < 1`; zeros, infinities and NaN come back
/// as `(x, 0)`.
fn frexp(x: f64) -> (f64, i32) {
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    match exp {
        0 if x == 0.0 => (x, 0),
        0 => {
            // Subnormal: scale by 2^64 into the normal range first.
            let (m, e) = frexp(x * f64::from_bits(0x43f0_0000_0000_0000));
            (m, e - 64)
        }
        0x7ff => (x, 0),
        _ => (
            f64::from_bits(bits & !(0x7ff << 52) | (0x3fe << 52)),
            exp - 0x3fe,
        ),
    }
}

/// `x * 2^n` with one rounding (musl's `scalbn`): `2^n` is out of range for
/// large `|n|`, so scale in steps first.
fn ldexp(mut x: f64, mut n: i32) -> f64 {
    let p1023 = f64::from_bits(0x7fe0_0000_0000_0000);
    // 2^-1022 * 2^53: the extra 2^53 keeps a subnormal result from rounding
    // in both steps.
    let pm969 = f64::from_bits(0x0360_0000_0000_0000);
    if n > 1023 {
        x *= p1023;
        n -= 1023;
        if n > 1023 {
            x *= p1023;
            n = (n - 1023).min(1023);
        }
    } else if n < -1022 {
        x *= pm969;
        n += 1022 - 53;
        if n < -1022 {
            x *= pm969;
            n = (n + 1022 - 53).max(-1022);
        }
    }
    x * f64::from_bits(((0x3ff + n) as u64) << 52)
}

/// `frexp(x)` — [`frexp`]'s `(m, e)`, `e` as an integer.
fn lua_frexp<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let x = check_number(ctx, stack.get(0), "frexp", 1)?;
    let (m, e) = frexp(x);
    stack.replace(&[Value::float(m), Value::integer(ctx.mutation(), e.into())]);
    Ok(())
}

/// `ldexp(m, e)` — `m * 2^e`.
fn lua_ldexp<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let m = check_number(ctx, stack.get(0), "ldexp", 1)?;
    // Truncated to `int`, as lmathlib casts it.
    let e = check_integer(ctx, stack.get(1), "ldexp", 2)? as i32;
    stack.ret1(Value::float(ldexp(m, e)));
    Ok(())
}

fn lua_ceil<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    round_to_int(ctx, stack, "ceil", f64::ceil)
}

fn lua_floor<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    round_to_int(ctx, stack, "floor", f64::floor)
}

/// Shared `floor`/`ceil` body: integers pass through unchanged; floats are
/// rounded, then returned as an integer when the result fits in `i64`, else as
/// a float (Lua's `pushnumint` — note this does *not* error on huge values).
fn round_to_int<'gc>(
    ctx: Context<'gc>,
    mut stack: Stack<'gc, '_>,
    fname: &str,
    round: fn(f64) -> f64,
) -> Result<(), Error<'gc>> {
    let v = stack.get(0);
    let result = if let Some(i) = v.get_integer() {
        Value::integer(ctx.mutation(), i)
    } else {
        num_to_value(ctx.mutation(), round(check_number(ctx, v, fname, 1)?))
    };
    stack.ret1(result);
    Ok(())
}

fn lua_max<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    select_extreme(ctx, stack, "max", false)
}

fn lua_min<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    select_extreme(ctx, stack, "min", true)
}

/// Shared `max`/`min` body: returns the argument that is largest (or smallest),
/// preserving its original subtype. Per the manual the result is selected "with
/// the `<` operator", so comparisons use Lua ordering semantics (no string→
/// number coercion); a non-orderable pair raises the VM's "attempt to compare"
/// error rather than a bad-argument error. Requires at least one argument.
fn select_extreme<'gc>(
    ctx: Context<'gc>,
    mut stack: Stack<'gc, '_>,
    fname: &str,
    want_min: bool,
) -> Result<(), Error<'gc>> {
    check_any(ctx, &stack, fname, 1)?;
    let n = stack.len();
    let mut best = stack.get(0);
    for i in 1..n {
        let v = stack.get(i);
        // max keeps `v` when best < v; min keeps `v` when v < best.
        let (lhs, rhs) = if want_min { (v, best) } else { (best, v) };
        let lt = if let (Some(x), Some(y)) = (lhs.get_integer(), rhs.get_integer()) {
            x < y
        } else if let (Some(x), Some(y)) = (lhs.get_float(), rhs.get_float()) {
            x < y
        } else if let (Some(x), Some(y)) = (lhs.get_integer(), rhs.get_float()) {
            num::lt_int_float(x, y)
        } else if let (Some(x), Some(y)) = (lhs.get_float(), rhs.get_integer()) {
            num::lt_float_int(x, y)
        } else if let (Some(x), Some(y)) = (lhs.get_string(), rhs.get_string()) {
            x < y
        } else {
            return Err(Error::from_str(ctx, &compare_error_msg(lhs, rhs)));
        };
        if lt {
            best = v;
        }
    }
    stack.ret1(best);
    Ok(())
}

/// `tointeger(x)` — the integer value of `x` if it has one, else `nil`. No
/// string coercion, matching `lua_tointegerx`.
fn lua_tointeger<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let v = stack.get(0);
    let mc = ctx.mutation();
    let result = if let Some(i) = v.get_integer() {
        Value::integer(mc, i)
    } else if let Some(f) = v.get_float() {
        float_to_integer(f).map_or(Value::nil(), |i| Value::integer(mc, i))
    } else if let Some(s) = v.get_string() {
        // Lua coerces a numeric string, then applies the same int/float rule.
        crate::builtin::util::str_to_number(s.as_bytes())
            .and_then(|n| n.to_integer())
            .map_or(Value::nil(), |i| Value::integer(mc, i))
    } else {
        Value::nil()
    };
    stack.ret1(result);
    Ok(())
}

/// `type(x)` — `"integer"`, `"float"`, or `nil` if `x` is not a number.
fn lua_type<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    check_any(ctx, &stack, "type", 1)?;
    let v = stack.get(0);
    let result = if v.get_integer().is_some() {
        Value::string(LuaString::new(ctx, b"integer"))
    } else if v.get_float().is_some() {
        Value::string(LuaString::new(ctx, b"float"))
    } else {
        Value::nil()
    };
    stack.ret1(result);
    Ok(())
}

/// `ult(m, n)` — unsigned `m < n` over the two integers' bit patterns.
fn lua_ult<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let m = check_integer(ctx, stack.get(0), "ult", 1)?;
    let n = check_integer(ctx, stack.get(1), "ult", 2)?;
    stack.ret1(Value::boolean((m as u64) < (n as u64)));
    Ok(())
}

// ---------------------------------------------------------------------------
// PRNG (`random` / `randomseed`)
// ---------------------------------------------------------------------------
//
// Lua 5.5 ships a per-state xoshiro256**. We deliberately back ours with
// `rand_pcg` instead, so the value *stream* differs from PUC-Lua, but every
// observable *semantic* matches: `random()` floats in `[0,1)`, inclusive
// integer ranges, the `random(0)` full-width case, the rejection-sampling
// projection, and the two-integer `randomseed` return. The state lives in a
// `RefCell` inside a `Userdata` shared as upvalue 0 of both functions —
// the direct analogue of Lua's shared `RanState` upvalue.

struct RngState {
    rng: Pcg64,
}

impl RngState {
    fn from_seeds(n1: u64, n2: u64) -> Self {
        // Map the two seed words onto PCG's 128-bit state/stream, then discard
        // a few outputs to wash out the low-quality initial state (Lua discards
        // 16 nextrand values after seeding for the same reason).
        let state = ((n1 as u128) << 64) | n2 as u128;
        let stream = ((n2 as u128) << 64) | 0x9e37_79b9_7f4a_7c15;
        let mut rng = Pcg64::new(state, stream);
        for _ in 0..16 {
            rng.next_u64();
        }
        RngState { rng }
    }

    fn from_entropy() -> Self {
        let mut buf = [0u8; 16];
        getrandom::fill(&mut buf).expect("OS entropy source unavailable");
        let seed = u128::from_ne_bytes(buf);
        Self::from_seeds((seed >> 64) as u64, seed as u64)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }
}

/// Seed word from the OS entropy source.
fn os_seed() -> u64 {
    getrandom::u64().expect("OS entropy source unavailable")
}

/// A 53-bit random value scaled into `[0, 1)` (Lua's `I2d`).
#[inline]
fn unit_float(rv: u64) -> f64 {
    (rv >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Lua's `project`: rejection-sample `ran` uniformly into `[0, n]`. `lim` is the
/// smallest Mersenne number `>= n`; we keep drawing while the masked value
/// exceeds `n`.
fn project(mut ran: u64, n: u64, st: &mut RngState) -> u64 {
    let mut lim = n;
    let mut sh = 1u32;
    while lim & lim.wrapping_add(1) != 0 {
        lim |= lim >> sh;
        sh <<= 1;
    }
    loop {
        ran &= lim;
        if ran <= n {
            return ran;
        }
        ran = st.next_u64();
    }
}

#[inline]
fn rng_state<'gc>(closure: &NativeClosure<'gc>) -> Userdata<'gc> {
    closure.upvalues()[0]
        .get_userdata()
        .expect("random/randomseed upvalue 0 must be the RNG userdata")
}

/// `random([m [, n]])` — float in `[0,1)` (no args); a full-width random integer
/// (`random(0)`); or an integer in `[1,m]` / `[m,n]`.
fn lua_random<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    // Parse + validate before drawing so error paths don't perturb the stream.
    #[derive(Clone, Copy)]
    enum Mode {
        Float,
        Bits,
        Range(i64, i64),
    }
    let mode = match stack.len() {
        0 => Mode::Float,
        1 => {
            let up = check_integer(ctx, stack.get(0), "random", 1)?;
            if up == 0 {
                Mode::Bits
            } else {
                Mode::Range(1, up)
            }
        }
        2 => {
            let low = check_integer(ctx, stack.get(0), "random", 1)?;
            let up = check_integer(ctx, stack.get(1), "random", 2)?;
            Mode::Range(low, up)
        }
        _ => return Err(Error::from_str(ctx, "wrong number of arguments")),
    };
    if let Mode::Range(low, up) = mode
        && low > up
    {
        return Err(arg_error(ctx, "random", 1, "interval is empty"));
    }

    let result = rng_state(closure)
        .with_data::<RefCell<RngState>, Value<'gc>>(|cell| {
            let mut st = cell.borrow_mut();
            let rv = st.next_u64();
            match mode {
                Mode::Float => Value::float(unit_float(rv)),
                Mode::Bits => Value::integer(ctx.mutation(), rv as i64),
                Mode::Range(low, up) => {
                    let span = (up as u64).wrapping_sub(low as u64);
                    let p = project(rv, span, &mut st);
                    Value::integer(ctx.mutation(), p.wrapping_add(low as u64) as i64)
                }
            }
        })
        .expect("RNG userdata payload type mismatch");

    stack.ret1(result);
    Ok(())
}

/// `randomseed([x [, y]])` — reseed (from entropy with no argument) and return
/// the two seed integers actually used.
fn lua_randomseed<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    // No argument at all → entropy reseed; an explicit arg (even nil) goes
    // through `check_integer`, like Lua's `luaL_checkinteger`.
    let provided = if stack.is_empty() {
        None
    } else {
        let n1 = check_integer(ctx, stack.get(0), "randomseed", 1)? as u64;
        let n2 = if stack.get(1).is_nil() {
            0
        } else {
            check_integer(ctx, stack.get(1), "randomseed", 2)? as u64
        };
        Some((n1, n2))
    };

    let (s1, s2) = rng_state(closure)
        .with_data::<RefCell<RngState>, (u64, u64)>(|cell| {
            let mut st = cell.borrow_mut();
            let seeds = match provided {
                Some(pair) => pair,
                None => (os_seed(), st.next_u64()),
            };
            *st = RngState::from_seeds(seeds.0, seeds.1);
            seeds
        })
        .expect("RNG userdata payload type mismatch");

    stack.replace(&[
        Value::integer(ctx.mutation(), s1 as i64),
        Value::integer(ctx.mutation(), s2 as i64),
    ]);
    Ok(())
}
