use std::cell::RefCell;

use rand_core::Rng;
use rand_pcg::Pcg64;

use crate::Context;
use crate::builtin::util::{
    self, check_integer, check_number, compare_error_msg, float_to_integer, num_to_value,
};
use crate::env::{
    Error, Function, LuaString, NativeContext, NativeFn, Stack, Table, Userdata, Value,
};
use crate::vm::sequence::CallbackAction;

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("abs", lua_abs),
        ("acos", lua_acos),
        ("asin", lua_asin),
        ("atan", lua_atan),
        ("ceil", lua_ceil),
        ("cos", lua_cos),
        ("deg", lua_deg),
        ("exp", lua_exp),
        ("floor", lua_floor),
        ("fmod", lua_fmod),
        ("log", lua_log),
        ("max", lua_max),
        ("min", lua_min),
        ("modf", lua_modf),
        ("rad", lua_rad),
        ("random", lua_random),
        ("randomseed", lua_randomseed),
        ("sin", lua_sin),
        ("sqrt", lua_sqrt),
        ("tan", lua_tan),
        ("tointeger", lua_tointeger),
        ("type", lua_type),
        ("ult", lua_ult),
    ];

    // Shared PRNG state for `random`/`randomseed`, mirroring Lua's per-closure
    // `RanState` userdata held as upvalue 0 of both functions.
    let rng = Userdata::new(ctx.mutation(), RefCell::new(RngState::from_entropy()), 0);

    let lib = Table::new(ctx);
    for &(name, handler) in fns {
        let upvalues: Box<[Value<'gc>]> = if name == "random" || name == "randomseed" {
            Box::new([Value::userdata(rng)])
        } else {
            Box::new([])
        };
        let handler = Function::new_native(ctx.mutation(), handler, upvalues);
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(handler));
    }

    let set = |name: &str, v: Value<'gc>| {
        lib.raw_set(ctx, Value::string(LuaString::new(ctx, name.as_bytes())), v);
    };
    set("pi", Value::float(std::f64::consts::PI));
    set("huge", Value::float(f64::INFINITY));
    set("maxinteger", Value::integer(i64::MAX));
    set("mininteger", Value::integer(i64::MIN));

    let lib_name = Value::string(LuaString::new(ctx, b"math"));
    ctx.globals().raw_set(ctx, lib_name, Value::table(lib));
}

// ---------------------------------------------------------------------------
// Functions returning a float of a single numeric argument
// ---------------------------------------------------------------------------

macro_rules! float_unary {
    ($name:ident, $fname:literal, $op:expr) => {
        fn $name<'gc>(
            nctx: NativeContext<'gc, '_>,
            mut stack: Stack<'gc, '_>,
        ) -> Result<CallbackAction<'gc>, Error<'gc>> {
            let x = check_number(nctx.ctx, stack.get(0), $fname, 1)?;
            let f: fn(f64) -> f64 = $op;
            stack.replace(&[Value::float(f(x))]);
            Ok(CallbackAction::Return)
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
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let v = stack.get(0);
    let result = if let Some(i) = v.get_integer() {
        // Wrapping matches Lua: abs(mininteger) == mininteger.
        Value::integer(i.wrapping_abs())
    } else {
        Value::float(check_number(nctx.ctx, v, "abs", 1)?.abs())
    };
    stack.replace(&[result]);
    Ok(CallbackAction::Return)
}

/// `atan(y [, x])` — two-argument form is `atan2`; `x` defaults to 1.
fn lua_atan<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let y = check_number(nctx.ctx, stack.get(0), "atan", 1)?;
    let x_arg = stack.get(1);
    let x = if x_arg.is_nil() {
        1.0
    } else {
        check_number(nctx.ctx, x_arg, "atan", 2)?
    };
    stack.replace(&[Value::float(y.atan2(x))]);
    Ok(CallbackAction::Return)
}

/// `log(x [, base])`. Special-cases bases 2 and 10 to their dedicated libm
/// routines, matching PUC-Lua's accuracy.
fn lua_log<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let x = check_number(nctx.ctx, stack.get(0), "log", 1)?;
    let base_arg = stack.get(1);
    let result = if base_arg.is_nil() {
        x.ln()
    } else {
        let base = check_number(nctx.ctx, base_arg, "log", 2)?;
        if base == 2.0 {
            x.log2()
        } else if base == 10.0 {
            x.log10()
        } else {
            x.ln() / base.ln()
        }
    };
    stack.replace(&[Value::float(result)]);
    Ok(CallbackAction::Return)
}

/// `fmod(x, y)` — C `fmod` for floats; for two integers, the C `%` remainder
/// (sign of the dividend), with `y == 0` an error and `y == -1` short-circuited
/// to avoid overflow on `mininteger % -1`.
fn lua_fmod<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let a = stack.get(0);
    let b = stack.get(1);
    let result = if let (Some(x), Some(y)) = (a.get_integer(), b.get_integer()) {
        if y == 0 {
            return Err(Error::from_str(
                nctx.ctx,
                "bad argument #2 to 'fmod' (zero)",
            ));
        } else if y == -1 {
            Value::integer(0)
        } else {
            Value::integer(x % y)
        }
    } else {
        let x = check_number(nctx.ctx, a, "fmod", 1)?;
        let y = check_number(nctx.ctx, b, "fmod", 2)?;
        Value::float(x % y)
    };
    stack.replace(&[result]);
    Ok(CallbackAction::Return)
}

/// `modf(x)` — `(integral_part, fractional_part)`, both floats.
fn lua_modf<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let x = check_number(nctx.ctx, stack.get(0), "modf", 1)?;
    // Lua 5.5 returns the integral part as an integer when it fits (pushnumint).
    let (ip, fp) = if x.is_infinite() {
        // Lua's modf returns the integral part (±inf) and a +0.0 fractional
        // part (it special-cases `n == ip`), regardless of sign.
        (Value::float(x), 0.0_f64)
    } else {
        (num_to_value(x.trunc()), x.fract())
    };
    stack.replace(&[ip, Value::float(fp)]);
    Ok(CallbackAction::Return)
}

fn lua_ceil<'gc>(
    nctx: NativeContext<'gc, '_>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    round_to_int(nctx, stack, "ceil", f64::ceil)
}

fn lua_floor<'gc>(
    nctx: NativeContext<'gc, '_>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    round_to_int(nctx, stack, "floor", f64::floor)
}

/// Shared `floor`/`ceil` body: integers pass through unchanged; floats are
/// rounded, then returned as an integer when the result fits in `i64`, else as
/// a float (Lua's `pushnumint` — note this does *not* error on huge values).
fn round_to_int<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
    fname: &str,
    round: fn(f64) -> f64,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let v = stack.get(0);
    let result = if let Some(i) = v.get_integer() {
        Value::integer(i)
    } else {
        num_to_value(round(check_number(nctx.ctx, v, fname, 1)?))
    };
    stack.replace(&[result]);
    Ok(CallbackAction::Return)
}

fn lua_max<'gc>(
    nctx: NativeContext<'gc, '_>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    select_extreme(nctx, stack, "max", false)
}

fn lua_min<'gc>(
    nctx: NativeContext<'gc, '_>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    select_extreme(nctx, stack, "min", true)
}

/// Shared `max`/`min` body: returns the argument that is largest (or smallest),
/// preserving its original subtype. Per the manual the result is selected "with
/// the `<` operator", so comparisons use Lua ordering semantics (no string→
/// number coercion); a non-orderable pair raises the VM's "attempt to compare"
/// error rather than a bad-argument error. Requires at least one argument.
fn select_extreme<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
    fname: &str,
    want_min: bool,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let n = stack.len();
    if n == 0 {
        return Err(Error::from_str(
            nctx.ctx,
            &format!("bad argument #1 to '{fname}' (value expected)"),
        ));
    }
    let mut best = stack.get(0);
    for i in 1..n {
        let v = stack.get(i);
        // max keeps `v` when best < v; min keeps `v` when v < best.
        let (lhs, rhs) = if want_min { (v, best) } else { (best, v) };
        let lt = match util::num_lt(lhs, rhs) {
            Some(r) => r,
            None => match (lhs.get_string(), rhs.get_string()) {
                (Some(x), Some(y)) => x < y,
                _ => return Err(Error::from_str(nctx.ctx, &compare_error_msg(lhs, rhs))),
            },
        };
        if lt {
            best = v;
        }
    }
    stack.replace(&[best]);
    Ok(CallbackAction::Return)
}

/// `tointeger(x)` — the integer value of `x` if it has one, else `nil`. No
/// string coercion, matching `lua_tointegerx`.
fn lua_tointeger<'gc>(
    _nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let v = stack.get(0);
    let result = if let Some(i) = v.get_integer() {
        Value::integer(i)
    } else if let Some(f) = v.get_float() {
        float_to_integer(f).map_or(Value::nil(), Value::integer)
    } else if let Some(s) = v.get_string() {
        // Lua coerces a numeric string, then applies the same int/float rule.
        match crate::builtin::util::str_to_number(s.as_bytes()) {
            Some(n) if n.get_integer().is_some() => n,
            Some(n) => n
                .get_float()
                .and_then(float_to_integer)
                .map_or(Value::nil(), Value::integer),
            None => Value::nil(),
        }
    } else {
        Value::nil()
    };
    stack.replace(&[result]);
    Ok(CallbackAction::Return)
}

/// `type(x)` — `"integer"`, `"float"`, or `nil` if `x` is not a number.
fn lua_type<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let v = stack.get(0);
    let result = if v.get_integer().is_some() {
        Value::string(LuaString::new(nctx.ctx, b"integer"))
    } else if v.get_float().is_some() {
        Value::string(LuaString::new(nctx.ctx, b"float"))
    } else {
        Value::nil()
    };
    stack.replace(&[result]);
    Ok(CallbackAction::Return)
}

/// `ult(m, n)` — unsigned `m < n` over the two integers' bit patterns.
fn lua_ult<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let m = check_integer(nctx.ctx, stack.get(0), "ult", 1)?;
    let n = check_integer(nctx.ctx, stack.get(1), "ult", 2)?;
    stack.replace(&[Value::boolean((m as u64) < (n as u64))]);
    Ok(CallbackAction::Return)
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
        let n1 = make_seed();
        let n2 = make_seed()
            .rotate_left(17)
            .wrapping_add(0x9e37_79b9_7f4a_7c15);
        Self::from_seeds(n1, n2)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }
}

/// Non-cryptographic entropy for `randomseed()` with no argument, mixing the
/// wall clock with a stack address (the spirit of `luaL_makeseed`).
fn make_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let addr = &nanos as *const u64 as u64;
    nanos ^ addr.rotate_left(32)
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
fn rng_state<'gc>(nctx: &NativeContext<'gc, '_>) -> Userdata<'gc> {
    nctx.upvalues[0]
        .get_userdata()
        .expect("random/randomseed upvalue 0 must be the RNG userdata")
}

/// `random([m [, n]])` — float in `[0,1)` (no args); a full-width random integer
/// (`random(0)`); or an integer in `[1,m]` / `[m,n]`.
fn lua_random<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
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
            let up = check_integer(nctx.ctx, stack.get(0), "random", 1)?;
            if up == 0 {
                Mode::Bits
            } else {
                Mode::Range(1, up)
            }
        }
        2 => {
            let low = check_integer(nctx.ctx, stack.get(0), "random", 1)?;
            let up = check_integer(nctx.ctx, stack.get(1), "random", 2)?;
            Mode::Range(low, up)
        }
        _ => return Err(Error::from_str(nctx.ctx, "wrong number of arguments")),
    };
    if let Mode::Range(low, up) = mode
        && low > up
    {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'random' (interval is empty)",
        ));
    }

    let result = rng_state(&nctx)
        .with_data::<RefCell<RngState>, Value<'gc>>(|cell| {
            let mut st = cell.borrow_mut();
            let rv = st.next_u64();
            match mode {
                Mode::Float => Value::float(unit_float(rv)),
                Mode::Bits => Value::integer(rv as i64),
                Mode::Range(low, up) => {
                    let span = (up as u64).wrapping_sub(low as u64);
                    let p = project(rv, span, &mut st);
                    Value::integer(p.wrapping_add(low as u64) as i64)
                }
            }
        })
        .expect("RNG userdata payload type mismatch");

    stack.replace(&[result]);
    Ok(CallbackAction::Return)
}

/// `randomseed([x [, y]])` — reseed (from entropy with no argument) and return
/// the two seed integers actually used.
fn lua_randomseed<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    // No argument at all → entropy reseed; an explicit arg (even nil) goes
    // through `check_integer`, like Lua's `luaL_checkinteger`.
    let provided = if stack.len() == 0 {
        None
    } else {
        let n1 = check_integer(nctx.ctx, stack.get(0), "randomseed", 1)? as u64;
        let n2 = if stack.get(1).is_nil() {
            0
        } else {
            check_integer(nctx.ctx, stack.get(1), "randomseed", 2)? as u64
        };
        Some((n1, n2))
    };

    let (s1, s2) = rng_state(&nctx)
        .with_data::<RefCell<RngState>, (u64, u64)>(|cell| {
            let mut st = cell.borrow_mut();
            let seeds = match provided {
                Some(pair) => pair,
                None => (make_seed(), st.next_u64()),
            };
            *st = RngState::from_seeds(seeds.0, seeds.1);
            seeds
        })
        .expect("RNG userdata payload type mismatch");

    stack.replace(&[Value::integer(s1 as i64), Value::integer(s2 as i64)]);
    Ok(CallbackAction::Return)
}
