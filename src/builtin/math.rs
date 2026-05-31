use crate::Context;
use crate::builtin::util::{check_integer, check_number, float_to_integer, num_to_value};
use crate::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Table, Value};
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

    let lib = Table::new(ctx);
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, Box::new([]));
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
        // C modf: integral part is ±inf, fractional part is ±0.
        (Value::float(x), 0.0_f64.copysign(x))
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
/// preserving its original integer/float subtype. Requires at least one
/// argument and that all arguments be numbers.
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
            &format!("bad argument #1 to '{fname}' (number expected, got no value)"),
        ));
    }
    let mut best = stack.get(0);
    let mut best_key = check_number(nctx.ctx, best, fname, 1)?;
    for i in 1..n {
        let v = stack.get(i);
        let key = check_number(nctx.ctx, v, fname, i + 1)?;
        let take = if want_min {
            key < best_key
        } else {
            key > best_key
        };
        if take {
            best = v;
            best_key = key;
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

// `math.random` / `math.randomseed` need a home for the PRNG state (Lua 5.5
// uses a per-state xoshiro256**). Where that state lives — function upvalues,
// a `State` field, or a thread-local — is a design decision left for #27
// follow-up; stubbed for now.
fn lua_random<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!("math.random: PRNG state location is an open design decision")
}

fn lua_randomseed<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!("math.randomseed: PRNG state location is an open design decision")
}
