//! Shared helpers for the standard library: Lua-faithful number/value
//! stringification and small argument-coercion routines used across the
//! `basic`, `string`, `table`, and `io` libraries.

use crate::dmm::Gc;
use crate::env::{Error, LuaString, Value};
use crate::lua::Context;

/// Append the canonical Lua textual form of an integer.
pub(crate) fn push_int(out: &mut Vec<u8>, i: i64) {
    out.extend_from_slice(i.to_string().as_bytes());
}

/// Append the canonical Lua 5.5 textual form of a float. Lua picks the
/// shortest `"%.Ng"` that reads back to the same value, scanning `N` upward
/// from the historical default of 14 to the round-trip-sufficient 17, then
/// tags any integer-looking result with a trailing `".0"`.
pub(crate) fn push_float(out: &mut Vec<u8>, f: f64) {
    if f.is_nan() {
        out.extend_from_slice(b"nan");
        return;
    }
    if f.is_infinite() {
        out.extend_from_slice(if f < 0.0 { b"-inf" } else { b"inf" });
        return;
    }
    // 17 significant digits always round-trip an f64, so the loop terminates
    // with a faithful representation even if 14..16 never match.
    let mut s = format_g(f, 17);
    for p in 14..17 {
        let cand = format_g(f, p);
        if cand.parse::<f64>() == Ok(f) {
            s = cand;
            break;
        }
    }
    // Tag an otherwise integer-looking float so round-trips stay floats.
    if !s
        .bytes()
        .any(|b| matches!(b, b'.' | b'e' | b'E' | b'n' | b'N' | b'i' | b'I'))
    {
        s.push_str(".0");
    }
    out.extend_from_slice(s.as_bytes());
}

/// C `printf` `%.*g` for finite `f` with `prec` significant digits. The `%e`
/// vs `%f` choice and trailing-zero trimming follow the C standard; the
/// exponent is rendered C-style (signed, at least two digits).
fn format_g(f: f64, prec: usize) -> String {
    let p = prec.max(1);
    // Format in scientific first to read off the decimal exponent.
    let sci = format!("{:.*e}", p - 1, f);
    let e = sci.find('e').expect("scientific format always has 'e'");
    let exp: i32 = sci[e + 1..].parse().expect("valid exponent");
    if exp < -4 || exp >= p as i32 {
        let mut mantissa = sci[..e].to_string();
        strip_trailing_zeros(&mut mantissa);
        let sign = if exp < 0 { '-' } else { '+' };
        let mag = exp.unsigned_abs();
        if mag < 10 {
            format!("{mantissa}e{sign}0{mag}")
        } else {
            format!("{mantissa}e{sign}{mag}")
        }
    } else {
        let dec = (p as i32 - 1 - exp).max(0) as usize;
        let mut s = format!("{f:.dec$}");
        strip_trailing_zeros(&mut s);
        s
    }
}

fn strip_trailing_zeros(s: &mut String) {
    if !s.contains('.') {
        return;
    }
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
}

/// `tostring` without metamethod dispatch: the default representation Lua
/// uses when there is no `__tostring`/`__name`. Numbers and strings get their
/// literal form; all reference types get `"<type>: 0x<addr>"`, with native
/// functions tagged `"function: builtin: ..."` to match PUC-Lua.
pub(crate) fn basic_tostring<'gc>(ctx: Context<'gc>, v: Value<'gc>) -> LuaString<'gc> {
    if let Some(s) = v.get_string() {
        return s;
    }
    let mut out: Vec<u8> = Vec::new();
    if v.is_nil() {
        out.extend_from_slice(b"nil");
    } else if let Some(b) = v.get_boolean() {
        out.extend_from_slice(if b { b"true" } else { b"false" });
    } else if let Some(i) = v.get_integer() {
        push_int(&mut out, i);
    } else if let Some(f) = v.get_float() {
        push_float(&mut out, f);
    } else if let Some(t) = v.get_table() {
        push_addr(&mut out, "table", Gc::as_ptr(t.inner()) as *const ());
    } else if let Some(f) = v.get_function() {
        if f.as_native().is_some() {
            out.extend_from_slice(b"function: builtin: ");
            push_ptr(&mut out, Gc::as_ptr(f.inner()) as *const ());
        } else {
            push_addr(&mut out, "function", Gc::as_ptr(f.inner()) as *const ());
        }
    } else if let Some(t) = v.get_thread() {
        push_addr(&mut out, "thread", Gc::as_ptr(t.inner()) as *const ());
    } else if let Some(u) = v.get_userdata() {
        push_addr(&mut out, "userdata", Gc::as_ptr(u.inner()) as *const ());
    }
    LuaString::new(ctx, &out)
}

// ---------------------------------------------------------------------------
// Argument coercion (shared `luaL_check*` analogues)
// ---------------------------------------------------------------------------

/// Coerce `v` to a float, mirroring `luaL_checknumber` (numeric strings
/// included). `fname`/`n` build the standard bad-argument message on failure.
pub(crate) fn check_number<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<f64, Error<'gc>> {
    to_number(v).ok_or_else(|| {
        Error::from_str(
            ctx,
            &format!(
                "bad argument #{n} to '{fname}' (number expected, got {})",
                v.type_name()
            ),
        )
    })
}

/// Coerce `v` to an integer, mirroring `luaL_checkinteger`: integers pass
/// through, floats must be exactly integral, numeric strings are parsed. A
/// non-integral number is reported distinctly from a non-number.
pub(crate) fn check_integer<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<i64, Error<'gc>> {
    if let Some(i) = to_integer(v) {
        return Ok(i);
    }
    if v.get_float().is_some() {
        return Err(Error::from_str(
            ctx,
            &format!("bad argument #{n} to '{fname}' (number has no integer representation)"),
        ));
    }
    Err(Error::from_str(
        ctx,
        &format!(
            "bad argument #{n} to '{fname}' (number expected, got {})",
            v.type_name()
        ),
    ))
}

pub(crate) fn to_number<'gc>(v: Value<'gc>) -> Option<f64> {
    if let Some(i) = v.get_integer() {
        Some(i as f64)
    } else if let Some(f) = v.get_float() {
        Some(f)
    } else if let Some(s) = v.get_string() {
        std::str::from_utf8(s.as_bytes()).ok()?.trim().parse().ok()
    } else {
        None
    }
}

/// Integer view of a value: integers as-is, floats with an exact integral
/// value, and numeric strings that name an integer.
pub(crate) fn to_integer<'gc>(v: Value<'gc>) -> Option<i64> {
    if let Some(i) = v.get_integer() {
        return Some(i);
    }
    if let Some(f) = v.get_float() {
        return float_to_integer(f);
    }
    if let Some(s) = v.get_string() {
        let t = std::str::from_utf8(s.as_bytes()).ok()?.trim();
        if let Ok(i) = t.parse::<i64>() {
            return Some(i);
        }
        return t.parse::<f64>().ok().and_then(float_to_integer);
    }
    None
}

/// Exact float→integer conversion (`lua_numbertointeger`): succeeds only when
/// `f` is integral and within `i64` range.
pub(crate) fn float_to_integer(f: f64) -> Option<i64> {
    if f.fract() == 0.0 && f >= -(2f64.powi(63)) && f < 2f64.powi(63) {
        Some(f as i64)
    } else {
        None
    }
}

/// Lua's `pushnumint`: an integral float collapses to an integer when it fits
/// in `i64`, otherwise stays a float. Used by `math.floor`/`ceil`/`modf`.
pub(crate) fn num_to_value<'gc>(f: f64) -> Value<'gc> {
    match float_to_integer(f) {
        Some(i) => Value::integer(i),
        None => Value::float(f),
    }
}

/// Lua raw equality (`==` without metamethods): numbers compare by value across
/// the integer/float divide, everything else by identity/content.
pub(crate) fn raw_eq<'gc>(a: Value<'gc>, b: Value<'gc>) -> bool {
    use crate::env::ValueKind::{Float, Integer};
    match (a.kind(), b.kind()) {
        (Integer, Integer) => a.get_integer() == b.get_integer(),
        // Bitwise `Value` eq would mishandle NaN and -0.0, so compare as f64.
        (Float, Float) => a.get_float() == b.get_float(),
        (Integer, Float) => float_eq_int(b.get_float().unwrap(), a.get_integer().unwrap()),
        (Float, Integer) => float_eq_int(a.get_float().unwrap(), b.get_integer().unwrap()),
        (ka, kb) if ka == kb => a == b,
        _ => false,
    }
}

fn float_eq_int(f: f64, i: i64) -> bool {
    float_to_integer(f) == Some(i)
}

fn push_addr(out: &mut Vec<u8>, kind: &str, ptr: *const ()) {
    out.extend_from_slice(kind.as_bytes());
    out.extend_from_slice(b": ");
    push_ptr(out, ptr);
}

fn push_ptr(out: &mut Vec<u8>, ptr: *const ()) {
    out.extend_from_slice(format!("{ptr:p}").as_bytes());
}
