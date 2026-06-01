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

/// Convert a string to a Lua number following the lexer's rules: optional
/// surrounding whitespace and sign, decimal integer/float, and `0x` hex
/// integer (wrapping, per Lua) / hex float (`0x1.8p3`). Returns `None` for
/// anything non-numeric — notably `"inf"`/`"nan"`, which Rust's `f64::parse`
/// would otherwise accept but Lua rejects. Shared by `tonumber` and
/// `math.tointeger`.
pub(crate) fn str_to_number<'gc>(b: &[u8]) -> Option<Value<'gc>> {
    let s = std::str::from_utf8(b)
        .ok()?
        .trim_matches(|c: char| c.is_ascii_whitespace());
    if s.is_empty() {
        return None;
    }
    let (neg, body) = match s.as_bytes()[0] {
        b'+' => (false, &s[1..]),
        b'-' => (true, &s[1..]),
        _ => (false, s),
    };
    if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        if hex.is_empty() {
            return None;
        }
        // `.`/`p` means a hex float; otherwise a (wrapping) hex integer.
        if hex.contains(['.', 'p', 'P']) {
            let f = crate::parser::lit::parse_hex_float(body)?;
            return Some(Value::float(if neg { -f } else { f }));
        }
        let mut acc: u64 = 0;
        for c in hex.bytes() {
            let d = (c as char).to_digit(16)? as u64;
            acc = acc.wrapping_mul(16).wrapping_add(d);
        }
        let i = acc as i64;
        return Some(Value::integer(if neg { i.wrapping_neg() } else { i }));
    }
    // Decimal. Restrict to numeric characters so `f64::parse` can't sneak in
    // `inf`/`nan`/`infinity`.
    if !body
        .bytes()
        .all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
        || !body.bytes().any(|c| c.is_ascii_digit())
    {
        return None;
    }
    if let Ok(i) = s.parse::<i64>() {
        return Some(Value::integer(i));
    }
    s.parse::<f64>().ok().map(Value::float)
}

/// Parse `b` as an integer written in `base` (2..=36), with optional
/// surrounding whitespace and sign, accumulating with wrapping arithmetic
/// (Lua's `l_str2int`). Letters `a..z`/`A..Z` are digits 10..35. Returns
/// `None` on an empty string or any digit `>= base`.
pub(crate) fn str_to_int_base(b: &[u8], base: u32) -> Option<i64> {
    let trimmed = {
        let s = b;
        let mut start = 0;
        let mut end = s.len();
        while start < end && s[start].is_ascii_whitespace() {
            start += 1;
        }
        while end > start && s[end - 1].is_ascii_whitespace() {
            end -= 1;
        }
        &s[start..end]
    };
    if trimmed.is_empty() {
        return None;
    }
    let (neg, digits) = match trimmed[0] {
        b'+' => (false, &trimmed[1..]),
        b'-' => (true, &trimmed[1..]),
        _ => (false, trimmed),
    };
    if digits.is_empty() {
        return None;
    }
    let mut acc: i64 = 0;
    for &c in digits {
        let d = match c {
            b'0'..=b'9' => (c - b'0') as u32,
            b'a'..=b'z' => (c - b'a') as u32 + 10,
            b'A'..=b'Z' => (c - b'A') as u32 + 10,
            _ => return None,
        };
        if d >= base {
            return None;
        }
        acc = acc.wrapping_mul(base as i64).wrapping_add(d as i64);
    }
    Some(if neg { acc.wrapping_neg() } else { acc })
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
        // Via the lexer rules (`str_to_number`), not raw `f64::parse`, so
        // `"inf"`/`"nan"` are rejected as Lua's `luaL_checknumber` does.
        let n = str_to_number(s.as_bytes())?;
        n.get_integer().map(|i| i as f64).or_else(|| n.get_float())
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
        // Same lexer-rule coercion as `to_number`, then the int/float rule.
        let n = str_to_number(s.as_bytes())?;
        return n
            .get_integer()
            .or_else(|| n.get_float().and_then(float_to_integer));
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
