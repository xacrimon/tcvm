//! Shared helpers for the standard library: Lua-faithful number/value
//! stringification and small argument-coercion routines used across the
//! `basic`, `string`, `table`, and `io` libraries.

use std::pin::Pin;

use crate::dmm::{Collect, Gc, Mutation, Trace};
use crate::env::{Error, Function, LuaString, Stack, Value};
use crate::lua::{Context, StashedError, StashedValue};
use crate::vm::async_sequence::AsyncSequence;
use crate::vm::interp::MAX_TAG_LOOP;
use crate::vm::sequence::{Execution, Sequence, SequencePoll};

/// Append the canonical Lua textual form of an integer.
pub(crate) fn push_int(out: &mut Vec<u8>, i: i64) {
    out.extend_from_slice(i.to_string().as_bytes());
}

/// The VM's "attempt to compare …" runtime-error message for an ordering of two
/// non-comparable values, in operand order (`a < b`). Used where native code
/// applies the `<` operator (`math.max`/`min`, `table.sort`'s default order).
pub(crate) fn compare_error_msg<'gc>(a: Value<'gc>, b: Value<'gc>) -> String {
    let (ta, tb) = (a.type_name(), b.type_name());
    if ta == tb {
        format!("attempt to compare two {ta} values")
    } else {
        format!("attempt to compare {ta} with {tb}")
    }
}

/// Convert a string to a Lua number following the lexer's rules: optional
/// surrounding whitespace and sign, decimal integer/float, and `0x` hex
/// integer (wrapping, per Lua) / hex float (`0x1.8p3`). Returns `None` for
/// anything non-numeric — notably `"inf"`/`"nan"`, which Rust's `f64::parse`
/// would otherwise accept but Lua rejects. Shared by `tonumber` and
/// `math.tointeger`.
pub(crate) fn str_to_number(b: &[u8]) -> Option<Number> {
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
            return Some(Number::Float(if neg { -f } else { f }));
        }
        let mut acc: u64 = 0;
        for c in hex.bytes() {
            let d = (c as char).to_digit(16)? as u64;
            acc = acc.wrapping_mul(16).wrapping_add(d);
        }
        let i = acc as i64;
        return Some(Number::Int(if neg { i.wrapping_neg() } else { i }));
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
        return Some(Number::Int(i));
    }
    s.parse::<f64>().ok().map(Number::Float)
}

/// A parsed numeral before it is materialized as a `Value`.
#[derive(Clone, Copy)]
pub(crate) enum Number {
    Int(i64),
    Float(f64),
}

impl Number {
    pub(crate) fn into_value<'gc>(self, mc: &Mutation<'gc>) -> Value<'gc> {
        match self {
            Number::Int(i) => Value::integer(mc, i),
            Number::Float(f) => Value::float(f),
        }
    }

    pub(crate) fn to_float(self) -> f64 {
        match self {
            Number::Int(i) => i as f64,
            Number::Float(f) => f,
        }
    }

    /// The int/float rule of `math.tointeger`: floats only if exactly integral.
    pub(crate) fn to_integer(self) -> Option<i64> {
        match self {
            Number::Int(i) => Some(i),
            Number::Float(f) => float_to_integer(f),
        }
    }
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

/// Append the canonical Lua 5.5 textual form of a float. Lua formats with
/// `LUA_NUMBER_FMT` (`"%.15g"`) and, only if that fails to read back exactly,
/// falls straight to `LUA_NUMBER_FMT_N` (`"%.17g"`) — there is no scan through
/// the intermediate precisions, so e.g. `1/3` prints `0.33333333333333331`, not
/// the shorter `%.16g` form. Any integer-looking result gets a trailing `".0"`.
pub(crate) fn push_float(out: &mut Vec<u8>, f: f64) {
    if f.is_nan() {
        out.extend_from_slice(b"nan");
        return;
    }
    if f.is_infinite() {
        out.extend_from_slice(if f < 0.0 { b"-inf" } else { b"inf" });
        return;
    }
    // `%.15g`, else `%.17g` (which always round-trips an f64).
    let s15 = format_g(f, 15);
    let mut s = if s15.parse::<f64>() == Ok(f) {
        s15
    } else {
        format_g(f, 17)
    };
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

/// `luaL_tolstring` short of calling `__tostring`: numbers and strings in
/// their literal form, other values as `"<__name or type>: 0x<addr>"`.
pub(crate) fn basic_tostring<'gc>(ctx: Context<'gc>, v: Value<'gc>) -> LuaString<'gc> {
    if let Some(s) = v.get_string() {
        return s;
    }
    let mut out: Vec<u8> = Vec::new();
    let ptr = if v.is_nil() {
        out.extend_from_slice(b"nil");
        None
    } else if let Some(b) = v.get_boolean() {
        out.extend_from_slice(if b { b"true" } else { b"false" });
        None
    } else if let Some(i) = v.get_integer() {
        push_int(&mut out, i);
        None
    } else if let Some(f) = v.get_float() {
        push_float(&mut out, f);
        None
    } else if let Some(t) = v.get_table() {
        Some(Gc::as_ptr(t.inner()) as *const ())
    } else if let Some(f) = v.get_function() {
        Some(Gc::as_ptr(f.inner()) as *const ())
    } else if let Some(t) = v.get_thread() {
        Some(Gc::as_ptr(t.inner()) as *const ())
    } else {
        v.get_userdata().map(|u| Gc::as_ptr(u.inner()) as *const ())
    };
    if let Some(ptr) = ptr {
        let name = ctx.metamethod_of(v, LuaString::new(ctx, b"__name"));
        match name.get_string() {
            Some(name) => out.extend_from_slice(name.as_bytes()),
            None => out.extend_from_slice(v.type_name().as_bytes()),
        }
        out.extend_from_slice(b": ");
        out.extend_from_slice(format!("{ptr:p}").as_bytes());
    }
    LuaString::new(ctx, &out)
}

/// The function a call to `f` reaches through its `__call` chain, with each
/// callable object prepended to `args` (`tryfuncTM`).
pub(crate) fn callable<'gc>(
    ctx: Context<'gc>,
    mut f: Value<'gc>,
    args: &mut Vec<Value<'gc>>,
) -> Result<Function<'gc>, Error<'gc>> {
    for _ in 0..MAX_TAG_LOOP {
        if let Some(func) = f.get_function() {
            return Ok(func);
        }
        let call = ctx.metamethod_of(f, ctx.symbols().mm_call);
        if call.is_nil() {
            break;
        }
        args.insert(0, f);
        f = call;
    }
    Err(Error::from_str(
        ctx,
        &format!("attempt to call a {} value", f.type_name()),
    ))
}

/// Call `v`'s `__tostring` metamethod `mm` using the stack from `bottom` up,
/// and return its result as `luaL_tolstring` accepts it: a string, or a
/// number converted to one.
pub(crate) async fn call_tostring(
    seq: &mut AsyncSequence,
    mm: &StashedValue,
    v: &StashedValue,
    bottom: usize,
) -> Result<Vec<u8>, StashedError> {
    let f = seq.try_enter(|ctx, locals, _exec, mut stack| {
        let mut args = vec![locals.fetch(ctx.mutation(), v)];
        let f = callable(ctx, locals.fetch(ctx.mutation(), mm), &mut args)?;
        stack.truncate(bottom);
        stack.extend(args);
        Ok(locals.stash(ctx.mutation(), f))
    })?;
    seq.call(&f, bottom).await?;
    seq.try_enter(|ctx, _locals, _exec, mut stack| {
        let r = stack.get(bottom);
        stack.truncate(bottom);
        Ok(tostring_result(ctx, r)?.as_bytes().to_vec())
    })
}

/// A `__tostring` result as `luaL_tolstring` accepts it: a string, or a number
/// converted to one.
fn tostring_result<'gc>(ctx: Context<'gc>, r: Value<'gc>) -> Result<LuaString<'gc>, Error<'gc>> {
    if r.get_string().is_none() && r.get_integer().is_none() && !r.is_float() {
        return Err(Error::from_str(ctx, "'__tostring' must return a string"));
    }
    Ok(basic_tostring(ctx, r))
}

/// Follow-up for a `Call` of a `__tostring` metamethod: its first result,
/// checked and converted by [`tostring_result`].
pub(crate) struct ToStringResult;

unsafe impl<'gc> Collect<'gc> for ToStringResult {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> Sequence<'gc> for ToStringResult {
    fn trace_pointers(&self, _cc: &mut dyn Trace<'gc>) {}

    fn poll(
        self: Pin<&mut Self>,
        ctx: Context<'gc>,
        _exec: Execution<'gc>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        let s = tostring_result(ctx, stack.get(0))?;
        stack.ret1(Value::string(s));
        Ok(SequencePoll::Return)
    }
}

/// Follow-up for a `Call` that adjusts the callee's results to exactly `.0`
/// values, as `lua_call(L, nargs, n)` does.
pub(crate) struct AdjustResults(pub(crate) usize);

unsafe impl<'gc> Collect<'gc> for AdjustResults {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> Sequence<'gc> for AdjustResults {
    fn trace_pointers(&self, _cc: &mut dyn Trace<'gc>) {}

    fn poll(
        self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        stack.truncate(self.0);
        while stack.len() < self.0 {
            stack.push(Value::nil());
        }
        Ok(SequencePoll::Return)
    }
}

// ---------------------------------------------------------------------------
// Argument coercion (shared `luaL_check*` analogues)
// ---------------------------------------------------------------------------

/// `luaL_typeerror`: argument `n` of `fname` should have been `expected`.
/// `got` is `None` for a missing argument; a value whose metatable has a
/// string `__name` is reported by that name.
pub(crate) fn type_error<'gc>(
    ctx: Context<'gc>,
    fname: &str,
    n: usize,
    expected: &str,
    got: Option<Value<'gc>>,
) -> Error<'gc> {
    let got = match got {
        None => "no value".into(),
        Some(v) => match ctx
            .metamethod_of(v, LuaString::new(ctx, b"__name"))
            .get_string()
        {
            Some(name) => String::from_utf8_lossy(name.as_bytes()),
            None => v.type_name().into(),
        },
    };
    Error::from_str(
        ctx,
        &format!("bad argument #{n} to '{fname}' ({expected} expected, got {got})"),
    )
}

/// Coerce `v` to a float, mirroring `luaL_checknumber` (numeric strings
/// included). `fname`/`n` build the standard bad-argument message on failure.
/// Floats and small integers are inlined; everything else (boxed integers,
/// strings, the `format!`) stays out of line so callers like `math.sqrt` stay
/// small.
#[inline(always)]
pub(crate) fn check_number<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<f64, Error<'gc>> {
    if let Some(f) = v.get_float() {
        return Ok(f);
    }
    if let Some(i) = v.get_small() {
        return Ok(i as f64);
    }
    check_number_slow(ctx, v, fname, n)
}

#[cold]
#[inline(never)]
fn check_number_slow<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<f64, Error<'gc>> {
    to_number(v).ok_or_else(|| type_error(ctx, fname, n, "number", Some(v)))
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
    Err(type_error(ctx, fname, n, "number", Some(v)))
}

pub(crate) fn to_number<'gc>(v: Value<'gc>) -> Option<f64> {
    if let Some(i) = v.get_integer() {
        Some(i as f64)
    } else if let Some(f) = v.get_float() {
        Some(f)
    } else if let Some(s) = v.get_string() {
        // Via the lexer rules (`str_to_number`), not raw `f64::parse`, so
        // `"inf"`/`"nan"` are rejected as Lua's `luaL_checknumber` does.
        Some(str_to_number(s.as_bytes())?.to_float())
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
        return str_to_number(s.as_bytes())?.to_integer();
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
pub(crate) fn num_to_value<'gc>(mc: &Mutation<'gc>, f: f64) -> Value<'gc> {
    match float_to_integer(f) {
        Some(i) => Value::integer(mc, i),
        None => Value::float(f),
    }
}

pub(crate) use crate::vm::num::raw_eq;
