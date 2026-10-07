//! Shared helpers for the standard library: Lua-faithful number/value
//! stringification and small argument-coercion routines used across the
//! `basic`, `string`, `table`, and `io` libraries.

use crate::builtin::strfmt_num;
use crate::dmm::{Gc, Mutation};
use crate::env::{Error, Function, LuaString, MetamethodBits, NativeClosure, Stack, Table, Value};
use crate::lua::Context;
use crate::vm::async_native::{AsyncError, Cx};
use crate::vm::debug::object_type_name;
use crate::vm::{IndexChain, NewIndexChain, walk_index_chain, walk_newindex_chain};

/// Append the canonical Lua textual form of an integer.
pub(crate) fn push_int(out: &mut Vec<u8>, i: i64) {
    out.extend_from_slice(i.format_into(&mut core::fmt::NumBuffer::new()).as_bytes());
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

/// Append the textual form of a float: `%.14g` as in LuaJIT, where Lua 5.5 uses `%.15g`
/// and falls back to `%.17g` when that doesn't read back exactly, plus `.0` on an
/// integer-looking result.
pub(crate) fn push_float(out: &mut Vec<u8>, f: f64) {
    let start = out.len();
    strfmt_num::put_fnum(out, strfmt_num::STRFMT_G14, f);
    // Tag an otherwise integer-looking float so round-trips stay floats.
    if out[start..].iter().all(|b| matches!(b, b'-' | b'0'..=b'9')) {
        out.extend_from_slice(b".0");
    }
}

/// `luaL_tolstring` short of calling `__tostring`: numbers and strings in
/// their literal form, other values as `"<__name or type>: 0x<addr>"`.
pub(crate) fn basic_tostring<'gc>(ctx: Context<'gc>, v: Value<'gc>) -> LuaString<'gc> {
    if let Some(s) = v.get_string() {
        return s;
    }
    ctx.build_string(|out| {
        if v.is_nil() {
            out.extend_from_slice(b"nil");
        } else if let Some(b) = v.get_boolean() {
            out.extend_from_slice(if b { b"true" } else { b"false" });
        } else if let Some(i) = v.get_integer() {
            push_int(out, i);
        } else if let Some(f) = v.get_float() {
            push_float(out, f);
        } else {
            let ptr = to_pointer(v).expect("every other type is an object");
            match ctx.mm_of(v, MetamethodBits::NAME).get_string() {
                Some(name) => out.extend_from_slice(name.as_bytes()),
                None => out.extend_from_slice(v.type_name().as_bytes()),
            }
            out.extend_from_slice(format!(": {ptr:p}").as_bytes());
        }
    })
}

/// `lua_topointer`: an object's address, `None` for nil, booleans and numbers.
pub(crate) fn to_pointer(v: Value<'_>) -> Option<*const ()> {
    Some(if let Some(s) = v.get_string() {
        Gc::as_ptr(s.inner()) as *const ()
    } else if let Some(t) = v.get_table() {
        Gc::as_ptr(t.inner()) as *const ()
    } else if let Some(f) = v.get_function() {
        Gc::as_ptr(f.inner()) as *const ()
    } else if let Some(t) = v.get_thread() {
        Gc::as_ptr(t.inner()) as *const ()
    } else {
        Gc::as_ptr(v.get_userdata()?.inner()) as *const ()
    })
}

/// `luaL_tolstring` of the value at window index `i`, calling its
/// `__tostring` with the window from `bottom` (above `i`) up.
pub(crate) async fn tolstring(cx: &Cx, i: usize, bottom: usize) -> Result<Vec<u8>, AsyncError> {
    let call = cx.enter(|ctx, mut stack| {
        let v = stack.get(i);
        let mm = ctx.mm_of(v, MetamethodBits::TOSTRING);
        if mm.is_nil() {
            return Err(basic_tostring(ctx, v).as_bytes().to_vec());
        }
        stack.truncate(bottom);
        stack.extend([mm, v]);
        Ok(())
    });
    if let Err(bytes) = call {
        return Ok(bytes);
    }
    cx.call(bottom).await;
    cx.try_enter(|ctx, stack| {
        let r = stack.get(bottom);
        stack.truncate(bottom);
        Ok(tostring_result(ctx, r)?.as_bytes().to_vec())
    })
}

/// A `__tostring` result as `luaL_tolstring` accepts it: a string, or a number
/// converted to one.
pub(crate) fn tostring_result<'gc>(
    ctx: Context<'gc>,
    r: Value<'gc>,
) -> Result<LuaString<'gc>, Error<'gc>> {
    if r.get_string().is_none() && r.get_integer().is_none() && !r.is_float() {
        return Err(Error::from_str(ctx, "'__tostring' must return a string"));
    }
    Ok(basic_tostring(ctx, r))
}

// ---------------------------------------------------------------------------
// Metamethod-aware access for async natives (`lua_geti`, `lua_getfield`,
// `lua_seti`, `lua_setfield`, `luaL_len`). Values move through the native's
// window, as in the C API.
// ---------------------------------------------------------------------------

/// `lua_geti`: push `window[idx][i]`, going through `__index`.
pub(crate) async fn geti(cx: &Cx, idx: usize, i: i64) -> Result<(), AsyncError> {
    get(cx, idx, |ctx| Value::integer(ctx.mutation(), i)).await
}

/// `lua_getfield`: push `window[idx][k]`, going through `__index`.
pub(crate) async fn getfield(cx: &Cx, idx: usize, k: &[u8]) -> Result<(), AsyncError> {
    get(cx, idx, |ctx| Value::string(LuaString::new(ctx, k))).await
}

async fn get(
    cx: &Cx,
    idx: usize,
    key: impl for<'gc> FnOnce(Context<'gc>) -> Value<'gc>,
) -> Result<(), AsyncError> {
    let call = cx.try_enter(|ctx, stack| {
        let t = stack.get(idx);
        let key = key(ctx);
        if let Some(tbl) = t.get_table() {
            let v = tbl.raw_get(key);
            if !v.is_nil() {
                stack.push(v);
                return Ok(None);
            }
        }
        match walk_index_chain(ctx, t, key) {
            IndexChain::Resolved(v) => {
                stack.push(v);
                Ok(None)
            }
            IndexChain::Invoke { func, receiver } => {
                let at = stack.len();
                stack.extend([Value::function(func), receiver, key]);
                Ok(Some(at))
            }
            IndexChain::NotIndexable(v) => Err(index_error(ctx, v)),
            IndexChain::Exhausted => Err(runtime_error(
                ctx,
                "'__index' chain too long; possible loop",
            )),
        }
    })?;
    if let Some(at) = call {
        cx.call(at).await;
        cx.enter(|_, mut stack| {
            stack.truncate(at + 1);
            if stack.len() == at {
                stack.push(Value::nil());
            }
        });
    }
    Ok(())
}

/// `lua_seti`: pop the top value into `window[idx][i]`, going through
/// `__newindex`.
pub(crate) async fn seti(cx: &Cx, idx: usize, i: i64) -> Result<(), AsyncError> {
    set(cx, idx, |ctx| Value::integer(ctx.mutation(), i)).await
}

/// `lua_setfield`: pop the top value into `window[idx][k]`, going through
/// `__newindex`.
pub(crate) async fn setfield(cx: &Cx, idx: usize, k: &[u8]) -> Result<(), AsyncError> {
    set(cx, idx, |ctx| Value::string(LuaString::new(ctx, k))).await
}

async fn set(
    cx: &Cx,
    idx: usize,
    key: impl for<'gc> FnOnce(Context<'gc>) -> Value<'gc>,
) -> Result<(), AsyncError> {
    let call = cx.try_enter(|ctx, stack| {
        let t = stack.get(idx);
        let v = stack.pop();
        let at = stack.len();
        let key = key(ctx);
        // Tables without `__newindex` skip `walk_newindex_chain`, which would
        // look the key up first for nothing.
        if let Some(tbl) = t.get_table()
            && !tbl.shape().has_mm(MetamethodBits::NEWINDEX)
        {
            tbl.raw_set(ctx, key, v);
            return Ok(None);
        }
        match walk_newindex_chain(ctx, t, key) {
            NewIndexChain::RawSet(tbl) => {
                tbl.raw_set(ctx, key, v);
                Ok(None)
            }
            NewIndexChain::Invoke { func, receiver } => {
                stack.extend([Value::function(func), receiver, key, v]);
                Ok(Some(at))
            }
            NewIndexChain::NotIndexable(v) => Err(index_error(ctx, v)),
            NewIndexChain::Exhausted => Err(runtime_error(
                ctx,
                "'__newindex' chain too long; possible loop",
            )),
        }
    })?;
    if let Some(at) = call {
        cx.call(at).await;
        cx.enter(|_, mut stack| stack.truncate(at));
    }
    Ok(())
}

/// `luaL_len`: `#window[idx]` through `__len`, which must give an integer.
pub(crate) async fn len(cx: &Cx, idx: usize) -> Result<i64, AsyncError> {
    let call = cx.try_enter(|ctx, stack| {
        let v = stack.get(idx);
        if let Some(s) = v.get_string() {
            return Ok(Err(s.len() as i64));
        }
        let mm = ctx.mm_of(v, MetamethodBits::LEN);
        if mm.is_nil() {
            return match v.get_table() {
                Some(t) => Ok(Err(t.raw_len() as i64)),
                None => Err(runtime_error(
                    ctx,
                    &format!(
                        "attempt to get length of a {} value",
                        object_type_name(ctx, v)
                    ),
                )),
            };
        }
        let at = stack.len();
        stack.extend([mm, v, v]);
        Ok(Ok(at))
    })?;
    let at = match call {
        Ok(at) => at,
        Err(n) => return Ok(n),
    };
    cx.call(at).await;
    cx.try_enter(|ctx, stack| {
        let r = stack.get(at);
        stack.truncate(at);
        to_integer(r).ok_or_else(|| Error::from_str(ctx, "object length is not an integer"))
    })
}

fn index_error<'gc>(ctx: Context<'gc>, v: Value<'gc>) -> Error<'gc> {
    runtime_error(
        ctx,
        &format!("attempt to index a {} value", object_type_name(ctx, v)),
    )
}

/// A library function tcvm doesn't implement yet: calling it raises
/// "`name` is not implemented".
pub(crate) fn not_implemented<'gc>(ctx: Context<'gc>, name: &str) -> Function<'gc> {
    let name = Value::string(LuaString::new(ctx, name.as_bytes()));
    Function::new_native(ctx.mutation(), raise_not_implemented, &[name])
}

fn raise_not_implemented<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let name = closure.upvalues()[0]
        .get_string()
        .map_or(&[][..], |s| s.as_bytes());
    let name = String::from_utf8_lossy(name);
    Err(Error::from_str(ctx, &format!("{name} is not implemented")))
}

/// Set each of `names` in `lib` to [`not_implemented`], named `lib_name.name`.
pub(crate) fn set_not_implemented<'gc>(
    ctx: Context<'gc>,
    lib: Table<'gc>,
    lib_name: &str,
    names: &[&str],
) {
    for name in names {
        let stub = not_implemented(ctx, &format!("{lib_name}.{name}"));
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(stub));
    }
}

/// An error the VM itself would raise (`luaG_runerror`): raised from a native
/// it carries no position, unlike `luaL_error`'s [`Error::from_str`].
pub(crate) fn runtime_error<'gc>(ctx: Context<'gc>, msg: &str) -> Error<'gc> {
    Error::from_str(ctx, msg).with_level(0)
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
        Some(v) => match ctx.mm_of(v, MetamethodBits::NAME).get_string() {
            Some(name) => String::from_utf8_lossy(name.as_bytes()),
            None => v.type_name().into(),
        },
    };
    arg_error(ctx, fname, n, &format!("{expected} expected, got {got}"))
}

/// `luaL_argerror`: "bad argument #`n` to '`fname`' (`msg`)".
pub(crate) fn arg_error<'gc>(ctx: Context<'gc>, fname: &str, n: usize, msg: &str) -> Error<'gc> {
    Error::from_str(ctx, &format!("bad argument #{n} to '{fname}' ({msg})"))
}

/// `luaL_checkany`: argument `n` (1-based) must be present, even if nil.
pub(crate) fn check_any<'gc>(
    ctx: Context<'gc>,
    stack: &Stack<'gc, '_>,
    fname: &str,
    n: usize,
) -> Result<(), Error<'gc>> {
    if stack.len() < n {
        return Err(arg_error(ctx, fname, n, "value expected"));
    }
    Ok(())
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
        return Err(arg_error(
            ctx,
            fname,
            n,
            "number has no integer representation",
        ));
    }
    Err(type_error(ctx, fname, n, "number", Some(v)))
}

/// `lua_tolstring` without metamethods: strings, and numbers converted.
pub(crate) fn to_lstring<'gc>(ctx: Context<'gc>, v: Value<'gc>) -> Option<LuaString<'gc>> {
    if let Some(s) = v.get_string() {
        Some(s)
    } else if v.get_integer().is_some() || v.get_float().is_some() {
        Some(basic_tostring(ctx, v))
    } else {
        None
    }
}

/// `luaL_checkstring`: a string, or a number converted.
pub(crate) fn check_string<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<LuaString<'gc>, Error<'gc>> {
    to_lstring(ctx, v).ok_or_else(|| type_error(ctx, fname, n, "string", Some(v)))
}

/// `luaL_optstring`: [`check_string`], or `None` for nil.
pub(crate) fn opt_string<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<Option<LuaString<'gc>>, Error<'gc>> {
    if v.is_nil() {
        return Ok(None);
    }
    check_string(ctx, v, fname, n).map(Some)
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
