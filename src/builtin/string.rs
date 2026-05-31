use crate::Context;
use crate::builtin::util;
use crate::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Table, Value};
use crate::vm::sequence::CallbackAction;

/// Coerce a string-or-number argument to a `LuaString`, mirroring
/// `luaL_checkstring` (numbers are accepted and stringified).
fn check_str<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<LuaString<'gc>, Error<'gc>> {
    if let Some(s) = v.get_string() {
        Ok(s)
    } else if v.get_integer().is_some() || v.get_float().is_some() {
        Ok(util::basic_tostring(ctx, v))
    } else {
        Err(Error::from_str(
            ctx,
            &format!(
                "bad argument #{n} to '{fname}' (string expected, got {})",
                v.type_name()
            ),
        ))
    }
}

/// Lua's `posrelat`: translate a possibly-negative 1-based string position into
/// an absolute 1-based position (negatives count from the end; 0 stays 0).
fn posrelat(pos: i64, len: usize) -> i64 {
    if pos >= 0 {
        pos
    } else if pos.unsigned_abs() > len as u64 {
        0
    } else {
        len as i64 + pos + 1
    }
}

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("byte", lua_byte),
        ("char", lua_char),
        ("dump", lua_dump),
        ("find", lua_find),
        ("format", lua_format),
        ("gmatch", lua_gmatch),
        ("gsub", lua_gsub),
        ("len", lua_len),
        ("lower", lua_lower),
        ("match", lua_match),
        ("pack", lua_pack),
        ("packsize", lua_packsize),
        ("rep", lua_rep),
        ("reverse", lua_reverse),
        ("sub", lua_sub),
        ("unpack", lua_unpack),
        ("upper", lua_upper),
    ];

    let lib = Table::new(ctx);
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, Box::new([]));
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(handler));
    }

    let lib_name = Value::string(LuaString::new(ctx, b"string"));
    ctx.globals().raw_set(ctx, lib_name, Value::table(lib));
}

/// `byte(s [, i [, j]])` — the numeric codes of `s[i..j]` (1-based, negatives
/// from the end; `i` defaults to 1, `j` to `i`).
fn lua_byte<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = check_str(nctx.ctx, stack.get(0), "byte", 1)?;
    let bytes = s.as_bytes();
    let len = bytes.len();
    let i_arg = stack.get(1);
    let i = if i_arg.is_nil() {
        1
    } else {
        util::check_integer(nctx.ctx, i_arg, "byte", 2)?
    };
    let j_arg = stack.get(2);
    let j = if j_arg.is_nil() {
        i
    } else {
        util::check_integer(nctx.ctx, j_arg, "byte", 3)?
    };
    let start = posrelat(i, len).max(1);
    let end = posrelat(j, len).min(len as i64);
    let mut out = Vec::new();
    let mut k = start;
    while k <= end {
        out.push(Value::integer(bytes[(k - 1) as usize] as i64));
        k += 1;
    }
    stack.replace(&out);
    Ok(CallbackAction::Return)
}

/// `char(...)` — a string built from the given byte values (each 0–255).
fn lua_char<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let n = stack.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let c = util::check_integer(nctx.ctx, stack.get(i), "char", i + 1)?;
        if !(0..=255).contains(&c) {
            return Err(Error::from_str(
                nctx.ctx,
                &format!("bad argument #{} to 'char' (value out of range)", i + 1),
            ));
        }
        out.push(c as u8);
    }
    let r = LuaString::new(nctx.ctx, &out);
    stack.replace(&[Value::string(r)]);
    Ok(CallbackAction::Return)
}

fn lua_dump<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_find<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_format<'gc>(
    ctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let fmt_val = stack.get(0);
    let fmt_str = fmt_val.get_string().ok_or_else(|| {
        Error::from_str(
            ctx.ctx,
            &format!(
                "bad argument #1 to 'format' (string expected, got {})",
                fmt_val.type_name()
            ),
        )
    })?;
    let fmt = fmt_str.as_bytes();

    let mut out: Vec<u8> = Vec::with_capacity(fmt.len() + 16);
    let mut arg_idx = 1usize;
    let mut i = 0usize;
    while i < fmt.len() {
        let b = fmt[i];
        if b != b'%' {
            out.push(b);
            i += 1;
            continue;
        }
        // parse flags/width/precision/conv starting at fmt[i+1]
        let (spec, next) = parse_spec(ctx.ctx, fmt, i + 1)?;
        i = next;
        if spec.conv == b'%' {
            out.push(b'%');
            continue;
        }
        let arg = stack.get(arg_idx);
        arg_idx += 1;
        format_one(ctx.ctx, &mut out, &spec, arg)?;
    }

    let s = LuaString::new(ctx.ctx, &out);
    stack.replace(&[Value::string(s)]);
    Ok(CallbackAction::Return)
}

#[derive(Default)]
struct FmtSpec {
    flag_minus: bool,
    flag_plus: bool,
    flag_space: bool,
    flag_hash: bool,
    flag_zero: bool,
    width: usize,
    precision: Option<usize>,
    conv: u8,
}

fn parse_spec<'gc>(
    ctx: Context<'gc>,
    fmt: &[u8],
    mut i: usize,
) -> Result<(FmtSpec, usize), Error<'gc>> {
    let mut spec = FmtSpec::default();
    while i < fmt.len() {
        match fmt[i] {
            b'-' => spec.flag_minus = true,
            b'+' => spec.flag_plus = true,
            b' ' => spec.flag_space = true,
            b'#' => spec.flag_hash = true,
            b'0' => spec.flag_zero = true,
            _ => break,
        }
        i += 1;
    }
    while i < fmt.len() && fmt[i].is_ascii_digit() {
        spec.width = spec.width * 10 + (fmt[i] - b'0') as usize;
        if spec.width > 99 {
            return Err(Error::from_str(ctx, "invalid format (width too large)"));
        }
        i += 1;
    }
    if i < fmt.len() && fmt[i] == b'.' {
        i += 1;
        let mut p = 0usize;
        while i < fmt.len() && fmt[i].is_ascii_digit() {
            p = p * 10 + (fmt[i] - b'0') as usize;
            if p > 99 {
                return Err(Error::from_str(ctx, "invalid format (precision too large)"));
            }
            i += 1;
        }
        spec.precision = Some(p);
    }
    if i >= fmt.len() {
        return Err(Error::from_str(ctx, "invalid conversion '%' (missing)"));
    }
    spec.conv = fmt[i];
    Ok((spec, i + 1))
}

fn format_one<'gc>(
    ctx: Context<'gc>,
    out: &mut Vec<u8>,
    spec: &FmtSpec,
    arg: Value<'gc>,
) -> Result<(), Error<'gc>> {
    match spec.conv {
        b'd' | b'i' | b'u' => {
            let n = to_integer(arg).ok_or_else(|| arg_type_err(ctx, "integer", &arg))?;
            fmt_int_signed(out, spec, n);
        }
        b'o' => {
            let n = to_integer(arg).ok_or_else(|| arg_type_err(ctx, "integer", &arg))?;
            fmt_int_unsigned(out, spec, n as u64, 8, false);
        }
        b'x' => {
            let n = to_integer(arg).ok_or_else(|| arg_type_err(ctx, "integer", &arg))?;
            fmt_int_unsigned(out, spec, n as u64, 16, false);
        }
        b'X' => {
            let n = to_integer(arg).ok_or_else(|| arg_type_err(ctx, "integer", &arg))?;
            fmt_int_unsigned(out, spec, n as u64, 16, true);
        }
        b'c' => {
            let n = to_integer(arg).ok_or_else(|| arg_type_err(ctx, "integer", &arg))?;
            if !(0..=255).contains(&n) {
                return Err(Error::from_str(
                    ctx,
                    "bad argument to 'format' (value out of range)",
                ));
            }
            out.push(n as u8);
        }
        b'f' | b'F' => {
            let f = to_float(arg).ok_or_else(|| arg_type_err(ctx, "number", &arg))?;
            fmt_float_fixed(out, spec, f);
        }
        b'e' | b'E' => {
            let f = to_float(arg).ok_or_else(|| arg_type_err(ctx, "number", &arg))?;
            fmt_float_exp(out, spec, f, spec.conv == b'E');
        }
        b'g' | b'G' => {
            let f = to_float(arg).ok_or_else(|| arg_type_err(ctx, "number", &arg))?;
            fmt_float_g(out, spec, f, spec.conv == b'G');
        }
        b's' => {
            fmt_string(out, spec, arg);
        }
        b'q' => {
            fmt_q(out, arg);
        }
        c => {
            return Err(Error::from_str(
                ctx,
                &format!("invalid conversion '%{}' to 'format'", c as char),
            ));
        }
    }
    Ok(())
}

fn arg_type_err<'gc>(ctx: Context<'gc>, expected: &str, arg: &Value<'gc>) -> Error<'gc> {
    Error::from_str(
        ctx,
        &format!(
            "bad argument to 'format' ({} expected, got {})",
            expected,
            arg.type_name()
        ),
    )
}

fn to_integer<'gc>(v: Value<'gc>) -> Option<i64> {
    if let Some(i) = v.get_integer() {
        return Some(i);
    }
    if let Some(f) = v.get_float() {
        if f.is_finite() && f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
            return Some(f as i64);
        }
        return None;
    }
    if let Some(s) = v.get_string() {
        let t = std::str::from_utf8(s.as_bytes()).ok()?.trim();
        return t.parse::<i64>().ok().or_else(|| {
            t.parse::<f64>()
                .ok()
                .filter(|f| f.fract() == 0.0)
                .map(|f| f as i64)
        });
    }
    None
}

fn to_float<'gc>(v: Value<'gc>) -> Option<f64> {
    if let Some(i) = v.get_integer() {
        return Some(i as f64);
    }
    if let Some(f) = v.get_float() {
        return Some(f);
    }
    if let Some(s) = v.get_string() {
        return std::str::from_utf8(s.as_bytes()).ok()?.trim().parse().ok();
    }
    None
}

// ---------- integer formatting ----------

fn fmt_int_signed(out: &mut Vec<u8>, spec: &FmtSpec, n: i64) {
    let negative = n < 0;
    let abs = n.unsigned_abs();
    let mut digits = format!("{abs}");
    if let Some(p) = spec.precision {
        if digits.len() < p {
            let pad = p - digits.len();
            let mut zeros = "0".repeat(pad);
            zeros.push_str(&digits);
            digits = zeros;
        }
        if p == 0 && abs == 0 {
            digits.clear();
        }
    }
    let sign = if negative {
        "-"
    } else if spec.flag_plus {
        "+"
    } else if spec.flag_space {
        " "
    } else {
        ""
    };
    apply_width(out, spec, sign.as_bytes(), b"", digits.as_bytes());
}

fn fmt_int_unsigned(out: &mut Vec<u8>, spec: &FmtSpec, n: u64, radix: u32, upper: bool) {
    let mut digits = match radix {
        8 => format!("{n:o}"),
        16 if upper => format!("{n:X}"),
        16 => format!("{n:x}"),
        _ => unreachable!(),
    };
    if let Some(p) = spec.precision {
        if digits.len() < p {
            let pad = p - digits.len();
            let mut zeros = "0".repeat(pad);
            zeros.push_str(&digits);
            digits = zeros;
        }
        if p == 0 && n == 0 {
            digits.clear();
        }
    }
    let prefix: &[u8] = if spec.flag_hash && n != 0 {
        match (radix, upper) {
            (16, false) => b"0x",
            (16, true) => b"0X",
            (8, _) => b"0",
            _ => b"",
        }
    } else {
        b""
    };
    apply_width(out, spec, b"", prefix, digits.as_bytes());
}

// ---------- float formatting ----------

fn fmt_float_fixed(out: &mut Vec<u8>, spec: &FmtSpec, f: f64) {
    let prec = spec.precision.unwrap_or(6);
    if let Some(s) = special_float(f, spec) {
        apply_width(out, spec, b"", b"", s.as_bytes());
        return;
    }
    let (sign, abs) = sign_split(f, spec);
    let mut body = format!("{abs:.prec$}");
    if spec.flag_hash && prec == 0 {
        body.push('.');
    }
    apply_width(out, spec, sign.as_bytes(), "".as_bytes(), body.as_bytes());
}

fn fmt_float_exp(out: &mut Vec<u8>, spec: &FmtSpec, f: f64, upper: bool) {
    let prec = spec.precision.unwrap_or(6);
    if let Some(s) = special_float(f, spec) {
        apply_width(out, spec, b"", b"", s.as_bytes());
        return;
    }
    let (sign, abs) = sign_split(f, spec);
    let body = format_exp_abs(abs, prec, upper, spec.flag_hash);
    apply_width(out, spec, sign.as_bytes(), b"", body.as_bytes());
}

fn fmt_float_g(out: &mut Vec<u8>, spec: &FmtSpec, f: f64, upper: bool) {
    let raw_prec = spec.precision.unwrap_or(6);
    let prec = if raw_prec == 0 { 1 } else { raw_prec };
    if let Some(s) = special_float(f, spec) {
        apply_width(out, spec, b"", b"", s.as_bytes());
        return;
    }
    let (sign, abs) = sign_split(f, spec);
    // X = decimal exponent of `abs` when formatted as %e
    let x: i32 = if abs == 0.0 {
        0
    } else {
        abs.abs().log10().floor() as i32
    };
    let mut body = if (x < -4) || (x >= prec as i32) {
        // %e with precision = prec - 1
        format_exp_abs(abs, prec - 1, upper, false)
    } else {
        // %f with precision = prec - 1 - X
        let p = (prec as i32 - 1 - x).max(0) as usize;
        format!("{abs:.p$}")
    };
    if !spec.flag_hash {
        strip_trailing_zeros(&mut body);
    }
    apply_width(out, spec, sign.as_bytes(), b"", body.as_bytes());
}

fn special_float(f: f64, spec: &FmtSpec) -> Option<&'static str> {
    if f.is_nan() {
        Some(if spec.conv.is_ascii_uppercase() {
            "NAN"
        } else {
            "nan"
        })
    } else if f.is_infinite() {
        Some(if f.is_sign_negative() {
            if spec.conv.is_ascii_uppercase() {
                "-INF"
            } else {
                "-inf"
            }
        } else if spec.conv.is_ascii_uppercase() {
            "INF"
        } else {
            "inf"
        })
    } else {
        None
    }
}

fn sign_split(f: f64, spec: &FmtSpec) -> (&'static str, f64) {
    if f.is_sign_negative() && f != 0.0 {
        ("-", -f)
    } else if spec.flag_plus {
        ("+", f)
    } else if spec.flag_space {
        (" ", f)
    } else {
        ("", f)
    }
}

fn format_exp_abs(abs: f64, prec: usize, upper: bool, hash: bool) -> String {
    // Produce mantissa and exponent matching C printf semantics:
    //   d.ddd e ±NN  (exponent always signed, at least 2 digits)
    let (mantissa, exp) = decompose_exp(abs, prec);
    let e_char = if upper { 'E' } else { 'e' };
    let mut s = mantissa;
    if hash && prec == 0 && !s.contains('.') {
        s.push('.');
    }
    let sign = if exp < 0 { '-' } else { '+' };
    let mag = exp.unsigned_abs();
    if mag < 10 {
        format!("{s}{e_char}{sign}0{mag}")
    } else {
        format!("{s}{e_char}{sign}{mag}")
    }
}

fn decompose_exp(abs: f64, prec: usize) -> (String, i32) {
    // Use Rust's `{:e}` (which formats as "d.dddde-N"/"d.ddddeN") and then
    // re-shape the mantissa to a fixed precision. This avoids reimplementing
    // Grisu/Dragon for now; precision adjustment is handled by `format!`.
    if abs == 0.0 {
        let mantissa = if prec == 0 {
            "0".to_string()
        } else {
            format!("0.{}", "0".repeat(prec))
        };
        return (mantissa, 0);
    }
    let exp = abs.abs().log10().floor() as i32;
    let scale = 10f64.powi(-exp);
    let mantissa_val = abs * scale;
    // Rounding can push mantissa to >=10; renormalize.
    let mut m = format!("{mantissa_val:.prec$}");
    let mut e = exp;
    if m.starts_with("10") {
        // Rounding overflow, e.g. "10.000" — shift to "1.000" with exp+1.
        let new_val = mantissa_val / 10.0;
        m = format!("{new_val:.prec$}");
        e += 1;
    }
    (m, e)
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

// ---------- string and q ----------

fn fmt_string<'gc>(out: &mut Vec<u8>, spec: &FmtSpec, arg: Value<'gc>) {
    let owned: String;
    let bytes: &[u8] = if let Some(s) = arg.get_string() {
        s.as_bytes()
    } else if arg.is_nil() {
        b"nil"
    } else if let Some(b) = arg.get_boolean() {
        if b { b"true" } else { b"false" }
    } else if let Some(i) = arg.get_integer() {
        owned = format!("{i}");
        owned.as_bytes()
    } else if let Some(f) = arg.get_float() {
        owned = format!("{f}");
        owned.as_bytes()
    } else {
        b"<value>"
    };
    let trimmed: &[u8] = if let Some(p) = spec.precision {
        &bytes[..bytes.len().min(p)]
    } else {
        bytes
    };
    apply_width(out, spec, b"", b"", trimmed);
}

fn fmt_q<'gc>(out: &mut Vec<u8>, arg: Value<'gc>) {
    if arg.is_nil() {
        out.extend_from_slice(b"nil");
    } else if let Some(b) = arg.get_boolean() {
        out.extend_from_slice(if b { b"true" } else { b"false" });
    } else if let Some(i) = arg.get_integer() {
        let s = format!("{i}");
        out.extend_from_slice(s.as_bytes());
    } else if let Some(f) = arg.get_float() {
        // TODO: use %a (hex float) for exact round-trip per Lua spec.
        let s = format!("{f:?}");
        out.extend_from_slice(s.as_bytes());
    } else if let Some(s) = arg.get_string() {
        out.push(b'"');
        for &b in s.as_bytes() {
            match b {
                b'"' => out.extend_from_slice(b"\\\""),
                b'\\' => out.extend_from_slice(b"\\\\"),
                b'\n' => out.extend_from_slice(b"\\n"),
                b'\r' => out.extend_from_slice(b"\\r"),
                0 => out.extend_from_slice(b"\\0"),
                b if b < 0x20 || b == 0x7f => {
                    let s = format!("\\{}", b);
                    out.extend_from_slice(s.as_bytes());
                }
                b => out.push(b),
            }
        }
        out.push(b'"');
    } else {
        let s = format!("<{}>", arg.type_name());
        out.extend_from_slice(s.as_bytes());
    }
}

// ---------- shared width/padding ----------

fn apply_width(out: &mut Vec<u8>, spec: &FmtSpec, sign: &[u8], prefix: &[u8], body: &[u8]) {
    let content_len = sign.len() + prefix.len() + body.len();
    let pad = spec.width.saturating_sub(content_len);
    if pad == 0 {
        out.extend_from_slice(sign);
        out.extend_from_slice(prefix);
        out.extend_from_slice(body);
        return;
    }
    if spec.flag_minus {
        out.extend_from_slice(sign);
        out.extend_from_slice(prefix);
        out.extend_from_slice(body);
        for _ in 0..pad {
            out.push(b' ');
        }
    } else if spec.flag_zero && spec.precision.is_none() {
        out.extend_from_slice(sign);
        out.extend_from_slice(prefix);
        for _ in 0..pad {
            out.push(b'0');
        }
        out.extend_from_slice(body);
    } else {
        for _ in 0..pad {
            out.push(b' ');
        }
        out.extend_from_slice(sign);
        out.extend_from_slice(prefix);
        out.extend_from_slice(body);
    }
}

fn lua_gmatch<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_gsub<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

/// `len(s)` — number of bytes in `s`.
fn lua_len<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = check_str(nctx.ctx, stack.get(0), "len", 1)?;
    stack.replace(&[Value::integer(s.len() as i64)]);
    Ok(CallbackAction::Return)
}

/// `lower(s)` — ASCII-lowercased copy of `s` (C locale).
fn lua_lower<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = check_str(nctx.ctx, stack.get(0), "lower", 1)?;
    let lowered: Vec<u8> = s.as_bytes().iter().map(u8::to_ascii_lowercase).collect();
    stack.replace(&[Value::string(LuaString::new(nctx.ctx, &lowered))]);
    Ok(CallbackAction::Return)
}

fn lua_match<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_pack<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_packsize<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

/// `rep(s, n [, sep])` — `s` repeated `n` times, with `sep` between copies.
/// `n <= 0` yields the empty string.
fn lua_rep<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = check_str(nctx.ctx, stack.get(0), "rep", 1)?;
    let n = util::check_integer(nctx.ctx, stack.get(1), "rep", 2)?;
    let sep_arg = stack.get(2);
    let sep = if sep_arg.is_nil() {
        Vec::new()
    } else {
        check_str(nctx.ctx, sep_arg, "rep", 3)?.as_bytes().to_vec()
    };
    let out = if n <= 0 {
        Vec::new()
    } else {
        let n = n as usize;
        let body = s.as_bytes();
        // Guard against overflow on pathological sizes (Lua: "too large").
        let total = body
            .len()
            .checked_add(sep.len())
            .and_then(|per| per.checked_mul(n))
            .map(|t| t - sep.len());
        let Some(total) = total else {
            return Err(Error::from_str(nctx.ctx, "resulting string too large"));
        };
        let mut out = Vec::with_capacity(total);
        for k in 0..n {
            if k > 0 {
                out.extend_from_slice(&sep);
            }
            out.extend_from_slice(body);
        }
        out
    };
    stack.replace(&[Value::string(LuaString::new(nctx.ctx, &out))]);
    Ok(CallbackAction::Return)
}

/// `reverse(s)` — `s` with its bytes in reverse order.
fn lua_reverse<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = check_str(nctx.ctx, stack.get(0), "reverse", 1)?;
    let mut bytes = s.as_bytes().to_vec();
    bytes.reverse();
    stack.replace(&[Value::string(LuaString::new(nctx.ctx, &bytes))]);
    Ok(CallbackAction::Return)
}

/// `sub(s, i [, j])` — substring `s[i..j]` (1-based, negatives from the end;
/// `i` defaults to 1, `j` to -1).
fn lua_sub<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = check_str(nctx.ctx, stack.get(0), "sub", 1)?;
    let bytes = s.as_bytes();
    let len = bytes.len();
    let i_arg = stack.get(1);
    let i = if i_arg.is_nil() {
        1
    } else {
        util::check_integer(nctx.ctx, i_arg, "sub", 2)?
    };
    let j_arg = stack.get(2);
    let j = if j_arg.is_nil() {
        -1
    } else {
        util::check_integer(nctx.ctx, j_arg, "sub", 3)?
    };
    let start = posrelat(i, len).max(1);
    let end = posrelat(j, len).min(len as i64);
    let result = if start <= end {
        LuaString::new(nctx.ctx, &bytes[(start - 1) as usize..end as usize])
    } else {
        LuaString::new(nctx.ctx, b"")
    };
    stack.replace(&[Value::string(result)]);
    Ok(CallbackAction::Return)
}

fn lua_unpack<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

/// `upper(s)` — ASCII-uppercased copy of `s` (C locale).
fn lua_upper<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = check_str(nctx.ctx, stack.get(0), "upper", 1)?;
    let uppered: Vec<u8> = s.as_bytes().iter().map(u8::to_ascii_uppercase).collect();
    stack.replace(&[Value::string(LuaString::new(nctx.ctx, &uppered))]);
    Ok(CallbackAction::Return)
}
