use crate::Context;
use crate::builtin::util;
// `%d`/`%f` argument coercion reuses the shared `util` helpers so the
// integer-representation and numeric-string rules (including `inf`/`nan`
// rejection) match `tonumber`/`math.*` and don't drift.
use crate::builtin::util::{to_integer, to_number as to_float};
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
        // A conversion consumes the next argument; a missing one is an error
        // ("no value"), distinct from an explicitly-passed nil.
        if arg_idx >= stack.len() {
            return Err(Error::from_str(
                ctx.ctx,
                &format!("bad argument #{} to 'format' (no value)", arg_idx + 1),
            ));
        }
        let arg = stack.get(arg_idx);
        arg_idx += 1;
        // After the increment, `arg_idx` is the 1-based Lua argument number of
        // the argument just consumed (the format string is #1).
        format_one(ctx.ctx, &mut out, &spec, arg, arg_idx)?;
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
    let start = i - 1; // the '%'
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
    // Lua's `get2digits`: only the first two digits set the value; a third digit
    // is left in place so the conversion char ends up non-alpha and the spec is
    // rejected as malformed below (matches `checkformat`). Max width/prec = 99.
    let mut wdigits = 0;
    while i < fmt.len() && fmt[i].is_ascii_digit() {
        if wdigits < 2 {
            spec.width = spec.width * 10 + (fmt[i] - b'0') as usize;
        }
        wdigits += 1;
        i += 1;
    }
    let mut pdigits = 0;
    if i < fmt.len() && fmt[i] == b'.' {
        i += 1;
        let mut p = 0usize;
        while i < fmt.len() && fmt[i].is_ascii_digit() {
            if pdigits < 2 {
                p = p * 10 + (fmt[i] - b'0') as usize;
            }
            pdigits += 1;
            i += 1;
        }
        spec.precision = Some(p);
    }
    if i >= fmt.len() {
        return Err(invalid_conv_spec(ctx, &fmt[start..]));
    }
    spec.conv = fmt[i];
    // `%%` is handled by the caller; no flags or modifiers apply to it.
    if spec.conv == b'%' {
        return Ok((spec, i + 1));
    }
    // `%q` accepts no flags, width, or precision at all (its own error).
    if spec.conv == b'q' {
        let has_mods = spec.flag_minus
            || spec.flag_plus
            || spec.flag_space
            || spec.flag_hash
            || spec.flag_zero
            || spec.width != 0
            || spec.precision.is_some();
        if has_mods {
            return Err(Error::from_str(ctx, "specifier '%q' cannot have modifiers"));
        }
        return Ok((spec, i + 1));
    }
    let form = &fmt[start..=i];
    // The conversion char must be a letter and width/precision at most two
    // digits (Lua's `get2digits` caps both at 2).
    if !spec.conv.is_ascii_alphabetic() || wdigits > 2 || pdigits > 2 {
        return Err(invalid_conv_spec(ctx, form));
    }
    // Per-specifier flag/precision validation (Lua's `checkformat`): each
    // conversion accepts only a subset of flags, and `c` forbids a precision.
    let (allowed_flags, precision_ok): (&[u8], bool) = match spec.conv {
        b'd' | b'i' => (b"-+0 ", true),
        b'u' => (b"-0", true),
        b'o' | b'x' | b'X' => (b"-#0", true),
        b'a' | b'A' | b'e' | b'E' | b'f' | b'F' | b'g' | b'G' => (b"-+#0 ", true),
        b'c' => (b"-", false),
        b's' => (b"-", true),
        // Unknown letters fall through to `format_one`'s
        // "invalid conversion '%c' to 'format'".
        _ => return Ok((spec, i + 1)),
    };
    let flag_rejected = (spec.flag_minus && !allowed_flags.contains(&b'-'))
        || (spec.flag_plus && !allowed_flags.contains(&b'+'))
        || (spec.flag_space && !allowed_flags.contains(&b' '))
        || (spec.flag_hash && !allowed_flags.contains(&b'#'))
        || (spec.flag_zero && !allowed_flags.contains(&b'0'));
    if flag_rejected || (spec.precision.is_some() && !precision_ok) {
        return Err(invalid_conv_spec(ctx, form));
    }
    Ok((spec, i + 1))
}

/// Lua's "invalid conversion specification: '%...'" error, echoing the offending
/// spec verbatim (`form` includes the leading `%`).
fn invalid_conv_spec<'gc>(ctx: Context<'gc>, form: &[u8]) -> Error<'gc> {
    Error::from_str(
        ctx,
        &format!(
            "invalid conversion specification: '{}'",
            String::from_utf8_lossy(form)
        ),
    )
}

fn format_one<'gc>(
    ctx: Context<'gc>,
    out: &mut Vec<u8>,
    spec: &FmtSpec,
    arg: Value<'gc>,
    arg_num: usize,
) -> Result<(), Error<'gc>> {
    match spec.conv {
        b'd' | b'i' => {
            let n = check_fmt_int(ctx, arg, arg_num)?;
            fmt_int_signed(out, spec, n);
        }
        b'u' => {
            // `%u` formats the integer's unsigned 64-bit value, not its signed
            // form: `-1` -> "18446744073709551615".
            let n = check_fmt_int(ctx, arg, arg_num)?;
            fmt_int_unsigned(out, spec, n as u64, 10, false);
        }
        b'o' => {
            let n = check_fmt_int(ctx, arg, arg_num)?;
            fmt_int_unsigned(out, spec, n as u64, 8, false);
        }
        b'x' => {
            let n = check_fmt_int(ctx, arg, arg_num)?;
            fmt_int_unsigned(out, spec, n as u64, 16, false);
        }
        b'X' => {
            let n = check_fmt_int(ctx, arg, arg_num)?;
            fmt_int_unsigned(out, spec, n as u64, 16, true);
        }
        b'c' => {
            // C's `%c` casts the integer to `unsigned char`; Lua adds no range
            // check, so out-of-range values wrap to the low byte. Width/`-`
            // flags still apply (via apply_width).
            let n = check_fmt_int(ctx, arg, arg_num)?;
            apply_width(out, spec, b"", b"", &[n as u8]);
        }
        b'f' | b'F' => {
            let f = to_float(arg).ok_or_else(|| arg_type_err(ctx, "number", &arg, arg_num))?;
            fmt_float_fixed(out, spec, f);
        }
        b'e' | b'E' => {
            let f = to_float(arg).ok_or_else(|| arg_type_err(ctx, "number", &arg, arg_num))?;
            fmt_float_exp(out, spec, f, spec.conv == b'E');
        }
        b'g' | b'G' => {
            let f = to_float(arg).ok_or_else(|| arg_type_err(ctx, "number", &arg, arg_num))?;
            fmt_float_g(out, spec, f, spec.conv == b'G');
        }
        b'a' | b'A' => {
            let f = to_float(arg).ok_or_else(|| arg_type_err(ctx, "number", &arg, arg_num))?;
            fmt_hex_float(out, spec, f, spec.conv == b'A');
        }
        b's' => {
            fmt_string(ctx, out, spec, arg);
        }
        b'q' => {
            fmt_q(ctx, out, arg, arg_num)?;
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

fn arg_type_err<'gc>(
    ctx: Context<'gc>,
    expected: &str,
    arg: &Value<'gc>,
    arg_num: usize,
) -> Error<'gc> {
    Error::from_str(
        ctx,
        &format!(
            "bad argument #{arg_num} to 'format' ({} expected, got {})",
            expected,
            arg.type_name()
        ),
    )
}

/// Coerce a `%d`/`%x`/`%c`/… argument to an integer, distinguishing — as Lua
/// does — a non-number ("number expected, got X") from a number with no exact
/// integer value ("number has no integer representation").
fn check_fmt_int<'gc>(
    ctx: Context<'gc>,
    arg: Value<'gc>,
    arg_num: usize,
) -> Result<i64, Error<'gc>> {
    if let Some(i) = to_integer(arg) {
        return Ok(i);
    }
    let msg = if to_float(arg).is_some() {
        format!("bad argument #{arg_num} to 'format' (number has no integer representation)")
    } else {
        format!(
            "bad argument #{arg_num} to 'format' (number expected, got {})",
            arg.type_name()
        )
    };
    Err(Error::from_str(ctx, &msg))
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
        10 => format!("{n}"),
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
        strip_g_trailing_zeros(&mut body);
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
    // Rust's `{:e}` formatting rounds, renormalizes (e.g. 9.9 at prec 0 -> "1e1"),
    // and handles subnormals correctly. Scaling the mantissa by `10^-exp` by hand
    // instead overflowed to `inf` for denormal exponents (~1e-309 and below),
    // producing garbage like "infe-309". Split the mantissa from the exponent.
    // `.abs()` strips a `-0.0` sign bit, which the caller already accounts for and
    // which `{:e}` would otherwise re-emit as a duplicate '-'.
    let s = format!("{:.prec$e}", abs.abs());
    let epos = s.find('e').expect("exponential format always contains 'e'");
    let exp = s[epos + 1..]
        .parse::<i32>()
        .expect("exponent is a valid integer");
    (s[..epos].to_string(), exp)
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

/// `%g` trailing-zero trimming: strip only the *mantissa*, never the exponent
/// digits. `strip_trailing_zeros` on a whole `"1.20000e+20"` would chew the
/// `0` off the exponent (`...e+2`); split at `e`/`E` and trim just the front.
fn strip_g_trailing_zeros(s: &mut String) {
    match s.find(['e', 'E']) {
        Some(epos) => {
            let mut mant = s[..epos].to_string();
            strip_trailing_zeros(&mut mant);
            mant.push_str(&s[epos..]);
            *s = mant;
        }
        None => strip_trailing_zeros(s),
    }
}

/// Format `abs` (finite, non-negative) as a C `%a`/`%A` hex float — `0x1.fep+7`
/// style. Without `prec`, emits the minimal fraction digits (trailing zeros
/// trimmed); with `prec`, emits exactly that many fraction nibbles.
fn format_hex_float(abs: f64, upper: bool, prec: Option<usize>) -> String {
    let bits = abs.to_bits();
    let exp_bits = ((bits >> 52) & 0x7ff) as i64;
    let raw_frac = bits & 0x000f_ffff_ffff_ffff;
    // (leading hex digit, unbiased exponent, 52-bit fraction below the lead).
    let (mut lead, exp, frac) = if exp_bits == 0 {
        if raw_frac == 0 {
            (0u8, 0i64, 0u64) // ±0
        } else {
            // Subnormal: normalize to a leading `1` digit (matches C/Lua).
            let hb = 63 - raw_frac.leading_zeros() as i64; // highest set bit, 0..=51
            let frac = (raw_frac << (52 - hb as u32)) & 0x000f_ffff_ffff_ffff;
            (1u8, hb - 1074, frac)
        }
    } else {
        (1u8, exp_bits - 1023, raw_frac)
    };
    // 13 hex digits of the fraction, most-significant nibble first.
    let mut digits: Vec<u8> = (0..13)
        .map(|i| ((frac >> (48 - i * 4)) & 0xf) as u8)
        .collect();
    match prec {
        Some(p) => {
            // Round to `p` fraction nibbles using round-half-to-even (the IEEE
            // 754 default). This is chosen for cross-platform determinism: C's
            // `%a` tie-breaking is host-`printf`-defined (glibc rounds to even,
            // macOS toward zero), which we deliberately do not inherit. The
            // discarded part is exactly `digits[p..]` — the 52-bit fraction is
            // 13 nibbles, so nothing lies beyond. A carry can ripple into `lead`.
            if p < digits.len() {
                let first = digits[p];
                let rest_nonzero = digits[p + 1..].iter().any(|&d| d != 0);
                let round_up = if first > 8 {
                    true
                } else if first < 8 {
                    false
                } else if rest_nonzero {
                    true // past the halfway point
                } else {
                    // Exactly halfway: round so the last kept nibble is even.
                    let last_kept = if p == 0 { lead } else { digits[p - 1] };
                    last_kept & 1 == 1
                };
                digits.truncate(p);
                if round_up {
                    let mut carry = true;
                    let mut i = digits.len();
                    while carry && i > 0 {
                        i -= 1;
                        if digits[i] == 0xf {
                            digits[i] = 0;
                        } else {
                            digits[i] += 1;
                            carry = false;
                        }
                    }
                    if carry {
                        lead += 1;
                    }
                }
            } else {
                digits.truncate(p);
            }
        }
        None => {
            while digits.last() == Some(&0) {
                digits.pop();
            }
        }
    }
    let hexdig = |d: u8| -> char {
        let c = if d < 10 {
            b'0' + d
        } else if upper {
            b'A' + (d - 10)
        } else {
            b'a' + (d - 10)
        };
        c as char
    };
    let mut s = String::from(if upper { "0X" } else { "0x" });
    s.push((b'0' + lead) as char);
    if !digits.is_empty() || matches!(prec, Some(p) if p > 0) {
        s.push('.');
        for &d in &digits {
            s.push(hexdig(d));
        }
        if let Some(p) = prec {
            for _ in digits.len()..p {
                s.push('0');
            }
        }
    }
    s.push(if upper { 'P' } else { 'p' });
    s.push(if exp < 0 { '-' } else { '+' });
    s.push_str(&exp.unsigned_abs().to_string());
    s
}

fn fmt_hex_float(out: &mut Vec<u8>, spec: &FmtSpec, f: f64, upper: bool) {
    if let Some(s) = special_float(f, spec) {
        apply_width(out, spec, b"", b"", s.as_bytes());
        return;
    }
    // Unlike the decimal float specs, `%a` keeps the sign of `-0.0`.
    let sign: &str = if f.is_sign_negative() {
        "-"
    } else if spec.flag_plus {
        "+"
    } else if spec.flag_space {
        " "
    } else {
        ""
    };
    let body = format_hex_float(f.abs(), upper, spec.precision);
    // Split the `0x`/`0X` prefix so the `0` flag zero-pads *after* it
    // (`%010a` -> `0x00001p+0`, not `00000x1p+0`).
    let (prefix, rest) = body.split_at(2);
    apply_width(
        out,
        spec,
        sign.as_bytes(),
        prefix.as_bytes(),
        rest.as_bytes(),
    );
}

// ---------- string and q ----------

fn fmt_string<'gc>(ctx: Context<'gc>, out: &mut Vec<u8>, spec: &FmtSpec, arg: Value<'gc>) {
    // Lua's %s applies tostring (`luaL_tolstring`) to non-strings — booleans,
    // numbers (in Lua form), nil, and the `type: 0xADDR` form for the rest.
    // Honoring `__tostring` needs native->Lua calls (deferred with #27).
    let ls = arg
        .get_string()
        .unwrap_or_else(|| crate::builtin::util::basic_tostring(ctx, arg));
    let bytes = ls.as_bytes();
    let trimmed: &[u8] = if let Some(p) = spec.precision {
        &bytes[..bytes.len().min(p)]
    } else {
        bytes
    };
    apply_width(out, spec, b"", b"", trimmed);
}

fn fmt_q<'gc>(
    ctx: Context<'gc>,
    out: &mut Vec<u8>,
    arg: Value<'gc>,
    arg_num: usize,
) -> Result<(), Error<'gc>> {
    if arg.is_nil() {
        out.extend_from_slice(b"nil");
    } else if let Some(b) = arg.get_boolean() {
        out.extend_from_slice(if b { b"true" } else { b"false" });
    } else if let Some(i) = arg.get_integer() {
        // `LUA_MININTEGER` has no decimal literal form (`-9223372036854775808`
        // parses as negation of an out-of-range literal), so Lua emits it as
        // unsigned hex, which reads back as the same integer.
        if i == i64::MIN {
            out.extend_from_slice(b"0x8000000000000000");
        } else {
            let s = format!("{i}");
            out.extend_from_slice(s.as_bytes());
        }
    } else if let Some(f) = arg.get_float() {
        // %q must read back to the exact value: hex-float for finite numbers,
        // Lua's literal forms for the specials.
        if f.is_nan() {
            out.extend_from_slice(b"(0/0)");
        } else if f.is_infinite() {
            out.extend_from_slice(if f < 0.0 { b"-1e9999" } else { b"1e9999" });
        } else {
            if f.is_sign_negative() {
                out.push(b'-');
            }
            out.extend_from_slice(format_hex_float(f.abs(), false, None).as_bytes());
        }
    } else if let Some(s) = arg.get_string() {
        // Mirror Lua's `addquoted`: `"` / `\` / `\n` -> backslash + the char;
        // control bytes (0..31, 127) -> decimal `\ddd`, zero-padded to 3
        // digits *only* when the next byte is an ASCII digit (so they can't
        // merge); everything else (printable and high bytes) emitted raw.
        let bytes = s.as_bytes();
        out.push(b'"');
        for (i, &b) in bytes.iter().enumerate() {
            match b {
                b'"' | b'\\' | b'\n' => {
                    out.push(b'\\');
                    out.push(b);
                }
                b if b < 0x20 || b == 0x7f => {
                    let next_is_digit = bytes.get(i + 1).is_some_and(u8::is_ascii_digit);
                    let esc = if next_is_digit {
                        format!("\\{b:03}")
                    } else {
                        format!("\\{b}")
                    };
                    out.extend_from_slice(esc.as_bytes());
                }
                b => out.push(b),
            }
        }
        out.push(b'"');
    } else {
        // Tables, functions, threads, userdata have no literal form.
        return Err(Error::from_str(
            ctx,
            &format!("bad argument #{arg_num} to 'format' (value has no literal form)"),
        ));
    }
    Ok(())
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
    // C printf: a precision suppresses the `0` flag only for the integer
    // conversions (d/i/o/u/x/X); for floats (f/e/g/a) the `0` flag still pads.
    let int_conv = matches!(spec.conv, b'd' | b'i' | b'u' | b'o' | b'x' | b'X');
    let zero_pad = spec.flag_zero && !(int_conv && spec.precision.is_some());
    if spec.flag_minus {
        out.extend_from_slice(sign);
        out.extend_from_slice(prefix);
        out.extend_from_slice(body);
        for _ in 0..pad {
            out.push(b' ');
        }
    } else if zero_pad {
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
        // Bound the result by Lua's `MAX_SIZE` (= `LUA_MAXINTEGER` on 64-bit),
        // not just by `usize` arithmetic: `(len+sep)*n` can fit `usize` yet still
        // be an absurd allocation (e.g. `rep("ab", maxinteger)`), so cap at
        // `i64::MAX` to raise a catchable error instead of aborting on a
        // `Vec::with_capacity` overflow.
        let total = body
            .len()
            .checked_add(sep.len())
            .and_then(|per| per.checked_mul(n))
            .and_then(|t| t.checked_sub(sep.len()))
            .filter(|&t| t <= i64::MAX as usize);
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
