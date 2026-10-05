use core::fmt::NumBuffer;
use std::cell::{Cell, RefCell};

use crate::Context;
use crate::builtin::strfmt_num::{
    self, SFormat, STRFMT_F_ALT, STRFMT_F_LEFT, STRFMT_F_PLUS, STRFMT_F_SPACE, STRFMT_F_UPPER,
    STRFMT_F_ZERO, STRFMT_SH_PREC, STRFMT_SH_WIDTH, STRFMT_T_FP_A, STRFMT_T_FP_E, STRFMT_T_FP_F,
    STRFMT_T_FP_G,
};
use crate::builtin::util;
// `%d`/`%f` argument coercion reuses the shared `util` helpers so the
// integer-representation and numeric-string rules (including `inf`/`nan`
// rejection) match `tonumber`/`math.*` and don't drift.
use crate::builtin::util::{to_integer, to_number as to_float};
use crate::env::{
    Error, Function, LuaString, NativeClosure, NativeFn, Stack, Table, Userdata, Value,
};
use crate::lua::{StashedError, StashedFunction, StashedTable, StashedValue};
use crate::vm::async_sequence::{AsyncSequence, SequenceReturn, async_sequence};
use crate::vm::interp::{IndexChain, walk_index_chain};
use crate::vm::sequence::CallbackAction;

mod meta;
mod pack;
mod pattern;
use pattern::{CapValue, MatchState, PatError};

/// Lua's `posrelat`: translate a possibly-negative 1-based string position into
/// an absolute 1-based position (negatives count from the end; 0 stays 0).
pub(super) fn posrelat(pos: i64, len: usize) -> i64 {
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
        ("find", lua_find),
        ("format", lua_format),
        ("gmatch", lua_gmatch),
        ("gsub", lua_gsub),
        ("len", lua_len),
        ("lower", lua_lower),
        ("match", lua_match),
        ("pack", pack::lua_pack),
        ("packsize", pack::lua_packsize),
        ("rep", lua_rep),
        ("reverse", lua_reverse),
        ("sub", lua_sub),
        ("unpack", pack::lua_unpack),
        ("upper", lua_upper),
    ];

    // `format`'s reusable output buffer, upvalue 0.
    let fmt_buf = Userdata::new(ctx.mutation(), Cell::new(Vec::<u8>::new()), 0);
    let lib = Table::new(ctx);
    for &(name, handler) in fns {
        let upvalues: &[Value<'gc>] = if name == "format" {
            &[Value::userdata(fmt_buf)]
        } else {
            &[]
        };
        let handler = Function::new_native(ctx.mutation(), handler, upvalues);
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(handler));
    }

    util::set_not_implemented(ctx, lib, "string", &["dump"]);

    meta::install(ctx, lib);

    let lib_name = Value::string(LuaString::new(ctx, b"string"));
    ctx.globals().raw_set(ctx, lib_name, Value::table(lib));
}

/// `byte(s [, i [, j]])` — the numeric codes of `s[i..j]` (1-based, negatives
/// from the end; `i` defaults to 1, `j` to `i`).
fn lua_byte<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "byte", 1)?;
    let bytes = s.as_bytes();
    let len = bytes.len();
    let i_arg = stack.get(1);
    let i = if i_arg.is_nil() {
        1
    } else {
        util::check_integer(ctx, i_arg, "byte", 2)?
    };
    let j_arg = stack.get(2);
    let j = if j_arg.is_nil() {
        i
    } else {
        util::check_integer(ctx, j_arg, "byte", 3)?
    };
    let start = posrelat(i, len).max(1);
    let end = posrelat(j, len).min(len as i64);
    let slice = if start <= end {
        &bytes[(start - 1) as usize..end as usize]
    } else {
        &[]
    };
    let Some(out) = stack.replace_slots(slice.len()) else {
        return Err(Error::from_str(ctx, "string slice too long"));
    };
    for (slot, &b) in out.iter_mut().zip(slice) {
        *slot = Value::integer(ctx.mutation(), b as i64);
    }
    Ok(CallbackAction::Return)
}

/// `char(...)` — a string built from the given byte values (each 0–255).
fn lua_char<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let n = stack.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let c = util::check_integer(ctx, stack.get(i), "char", i + 1)?;
        if !(0..=255).contains(&c) {
            return Err(util::arg_error(ctx, "char", i + 1, "value out of range"));
        }
        out.push(c as u8);
    }
    let r = LuaString::new(ctx, &out);
    stack.ret1(Value::string(r));
    Ok(CallbackAction::Return)
}

// ---------- pattern matching: shared helpers ----------

/// Turn a `PatError` from the matcher into a Lua error at the call boundary.
fn pat_err<'gc>(ctx: Context<'gc>, e: PatError) -> Error<'gc> {
    Error::from_str(ctx, &e.message())
}

/// Build a `Value` from a resolved capture (substring or position).
fn cap_to_value<'gc>(ctx: Context<'gc>, src: &[u8], cv: CapValue) -> Value<'gc> {
    match cv {
        CapValue::Str { start, end } => Value::string(LuaString::new(ctx, &src[start..end])),
        CapValue::Pos(n) => Value::integer(ctx.mutation(), n),
    }
}

/// `posrelatI(pos, len) - 1`: the 0-based start index for `find`/`match`/
/// `gmatch`. Returns `None` when the requested start is past the end of the
/// string (`init > len`), which the callers treat as "no match" (or, for
/// `gmatch`, "iterate nothing").
fn init_pos(pos: i64, len: usize) -> Option<usize> {
    // Mirrors `posrelatI`: positive is absolute, 0 -> 1, very negative clips to
    // 1, otherwise counts from the end. The i128 compare avoids any overflow.
    let one_based: usize = if pos > 0 {
        pos as usize
    } else if pos == 0 || (pos as i128) < -(len as i128) {
        1 // 0 -> 1; anything more negative than -len clips to 1
    } else {
        (len as i64 + pos + 1) as usize // count from the end
    };
    let init = one_based - 1;
    (init <= len).then_some(init)
}

/// `find(s, pattern [, init [, plain]])` — locate `pattern` in `s` from `init`
/// (1-based, negatives from the end). Returns the 1-based start/end indices
/// plus any captures, or nil. A `plain` flag — or a pattern with no magic
/// characters — switches to a plain substring search.
fn lua_find<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "find", 1)?;
    let p = util::check_string(ctx, stack.get(1), "find", 2)?;
    let src = s.as_bytes();
    let pat = p.as_bytes();

    let init_arg = stack.get(2);
    let init_raw = if init_arg.is_nil() {
        1
    } else {
        util::check_integer(ctx, init_arg, "find", 3)?
    };
    let Some(init) = init_pos(init_raw, src.len()) else {
        stack.ret1(Value::nil());
        return Ok(CallbackAction::Return);
    };

    // Plain search on explicit request or a pattern with no magic chars.
    if !stack.get(3).is_falsy() || pattern::nospecials(pat) {
        match pattern::plain_find(&src[init..], pat) {
            Some(off) => {
                let start = init + off;
                stack.replace(&[
                    Value::integer(ctx.mutation(), start as i64 + 1),
                    Value::integer(ctx.mutation(), (start + pat.len()) as i64),
                ]);
            }
            None => stack.replace(&[Value::nil()]),
        }
        return Ok(CallbackAction::Return);
    }

    let anchor = pat.first() == Some(&b'^');
    let body = if anchor { &pat[1..] } else { pat };
    let mut ms = MatchState::new(src, body);
    let mut s1 = init;
    loop {
        if let Some(e) = ms.match_at(s1).map_err(|e| pat_err(ctx, e))? {
            // start, end, then the explicit captures (no whole-match fallback).
            let mut out = vec![
                Value::integer(ctx.mutation(), s1 as i64 + 1),
                Value::integer(ctx.mutation(), e as i64),
            ];
            for i in 0..ms.num_captures(false) {
                let cv = ms.get_onecapture(i, s1, e).map_err(|e| pat_err(ctx, e))?;
                out.push(cap_to_value(ctx, src, cv));
            }
            stack.replace(&out);
            return Ok(CallbackAction::Return);
        }
        if anchor || s1 >= src.len() {
            break;
        }
        s1 += 1;
    }
    stack.ret1(Value::nil());
    Ok(CallbackAction::Return)
}

/// The most `format` keeps allocated in its buffer between calls.
const FORMAT_BUF_KEEP: usize = 64 * 1024;

fn lua_format<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let fmt_val = stack.get(0);
    let fmt_str = fmt_val
        .get_string()
        .ok_or_else(|| util::type_error(ctx, "format", 1, "string", stack.arg(0)))?;
    let fmt = fmt_str.as_bytes();

    // Taken rather than borrowed: a `__tostring` below can re-enter `format`, which then
    // finds the cell empty and allocates its own.
    let buf = closure.upvalues[0]
        .get_userdata()
        .expect("format upvalue 0 must be its buffer");
    let mut out = buf
        .with_data(|c: &Cell<Vec<u8>>| c.take())
        .expect("format upvalue 0 must be its buffer");
    out.reserve(64);
    let mut f = Formatter {
        i: 0,
        arg_idx: 1,
        out,
    };
    let res = f.run(ctx, fmt, &stack);
    let give_back = |mut out: Vec<u8>| {
        if out.capacity() <= FORMAT_BUF_KEEP {
            out.clear();
            buf.with_data(|c: &Cell<Vec<u8>>| c.set(out));
        }
    };
    let pending = match res {
        Ok(Some(pending)) => pending,
        Ok(None) => {
            let s = LuaString::new(ctx, &f.out);
            give_back(f.out);
            stack.ret1(Value::string(s));
            return Ok(CallbackAction::Return);
        }
        Err(e) => {
            give_back(f.out);
            return Err(e);
        }
    };
    // A conversion needs `__tostring`: finish in a sequence that can call it,
    // continuing from where `run` stopped. The buffer goes with it and is not
    // returned. The arguments, format string included, stay on the stack below
    // `n` throughout.
    let n = stack.len();
    let seq = async_sequence(ctx.mutation(), move |_locals, mut seq| async move {
        let mut pending = Some(pending);
        while let Some((spec, arg)) = pending {
            let bytes = util::tolstring(&mut seq, arg, n).await?;
            // `%q` adds the result as-is.
            if spec.conv == b'q' {
                f.out.extend_from_slice(&bytes);
            } else {
                fmt_str_bytes(&mut f.out, &spec, &bytes);
            }
            pending = seq.try_enter(|ctx, _locals, _exec, stack| {
                let fmt = stack.get(0).get_string().expect("checked on entry");
                f.run(ctx, fmt.as_bytes(), &stack)
            })?;
        }
        seq.enter(|ctx, _locals, _exec, mut stack| {
            stack.replace(&[Value::string(LuaString::new(ctx, &f.out))]);
        });
        Ok(SequenceReturn::Return)
    });
    Ok(CallbackAction::sequence(seq))
}

struct Formatter {
    i: usize,
    arg_idx: usize,
    out: Vec<u8>,
}

impl Formatter {
    /// Format `fmt` from where the previous call stopped, until the end or a
    /// `luaL_tolstring` conversion whose argument has `__tostring`; that
    /// conversion's spec and stack index are returned for the caller to finish.
    fn run<'gc>(
        &mut self,
        ctx: Context<'gc>,
        fmt: &[u8],
        args: &Stack<'gc, '_>,
    ) -> Result<Option<(FmtSpec, usize)>, Error<'gc>> {
        while self.i < fmt.len() {
            let lit = fmt[self.i..].iter().position(|&b| b == b'%');
            let end = lit.map_or(fmt.len(), |n| self.i + n);
            if end > self.i {
                self.out.extend_from_slice(&fmt[self.i..end]);
                self.i = end;
            }
            if lit.is_none() {
                break;
            }
            let (spec, next) = match fmt.get(end + 1) {
                // A bare conversion has nothing for `parse_spec` to check.
                Some(&conv)
                    if matches!(
                        conv,
                        b'd' | b'i'
                            | b'u'
                            | b'o'
                            | b'x'
                            | b'X'
                            | b'c'
                            | b'p'
                            | b's'
                            | b'q'
                            | b'a'
                            | b'A'
                            | b'e'
                            | b'f'
                            | b'g'
                            | b'G'
                            | b'E'
                    ) =>
                {
                    (
                        FmtSpec {
                            conv,
                            ..FmtSpec::default()
                        },
                        end + 2,
                    )
                }
                _ => parse_spec(ctx, fmt, end + 1)?,
            };
            self.i = next;
            if spec.conv == b'%' {
                self.out.push(b'%');
                continue;
            }
            // A conversion consumes the next argument; a missing one is an error
            // ("no value"), distinct from an explicitly-passed nil.
            if self.arg_idx >= args.len() {
                return Err(util::arg_error(ctx, "format", self.arg_idx + 1, "no value"));
            }
            let arg = args.get(self.arg_idx);
            self.arg_idx += 1;
            let tolstring = spec.conv == b's'
                || (spec.conv == b'q' && (arg.is_nil() || arg.get_boolean().is_some()));
            if tolstring && !ctx.metamethod_of(arg, ctx.symbols().mm_tostring).is_nil() {
                return Ok(Some((spec, self.arg_idx - 1)));
            }
            // After the increment, `arg_idx` is the 1-based Lua argument number of
            // the argument just consumed (the format string is #1).
            format_one(ctx, &mut self.out, &spec, arg, self.arg_idx)?;
        }
        Ok(None)
    }
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
        b'a' | b'A' | b'e' | b'E' | b'f' | b'g' | b'G' => (b"-+#0 ", true),
        b'c' | b'p' => (b"-", false),
        b's' => (b"-", true),
        _ => {
            return Err(Error::from_str(
                ctx,
                &format!(
                    "invalid conversion '{}' to 'format'",
                    String::from_utf8_lossy(form)
                ),
            ));
        }
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
            apply_width(out, spec, b"", b"", 0, &[n as u8]);
        }
        b'a' | b'A' | b'e' | b'E' | b'f' | b'g' | b'G' => {
            let f = to_float(arg).ok_or_else(|| arg_type_err(ctx, "number", &arg, arg_num))?;
            strfmt_num::put_fnum(out, float_sformat(spec), f);
        }
        b's' => {
            fmt_string(ctx, out, spec, arg);
        }
        b'p' => {
            let s = match util::to_pointer(arg) {
                Some(p) => format!("{p:p}"),
                None => "(null)".to_owned(),
            };
            apply_width(out, spec, b"", b"", 0, s.as_bytes());
        }
        b'q' => {
            fmt_q(ctx, out, arg, arg_num)?;
        }
        _ => unreachable!("parse_spec rejects other conversions"),
    }
    Ok(())
}

fn arg_type_err<'gc>(
    ctx: Context<'gc>,
    expected: &str,
    arg: &Value<'gc>,
    arg_num: usize,
) -> Error<'gc> {
    util::type_error(ctx, "format", arg_num, expected, Some(*arg))
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
    if to_float(arg).is_some() {
        return Err(util::arg_error(
            ctx,
            "format",
            arg_num,
            "number has no integer representation",
        ));
    }
    Err(util::type_error(
        ctx,
        "format",
        arg_num,
        "number",
        Some(arg),
    ))
}

// ---------- integer formatting ----------

fn fmt_int_signed(out: &mut Vec<u8>, spec: &FmtSpec, n: i64) {
    let mut dec = NumBuffer::new();
    let (zeros, digits) = int_digits(
        spec,
        n.unsigned_abs().format_into(&mut dec).as_bytes(),
        n == 0,
    );
    let sign: &[u8] = if n < 0 {
        b"-"
    } else if spec.flag_plus {
        b"+"
    } else if spec.flag_space {
        b" "
    } else {
        b""
    };
    apply_width(out, spec, sign, b"", zeros, digits);
}

fn fmt_int_unsigned(out: &mut Vec<u8>, spec: &FmtSpec, n: u64, radix: u32, upper: bool) {
    let mut dec = NumBuffer::new();
    let mut pow2 = [0; 22];
    let raw: &[u8] = match radix {
        10 => n.format_into(&mut dec).as_bytes(),
        // `format_into` is decimal-only.
        _ => {
            let (bits, alphabet): (u32, &[u8]) = match (radix, upper) {
                (8, _) => (3, b"01234567"),
                (16, false) => (4, b"0123456789abcdef"),
                (16, true) => (4, b"0123456789ABCDEF"),
                _ => unreachable!(),
            };
            let mut i = pow2.len();
            let mut m = n;
            loop {
                i -= 1;
                pow2[i] = alphabet[(m & ((1 << bits) - 1)) as usize];
                m >>= bits;
                if m == 0 {
                    break;
                }
            }
            &pow2[i..]
        }
    };
    let (zeros, digits) = int_digits(spec, raw, n == 0);
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
    apply_width(out, spec, b"", prefix, zeros, digits);
}

/// The zeros that extend `digits` to the spec's precision, and the digits; as in C, a zero
/// precision prints no digits for zero.
fn int_digits<'a>(spec: &FmtSpec, digits: &'a [u8], zero: bool) -> (usize, &'a [u8]) {
    match spec.precision {
        Some(0) if zero => (0, b""),
        Some(p) => (p.saturating_sub(digits.len()), digits),
        None => (0, digits),
    }
}

// ---------- float formatting ----------

/// The `SFormat` of a float conversion, for `strfmt_num`.
fn float_sformat(spec: &FmtSpec) -> SFormat {
    let mut sf = match spec.conv.to_ascii_lowercase() {
        b'a' => STRFMT_T_FP_A,
        b'e' => STRFMT_T_FP_E,
        b'f' => STRFMT_T_FP_F,
        _ => STRFMT_T_FP_G,
    };
    for (on, flag) in [
        (spec.conv.is_ascii_uppercase(), STRFMT_F_UPPER),
        (spec.flag_minus, STRFMT_F_LEFT),
        (spec.flag_plus, STRFMT_F_PLUS),
        (spec.flag_zero, STRFMT_F_ZERO),
        (spec.flag_space, STRFMT_F_SPACE),
        (spec.flag_hash, STRFMT_F_ALT),
    ] {
        if on {
            sf |= flag;
        }
    }
    // `parse_spec` caps both at 99, inside the 8-bit fields.
    sf |= (spec.width as u32) << STRFMT_SH_WIDTH;
    if let Some(p) = spec.precision {
        sf |= (p as u32 + 1) << STRFMT_SH_PREC;
    }
    sf
}

// ---------- string and q ----------

/// `%s` of a value without `__tostring` (`Formatter::run` diverts the rest).
fn fmt_string<'gc>(ctx: Context<'gc>, out: &mut Vec<u8>, spec: &FmtSpec, arg: Value<'gc>) {
    fmt_str_bytes(out, spec, util::basic_tostring(ctx, arg).as_bytes());
}

fn fmt_str_bytes(out: &mut Vec<u8>, spec: &FmtSpec, bytes: &[u8]) {
    let trimmed: &[u8] = if let Some(p) = spec.precision {
        &bytes[..bytes.len().min(p)]
    } else {
        bytes
    };
    apply_width(out, spec, b"", b"", 0, trimmed);
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
            out.extend_from_slice(i.format_into(&mut NumBuffer::new()).as_bytes());
        }
    } else if let Some(f) = arg.get_float() {
        // %q must read back to the exact value: hex-float for finite numbers,
        // Lua's literal forms for the specials.
        if f.is_nan() {
            out.extend_from_slice(b"(0/0)");
        } else if f.is_infinite() {
            out.extend_from_slice(if f < 0.0 { b"-1e9999" } else { b"1e9999" });
        } else {
            strfmt_num::put_fnum(out, STRFMT_T_FP_A, f);
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
        return Err(util::arg_error(
            ctx,
            "format",
            arg_num,
            "value has no literal form",
        ));
    }
    Ok(())
}

// ---------- shared width/padding ----------

/// Append `sign`, `prefix`, `zeros` zero digits and `body`, padded to the spec's width.
fn apply_width(
    out: &mut Vec<u8>,
    spec: &FmtSpec,
    sign: &[u8],
    prefix: &[u8],
    zeros: usize,
    body: &[u8],
) {
    let content_len = sign.len() + prefix.len() + zeros + body.len();
    let pad = spec.width.saturating_sub(content_len);
    if pad == 0 && zeros == 0 {
        out.extend_from_slice(sign);
        out.extend_from_slice(prefix);
        out.extend_from_slice(body);
        return;
    }
    // C printf: a precision suppresses the `0` flag only for the integer
    // conversions (d/i/o/u/x/X).
    let int_conv = matches!(spec.conv, b'd' | b'i' | b'u' | b'o' | b'x' | b'X');
    let zero_pad = spec.flag_zero && !spec.flag_minus && !(int_conv && spec.precision.is_some());
    if !spec.flag_minus && !zero_pad {
        out.resize(out.len() + pad, b' ');
    }
    out.extend_from_slice(sign);
    out.extend_from_slice(prefix);
    out.resize(out.len() + zeros + if zero_pad { pad } else { 0 }, b'0');
    out.extend_from_slice(body);
    if spec.flag_minus {
        out.resize(out.len() + pad, b' ');
    }
}

/// Iterator state for `gmatch`, owned by the closure's userdata upvalue. The
/// subject and pattern bytes are copied in so the closure needn't keep the
/// argument strings rooted (mirroring PUC's `lua_settop` + userdata).
struct GmatchState {
    src: Vec<u8>,
    pat: Vec<u8>,
    /// Next subject byte to try matching from.
    pos: usize,
    /// End of the previous match — empty matches at this exact spot are
    /// skipped so iteration always advances (PUC's `e != lastmatch`).
    lastmatch: Option<usize>,
}

/// `gmatch(s, pattern [, init])` — return an iterator that, on each call,
/// yields the captures of the next match (whole match if no captures). Unlike
/// `find`/`match`/`gsub`, a leading `^` is *not* an anchor here — it matches a
/// literal `^` — because the iterator never strips it.
fn lua_gmatch<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "gmatch", 1)?;
    let p = util::check_string(ctx, stack.get(1), "gmatch", 2)?;
    let init_arg = stack.get(2);
    let init_raw = if init_arg.is_nil() {
        1
    } else {
        util::check_integer(ctx, init_arg, "gmatch", 3)?
    };
    // `init > len` clamps to len+1 (iterate nothing) rather than erroring.
    let pos = init_pos(init_raw, s.len()).unwrap_or(s.len() + 1);
    let state = GmatchState {
        src: s.as_bytes().to_vec(),
        pat: p.as_bytes().to_vec(),
        pos,
        lastmatch: None,
    };
    let ud = Userdata::new(ctx.mutation(), RefCell::new(state), 0);
    let iter = Function::new_native(ctx.mutation(), gmatch_aux, &[Value::userdata(ud)]);
    stack.ret1(Value::function(iter));
    Ok(CallbackAction::Return)
}

/// One step of a `gmatch` iterator: advance from the stored position to the
/// next match and return its captures, or nothing when exhausted.
fn gmatch_aux<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let ud = closure.upvalues[0]
        .get_userdata()
        .expect("gmatch iterator upvalue must be a userdata");
    let result = ud
        .with_data::<RefCell<GmatchState>, _>(|cell| {
            let mut st = cell.borrow_mut();
            let st = &mut *st;
            let mut ms = MatchState::new(&st.src, &st.pat);
            let mut src_pos = st.pos;
            loop {
                if src_pos > st.src.len() {
                    return Ok(None);
                }
                match ms.match_at(src_pos)? {
                    Some(e) if Some(e) != st.lastmatch => {
                        let n = ms.num_captures(true);
                        let mut caps = Vec::with_capacity(n);
                        for i in 0..n {
                            let cv = ms.get_onecapture(i, src_pos, e)?;
                            caps.push(cap_to_value(ctx, &st.src, cv));
                        }
                        // `ms` is unused past here, so its borrow of `st.src`/
                        // `st.pat` ends and these field writes are allowed.
                        st.pos = e;
                        st.lastmatch = Some(e);
                        return Ok(Some(caps));
                    }
                    _ => {}
                }
                src_pos += 1;
            }
        })
        .expect("gmatch userdata payload type mismatch");
    match result {
        Err(e) => Err(pat_err(ctx, e)),
        Ok(None) => {
            stack.replace(&[]);
            Ok(CallbackAction::Return)
        }
        Ok(Some(caps)) => {
            stack.replace(&caps);
            Ok(CallbackAction::Return)
        }
    }
}

/// `gsub(s, pattern, repl [, n])` — replace up to `n` (default: all)
/// non-overlapping matches of `pattern` in `s`. `repl` may be a template
/// string (`%0`–`%9`, `%%`), a number, a table (indexed by the first capture,
/// honoring `__index`), or a function (called with the captures). Returns the
/// result string and the substitution count.
///
/// A string/number `repl` resolves entirely in Rust (a synchronous return). A
/// table/function `repl` re-enters the interpreter per match, so the work runs
/// as an async `Sequence` (the same mechanism `table.sort` uses for its
/// comparator).
fn lua_gsub<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "gsub", 1)?;
    let p = util::check_string(ctx, stack.get(1), "gsub", 2)?;
    let repl = stack.get(2);
    let n_arg = stack.get(3);
    // Default cap is `len+1`: at most one match per position plus the empty
    // match past the end, so this never truncates a real "replace all".
    let max_n = if n_arg.is_nil() {
        s.len() as i64 + 1
    } else {
        util::check_integer(ctx, n_arg, "gsub", 4)?
    };

    // Fast path: string/number replacement template, fully synchronous.
    if let Some(template) = util::to_lstring(ctx, repl) {
        let (result, count) =
            gsub_string(ctx, s.as_bytes(), p.as_bytes(), template.as_bytes(), max_n)?;
        stack.replace(&[result, Value::integer(ctx.mutation(), count)]);
        return Ok(CallbackAction::Return);
    }

    // Table/function replacement: re-enters the VM, so drive a sequence.
    let repl_fn = repl.get_function();
    let repl_tbl = repl.get_table();
    if repl_fn.is_none() && repl_tbl.is_none() {
        return Err(util::type_error(
            ctx,
            "gsub",
            3,
            "string/function/table",
            Some(repl),
        ));
    }

    let mc = ctx.mutation();
    let src = s.as_bytes().to_vec();
    let pat = p.as_bytes().to_vec();
    let seq = async_sequence(mc, move |locals, seq| {
        let repl = match (repl_fn, repl_tbl) {
            (Some(f), _) => Repl::Func(locals.stash(mc, f)),
            (_, Some(t)) => Repl::Table(locals.stash(mc, t)),
            _ => unreachable!("repl kind validated above"),
        };
        async move {
            let mut seq = seq;
            let (result, count) = gsub_run(&mut seq, src, pat, repl, max_n).await?;
            seq.enter(|ctx, locals, _exec, mut stack| {
                let result = locals.fetch(ctx.mutation(), &result);
                stack.replace(&[result, Value::integer(ctx.mutation(), count)]);
            });
            Ok(SequenceReturn::Return)
        }
    });
    Ok(CallbackAction::sequence(seq))
}

/// Synchronous `gsub` for a template `repl`. Returns the result string and the
/// substitution count. The result is freshly interned; because identical bytes
/// intern to the same `LuaString`, an all-misses run still yields the original
/// string object (so PUC's "return the original on no change" identity holds).
fn gsub_string<'gc>(
    ctx: Context<'gc>,
    src: &[u8],
    pat: &[u8],
    template: &[u8],
    max_n: i64,
) -> Result<(Value<'gc>, i64), Error<'gc>> {
    let anchor = pat.first() == Some(&b'^');
    let body = if anchor { &pat[1..] } else { pat };
    let mut ms = MatchState::new(src, body);
    let mut out: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    let mut lastmatch: Option<usize> = None;
    let mut count: i64 = 0;
    while count < max_n {
        let m = ms.match_at(pos).map_err(|e| pat_err(ctx, e))?;
        match m {
            Some(e) if Some(e) != lastmatch => {
                count += 1;
                add_s(ctx, &mut out, &ms, src, pos, e, template)?;
                pos = e;
                lastmatch = Some(e);
            }
            _ => {
                if pos < src.len() {
                    out.push(src[pos]);
                    pos += 1;
                } else {
                    break;
                }
            }
        }
        if anchor {
            break;
        }
    }
    out.extend_from_slice(&src[pos..]);
    Ok((Value::string(LuaString::new(ctx, &out)), count))
}

/// Expand a replacement template (`add_s`) for the match `src[s..e]` into
/// `out`: `%0` is the whole match, `%1`–`%9` the captures, `%%` a literal `%`;
/// any other `%x` is an error.
fn add_s<'gc>(
    ctx: Context<'gc>,
    out: &mut Vec<u8>,
    ms: &MatchState,
    src: &[u8],
    s: usize,
    e: usize,
    template: &[u8],
) -> Result<(), Error<'gc>> {
    let invalid = || Error::from_str(ctx, "invalid use of '%' in replacement string");
    let mut i = 0;
    while i < template.len() {
        let b = template[i];
        if b != b'%' {
            out.push(b);
            i += 1;
            continue;
        }
        i += 1;
        let c = *template.get(i).ok_or_else(invalid)?;
        i += 1;
        if c == b'%' {
            out.push(b'%');
        } else if c == b'0' {
            out.extend_from_slice(&src[s..e]);
        } else if c.is_ascii_digit() {
            match ms
                .get_onecapture((c - b'1') as usize, s, e)
                .map_err(|er| pat_err(ctx, er))?
            {
                CapValue::Str { start, end } => out.extend_from_slice(&src[start..end]),
                CapValue::Pos(n) => util::push_int(out, n),
            }
        } else {
            return Err(invalid());
        }
    }
    Ok(())
}

/// Stashed table/function replacement target for the sequence path.
enum Repl {
    Func(StashedFunction),
    Table(StashedTable),
}

/// A capture extracted into owned bytes (or a position) so it can outlive the
/// transient `MatchState` borrow across an `.await`.
enum OwnedCap {
    Bytes(Vec<u8>),
    Pos(i64),
}

impl OwnedCap {
    fn to_value<'gc>(&self, ctx: Context<'gc>) -> Value<'gc> {
        match self {
            OwnedCap::Bytes(b) => Value::string(LuaString::new(ctx, b)),
            OwnedCap::Pos(n) => Value::integer(ctx.mutation(), *n),
        }
    }
}

/// What a single match's replacement resolves to.
enum ReplResult {
    /// Keep the original matched text (function/table returned nil/false).
    Keep,
    /// Substitute these bytes.
    Bytes(Vec<u8>),
}

/// One iteration's matching decision, extracted synchronously so the borrow of
/// the subject/pattern by `MatchState` never spans an `.await`.
enum GsubStep {
    /// A match ending at `e`; `whole` is the matched text and `caps` the
    /// replacement arguments (all captures for a function, just the first for
    /// a table).
    Replace {
        whole: Vec<u8>,
        caps: Vec<OwnedCap>,
        e: usize,
    },
    /// No match here: copy one subject byte and advance.
    SkipChar,
    /// End of subject: stop.
    End,
}

/// Async `gsub` for a table/function `repl`. Mirrors `gsub_string`'s loop but
/// resolves each replacement by re-entering the VM (`seq.call` for a function,
/// a `__index`-honoring index for a table).
async fn gsub_run(
    seq: &mut AsyncSequence,
    src: Vec<u8>,
    pat: Vec<u8>,
    repl: Repl,
    max_n: i64,
) -> Result<(StashedValue, i64), StashedError> {
    let anchor = pat.first() == Some(&b'^');
    let pat_off = if anchor { 1 } else { 0 };
    let is_func = matches!(repl, Repl::Func(_));

    let mut out: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    let mut lastmatch: Option<usize> = None;
    let mut count: i64 = 0;

    while count < max_n {
        // Synchronous matching block — `MatchState` lives and dies here, so its
        // borrow of `src`/`pat` never crosses the awaits below.
        let step: Result<GsubStep, PatError> = (|| {
            let mut ms = MatchState::new(&src, &pat[pat_off..]);
            match ms.match_at(pos)? {
                Some(e) if Some(e) != lastmatch => {
                    // Function: all captures as call args. Table: just the
                    // first capture (whole match if there are no captures).
                    let n = if is_func { ms.num_captures(true) } else { 1 };
                    let mut caps = Vec::with_capacity(n);
                    for i in 0..n {
                        caps.push(match ms.get_onecapture(i, pos, e)? {
                            CapValue::Str { start, end } => {
                                OwnedCap::Bytes(src[start..end].to_vec())
                            }
                            CapValue::Pos(p) => OwnedCap::Pos(p),
                        });
                    }
                    Ok(GsubStep::Replace {
                        whole: src[pos..e].to_vec(),
                        caps,
                        e,
                    })
                }
                _ => Ok(if pos < src.len() {
                    GsubStep::SkipChar
                } else {
                    GsubStep::End
                }),
            }
        })();

        match step.map_err(|e| stash_pat_err(seq, e))? {
            GsubStep::Replace { whole, caps, e } => {
                count += 1;
                let res = match &repl {
                    Repl::Func(f) => call_func_repl(seq, f, &caps).await?,
                    Repl::Table(t) => table_index_repl(seq, t, &caps[0]).await?,
                };
                match res {
                    ReplResult::Keep => out.extend_from_slice(&whole),
                    ReplResult::Bytes(b) => out.extend_from_slice(&b),
                }
                pos = e;
                lastmatch = Some(e);
            }
            GsubStep::SkipChar => {
                out.push(src[pos]);
                pos += 1;
            }
            GsubStep::End => break,
        }
        if anchor {
            break;
        }
    }
    out.extend_from_slice(&src[pos..]);

    let result = seq.enter(|ctx, locals, _exec, _stack| {
        locals.stash(ctx.mutation(), Value::string(LuaString::new(ctx, &out)))
    });
    Ok((result, count))
}

/// Call a function `repl` with the captures as arguments and classify its
/// first result.
async fn call_func_repl(
    seq: &mut AsyncSequence,
    f: &StashedFunction,
    caps: &[OwnedCap],
) -> Result<ReplResult, StashedError> {
    seq.enter(|ctx, _locals, _exec, mut stack| {
        stack.clear();
        for c in caps {
            stack.push(c.to_value(ctx));
        }
    });
    seq.call(f, 0).await?;
    seq.try_enter(|ctx, _locals, _exec, stack| classify_repl_result(ctx, stack.get(0)))
}

/// Index a table `repl` by `key` (honoring `__index`, which may itself be a
/// function requiring a call) and classify the result.
async fn table_index_repl(
    seq: &mut AsyncSequence,
    t: &StashedTable,
    key: &OwnedCap,
) -> Result<ReplResult, StashedError> {
    enum Plan {
        Resolved(ReplResult),
        CallIndex(StashedFunction),
    }
    let plan = seq.try_enter(|ctx, locals, _exec, mut stack| {
        let tbl = locals.fetch(ctx.mutation(), t);
        let key_val = key.to_value(ctx);
        let v = tbl.raw_get(key_val);
        if !v.is_nil() {
            return Ok(Plan::Resolved(classify_repl_result(ctx, v)?));
        }
        match walk_index_chain(ctx, Value::table(tbl), key_val) {
            IndexChain::Resolved(rv) => Ok(Plan::Resolved(classify_repl_result(ctx, rv)?)),
            IndexChain::Invoke { func, receiver } => {
                stack.replace(&[receiver, key_val]);
                Ok(Plan::CallIndex(locals.stash(ctx.mutation(), func)))
            }
            IndexChain::NotIndexable(v) => Err(Error::from_str(
                ctx,
                &format!("attempt to index a {} value", v.type_name()),
            )),
            IndexChain::Exhausted => Err(Error::from_str(
                ctx,
                "'__index' chain too long; possible loop",
            )),
        }
    })?;
    match plan {
        Plan::Resolved(r) => Ok(r),
        Plan::CallIndex(f) => {
            seq.call(&f, 0).await?;
            seq.try_enter(|ctx, _locals, _exec, stack| classify_repl_result(ctx, stack.get(0)))
        }
    }
}

/// Classify a function/table replacement result: nil/false keeps the original;
/// a string or number substitutes; anything else is an error (matching PUC's
/// `lua_isstring` test, which accepts numbers).
fn classify_repl_result<'gc>(ctx: Context<'gc>, v: Value<'gc>) -> Result<ReplResult, Error<'gc>> {
    if v.is_falsy() {
        Ok(ReplResult::Keep)
    } else if let Some(s) = util::to_lstring(ctx, v) {
        Ok(ReplResult::Bytes(s.as_bytes().to_vec()))
    } else {
        Err(Error::from_str(
            ctx,
            &format!("invalid replacement value (a {})", v.type_name()),
        ))
    }
}

/// Convert a matcher `PatError` into a `StashedError` from inside a sequence.
fn stash_pat_err(seq: &mut AsyncSequence, e: PatError) -> StashedError {
    seq.try_enter(|ctx, _locals, _exec, _stack| Result::<(), _>::Err(pat_err(ctx, e)))
        .unwrap_err()
}

/// `len(s)` — number of bytes in `s`.
fn lua_len<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "len", 1)?;
    stack.ret1(Value::integer(ctx.mutation(), s.len() as i64));
    Ok(CallbackAction::Return)
}

/// `lower(s)` — ASCII-lowercased copy of `s` (C locale).
fn lua_lower<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "lower", 1)?;
    let lowered: Vec<u8> = s.as_bytes().iter().map(u8::to_ascii_lowercase).collect();
    stack.ret1(Value::string(LuaString::new(ctx, &lowered)));
    Ok(CallbackAction::Return)
}

/// `match(s, pattern [, init])` — return the captures of the first match of
/// `pattern` in `s` from `init`, or the whole match if the pattern has no
/// captures; nil if there is no match.
fn lua_match<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "match", 1)?;
    let p = util::check_string(ctx, stack.get(1), "match", 2)?;
    let src = s.as_bytes();
    let pat = p.as_bytes();

    let init_arg = stack.get(2);
    let init_raw = if init_arg.is_nil() {
        1
    } else {
        util::check_integer(ctx, init_arg, "match", 3)?
    };
    let Some(init) = init_pos(init_raw, src.len()) else {
        stack.ret1(Value::nil());
        return Ok(CallbackAction::Return);
    };

    let anchor = pat.first() == Some(&b'^');
    let body = if anchor { &pat[1..] } else { pat };
    let mut ms = MatchState::new(src, body);
    let mut s1 = init;
    loop {
        if let Some(e) = ms.match_at(s1).map_err(|e| pat_err(ctx, e))? {
            let n = ms.num_captures(true); // whole match if no captures
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let cv = ms.get_onecapture(i, s1, e).map_err(|e| pat_err(ctx, e))?;
                out.push(cap_to_value(ctx, src, cv));
            }
            stack.replace(&out);
            return Ok(CallbackAction::Return);
        }
        if anchor || s1 >= src.len() {
            break;
        }
        s1 += 1;
    }
    stack.ret1(Value::nil());
    Ok(CallbackAction::Return)
}

/// `rep(s, n [, sep])` — `s` repeated `n` times, with `sep` between copies.
/// `n <= 0` yields the empty string.
fn lua_rep<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "rep", 1)?;
    let n = util::check_integer(ctx, stack.get(1), "rep", 2)?;
    let sep = util::opt_string(ctx, stack.get(2), "rep", 3)?
        .map_or(Vec::new(), |s| s.as_bytes().to_vec());
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
            return Err(Error::from_str(ctx, "resulting string too large"));
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
    stack.ret1(Value::string(LuaString::new(ctx, &out)));
    Ok(CallbackAction::Return)
}

/// `reverse(s)` — `s` with its bytes in reverse order.
fn lua_reverse<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "reverse", 1)?;
    let mut bytes = s.as_bytes().to_vec();
    bytes.reverse();
    stack.ret1(Value::string(LuaString::new(ctx, &bytes)));
    Ok(CallbackAction::Return)
}

/// `sub(s, i [, j])` — substring `s[i..j]` (1-based, negatives from the end;
/// `i` defaults to 1, `j` to -1).
fn lua_sub<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "sub", 1)?;
    let bytes = s.as_bytes();
    let len = bytes.len();
    let i_arg = stack.get(1);
    let i = if i_arg.is_nil() {
        1
    } else {
        util::check_integer(ctx, i_arg, "sub", 2)?
    };
    let j_arg = stack.get(2);
    let j = if j_arg.is_nil() {
        -1
    } else {
        util::check_integer(ctx, j_arg, "sub", 3)?
    };
    let start = posrelat(i, len).max(1);
    let end = posrelat(j, len).min(len as i64);
    let result = if start <= end {
        LuaString::new(ctx, &bytes[(start - 1) as usize..end as usize])
    } else {
        LuaString::new(ctx, b"")
    };
    stack.ret1(Value::string(result));
    Ok(CallbackAction::Return)
}

/// `upper(s)` — ASCII-uppercased copy of `s` (C locale).
fn lua_upper<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "upper", 1)?;
    let uppered: Vec<u8> = s.as_bytes().iter().map(u8::to_ascii_uppercase).collect();
    stack.ret1(Value::string(LuaString::new(ctx, &uppered)));
    Ok(CallbackAction::Return)
}
