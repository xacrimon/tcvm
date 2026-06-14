//! `string.pack` / `string.unpack` / `string.packsize` — binary
//! (de)serialization, mirroring Lua 5.5's `lstrlib.c`.
//!
//! The three entry points share a single format-string walker (`get_details`,
//! itself driving `get_option`) that yields, per option, its `(KOption, size,
//! padding-before)`. `pack` writes values + zero padding, `unpack` reads them
//! back (and returns the next byte position), `packsize` only sums sizes.

use super::{check_str, posrelat};
use crate::Context;
use crate::builtin::util;
use crate::env::{Error, LuaString, NativeContext, Stack, Value};
use crate::vm::sequence::CallbackAction;

// Native ABI (fixed, per project decision): LP64 widths, little-endian,
// native alignment 8. Unlike reference Lua these don't follow the host's C
// `sizeof`/endianness, so packed output is identical on every platform.
const NATIVE_LITTLE: bool = true;
const NATIVE_ALIGN: usize = 8;
/// Width of a Lua integer (`i64`) and of `size_t` — the boundary at which
/// integer overflow checks stop and sign-extension / fit checks take over.
const SZINT: usize = 8;
/// Upper bound on the byte count for the sized options `i`/`I`/`s`/`!`.
const MAXINTSIZE: usize = 16;
/// Reference Lua's `MAXSIZE` (= `LUA_MAXINTEGER` on the LP64 model). Bounds
/// numeral accumulation so an oversized numeral stops at the same digit as
/// reference, leaving the remainder to be parsed (and rejected) as options.
const MAXSIZE: usize = i64::MAX as usize;

#[derive(Clone, Copy, PartialEq, Eq)]
enum KOption {
    Int {
        signed: bool,
    },
    Float,
    /// `s` — body preceded by an unsigned length integer of `size` bytes.
    Str,
    /// `c` — fixed `size` bytes.
    Char,
    /// `z` — zero-terminated.
    Zstr,
    /// `x` — one padding byte.
    Padding,
    /// `X` — alignment-only, takes its alignment from the following option.
    PaddAlign,
    /// `<` `>` `=` `!` and space — state changes / ignored, no value.
    Nop,
}

/// In-progress format state: endianness and the maximum alignment (`!n`).
struct Header {
    little: bool,
    maxalign: usize,
}

fn bad_arg<'gc>(ctx: Context<'gc>, fname: &str, n: usize, msg: &str) -> Error<'gc> {
    Error::from_str(ctx, &format!("bad argument #{n} to '{fname}' ({msg})"))
}

/// Parse an optional decimal numeral at `*pos`, returning `df` when none is
/// present. The accumulation cap mirrors reference Lua's `getnum`: once the
/// running value would exceed `MAXSIZE`, the remaining digits are left for the
/// next option (and then rejected there), so the accept/reject boundary matches.
fn get_num(fmt: &[u8], pos: &mut usize, df: usize) -> usize {
    if *pos >= fmt.len() || !fmt[*pos].is_ascii_digit() {
        return df;
    }
    let mut a: usize = 0;
    while *pos < fmt.len() && fmt[*pos].is_ascii_digit() && a <= (MAXSIZE - 9) / 10 {
        a = a * 10 + (fmt[*pos] - b'0') as usize;
        *pos += 1;
    }
    a
}

/// `getnumlimit`: a numeral for `i`/`I`/`s`/`!`, constrained to `[1, 16]`.
fn get_num_limit<'gc>(
    ctx: Context<'gc>,
    fmt: &[u8],
    pos: &mut usize,
    df: usize,
) -> Result<usize, Error<'gc>> {
    let sz = get_num(fmt, pos, df);
    if !(1..=MAXINTSIZE).contains(&sz) {
        // The limit check sees the full value, but reference prints it through a
        // C `(int)` cast — match that truncation so a >2^31 numeral reports the
        // same (wrapped) size. Identity for every in-range size.
        let shown = sz as u32 as i32;
        return Err(Error::from_str(
            ctx,
            &format!("integral size ({shown}) out of limits [1,{MAXINTSIZE}]"),
        ));
    }
    Ok(sz)
}

/// Consume one option char (and any attached numeral), updating `h` for the
/// configuration options. Returns the option kind and its byte size.
/// Caller guarantees `*pos < fmt.len()`.
fn get_option<'gc>(
    ctx: Context<'gc>,
    h: &mut Header,
    fmt: &[u8],
    pos: &mut usize,
) -> Result<(KOption, usize), Error<'gc>> {
    let opt = fmt[*pos];
    *pos += 1;
    let signed = |s| KOption::Int { signed: s };
    let r = match opt {
        b'b' => (signed(true), 1),
        b'B' => (signed(false), 1),
        b'h' => (signed(true), 2),
        b'H' => (signed(false), 2),
        b'i' => (signed(true), get_num_limit(ctx, fmt, pos, 4)?),
        b'I' => (signed(false), get_num_limit(ctx, fmt, pos, 4)?),
        b'l' | b'j' => (signed(true), 8),
        b'L' | b'J' | b'T' => (signed(false), 8),
        b'f' => (KOption::Float, 4),
        b'd' | b'n' => (KOption::Float, 8),
        b'c' => {
            if *pos >= fmt.len() || !fmt[*pos].is_ascii_digit() {
                return Err(Error::from_str(ctx, "missing size for format option 'c'"));
            }
            (KOption::Char, get_num(fmt, pos, 0))
        }
        b's' => (KOption::Str, get_num_limit(ctx, fmt, pos, SZINT)?),
        b'z' => (KOption::Zstr, 0),
        b'x' => (KOption::Padding, 1),
        b'X' => (KOption::PaddAlign, 0),
        b' ' => (KOption::Nop, 0),
        b'<' => {
            h.little = true;
            (KOption::Nop, 0)
        }
        b'>' => {
            h.little = false;
            (KOption::Nop, 0)
        }
        b'=' => {
            h.little = NATIVE_LITTLE;
            (KOption::Nop, 0)
        }
        b'!' => {
            h.maxalign = get_num_limit(ctx, fmt, pos, NATIVE_ALIGN)?;
            (KOption::Nop, 0)
        }
        other => {
            return Err(Error::from_str(
                ctx,
                &format!("invalid format option '{}'", other as char),
            ));
        }
    };
    Ok(r)
}

/// `getdetails`: read the next option and compute the alignment padding needed
/// before it given the current `totalsize`. Returns `(opt, size, ntoalign)`.
fn get_details<'gc>(
    ctx: Context<'gc>,
    h: &mut Header,
    totalsize: usize,
    fmt: &[u8],
    pos: &mut usize,
    fname: &str,
) -> Result<(KOption, usize, usize), Error<'gc>> {
    let (opt, size) = get_option(ctx, h, fmt, pos)?;
    // Alignment usually follows the option size; `X` borrows it from the
    // following option (which is otherwise consumed and ignored).
    let mut align = size;
    if opt == KOption::PaddAlign {
        if *pos >= fmt.len() {
            return Err(bad_arg(ctx, fname, 1, "invalid next option for option 'X'"));
        }
        let (next_opt, next_size) = get_option(ctx, h, fmt, pos)?;
        align = next_size;
        if next_opt == KOption::Char || align == 0 {
            return Err(bad_arg(ctx, fname, 1, "invalid next option for option 'X'"));
        }
    }

    let ntoalign = if align <= 1 || opt == KOption::Char {
        0
    } else {
        if align > h.maxalign {
            align = h.maxalign;
        }
        if align & (align - 1) != 0 {
            return Err(bad_arg(
                ctx,
                fname,
                1,
                "format asks for alignment not power of 2",
            ));
        }
        (align - (totalsize & (align - 1))) & (align - 1)
    };
    Ok((opt, size, ntoalign))
}

/// Write `n` (a value's raw bits) as `size` bytes in `little`/big order. When
/// `neg` and `size` exceeds a Lua integer, the surplus high bytes are filled
/// with `0xFF` to sign-extend a negative value.
fn pack_int_bytes(out: &mut Vec<u8>, n: u64, little: bool, size: usize, neg: bool) {
    let base = out.len();
    out.resize(base + size, 0);
    let mut v = n;
    for i in 0..size {
        let idx = base + if little { i } else { size - 1 - i };
        out[idx] = (v & 0xff) as u8;
        v >>= 8; // exhausts to 0 once past the 8 bytes of `n`
    }
    if neg && size > SZINT {
        for i in SZINT..size {
            out[base + if little { i } else { size - 1 - i }] = 0xff;
        }
    }
}

/// Read a `size`-byte integer; sign-extends short signed values and checks that
/// values wider than a Lua integer actually fit one.
fn unpack_int<'gc>(
    ctx: Context<'gc>,
    bytes: &[u8],
    little: bool,
    size: usize,
    signed: bool,
) -> Result<i64, Error<'gc>> {
    let limit = size.min(SZINT);
    let mut res: u64 = 0;
    for i in (0..limit).rev() {
        res = (res << 8) | bytes[if little { i } else { size - 1 - i }] as u64;
    }
    if size < SZINT {
        if signed {
            let mask = 1u64 << (size * 8 - 1);
            res = (res ^ mask).wrapping_sub(mask);
        }
    } else if size > SZINT {
        // The unread high bytes must all match the sign extension of the
        // SZINT-byte value, else it doesn't round-trip through a Lua integer.
        let fill = if !signed || (res as i64) >= 0 {
            0
        } else {
            0xff
        };
        for i in limit..size {
            if bytes[if little { i } else { size - 1 - i }] != fill {
                return Err(Error::from_str(
                    ctx,
                    &format!("{size}-byte integer does not fit into Lua Integer"),
                ));
            }
        }
    }
    Ok(res as i64)
}

pub(super) fn lua_pack<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let ctx = nctx.ctx;
    let fmt = check_str(ctx, stack.get(0), "pack", 1)?;
    let fmt = fmt.as_bytes();
    let mut h = Header {
        little: NATIVE_LITTLE,
        maxalign: 1,
    };
    let mut fpos = 0;
    let mut out: Vec<u8> = Vec::new();
    // 1-based Lua argument number; `fmt` is #1, so values start at #2.
    let mut arg = 1usize;

    while fpos < fmt.len() {
        let (opt, size, ntoalign) = get_details(ctx, &mut h, out.len(), fmt, &mut fpos, "pack")?;
        out.resize(out.len() + ntoalign, 0);
        arg += 1;
        match opt {
            KOption::Int { signed } => {
                let n = util::check_integer(ctx, stack.get(arg - 1), "pack", arg)?;
                if signed {
                    if size < SZINT {
                        let lim = 1i64 << (size * 8 - 1);
                        if !(-lim..lim).contains(&n) {
                            return Err(bad_arg(ctx, "pack", arg, "integer overflow"));
                        }
                    }
                    pack_int_bytes(&mut out, n as u64, h.little, size, n < 0);
                } else {
                    if size < SZINT && (n as u64) >= (1u64 << (size * 8)) {
                        return Err(bad_arg(ctx, "pack", arg, "unsigned overflow"));
                    }
                    pack_int_bytes(&mut out, n as u64, h.little, size, false);
                }
            }
            KOption::Float => {
                let x = util::check_number(ctx, stack.get(arg - 1), "pack", arg)?;
                if size == 4 {
                    let b = x as f32;
                    out.extend_from_slice(&if h.little {
                        b.to_le_bytes()
                    } else {
                        b.to_be_bytes()
                    });
                } else {
                    out.extend_from_slice(&if h.little {
                        x.to_le_bytes()
                    } else {
                        x.to_be_bytes()
                    });
                }
            }
            KOption::Char => {
                let s = check_str(ctx, stack.get(arg - 1), "pack", arg)?;
                let b = s.as_bytes();
                if b.len() > size {
                    return Err(bad_arg(ctx, "pack", arg, "string longer than given size"));
                }
                out.extend_from_slice(b);
                // `c`'s size is unbounded (it skips the [1,16] limit), so the
                // zero padding can demand an arbitrarily large allocation. Reserve
                // fallibly so an impossible size surfaces as a catchable Lua error
                // (reference raises "not enough memory") rather than aborting.
                let pad = size - b.len();
                out.try_reserve(pad)
                    .map_err(|_| Error::from_str(ctx, "not enough memory"))?;
                out.resize(out.len() + pad, 0);
            }
            KOption::Str => {
                let s = check_str(ctx, stack.get(arg - 1), "pack", arg)?;
                let b = s.as_bytes();
                let len = b.len();
                if size < SZINT && (len as u64) >= (1u64 << (size * 8)) {
                    return Err(bad_arg(
                        ctx,
                        "pack",
                        arg,
                        "string length does not fit in given size",
                    ));
                }
                pack_int_bytes(&mut out, len as u64, h.little, size, false);
                out.extend_from_slice(b);
            }
            KOption::Zstr => {
                let s = check_str(ctx, stack.get(arg - 1), "pack", arg)?;
                let b = s.as_bytes();
                if b.contains(&0) {
                    return Err(bad_arg(ctx, "pack", arg, "string contains zeros"));
                }
                out.extend_from_slice(b);
                out.push(0);
            }
            KOption::Padding => {
                out.push(0);
                arg -= 1;
            }
            KOption::PaddAlign | KOption::Nop => arg -= 1,
        }
    }

    stack.replace(&[Value::string(LuaString::new(ctx, &out))]);
    Ok(CallbackAction::Return)
}

pub(super) fn lua_packsize<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let ctx = nctx.ctx;
    let fmt = check_str(ctx, stack.get(0), "packsize", 1)?;
    let fmt = fmt.as_bytes();
    let mut h = Header {
        little: NATIVE_LITTLE,
        maxalign: 1,
    };
    let mut fpos = 0;
    let mut total: usize = 0;

    while fpos < fmt.len() {
        let (opt, size, ntoalign) = get_details(ctx, &mut h, total, fmt, &mut fpos, "packsize")?;
        if matches!(opt, KOption::Str | KOption::Zstr) {
            return Err(bad_arg(ctx, "packsize", 1, "variable-length format"));
        }
        // Cap the running size at a Lua integer, matching reference's MAXSIZE.
        let step = ntoalign.checked_add(size);
        match step.and_then(|s| total.checked_add(s)) {
            Some(t) if t <= i64::MAX as usize => total = t,
            _ => return Err(bad_arg(ctx, "packsize", 1, "format result too large")),
        }
    }

    stack.replace(&[Value::integer(total as i64)]);
    Ok(CallbackAction::Return)
}

pub(super) fn lua_unpack<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let ctx = nctx.ctx;
    let fmt = check_str(ctx, stack.get(0), "unpack", 1)?;
    let fmt = fmt.as_bytes();
    let data = check_str(ctx, stack.get(1), "unpack", 2)?;
    let data = data.as_bytes();
    let ld = data.len();

    let init = match stack.get(2) {
        v if v.is_nil() => 1,
        v => util::check_integer(ctx, v, "unpack", 3)?,
    };
    // 1-based start; out-of-range high errors, low (<=0) clamps to the start.
    let p = posrelat(init, ld);
    if p > ld as i64 + 1 {
        return Err(bad_arg(ctx, "unpack", 3, "initial position out of string"));
    }
    let mut pos = (p.max(1) - 1) as usize;

    let mut h = Header {
        little: NATIVE_LITTLE,
        maxalign: 1,
    };
    let mut fpos = 0;
    let mut out: Vec<Value> = Vec::new();

    while fpos < fmt.len() {
        let (opt, size, ntoalign) = get_details(ctx, &mut h, pos, fmt, &mut fpos, "unpack")?;
        // Alignment padding plus the fixed part must be available.
        if ntoalign
            .checked_add(size)
            .is_none_or(|need| need > ld - pos)
        {
            return Err(bad_arg(ctx, "unpack", 2, "data string too short"));
        }
        pos += ntoalign;
        match opt {
            KOption::Int { signed } => {
                out.push(Value::integer(unpack_int(
                    ctx,
                    &data[pos..],
                    h.little,
                    size,
                    signed,
                )?));
            }
            KOption::Float => {
                let val = if size == 4 {
                    let b: [u8; 4] = data[pos..pos + 4].try_into().unwrap();
                    (if h.little {
                        f32::from_le_bytes(b)
                    } else {
                        f32::from_be_bytes(b)
                    }) as f64
                } else {
                    let b: [u8; 8] = data[pos..pos + 8].try_into().unwrap();
                    if h.little {
                        f64::from_le_bytes(b)
                    } else {
                        f64::from_be_bytes(b)
                    }
                };
                out.push(Value::float(val));
            }
            KOption::Char => {
                out.push(Value::string(LuaString::new(ctx, &data[pos..pos + size])));
            }
            KOption::Str => {
                let len = unpack_int(ctx, &data[pos..], h.little, size, false)? as u64 as usize;
                if len > ld - pos - size {
                    return Err(bad_arg(ctx, "unpack", 2, "data string too short"));
                }
                let body = &data[pos + size..pos + size + len];
                out.push(Value::string(LuaString::new(ctx, body)));
                pos += len;
            }
            KOption::Zstr => {
                let Some(len) = data[pos..].iter().position(|&b| b == 0) else {
                    return Err(bad_arg(
                        ctx,
                        "unpack",
                        2,
                        "unfinished string for format 'z'",
                    ));
                };
                out.push(Value::string(LuaString::new(ctx, &data[pos..pos + len])));
                pos += len + 1;
            }
            KOption::Padding | KOption::PaddAlign | KOption::Nop => {}
        }
        pos += size;
    }

    out.push(Value::integer(pos as i64 + 1));
    stack.replace(&out);
    Ok(CallbackAction::Return)
}
