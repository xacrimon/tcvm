use crate::Context;
use crate::builtin::util;
use crate::env::{Error, Function, LuaString, NativeClosure, NativeFn, Stack, Table, Value};
use crate::vm::sequence::CallbackAction;

/// Original-UTF-8 (Lua flavor) max code point: 6-byte sequences up to
/// `0x7FFFFFFF`. Stricter than Unicode's `0x10FFFF`, matching Lua's defaults
/// for `utf8.char`.
const MAX_CODEPOINT: i64 = 0x7FFF_FFFF;

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("char", lua_char),
        ("codes", lua_codes),
        ("codepoint", lua_codepoint),
        ("len", lua_len),
        ("offset", lua_offset),
    ];

    let lib = Table::new(ctx);
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, &[]);
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(handler));
    }

    // Matches any single UTF-8 byte sequence (lead byte + continuation bytes).
    let pat = LuaString::new(ctx, b"[\x00-\x7F\xC2-\xFD][\x80-\xBF]*");
    lib.raw_set(
        ctx,
        Value::string(LuaString::new(ctx, b"charpattern")),
        Value::string(pat),
    );

    let lib_name = Value::string(LuaString::new(ctx, b"utf8"));
    ctx.globals().raw_set(ctx, lib_name, Value::table(lib));
}

// ---------------------------------------------------------------------------
// Codec helpers
// ---------------------------------------------------------------------------

/// Decode one code point starting at 0-based `pos`, returning `(code,
/// next_pos)` or `None` for a malformed sequence. Mirrors PUC-Lua's
/// `utf8_decode`: it always rejects ill-formed structure and overlong
/// encodings; with `strict` it additionally rejects surrogates and code
/// points above `0x10FFFF` (the default for `codepoint`/`len`/`codes`).
fn decode(bytes: &[u8], pos: usize, strict: bool) -> Option<(u32, usize)> {
    // Minimum value representable by an N-byte sequence, indexed by the number
    // of continuation bytes (`lead - 1`); a result below it is overlong.
    const LIMITS: [u32; 6] = [0, 0x80, 0x800, 0x10000, 0x20_0000, 0x400_0000];
    let c = *bytes.get(pos)?;
    let lead = c.leading_ones();
    if lead == 0 {
        return Some((c as u32, pos + 1));
    }
    if lead == 1 || lead > 6 {
        return None; // a continuation byte or an over-long lead byte
    }
    let mut code = (c & (0xffu8 >> (lead + 1))) as u32;
    for k in 1..lead as usize {
        let cc = *bytes.get(pos + k)?;
        if cc & 0xC0 != 0x80 {
            return None;
        }
        code = (code << 6) | (cc & 0x3f) as u32;
    }
    if code < LIMITS[lead as usize - 1] {
        return None; // overlong encoding
    }
    if strict && (code > 0x10_FFFF || (0xD800..=0xDFFF).contains(&code)) {
        return None; // surrogate or above the Unicode range
    }
    Some((code, pos + lead as usize))
}

/// Encode one code point as original-UTF-8 (1–6 bytes), per Lua's
/// `luaO_utf8esc`.
fn encode(cp: u32, out: &mut Vec<u8>) {
    if cp < 0x80 {
        out.push(cp as u8);
        return;
    }
    let mut tail = Vec::new();
    let mut x = cp;
    let mut mfb: u32 = 0x3f;
    loop {
        tail.push(0x80 | (x & 0x3f) as u8);
        x >>= 6;
        mfb >>= 1;
        if x <= mfb {
            break;
        }
    }
    out.push((!mfb << 1) as u8 | x as u8);
    tail.reverse();
    out.extend_from_slice(&tail);
}

/// Lua's `posrelat` for byte positions.
fn posrelat(pos: i64, len: usize) -> i64 {
    if pos >= 0 {
        pos
    } else if pos.unsigned_abs() > len as u64 {
        0
    } else {
        len as i64 + pos + 1
    }
}

/// `iscontp` at 0-based `idx`; the end of the string is not a continuation byte.
fn iscont(bytes: &[u8], idx: usize) -> bool {
    bytes.get(idx).is_some_and(|b| b & 0xC0 == 0x80)
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/// `utf8.char(...)` — build a string from the given code points.
fn lua_char<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let n = stack.len();
    let mut out = Vec::new();
    for i in 0..n {
        let c = crate::builtin::util::check_integer(ctx, stack.get(i), "char", i + 1)?;
        if !(0..=MAX_CODEPOINT).contains(&c) {
            return Err(util::arg_error(ctx, "char", i + 1, "value out of range"));
        }
        encode(c as u32, &mut out);
    }
    stack.ret1(Value::string(LuaString::new(ctx, &out)));
    Ok(CallbackAction::Return)
}

/// `utf8.codepoint(s [, i [, j]])` — code points of the characters in byte
/// range `i..j` (`i` defaults to 1, `j` to `i`).
fn lua_codepoint<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "codepoint", 1)?;
    let bytes = s.as_bytes();
    let len = bytes.len();
    let i_arg = stack.get(1);
    let i = if i_arg.is_nil() {
        1
    } else {
        crate::builtin::util::check_integer(ctx, i_arg, "codepoint", 2)?
    };
    let j_arg = stack.get(2);
    let j = if j_arg.is_nil() {
        i
    } else {
        crate::builtin::util::check_integer(ctx, j_arg, "codepoint", 3)?
    };
    let posi = posrelat(i, len);
    let posj = posrelat(j, len);
    if posi < 1 {
        return Err(util::arg_error(ctx, "codepoint", 2, "out of bounds"));
    }
    if posj > len as i64 {
        return Err(util::arg_error(ctx, "codepoint", 3, "out of bounds"));
    }
    let strict = stack.get(3).is_falsy(); // optional `lax` flag (arg #4): lax ⇒ not strict
    let mut out = Vec::new();
    let mut pos = (posi - 1) as usize;
    let end = posj as usize;
    while pos < end {
        match decode(bytes, pos, strict) {
            Some((code, next)) => {
                out.push(Value::integer(ctx.mutation(), code as i64));
                pos = next;
            }
            None => return Err(Error::from_str(ctx, "invalid UTF-8 code")),
        }
    }
    stack.replace(&out);
    Ok(CallbackAction::Return)
}

/// `utf8.len(s [, i [, j]])` — number of characters in byte range `i..j`
/// (`i` defaults to 1, `j` to -1). On a malformed sequence, returns
/// `(nil, position)`.
fn lua_len<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "len", 1)?;
    let bytes = s.as_bytes();
    let len = bytes.len();
    let i_arg = stack.get(1);
    let i = if i_arg.is_nil() {
        1
    } else {
        crate::builtin::util::check_integer(ctx, i_arg, "len", 2)?
    };
    let j_arg = stack.get(2);
    let j = if j_arg.is_nil() {
        -1
    } else {
        crate::builtin::util::check_integer(ctx, j_arg, "len", 3)?
    };
    let mut posi = posrelat(i, len);
    let posj = posrelat(j, len);
    if posi < 1 || posi > len as i64 + 1 {
        return Err(util::arg_error(
            ctx,
            "len",
            2,
            "initial position out of bounds",
        ));
    }
    if posj > len as i64 {
        return Err(util::arg_error(
            ctx,
            "len",
            3,
            "final position out of bounds",
        ));
    }
    let strict = stack.get(3).is_falsy(); // optional `lax` flag (arg #4)
    let mut count = 0i64;
    while posi <= posj {
        match decode(bytes, (posi - 1) as usize, strict) {
            Some((_, next)) => {
                count += 1;
                posi = next as i64 + 1;
            }
            None => {
                stack.replace(&[Value::nil(), Value::integer(ctx.mutation(), posi)]);
                return Ok(CallbackAction::Return);
            }
        }
    }
    stack.ret1(Value::integer(ctx.mutation(), count));
    Ok(CallbackAction::Return)
}

/// `utf8.offset(s, n [, i])` — the byte position where the `n`-th character
/// (counting from byte `i`) begins; `n == 0` finds the start of the character
/// containing byte `i`. Returns `nil` when the position falls outside the
/// string.
fn lua_offset<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "offset", 1)?;
    let bytes = s.as_bytes();
    let len = bytes.len();
    let n = crate::builtin::util::check_integer(ctx, stack.get(1), "offset", 2)?;
    let default_i = if n >= 0 { 1 } else { len as i64 + 1 };
    let i_arg = stack.get(2);
    let i = if i_arg.is_nil() {
        default_i
    } else {
        crate::builtin::util::check_integer(ctx, i_arg, "offset", 3)?
    };
    let posi = posrelat(i, len);
    if posi < 1 || posi > len as i64 + 1 {
        return Err(util::arg_error(ctx, "offset", 3, "position out of bounds"));
    }
    // 0-based from here on, as in `byteoffset`.
    let mut posi = posi as usize - 1;
    let cont_err = || Error::from_str(ctx, "initial position is a continuation byte");
    let mut n = n;
    if n == 0 {
        while posi > 0 && iscont(bytes, posi) {
            posi -= 1;
        }
    } else {
        if iscont(bytes, posi) {
            return Err(cont_err());
        }
        if n < 0 {
            while n < 0 && posi > 0 {
                posi -= 1;
                while posi > 0 && iscont(bytes, posi) {
                    posi -= 1;
                }
                n += 1;
            }
        } else {
            n -= 1;
            while n > 0 && posi < len {
                posi += 1;
                while iscont(bytes, posi) {
                    posi += 1;
                }
                n -= 1;
            }
        }
    }
    if n != 0 {
        stack.replace(&[Value::nil()]);
        return Ok(CallbackAction::Return);
    }
    let start = posi;
    // A stray continuation byte reached by moving is caught here.
    if bytes.get(posi).is_some_and(|b| b & 0x80 != 0) {
        if iscont(bytes, posi) {
            return Err(cont_err());
        }
        while iscont(bytes, posi + 1) {
            posi += 1;
        }
    }
    stack.replace(&[
        Value::integer(ctx.mutation(), start as i64 + 1),
        Value::integer(ctx.mutation(), posi as i64 + 1),
    ]);
    Ok(CallbackAction::Return)
}

/// `utf8.codes(s [, lax])` — iterator triple `(iterator, s, 0)` yielding
/// `(byte_position, code_point)` for each character.
fn lua_codes<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let lax = !stack.get(1).is_falsy();
    let s = util::check_string(ctx, stack.get(0), "codes", 1)?;
    if iscont(s.as_bytes(), 0) {
        return Err(util::arg_error(ctx, "codes", 1, "invalid UTF-8 code"));
    }
    let aux: NativeFn = if lax { codes_lax } else { codes_strict };
    let iter = Function::new_native(ctx.mutation(), aux, &[]);
    stack.replace(&[
        Value::function(iter),
        Value::string(s),
        Value::integer(ctx.mutation(), 0),
    ]);
    Ok(CallbackAction::Return)
}

fn codes_strict<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    codes_aux(ctx, stack, true)
}

fn codes_lax<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    codes_aux(ctx, stack, false)
}

/// `iter_aux`: the control value is the previous character's 1-based position,
/// i.e. the 0-based index just past its lead byte; it ends at or past the end,
/// negative values included.
fn codes_aux<'gc>(
    ctx: Context<'gc>,
    mut stack: Stack<'gc, '_>,
    strict: bool,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = util::check_string(ctx, stack.get(0), "codes", 1)?;
    let bytes = s.as_bytes();
    let mut n = util::to_integer(stack.get(1)).unwrap_or(0) as u64 as usize;
    while n < bytes.len() && iscont(bytes, n) {
        n += 1;
    }
    if n >= bytes.len() {
        stack.replace(&[]);
        return Ok(CallbackAction::Return);
    }
    match decode(bytes, n, strict) {
        Some((code, next)) if !iscont(bytes, next) => {
            stack.replace(&[
                Value::integer(ctx.mutation(), n as i64 + 1),
                Value::integer(ctx.mutation(), code as i64),
            ]);
            Ok(CallbackAction::Return)
        }
        _ => Err(Error::from_str(ctx, "invalid UTF-8 code")),
    }
}
