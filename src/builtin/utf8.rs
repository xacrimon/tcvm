use crate::Context;
use crate::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Table, Value};
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
        let handler = Function::new_native(ctx.mutation(), handler, Box::new([]));
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

fn check_str<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<LuaString<'gc>, Error<'gc>> {
    v.get_string().ok_or_else(|| {
        Error::from_str(
            ctx,
            &format!(
                "bad argument #{n} to '{fname}' (string expected, got {})",
                v.type_name()
            ),
        )
    })
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

fn iscont(bytes: &[u8], idx1: i64) -> bool {
    idx1 >= 1 && ((idx1 - 1) as usize) < bytes.len() && bytes[(idx1 - 1) as usize] & 0xC0 == 0x80
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/// `utf8.char(...)` — build a string from the given code points.
fn lua_char<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let n = stack.len();
    let mut out = Vec::new();
    for i in 0..n {
        let c = crate::builtin::util::check_integer(nctx.ctx, stack.get(i), "char", i + 1)?;
        if !(0..=MAX_CODEPOINT).contains(&c) {
            return Err(Error::from_str(
                nctx.ctx,
                &format!("bad argument #{} to 'char' (value out of range)", i + 1),
            ));
        }
        encode(c as u32, &mut out);
    }
    stack.replace(&[Value::string(LuaString::new(nctx.ctx, &out))]);
    Ok(CallbackAction::Return)
}

/// `utf8.codepoint(s [, i [, j]])` — code points of the characters in byte
/// range `i..j` (`i` defaults to 1, `j` to `i`).
fn lua_codepoint<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = check_str(nctx.ctx, stack.get(0), "codepoint", 1)?;
    let bytes = s.as_bytes();
    let len = bytes.len();
    let i_arg = stack.get(1);
    let i = if i_arg.is_nil() {
        1
    } else {
        crate::builtin::util::check_integer(nctx.ctx, i_arg, "codepoint", 2)?
    };
    let j_arg = stack.get(2);
    let j = if j_arg.is_nil() {
        i
    } else {
        crate::builtin::util::check_integer(nctx.ctx, j_arg, "codepoint", 3)?
    };
    let posi = posrelat(i, len);
    let posj = posrelat(j, len);
    if posi < 1 {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #2 to 'codepoint' (out of bounds)",
        ));
    }
    if posj > len as i64 {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #3 to 'codepoint' (out of bounds)",
        ));
    }
    let strict = stack.get(3).is_falsy(); // optional `lax` flag (arg #4): lax ⇒ not strict
    let mut out = Vec::new();
    let mut pos = (posi - 1) as usize;
    let end = posj as usize;
    while pos < end {
        match decode(bytes, pos, strict) {
            Some((code, next)) => {
                out.push(Value::integer(code as i64));
                pos = next;
            }
            None => return Err(Error::from_str(nctx.ctx, "invalid UTF-8 code")),
        }
    }
    stack.replace(&out);
    Ok(CallbackAction::Return)
}

/// `utf8.len(s [, i [, j]])` — number of characters in byte range `i..j`
/// (`i` defaults to 1, `j` to -1). On a malformed sequence, returns
/// `(nil, position)`.
fn lua_len<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = check_str(nctx.ctx, stack.get(0), "len", 1)?;
    let bytes = s.as_bytes();
    let len = bytes.len();
    let i_arg = stack.get(1);
    let i = if i_arg.is_nil() {
        1
    } else {
        crate::builtin::util::check_integer(nctx.ctx, i_arg, "len", 2)?
    };
    let j_arg = stack.get(2);
    let j = if j_arg.is_nil() {
        -1
    } else {
        crate::builtin::util::check_integer(nctx.ctx, j_arg, "len", 3)?
    };
    let mut posi = posrelat(i, len);
    let posj = posrelat(j, len);
    if posi < 1 || posi > len as i64 + 1 {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #2 to 'len' (initial position out of bounds)",
        ));
    }
    if posj > len as i64 {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #3 to 'len' (final position out of bounds)",
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
                stack.replace(&[Value::nil(), Value::integer(posi)]);
                return Ok(CallbackAction::Return);
            }
        }
    }
    stack.replace(&[Value::integer(count)]);
    Ok(CallbackAction::Return)
}

/// `utf8.offset(s, n [, i])` — the byte position where the `n`-th character
/// (counting from byte `i`) begins; `n == 0` finds the start of the character
/// containing byte `i`. Returns `nil` when the position falls outside the
/// string.
fn lua_offset<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = check_str(nctx.ctx, stack.get(0), "offset", 1)?;
    let bytes = s.as_bytes();
    let len = bytes.len();
    let n = crate::builtin::util::check_integer(nctx.ctx, stack.get(1), "offset", 2)?;
    let default_i = if n >= 0 { 1 } else { len as i64 + 1 };
    let i_arg = stack.get(2);
    let i = if i_arg.is_nil() {
        default_i
    } else {
        crate::builtin::util::check_integer(nctx.ctx, i_arg, "offset", 3)?
    };
    let mut posi = posrelat(i, len);
    if posi < 1 || posi > len as i64 + 1 {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #3 to 'offset' (position out of bounds)",
        ));
    }

    let found = if n == 0 {
        while posi > 1 && iscont(bytes, posi) {
            posi -= 1;
        }
        Some(posi)
    } else if posi <= len as i64 && iscont(bytes, posi) {
        return Err(Error::from_str(
            nctx.ctx,
            "initial position is a continuation byte",
        ));
    } else if n > 0 {
        let mut n = n - 1;
        while n > 0 && posi <= len as i64 {
            posi += 1;
            while posi <= len as i64 && iscont(bytes, posi) {
                posi += 1;
            }
            n -= 1;
        }
        if n > 0 { None } else { Some(posi) }
    } else {
        let mut n = n;
        while n < 0 && posi > 1 {
            posi -= 1;
            while posi > 1 && iscont(bytes, posi) {
                posi -= 1;
            }
            n += 1;
        }
        if n < 0 { None } else { Some(posi) }
    };
    match found {
        // Lua 5.5 returns the start position AND the byte index of that
        // character's last byte (the start itself when past the end).
        Some(p) => {
            let end = if p > len as i64 {
                p
            } else {
                let mut e = p;
                while e < len as i64 && iscont(bytes, e + 1) {
                    e += 1;
                }
                e
            };
            stack.replace(&[Value::integer(p), Value::integer(end)]);
        }
        None => stack.replace(&[Value::nil()]),
    }
    Ok(CallbackAction::Return)
}

/// `utf8.codes(s)` — iterator triple `(iterator, s, 0)` yielding
/// `(byte_position, code_point)` for each character.
fn lua_codes<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = check_str(nctx.ctx, stack.get(0), "codes", 1)?;
    let iter = Function::new_native(nctx.ctx.mutation(), codes_aux, Box::new([]));
    stack.replace(&[Value::function(iter), Value::string(s), Value::integer(0)]);
    Ok(CallbackAction::Return)
}

/// Stateless iterator body for `utf8.codes`. `i` is the byte position (1-based)
/// of the previously yielded character, or 0 to start.
fn codes_aux<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let s = check_str(nctx.ctx, stack.get(0), "codes", 1)?;
    let bytes = s.as_bytes();
    let len = bytes.len();
    let i = stack.get(1).get_integer().unwrap_or(0);
    // `utf8.codes` decodes strictly by default (matching PUC-Lua).
    // Advance past the previously decoded character (if any).
    let pos = if i == 0 {
        0
    } else {
        match decode(bytes, (i - 1) as usize, true) {
            Some((_, next)) => next,
            None => return Err(Error::from_str(nctx.ctx, "invalid UTF-8 code")),
        }
    };
    if pos >= len {
        stack.replace(&[]);
        return Ok(CallbackAction::Return);
    }
    match decode(bytes, pos, true) {
        Some((code, _)) => {
            stack.replace(&[Value::integer(pos as i64 + 1), Value::integer(code as i64)]);
            Ok(CallbackAction::Return)
        }
        None => Err(Error::from_str(nctx.ctx, "invalid UTF-8 code")),
    }
}
