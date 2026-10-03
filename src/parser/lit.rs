use std::num::{ParseFloatError, ParseIntError};
use std::ops::Range;

/// Length of the line break at the start of `b`, or 0. `\n`, `\r`, `\r\n`
/// and `\n\r` are each one break, as in `inclinenumber` (llex.c).
pub(crate) fn line_break_len(b: &[u8]) -> usize {
    match b {
        [b'\n', b'\r', ..] | [b'\r', b'\n', ..] => 2,
        [b'\n' | b'\r', ..] => 1,
        _ => 0,
    }
}

pub fn parse_int(s: &str) -> Result<i64, ParseIntError> {
    s.parse()
}

// Hex integer literals wrap mod 2^64 per Lua 5.5 lexical conventions, so fold
// digit-by-digit ignoring overflow (matching `luaO_hexavalue`) rather than
// `from_str_radix`, which would reject >16 digits instead of keeping the low 64.
pub fn parse_hex_int(s: &str) -> i64 {
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    let mut acc: u64 = 0;
    for ch in s.chars() {
        // The HexInt lexer regex guarantees every digit is valid hex.
        let digit = ch.to_digit(16).expect("hex literal has only hex digits");
        acc = acc.wrapping_mul(16).wrapping_add(digit as u64);
    }
    acc as i64
}

pub fn parse_float(s: &str) -> Result<f64, ParseFloatError> {
    s.parse()
}

pub fn parse_hex_float(s: &str) -> Option<f64> {
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;

    let (mantissa, exp_str) = match s.find(['p', 'P']) {
        Some(i) => (&s[..i], Some(&s[i + 1..])),
        None => (s, None),
    };

    let (int_str, frac_str) = match mantissa.find('.') {
        Some(i) => (&mantissa[..i], Some(&mantissa[i + 1..])),
        None => (mantissa, None),
    };

    let mut value = if int_str.is_empty() {
        0.0
    } else {
        u64::from_str_radix(int_str, 16).ok()? as f64
    };

    if let Some(frac) = frac_str {
        let mut place = 1.0 / 16.0;
        for ch in frac.chars() {
            value += ch.to_digit(16)? as f64 * place;
            place /= 16.0;
        }
    }

    if let Some(exp) = exp_str {
        let exp: i32 = exp.parse().ok()?;
        value *= (2.0f64).powi(exp);
    }

    Some(value)
}

/// A malformed escape in a quoted string literal: Lua's `message`, for the
/// bytes `range` of the literal (the escape up to the offending byte).
#[derive(Debug, PartialEq, Eq)]
pub struct EscapeError {
    pub range: Range<usize>,
    pub message: &'static str,
}

/// Decode a quoted string literal (including the surrounding quotes), as
/// llex.c's `read_string` does.
pub fn parse_string(s: &str) -> Result<Vec<u8>, EscapeError> {
    let b = s.as_bytes();
    // The closing quote. The lexer never ends a literal on an escaped one,
    // so every escape is complete before it.
    let end = b.len() - 1;
    let mut out = Vec::with_capacity(end);
    let mut i = 1;
    while i < end {
        if b[i] != b'\\' {
            out.push(b[i]);
            i += 1;
            continue;
        }
        let start = i;
        let fail = |at: usize, message| EscapeError {
            range: start..s.ceil_char_boundary(at + 1),
            message,
        };
        let hex = |at: usize| (b[at] as char).to_digit(16);
        i += 1;
        // Escapes that stand for one byte evaluate to it; the rest `continue`.
        let c = match b[i] {
            b'a' => 0x07,
            b'b' => 0x08,
            b'f' => 0x0C,
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'v' => 0x0B,
            c @ (b'\\' | b'"' | b'\'') => c,
            b'\n' | b'\r' => {
                out.push(b'\n');
                i += line_break_len(&b[i..]);
                continue;
            }
            b'z' => {
                i += 1;
                while matches!(b[i], b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r') {
                    i += 1;
                }
                continue;
            }
            b'x' => {
                let mut c = 0;
                for _ in 0..2 {
                    i += 1;
                    c = c * 16 + hex(i).ok_or_else(|| fail(i, "hexadecimal digit expected"))?;
                }
                c as u8
            }
            b'u' => {
                i += 1;
                if b[i] != b'{' {
                    return Err(fail(i, "missing '{'"));
                }
                i += 1;
                let mut cp = hex(i).ok_or_else(|| fail(i, "hexadecimal digit expected"))?;
                i += 1;
                while let Some(d) = hex(i) {
                    if cp > 0x7FFF_FFFF >> 4 {
                        return Err(fail(i, "UTF-8 value too large"));
                    }
                    cp = cp << 4 | d;
                    i += 1;
                }
                if b[i] != b'}' {
                    return Err(fail(i, "missing '}'"));
                }
                utf8_encode(&mut out, cp);
                i += 1;
                continue;
            }
            b'0'..=b'9' => {
                let digits = b[i..]
                    .iter()
                    .take(3)
                    .take_while(|c| c.is_ascii_digit())
                    .count();
                let n: u32 = s[i..i + digits].parse().expect("ASCII digits");
                i += digits;
                if n > 255 {
                    return Err(fail(i, "decimal escape too large"));
                }
                out.push(n as u8);
                continue;
            }
            _ => return Err(fail(i, "invalid escape sequence")),
        };
        out.push(c);
        i += 1;
    }
    Ok(out)
}

/// Decode a long string literal (including its brackets). There are no
/// escapes; a line break right after the opening bracket is dropped and every
/// other one becomes `\n`, as in llex.c's `read_long_string`.
pub fn parse_long_string(s: &str) -> Vec<u8> {
    let level = s[1..].bytes().take_while(|&c| c == b'=').count();
    let mut b = &s.as_bytes()[level + 2..s.len() - level - 2];
    b = &b[line_break_len(b)..];
    let mut out = Vec::with_capacity(b.len());
    while let Some(i) = b.iter().position(|&c| matches!(c, b'\n' | b'\r')) {
        out.extend_from_slice(&b[..i]);
        out.push(b'\n');
        b = &b[i + line_break_len(&b[i..])..];
    }
    out.extend_from_slice(b);
    out
}

/// Lua's `luaO_utf8esc`: encode a code point (0..=0x7FFFFFFF) as 1–6 bytes via
/// the original, pre-Unicode UTF-8 scheme. Unlike `char::encode_utf8`, this
/// emits surrogates (U+D800..U+DFFF) and values beyond U+10FFFF verbatim, which
/// is what `\u{...}` does in lua 5.5.
fn utf8_encode(dst: &mut Vec<u8>, cp: u32) {
    if cp < 0x80 {
        dst.push(cp as u8);
        return;
    }
    const SZ: usize = 8;
    let mut buff = [0u8; SZ];
    let mut x = cp;
    let mut n = 1usize;
    let mut mfb: u32 = 0x3f; // max value representable in the lead byte
    loop {
        buff[SZ - n] = (0x80 | (x & 0x3f)) as u8; // a continuation byte
        n += 1;
        x >>= 6;
        mfb >>= 1; // one fewer bit available in the lead byte each round
        if x <= mfb {
            break;
        }
    }
    buff[SZ - n] = ((!mfb << 1) | x) as u8; // lead byte: count marker + residue
    dst.extend_from_slice(&buff[SZ - n..]);
}

#[cfg(test)]
mod tests {
    use super::{parse_hex_int, parse_int, parse_string};

    fn parse(s: &str) -> Vec<u8> {
        super::parse_string(s).expect("well-formed literal")
    }

    // Hex int literals wrap mod 2^64; >16 digits keep the low 64 bits.
    // Values cross-checked against lua 5.5.0.
    #[test]
    fn hex_int_wraps_mod_2_64() {
        assert_eq!(parse_hex_int("0xff"), 255);
        assert_eq!(parse_hex_int("0x7fffffffffffffff"), i64::MAX);
        assert_eq!(parse_hex_int("0x8000000000000000"), i64::MIN);
        assert_eq!(parse_hex_int("0xffffffffffffffff"), -1);
        assert_eq!(parse_hex_int("0x10000000000000000"), 0); // 2^64 -> 0
        assert_eq!(parse_hex_int("0xffffffffffffffffff"), -1); // >16 digits
    }

    // A decimal int at the i64 boundary stays integer; one past it overflows
    // (the caller then reparses it as a float).
    #[test]
    fn decimal_int_boundary_and_overflow() {
        assert_eq!(parse_int("9223372036854775807"), Ok(i64::MAX));
        assert!(parse_int("9223372036854775808").is_err());
        assert_eq!(
            super::parse_float("9223372036854775808"),
            Ok(9223372036854775808.0)
        );
    }

    // `parse_string` expects the surrounding quotes; the raw strings below are
    // the literal source bytes, so `r#""\195\169""#` is the Lua literal "\195\169".
    #[test]
    fn decimal_escape_multibyte() {
        assert_eq!(parse(r#""\195\169""#), vec![195, 169]);
    }

    #[test]
    fn decimal_escape_ascii() {
        assert_eq!(parse(r#""\65\66\67""#), b"ABC");
    }

    #[test]
    fn decimal_escape_embedded_nul() {
        assert_eq!(parse(r#""a\0b""#), vec![b'a', 0, b'b']);
    }

    #[test]
    fn decimal_escape_is_greedy_then_literal() {
        // `\065` munches three digits (= 65, 'A'), leaving '3' as a literal.
        assert_eq!(parse(r#""\0653""#), vec![65, b'3']);
    }

    #[test]
    fn decimal_escape_does_not_cross_escaped_backslash() {
        // `\\` is one escaped backslash; the following `65` stay literal.
        assert_eq!(parse(r#""\\65""#), vec![b'\\', b'6', b'5']);
    }

    #[test]
    fn decimal_escape_max_byte() {
        assert_eq!(parse(r#""\255""#), vec![255]);
    }

    #[test]
    fn decimal_escape_too_large_is_an_error() {
        assert_eq!(
            parse_string(r#""a\256""#).unwrap_err().message,
            "decimal escape too large"
        );
    }

    // Unicode escapes use Lua's extended UTF-8 (`luaO_utf8esc`): up to 6 bytes,
    // code points to 0x7FFFFFFF, surrogates and beyond-U+10FFFF included.
    // Expected bytes cross-checked against lua 5.5.0.
    #[test]
    fn unicode_escape_ascii() {
        assert_eq!(parse(r#""\u{48}""#), vec![0x48]); // 'H'
        assert_eq!(parse(r#""\u{0}""#), vec![0]);
        assert_eq!(parse(r#""\u{00048}""#), vec![0x48]); // leading zeros
    }

    #[test]
    fn unicode_escape_multibyte() {
        assert_eq!(parse(r#""\u{E9}""#), vec![0xC3, 0xA9]); // é
        assert_eq!(parse(r#""\u{20AC}""#), vec![0xE2, 0x82, 0xAC]); // €
        assert_eq!(parse(r#""\u{1F600}""#), vec![0xF0, 0x9F, 0x98, 0x80]); // emoji
    }

    #[test]
    fn unicode_escape_surrogates_and_beyond_unicode() {
        assert_eq!(parse(r#""\u{D800}""#), vec![0xED, 0xA0, 0x80]); // surrogate
        assert_eq!(parse(r#""\u{110000}""#), vec![0xF4, 0x90, 0x80, 0x80]);
        // 0x7FFFFFFF — the maximum Lua accepts — encodes to 6 bytes.
        assert_eq!(
            parse(r#""\u{7FFFFFFF}""#),
            vec![0xFD, 0xBF, 0xBF, 0xBF, 0xBF, 0xBF]
        );
    }

    #[test]
    fn unicode_escape_too_large_is_an_error() {
        for s in [r#""\u{80000000}""#, r#""\u{FFFFFFFFF}""#] {
            assert_eq!(
                parse_string(s).unwrap_err().message,
                "UTF-8 value too large"
            );
        }
    }
}
