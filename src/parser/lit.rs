use std::num::{ParseFloatError, ParseIntError};

use logos::Logos;

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

/// Decode a quoted string literal (including the surrounding quotes). Returns
/// `None` on a malformed escape (a too-large decimal/`\u{}` value) so the
/// caller can surface a parse error instead of the decoder panicking.
pub fn parse_string(s: &str) -> Option<Vec<u8>> {
    parse_string_fragment(&s[1..s.len() - 1])
}

pub fn parse_long_string(s: &str) -> Option<Vec<u8>> {
    let mut suffix = 1;
    let mut s = &s[1..s.len() - 1];

    while s.starts_with('=') {
        suffix += 1;
        s = &s[1..];
    }

    parse_string_fragment(&s[1..s.len() - suffix])
}

#[derive(Logos)]
enum StringToken {
    #[token("\\a")]
    Bell,

    #[token("\\b")]
    Backspace,

    #[token("\\f")]
    FormFeed,

    #[token("\\n")]
    Newline,

    #[token("\\r")]
    CarriageReturn,

    #[token("\\t")]
    Tab,

    #[token("\\v")]
    VerticalTab,

    #[token("\\\\")]
    Backslash,

    #[token("\\\"")]
    DoubleQuote,

    #[token("\\'")]
    Quote,

    #[token("\\[")]
    LeftBracket,

    #[token("\\]")]
    RightBracket,

    #[regex(r"\\x[0-9a-fA-F][0-9a-fA-F]")]
    Hex,

    // Greedy 1-3 decimal digits: `\195` takes all three, matching Lua, which
    // then rejects values > 255 rather than falling back to fewer digits.
    #[regex(r"\\[0-9][0-9]?[0-9]?")]
    Decimal,

    #[regex(r"\\u\{[0-9a-fA-F]+\}")]
    Unicode,
}

fn parse_string_fragment(s: &str) -> Option<Vec<u8>> {
    let b = s.as_bytes();
    let mut bytes = Vec::new();

    for (token, span) in StringToken::lexer(s).spanned() {
        match token {
            Ok(StringToken::Bell) => bytes.push(0x07),
            Ok(StringToken::Backspace) => bytes.push(0x08),
            Ok(StringToken::FormFeed) => bytes.push(0x0C),
            Ok(StringToken::Newline) => bytes.push(0x0A),
            Ok(StringToken::CarriageReturn) => bytes.push(0x0D),
            Ok(StringToken::Tab) => bytes.push(0x09),
            Ok(StringToken::VerticalTab) => bytes.push(0x0B),
            Ok(StringToken::Backslash) => bytes.push(0x5C),
            Ok(StringToken::DoubleQuote) => bytes.push(0x22),
            Ok(StringToken::Quote) => bytes.push(0x27),
            Ok(StringToken::LeftBracket) => bytes.push(0x5B),
            Ok(StringToken::RightBracket) => bytes.push(0x5D),
            Ok(StringToken::Hex) => parse_hex_escape(&mut bytes, &s[span]),
            Ok(StringToken::Decimal) => parse_decimal_escape(&mut bytes, &s[span])?,
            Ok(StringToken::Unicode) => parse_unicode_escape(&mut bytes, &s[span])?,
            Err(()) => bytes.extend_from_slice(&b[span]),
        }
    }

    Some(bytes)
}

fn parse_hex_escape(dst: &mut Vec<u8>, s: &str) {
    // The `\xHH` regex guarantees exactly two hex digits, so this fits a `u8`.
    let char = u8::from_str_radix(&s[2..], 16).expect("\\x escape has two hex digits");
    dst.push(char);
}

fn parse_decimal_escape(dst: &mut Vec<u8>, s: &str) -> Option<()> {
    // `\ddd` is 1–3 decimal digits; Lua rejects a value > 255 (`None` here so
    // the caller raises a parse error rather than the decoder panicking).
    let char = s[1..].parse::<u8>().ok()?;
    dst.push(char);
    Some(())
}

fn parse_unicode_escape(dst: &mut Vec<u8>, s: &str) -> Option<()> {
    // `s` is `\u{HEX}`; skip the 3-byte `\u{` prefix and the trailing `}`.
    let hex = &s[3..s.len() - 1];
    // Parse into `u64` so an over-long digit run yields `None` rather than
    // overflow-panicking. Lua accepts code points up to 0x7FFFFFFF.
    let cp = u64::from_str_radix(hex, 16).ok()?;
    if cp > 0x7FFF_FFFF {
        return None; // Lua: "UTF-8 value too large"
    }
    utf8_encode(dst, cp as u32);
    Some(())
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
    fn decimal_escape_too_large_is_none() {
        // `\256` > 255 — a parse error, not a decoder panic.
        assert_eq!(parse_string(r#""\256""#), None);
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
    fn unicode_escape_too_large_is_none() {
        assert_eq!(parse_string(r#""\u{80000000}""#), None); // > 0x7FFFFFFF
        assert_eq!(parse_string(r#""\u{FFFFFFFFF}""#), None); // overflows
    }
}
