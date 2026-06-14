use std::num::{ParseFloatError, ParseIntError};

use logos::Logos;

pub fn parse_int(s: &str) -> Result<i64, ParseIntError> {
    s.parse()
}

pub fn parse_hex_int(s: &str) -> Result<i64, ParseIntError> {
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    i64::from_str_radix(s, 16)
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

pub fn parse_string(s: &str) -> Vec<u8> {
    parse_string_fragment(&s[1..s.len() - 1])
}

pub fn parse_long_string(s: &str) -> Vec<u8> {
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

fn parse_string_fragment(s: &str) -> Vec<u8> {
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
            Ok(StringToken::Decimal) => parse_decimal_escape(&mut bytes, &s[span]),
            Ok(StringToken::Unicode) => parse_unicode_escape(&mut bytes, &s[span]),
            Err(()) => bytes.extend_from_slice(&b[span]),
        }
    }

    bytes
}

fn parse_hex_escape(dst: &mut Vec<u8>, s: &str) {
    let char = u8::from_str_radix(&s[2..], 16).unwrap();
    dst.push(char);
}

fn parse_decimal_escape(dst: &mut Vec<u8>, s: &str) {
    let char = s[1..].parse::<u8>().unwrap();
    dst.push(char);
}

fn parse_unicode_escape(dst: &mut Vec<u8>, s: &str) {
    let codepoint = u32::from_str_radix(&s[2..s.len() - 1], 16).unwrap();
    let ch = char::from_u32(codepoint).unwrap();
    let buf = &mut [0; 4];
    let subs = ch.encode_utf8(buf).as_bytes();
    dst.extend_from_slice(subs);
}

#[cfg(test)]
mod tests {
    use super::parse_string;

    // `parse_string` expects the surrounding quotes; the raw strings below are
    // the literal source bytes, so `r#""\195\169""#` is the Lua literal "\195\169".
    #[test]
    fn decimal_escape_multibyte() {
        assert_eq!(parse_string(r#""\195\169""#), vec![195, 169]);
    }

    #[test]
    fn decimal_escape_ascii() {
        assert_eq!(parse_string(r#""\65\66\67""#), b"ABC");
    }

    #[test]
    fn decimal_escape_embedded_nul() {
        assert_eq!(parse_string(r#""a\0b""#), vec![b'a', 0, b'b']);
    }

    #[test]
    fn decimal_escape_is_greedy_then_literal() {
        // `\065` munches three digits (= 65, 'A'), leaving '3' as a literal.
        assert_eq!(parse_string(r#""\0653""#), vec![65, b'3']);
    }

    #[test]
    fn decimal_escape_does_not_cross_escaped_backslash() {
        // `\\` is one escaped backslash; the following `65` stay literal.
        assert_eq!(parse_string(r#""\\65""#), vec![b'\\', b'6', b'5']);
    }

    #[test]
    fn decimal_escape_max_byte() {
        assert_eq!(parse_string(r#""\255""#), vec![255]);
    }
}
