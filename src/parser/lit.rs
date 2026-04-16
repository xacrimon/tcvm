use std::num::{ParseFloatError, ParseIntError};

use logos::Logos;

pub fn parse_int(s: &str) -> Result<i64, ParseIntError> {
    s.parse()
}

pub fn parse_hex_int(s: &str) -> Result<i64, ParseIntError> {
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
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
    parse_string_fragment(&s[1..s.len()])
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

    #[regex(r"\\u\{[0-9a-fA-F]+\}")]
    Unicode,

    #[token(".+", priority = 1000)]
    #[error]
    Other,
}

fn parse_string_fragment(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut bytes = Vec::new();

    for (token, span) in StringToken::lexer(s).spanned() {
        match token {
            StringToken::Bell => bytes.push(0x07),
            StringToken::Backspace => bytes.push(0x08),
            StringToken::FormFeed => bytes.push(0x0C),
            StringToken::Newline => bytes.push(0x0A),
            StringToken::CarriageReturn => bytes.push(0x0D),
            StringToken::Tab => bytes.push(0x09),
            StringToken::VerticalTab => bytes.push(0x0B),
            StringToken::Backslash => bytes.push(0x5C),
            StringToken::DoubleQuote => bytes.push(0x22),
            StringToken::Quote => bytes.push(0x27),
            StringToken::LeftBracket => bytes.push(0x5B),
            StringToken::RightBracket => bytes.push(0x5D),
            StringToken::Hex => parse_hex_escape(&mut bytes, &s[span]),
            StringToken::Unicode => parse_unicode_escape(&mut bytes, &s[span]),
            StringToken::Other => bytes.extend_from_slice(&b[span]),
        }
    }

    bytes
}

fn parse_hex_escape(dst: &mut Vec<u8>, s: &str) {
    let char = u8::from_str_radix(&s[2..], 16).unwrap();
    dst.push(char);
}

fn parse_unicode_escape(dst: &mut Vec<u8>, s: &str) {
    let codepoint = u32::from_str_radix(&s[2..s.len() - 1], 16).unwrap();
    let ch = char::from_u32(codepoint).unwrap();
    let buf = &mut [0; 4];
    let subs = ch.encode_utf8(buf).as_bytes();
    dst.extend_from_slice(subs);
}
