use std::num::{ParseFloatError, ParseIntError};

use hexponent::FloatLiteral;
use logos::Logos;

pub fn parse_int(s: &str) -> Result<i32, ParseIntError> {
    s.parse()
}

pub fn parse_hex_int(s: &str) -> Result<i32, ParseIntError> {
    i32::from_str_radix(s, 16)
}

pub fn parse_float(s: &str) -> Result<f32, ParseFloatError> {
    s.parse()
}

pub fn parse_hex_float(s: &str) -> Result<f32, hexponent::ParseError> {
    Ok(s.parse::<FloatLiteral>()?.convert::<f32>().inner())
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
