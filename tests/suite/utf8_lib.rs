//! `utf8.codes` and `utf8.offset` edge cases (#244). Expected outputs from
//! lua 5.5.1, apart from function names (#186).

use crate::common::{err, ok};

#[test]
fn codes_iterator_ends_past_the_end() {
    assert_eq!(
        ok("local f = utf8.codes('') \
            return cat(select('#', f('', 2)), select('#', f('', -1)), \
              select('#', f('abc', 10)), select('#', f('abc', math.mininteger)), \
              f('abc', 1.0), f('abc', '1'))"),
        "0 0 0 0 2 2 98"
    );
}

#[test]
fn codes_lax() {
    assert_eq!(
        ok("local out = {} \
            for p, c in utf8.codes('\\u{4000000}\\u{7FFFFFFF}', true) do out[#out + 1] = cat(p, c) end \
            return table.concat(out, ', ')"),
        "1 67108864, 7 2147483647"
    );
    assert_eq!(
        err("for p, c in utf8.codes('\\u{4000000}') do end"),
        "c:1: invalid UTF-8 code"
    );
}

#[test]
fn codes_rejects_stray_continuation_bytes() {
    assert_eq!(
        err("utf8.codes('\\x80abc')"),
        "c:1: bad argument #1 to 'codes' (invalid UTF-8 code)"
    );
    for s in ["a\\x80", "\\xe4\\xb8", "é\\x80\\x80"] {
        assert_eq!(
            err(&format!("for p, c in utf8.codes('{s}') do end")),
            "c:1: invalid UTF-8 code",
            "{s}"
        );
    }
}

#[test]
fn offset_over_continuation_bytes() {
    let cases = [
        ("'\\x9c', -1", "initial position is a continuation byte"),
        ("'a\\x9c', -1", "1 1"),
        (
            "'\\x9c\\x9c', -1",
            "initial position is a continuation byte",
        ),
        ("'aé', -1", "2 3"),
        ("'aé', 3", "4 4"),
        ("'aé', 1, 3", "initial position is a continuation byte"),
        ("'aé', -3", "nil"),
        ("'a\\x9cb', 2", "3 3"),
        ("'\\xe4\\xb8\\xad', 0, 3", "1 3"),
        ("'abc', -1, 2", "1 1"),
        ("'', -1", "nil"),
    ];
    for (args, expected) in cases {
        let src = format!(
            "local ok, a, b = pcall(utf8.offset, {args}) \
             return ok and cat(a, b):gsub(' nil$', '') or a"
        );
        assert_eq!(ok(&src), expected, "{args}");
    }
}
