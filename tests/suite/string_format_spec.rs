//! `string.format` spec parsing, integer conversions and error order. Expected outputs from lua
//! 5.5.1.

use crate::common::{err, ok};

#[test]
fn integer_flags() {
    assert_eq!(
        ok(
            "return string.format('[%#.0o][%#.3o][%#5.0x][%.d][%-#08.3x][%+05d][% -5i][%.3u]', \
            0, 8, 0, 0, 255, 42, 7, 9)"
        ),
        "[0][010][     ][][0x0ff   ][+0042][ 7   ][009]"
    );
    assert_eq!(
        ok("return string.format('%d %x %X %o %u', math.mininteger, -1, 255, 8, -1)"),
        "-9223372036854775808 ffffffffffffffff FF 10 18446744073709551615"
    );
}

#[test]
fn strings_and_chars() {
    assert_eq!(
        ok(
            "local t = setmetatable({}, {__tostring = function() return 'T' end}) \
            return string.format('%-5c|%5c|%5.1s|%-8s|%.0s|', 65, 65, t, t, 'abc')"
        ),
        "A    |    A|    T|T       ||"
    );
}

#[test]
fn malformed_specs() {
    for (spec, msg) in [
        ("%5%", "invalid conversion '%5%' to 'format'"),
        ("%5-d", "invalid conversion specification: '%5-d'"),
        ("%100d", "invalid conversion specification: '%100d'"),
        ("%-05s", "invalid conversion specification: '%-05s'"),
        ("%.3c", "invalid conversion specification: '%.3c'"),
        ("%#s", "invalid conversion specification: '%#s'"),
        ("%5q", "specifier '%q' cannot have modifiers"),
        ("%---------------------d", "invalid format (too long)"),
        ("%\0d", "invalid conversion '%' to 'format'"),
    ] {
        assert_eq!(
            err(&format!("string.format({spec:?}, 1)")),
            format!("c:1: {msg}"),
            "{spec}"
        );
    }
    assert_eq!(ok("return string.format('%--------------------d', 1)"), "1");
}

#[test]
fn error_order() {
    // A missing argument comes first; then integer and `%e`-style conversions check the
    // argument before the spec, `%c` and `%a` the spec first.
    assert_eq!(
        err("string.format('%k')"),
        "c:1: bad argument #2 to 'format' (no value)"
    );
    assert_eq!(
        err("string.format('%-q')"),
        "c:1: bad argument #2 to 'format' (no value)"
    );
    assert_eq!(
        err("string.format('%#d', 'x')"),
        "c:1: bad argument #2 to 'format' (number expected, got string)"
    );
    assert_eq!(
        err("string.format('%#e', {})"),
        "c:1: bad argument #2 to 'format' (number expected, got table)"
    );
    assert_eq!(
        err("string.format('%.3c', 'x')"),
        "c:1: invalid conversion specification: '%.3c'"
    );
    // `%s` converts with `__tostring` before checking.
    assert_eq!(
        err("string.format('%#s', setmetatable({}, {__tostring = function() error('ts') end}))"),
        "c:1: ts"
    );
}
