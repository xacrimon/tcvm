//! `string.pack` (#240). Expected messages from lua 5.5.1, apart from the
//! function name (#186).

use crate::common::err;

#[test]
fn result_too_long() {
    // Checked before each item, blaming the argument before it: the format
    // for the first item.
    for (src, msg) in [
        (
            "string.pack(string.format('xxxxxxxxxx c%d', math.maxinteger - 9))",
            "bad argument #1 to 'pack' (result too long)",
        ),
        (
            "string.pack(string.format('xxxxxxxxxx c%d', math.maxinteger - 9), 'a')",
            "bad argument #1 to 'pack' (result too long)",
        ),
        (
            "string.pack(string.format('!8 b i8 c%d', math.maxinteger - 15), 1, 2, '')",
            "bad argument #3 to 'pack' (result too long)",
        ),
    ] {
        assert_eq!(err(&format!("local s = {src}")), format!("c:1: {msg}"));
    }
}
