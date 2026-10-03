//! `table.create`'s size limits (#238). Expected messages from lua 5.5.1,
//! apart from the function name (#186).

use crate::common::{err, ok};

#[test]
fn sizes_out_of_range() {
    for (args, msg) in [
        ("1 << 31", "bad argument #1 to 'create' (out of range)"),
        ("-1", "bad argument #1 to 'create' (out of range)"),
        (
            "math.mininteger",
            "bad argument #1 to 'create' (out of range)",
        ),
        ("0, 1 << 31", "bad argument #2 to 'create' (out of range)"),
        ("0, -1", "bad argument #2 to 'create' (out of range)"),
    ] {
        assert_eq!(
            err(&format!("local t = table.create({args})")),
            format!("c:1: {msg}"),
            "{args}"
        );
    }
}

#[test]
fn hash_part_overflow() {
    // Raised by the table itself, so without a position.
    assert_eq!(
        err("local t = table.create(0, (1 << 31) - 1)"),
        "table overflow"
    );
    assert_eq!(
        err("local t = table.create(0, (1 << 30) + 1)"),
        "table overflow"
    );
    assert_eq!(
        ok("local t = table.create(10, 1 << 4) return cat(#t, next(t), #table.create(0, nil))"),
        "0 nil 0"
    );
}
