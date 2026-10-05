//! An error a builtin raises after calling Lua (`table.sort`'s invalid order
//! function) is positioned at the builtin's caller. Expected strings come
//! from `lua` 5.5.1 running the same chunk.

use crate::common::err;

#[test]
fn sort_error_gets_callers_position() {
    assert_eq!(
        err(
            "local t = {5, 4, 3, 2, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
             table.sort(t, function(a, b) return true end)"
        ),
        "c:2: invalid order function for sorting"
    );
}
