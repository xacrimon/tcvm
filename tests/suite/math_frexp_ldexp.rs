//! `math.frexp` / `math.ldexp` (#246). Expected outputs from lua 5.5.1.

use crate::common::{err, ok};

#[test]
fn frexp() {
    assert_eq!(
        ok("local out = {} \
            for _, x in ipairs{10, 0, -0.0, 1, -3.5, 1/0, -1/0, 5e-324, 2^1023 * 1.5, '8'} do \
              local m, e = math.frexp(x) out[#out + 1] = cat(m, e, math.type(e)) \
            end return table.concat(out, ', ')"),
        "0.625 4 integer, 0.0 0 integer, -0.0 0 integer, 0.5 1 integer, \
         -0.875 2 integer, inf 0 integer, -inf 0 integer, 0.5 -1073 integer, \
         0.75 1024 integer, 0.5 4 integer"
    );
    assert_eq!(
        err("math.frexp({})"),
        "c:1: bad argument #1 to 'frexp' (number expected, got table)"
    );
}

#[test]
fn ldexp() {
    // The exponent is truncated to a C int: 2^32 is 0, 2^32 + 3 is 3.
    assert_eq!(
        ok(
            "return cat(math.ldexp(0.625, 4), math.ldexp(1, -1075), math.ldexp(1, 1024), \
            math.ldexp(1, 2^32), math.ldexp(1, 2^32 + 3), math.ldexp(3, '2'), \
            math.ldexp(1, -(2^31)), math.ldexp(-0.0, 3), math.type(math.ldexp(1, 1)))"
        ),
        "10.0 0.0 inf 1.0 8.0 12.0 0.0 -0.0 float"
    );
    assert_eq!(
        err("math.ldexp(1, 1.5)"),
        "c:1: bad argument #2 to 'ldexp' (number has no integer representation)"
    );
}
