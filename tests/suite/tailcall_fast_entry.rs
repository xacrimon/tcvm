//! Regression for issue #356: a fast entry reached by a TAILCALL from a frame
//! with an open upvalue returned the TAILCALL's argument count of values.
//! Expected values from Lua 5.5.1.

use crate::common::ok;

#[test]
fn one_result_from_a_frame_with_a_captured_local() {
    let src = "local function f(c) local function h() c = 1 end return math.max(1.0, c) end
        local function g(s) local function h() s = 1 end return s:sub(1, 2) end
        local function a(c) local function h() c = 1 end return math.abs(c) end
        return cat(select('#', f(2)), f(2), select('#', g('abc')), g('abc'), select('#', a(-3)), a(-3))";
    assert_eq!(ok(src), "1 2 1 ab 1 3");
}

#[test]
fn assert_returns_its_arguments() {
    let src = "local function f(c) local function h() c = 1 end return assert(c, 'm', 3) end
        local function v(c, ...) local function h() c = 1 end return assert(...) end
        return cat(select('#', f(2)), cat(f(2)), select('#', v(0, 5, 6)), v(0, 5, 6))";
    assert_eq!(ok(src), "3 2 m 3 2 5 6");
}

#[test]
fn the_upvalue_is_closed_with_the_local() {
    let src =
        "local function tail(c) local h = function() return c end keep = h return math.abs(c) end
        local r = tail(-41)
        return cat(r, keep())";
    assert_eq!(ok(src), "41 -41");
}
