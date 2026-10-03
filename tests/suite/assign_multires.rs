//! Multiple assignment spreads a trailing call / method call / `...` across
//! the remaining targets (manual §3.3.3). Regression for #121.

use crate::common::{eval, ok};

fn run(src: &str) -> (i64, i64) {
    eval(src)
}

#[test]
fn trailing_func_call() {
    assert_eq!(
        run("local function f() return 10, 20 end\n\
             local a, b\n\
             a, b = f()\n\
             return a, b"),
        (10, 20)
    );
}

#[test]
fn trailing_method_call() {
    assert_eq!(
        run(
            "local t = setmetatable({}, {__index = {m = function() return 7, 8 end}})\n\
             local c, d\n\
             c, d = t:m()\n\
             return c, d"
        ),
        (7, 8)
    );
}

#[test]
fn trailing_method_call_on_temp_receiver() {
    // The receiver is a temp, so SELF lands the call block above the
    // assignment's base and the results must still reach both targets.
    assert_eq!(
        run("local mt = {__index = {m = function() return 7, 8 end}}\n\
             local c, d\n\
             c, d = setmetatable({}, mt):m()\n\
             return c, d"),
        (7, 8)
    );
}

#[test]
fn trailing_vararg() {
    assert_eq!(
        run("local e, g\n\
             local function vg(...) e, g = ... end\n\
             vg(100, 200)\n\
             return e, g"),
        (100, 200)
    );
}

#[test]
fn trailing_vararg_into_upvalue_and_index() {
    assert_eq!(
        run("local t = {}\n\
             local e\n\
             local function vg(...) e, t.x = ... end\n\
             vg(1, 2)\n\
             return e, t.x"),
        (1, 2)
    );
}

#[test]
fn parenthesized_call_truncates() {
    assert_eq!(
        run("local function f() return 10, 20 end\n\
             local a, b = 0, 0\n\
             a, b = (f())\n\
             return a, b == nil and -1 or b"),
        (10, -1)
    );
}

#[test]
fn extra_values_are_evaluated_and_dropped() {
    // More values than targets indexed past the targets (#237). Expected
    // output from lua 5.5.1.
    assert_eq!(
        ok("local n = 0\n\
            local function f() n = n + 1 return 'f' end\n\
            local t = {}\n\
            t.x, t.y = 1, 2, f()\n\
            local a; a = 1, 2\n\
            local x, y = 1, 2\n\
            x, y = y, x, x, y\n\
            local mt = setmetatable({}, {__index = function(_, k) n = n + 10 return k end})\n\
            local p; p = 3, mt.foo, mt.bar\n\
            g1, g2 = 'a', 'b', 'c', 'd'\n\
            return cat(t.x, t.y, a, x, y, p, n, g1, g2)"),
        "1 2 1 2 1 3 21 a b"
    );
}
