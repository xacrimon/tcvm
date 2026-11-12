//! Regression for issue #355: an `and`/`or` with a constant operand left a
//! jump aimed one instruction late (past the end of the function, or past the
//! operand that decides the value), and `repeat ... until (x and K)` lost its
//! back edge. Expected values from Lua 5.5.1.

use crate::common::ok;

#[test]
fn empty_if_on_constant_or() {
    let src = "local function f() if (100.25 or u1) then end end
        return cat(select('#', f()))";
    assert_eq!(ok(src), "0");
}

#[test]
fn empty_if_on_and_or_constant() {
    let src =
        "local function f(b) if (((math.type(b) == 'float') and math.max) or (-2)) then end end
        return cat(select('#', f(1)), select('#', f(1.5)))";
    assert_eq!(ok(src), "0 0");
}

#[test]
fn empty_elseif_keeps_registers() {
    let src = "local function id(...) return ... end
        local function f(b) if (7 < #'') then elseif (true or b) then end return id(b, 2) end
        return cat(f(1))";
    assert_eq!(ok(src), "1 2");
}

#[test]
fn falsy_constant_and_then_or_constant() {
    let src = "return cat(((false and u1) or 2.0) and 'y', ((nil and u1) or false) and 'y')";
    assert_eq!(ok(src), "y false");
}

#[test]
fn while_on_falsy_and_or_constant() {
    let src = "local n = 0
        while ((false and c) or '10') do n = n + 1 if n == 3 then break end end
        return cat(n)";
    assert_eq!(ok(src), "3");
}

#[test]
fn repeat_until_and_constant() {
    let src = "local n = 0
        repeat n = n + 1; if n > 5 then break end until (N and 1e300)
        local looped = n
        N, n = 1, 0
        repeat n = n + 1; if n > 5 then break end until (N and 1e300)
        N = nil
        return cat(looped, n)";
    assert_eq!(ok(src), "6 1");
}
