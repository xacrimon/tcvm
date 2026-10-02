//! `return e1, ..., call()` must return exactly the leading values followed
//! by every result of the trailing call: a call in a non-final position is
//! adjusted to one value and lands directly in its return slot, with no
//! stale temp between it and the MULTRET tail.

use crate::common::eval;

fn run(src: &str) -> (i64, i64, i64) {
    eval(src)
}

#[test]
fn call_then_call() {
    assert_eq!(
        run("local function id(...) return ... end\n\
             local function k() return id(1), id(2) end\n\
             return select('#', k()), k()"),
        (2, 1, 2)
    );
}

#[test]
fn native_call_then_native_call() {
    assert_eq!(
        run("local floor = math.floor\n\
             local function k() return floor(11.4), floor(2.4) end\n\
             return select('#', k()), k()"),
        (2, 11, 2)
    );
}

#[test]
fn call_local_then_multret_call() {
    assert_eq!(
        run("local function id(...) return ... end\n\
             local function k(x) local z = 3 return id(z), x, id(7, 8) end\n\
             return select('#', k(5)), k(5)"),
        (4, 3, 5)
    );
}

#[test]
fn three_calls() {
    assert_eq!(
        run("local function id(...) return ... end\n\
             local function k() return id(1), id(2), id(3) end\n\
             return select('#', k()), k()"),
        (3, 1, 2)
    );
}
