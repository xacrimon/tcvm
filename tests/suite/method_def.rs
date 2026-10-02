//! `function t:m(...)` declares a method with an implicit leading `self`
//! (manual §3.4.11). Regression for #79 (the parser used to panic).

use crate::common::eval;

fn run(src: &str) -> i64 {
    eval(src)
}

#[test]
fn implicit_self_and_params() {
    assert_eq!(
        run("local obj = {v = 5}\n\
             function obj:add(a, b) return self.v + a + b end\n\
             return obj:add(1, 2)"),
        8
    );
}

#[test]
fn self_is_the_receiver() {
    assert_eq!(
        run("local obj = {}\n\
             function obj:me() return self end\n\
             return obj:me() == obj and 1 or 0"),
        1
    );
}

#[test]
fn dotted_path_then_colon() {
    assert_eq!(
        run("local ns = {inner = {v = 3}}\n\
             function ns.inner:get(k) return self.v * k end\n\
             return ns.inner:get(4)"),
        12
    );
}

#[test]
fn vararg_method() {
    assert_eq!(
        run("local obj = {}\n\
             function obj:count(...) return select('#', ...) end\n\
             return obj:count(1, 2, 3)"),
        3
    );
}

#[test]
fn class_pattern() {
    assert_eq!(
        run("local Account = {}\n\
             Account.__index = Account\n\
             function Account.new(b) return setmetatable({balance = b}, Account) end\n\
             function Account:deposit(v) self.balance = self.balance + v return self end\n\
             local a = Account.new(10)\n\
             a:deposit(5)\n\
             return a.balance"),
        15
    );
}
