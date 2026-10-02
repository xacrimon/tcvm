//! `f"str"`, `f[[str]]` and `f{...}` call forms (manual §3.4.10). Regression
//! for #69.

use crate::common::eval;

fn run(src: &str) -> i64 {
    eval(src)
}

#[test]
fn string_and_table_args() {
    assert_eq!(
        run("local function len(x) return #x end\n\
             return len 'abc' * 100 + len [[de]] * 10 + len {1, 2, 3}"),
        323
    );
}

#[test]
fn method_call_and_chaining() {
    assert_eq!(
        run("local o = {m = function(self, t) return t[1] end}\n\
             local function g(a) return function(b) return a[1] * 10 + #b end end\n\
             return o:m {40} + g {1} \"xy\""),
        52
    );
}
