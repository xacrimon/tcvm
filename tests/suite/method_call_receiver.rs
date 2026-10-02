//! A method call whose receiver is a temp (call result, field, paren expr)
//! must not leave the receiver below the call block, where a MULTRET
//! consumer would see it as an extra leading value. Regression for #95.

use crate::common::eval;

fn run(src: &str) -> i64 {
    eval(src)
}

const PRELUDE: &str = "local t = setmetatable({}, {__index = {m = function(self, ...) return ... end}})\n\
                       local function getit() return t end\n";

#[test]
fn call_result_receiver_in_multret_arg() {
    assert_eq!(
        run(&format!("{PRELUDE}return select('#', getit():m(1, 2))")),
        2
    );
}

#[test]
fn call_result_receiver_in_return() {
    assert_eq!(
        run(&format!(
            "{PRELUDE}local function f() return getit():m(7) end\n\
                      local a, b = f()\n\
                      return a + (b == nil and 100 or 0)"
        )),
        107
    );
}

#[test]
fn field_receiver_in_multret_arg() {
    assert_eq!(
        run(&format!(
            "{PRELUDE}local h = {{inner = t}}\n\
                      return select('#', h.inner:m(1, 2, 3))"
        )),
        3
    );
}

#[test]
fn receiver_still_usable_after_call() {
    // Reusing the receiver's slot for `func` must only happen for temps,
    // never for a named local.
    assert_eq!(
        run(&format!(
            "{PRELUDE}local r = getit()\n\
                      local a = r:m(5)\n\
                      return a + r:m(6)"
        )),
        11
    );
}
