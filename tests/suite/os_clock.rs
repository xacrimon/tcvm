//! `os.clock` reports process CPU time via C `clock()`. The value itself is
//! machine-dependent, so only its shape is checked.

use crate::common::eval;

fn run_bool(src: &str) -> bool {
    eval(src)
}

#[test]
fn clock_is_nondecreasing_float() {
    assert!(run_bool(
        "local a = os.clock()\n\
         local x = 0\n\
         for i = 1, 100000 do x = x + i end\n\
         local b = os.clock()\n\
         return math.type(a) == 'float' and a >= 0 and b >= a"
    ));
}
