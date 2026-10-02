//! Multiple assignment spreads a trailing call / method call / `...` across
//! the remaining targets (manual §3.3.3). Regression for #121.

use tcvm::{Executor, LoadError, Lua};

fn run(src: &str) -> (i64, i64) {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("assign_multires"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
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
