//! `os.clock` reports process CPU time via C `clock()`. The value itself is
//! machine-dependent, so only its shape is checked.

use tcvm::{Executor, LoadError, Lua};

fn run_bool(src: &str) -> bool {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("os_clock_test"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute::<bool>(&ex).expect("run")
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
