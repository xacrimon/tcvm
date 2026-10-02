//! A statement must not leak temps: `function t.f() end` used to leave its
//! closure temp allocated, so the next `local` was bound above `nactvar` and
//! later treated as a dead temp by call setup.

use tcvm::{Executor, LoadError, Lua};

fn run(src: &str) -> (i64, i64) {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("stmt_temp_reclaim"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
}

#[test]
fn local_after_func_stmt_survives_call() {
    assert_eq!(
        run("local t = {}\n\
             function t.f() end\n\
             local id = function(x) return x end\n\
             local y = id(1)\n\
             return y, id(2)"),
        (1, 2)
    );
}

#[test]
fn local_after_method_stmt_on_temp_receiver() {
    assert_eq!(
        run("local t = {o = {m = function() end}}\n\
             t.o:m()\n\
             local id = function(x) return x end\n\
             local y = id(3)\n\
             return y, id(4)"),
        (3, 4)
    );
}
