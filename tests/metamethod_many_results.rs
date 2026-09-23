//! A metamethod or generic-for iterator returning more than 255 values: the
//! continuation reads its results up to `top`, so the count doesn't wrap at
//! 256. Expected values come from `lua` 5.5.1.

use tcvm::{Executor, LoadError, Lua};

fn run(src: &str) -> i64 {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
}

const PRELUDE: &str = "local t = {} for i = 1, 256 do t[i] = i end ";

#[test]
fn lua_iterator() {
    let src = format!(
        "{PRELUDE} for a, b in function() return table.unpack(t) end do return a * 10 + b end"
    );
    assert_eq!(run(&src), 12);
}

#[test]
fn native_iterator() {
    let src = format!("{PRELUDE} for a, b in table.unpack, t do return a * 10 + b end");
    assert_eq!(run(&src), 12);
}

#[test]
fn index_function() {
    let src = format!(
        "{PRELUDE} return setmetatable({{}}, {{ __index = function() return table.unpack(t) end }}).x"
    );
    assert_eq!(run(&src), 1);
}
