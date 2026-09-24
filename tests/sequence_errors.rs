//! Errors a sequence raises keep their pending position level across the
//! stash/fetch at the sequence boundary. Expected strings come from `lua`
//! 5.5.1 running the same chunk.

use tcvm::env::Value;
use tcvm::{Executor, LoadError, Lua, RuntimeError};

fn raise_str(src: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=c"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    match lua.execute::<()>(&ex) {
        Err(RuntimeError::Lua(e)) => lua.enter(|ctx| {
            let v: Value = ctx.fetch(&e).value();
            String::from_utf8_lossy(v.get_string().expect("string error").as_bytes()).into_owned()
        }),
        other => panic!("expected a Lua error for {src:?}, got {other:?}"),
    }
}

#[test]
fn sort_error_gets_callers_position() {
    assert_eq!(
        raise_str(
            "local t = {5, 4, 3, 2, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
             table.sort(t, function(a, b) return true end)"
        ),
        "c:2: invalid order function for sorting"
    );
}
