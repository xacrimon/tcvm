//! `next` resuming from a deleted large-integer key. The integer hash part
//! stores keys as raw `i64`, so a dead entry never refers to a swept box and
//! a freshly computed equal key still finds it.

use tcvm::env::{LuaString, Value};
use tcvm::{Lua, RuntimeError, StashedExecutor};

use crate::common::{start_on, yielding_lua};

fn setup(src: &str) -> (Lua, StashedExecutor) {
    let mut lua = yielding_lua();
    let ex = start_on(&mut lua, src);
    (lua, ex)
}

/// Allocate a lot of boxed-integer garbage so a use-after-free reads something
/// other than what happened to still be there — turning a latent bug into an
/// observable wrong value (or a crash) instead of a lucky pass.
fn churn_boxed_ints(lua: &mut Lua) {
    lua.enter(|ctx| {
        for i in 0..20_000i64 {
            let _ = Value::integer(ctx.mutation(), (1i64 << 41) + i);
        }
    });
    lua.collect_all();
}

fn finish_str(lua: &mut Lua, ex: &StashedExecutor) -> String {
    lua.resume(ex, ()).expect("resume to completion");
    lua.try_enter(|ctx| {
        let r = ctx.fetch(ex).take_result::<LuaString>(ctx)?;
        Ok::<_, RuntimeError>(String::from_utf8_lossy(r.as_bytes()).into_owned())
    })
    .expect("take result")
}

#[test]
fn dead_boxed_key_resumes_by_value() {
    let (mut lua, ex) = setup(
        "local t = {}\n\
         local big = 1 << 40\n\
         t[-(big+1)] = 1\n\
         t[-(big+2)] = 2\n\
         t[-(big+1)] = nil\n\
         yielder()\n\
         local ok, k = pcall(next, t, -(big+1))\n\
         return tostring(ok) .. ' ' .. tostring(k == nil or k == -(big+2))",
    );
    let err = lua.finish(&ex).expect_err("main should yield");
    assert!(matches!(err, RuntimeError::MainYielded), "got {err:?}");

    lua.collect_all();
    churn_boxed_ints(&mut lua);

    assert_eq!(finish_str(&mut lua, &ex), "true true");
}

#[test]
fn live_boxed_key_still_resumes_by_value() {
    let (mut lua, ex) = setup(
        "local t = {}\n\
         local big = 1 << 40\n\
         for i = 1, 5 do t[-(big + i)] = i end\n\
         local after\n\
         local k = next(t)\n\
         while k ~= nil do\n\
           local n = next(t, k)\n\
           if k == -(big + 2) then after = n end\n\
           k = n\n\
         end\n\
         local k2, v2 = next(t, -(big + 2))\n\
         return tostring(k2 == after) .. ' ' .. tostring(v2 == t[k2])",
    );
    lua.finish(&ex).expect("run to completion");
    let s = lua
        .try_enter(|ctx| {
            let r = ctx.fetch(&ex).take_result::<LuaString>(ctx)?;
            Ok::<_, RuntimeError>(String::from_utf8_lossy(r.as_bytes()).into_owned())
        })
        .expect("take result");
    // `next`'s order is unspecified, so resuming is checked against the table's own traversal.
    assert_eq!(s, "true true");
}
