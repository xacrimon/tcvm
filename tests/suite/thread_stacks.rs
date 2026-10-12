//! Every thread runs on a stack of its own: parked stacks survive
//! collections, `coroutine.close`, `os.exit` and the end of the `Lua`.

use tcvm::{Executor, Lua, RuntimeError, StepResult};

use crate::common::{ok, start_on};

// Expected values from `lua` 5.5.1.
#[test]
fn close_of_a_parked_coroutine_runs_its_close_variables() {
    assert_eq!(
        ok("local log = {}
            local function mk(i)
              return coroutine.create(function()
                local a <close> = setmetatable({}, {__close = function() log[#log + 1] = 'a' .. i end})
                local b <close> = setmetatable({}, {__close = function() log[#log + 1] = 'b' .. i end})
                for k = 1, 300 do coroutine.yield(k) end
              end)
            end
            local res = {}
            for i = 1, 3 do
              local co = mk(i)
              for k = 1, 5 do coroutine.resume(co) end
              res[#res + 1] = tostring(coroutine.close(co)) .. coroutine.status(co)
            end
            return table.concat(res, ' ') .. ' ' .. table.concat(log, ',')"),
        "truedead truedead truedead b1,a1,b2,a2,b3,a3"
    );
}

/// An exit from a coroutine resumed by a coroutine leaves both parked; the
/// state runs another chunk after it.
#[test]
fn exit_with_parked_coroutines() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = start_on(
        &mut lua,
        "local inner = coroutine.wrap(function() for i = 1, 10 do coroutine.yield(i) end os.exit(3) end)
         local outer = coroutine.wrap(function() while true do coroutine.yield(inner()) end end)
         for i = 1, 20 do outer() end",
    );
    assert!(matches!(lua.finish(&ex), Err(RuntimeError::Exit(3))));
    let ex = start_on(&mut lua, "return 'after'");
    lua.finish(&ex).unwrap();
    lua.enter(|ctx| {
        let v: tcvm::env::Value = ctx.fetch(&ex).take_result(ctx).unwrap();
        assert_eq!(v.get_string().unwrap().as_bytes(), b"after");
    });
}

/// A `Lua` dropped with coroutines suspended mid-call and a main thread
/// yielded to the host frees their stacks.
#[test]
fn dropping_lua_with_parked_stacks() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = start_on(
        &mut lua,
        "cos = {}
         local function deep(n) if n == 0 then return coroutine.yield(n) end return deep(n - 1) + 1 end
         for i = 1, 100 do cos[i] = coroutine.wrap(deep) cos[i](i) end
         coroutine.yield(1)",
    );
    let r = lua.enter(|ctx| {
        loop {
            match ctx.fetch(&ex).step(ctx) {
                Ok(StepResult::Yielded(_)) => break true,
                Ok(StepResult::Pending) => continue,
                _ => break false,
            }
        }
    });
    assert!(r);
    drop(lua);
}

/// A thread that never ran has no stack; one that ran gives it back.
#[test]
fn threads_get_stacks_when_they_run() {
    let mut lua = Lua::new();
    lua.load_all();
    for _ in 0..3 {
        let ex = lua.enter(|ctx| {
            let f = ctx
                .load(
                    "local t = {} for i = 1, 1000 do t[i] = coroutine.create(function() end) end
                     for i = 1, 1000, 2 do coroutine.resume(t[i]) end return #t",
                    Some("=c"),
                )
                .unwrap();
            ctx.stash(Executor::start(ctx, f, ()))
        });
        lua.finish(&ex).unwrap();
        lua.collect_all();
    }
}
