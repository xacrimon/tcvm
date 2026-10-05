//! Natives whose result count comes from their arguments stop at the stack
//! limit with their own error instead of growing the stack without bound.
//! Messages are LuaJIT 2.1's (no "stack overflow (…)" prefix as in `lua`
//! 5.5.1, whose larger stack also lets the 70000-result cases through).

use tcvm::env::{Error, Function, LuaString, NativeClosure, NativeFn, Stack, Value};
use tcvm::{Context, Lua, RuntimeError};

use crate::common::{err, ok, start_on};

#[test]
fn unpack_within_the_limit() {
    assert_eq!(
        ok("return cat(select('#', table.unpack({}, 1, 40000)))"),
        "40000"
    );
    assert_eq!(ok("return cat(table.unpack({10, 20, 30}, 2))"), "20 30");
}

#[test]
fn unpack_past_the_limit() {
    for range in ["1, 70000", "1, 2^31 - 2", "1, math.maxinteger"] {
        let src = format!("return table.unpack({{}}, {range})");
        assert_eq!(err(&src), "c:1: too many results to unpack", "{range}");
    }
}

#[test]
fn unpack_through_index_past_the_limit() {
    let src = "local t = setmetatable({}, {__index = function(_, k) return k end}) ";
    assert_eq!(
        ok(&format!("{src} return cat(table.unpack(t, 1, 3))")),
        "1 2 3"
    );
    assert_eq!(
        err(&format!("{src} return cat(table.unpack(t, 1, 70000))")),
        "c:1: too many results to unpack"
    );
}

#[test]
fn byte_past_the_limit() {
    assert_eq!(ok("return cat(string.byte('abc', 1, -1))"), "97 98 99");
    assert_eq!(ok("return cat(select('#', string.byte('abc', 3, 1)))"), "0");
    assert_eq!(
        err("return string.byte(('a'):rep(70000), 1, -1)"),
        "c:1: string slice too long"
    );
}

#[test]
fn codepoint_past_the_limit() {
    assert_eq!(
        ok("return cat(utf8.codepoint('aé€', 1, -1))"),
        "97 233 8364"
    );
    assert_eq!(
        err("return utf8.codepoint(('a'):rep(70000), 1, -1)"),
        "c:1: string slice too long"
    );
}

#[test]
fn string_unpack_past_the_limit() {
    assert_eq!(
        ok("return cat(string.unpack('bbx b', '\\1\\2\\0\\3'))"),
        "1 2 3 5"
    );
    assert_eq!(
        err("return string.unpack(('b'):rep(70000), ('\\0'):rep(70000))"),
        "c:1: too many results"
    );
}

/// Pushes 100000 values without checking for room.
fn flood<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    for _ in 0..100_000 {
        stack.push(Value::boolean(true));
    }
    Ok(())
}

#[test]
fn unchecked_pushes_overflow() {
    let mut lua = Lua::new();
    lua.load_all();
    lua.enter(|ctx| {
        let f = Function::new_native(ctx.mutation(), flood as NativeFn, &[]);
        let key = Value::string(LuaString::new(ctx, b"flood"));
        ctx.globals().raw_set(ctx, key, Value::function(f));
    });
    let ex = start_on(&mut lua, "error(select(2, pcall(flood)), 0)");
    match lua.finish(&ex) {
        Err(RuntimeError::Lua(e)) => lua.enter(|ctx| {
            let v = ctx.fetch(&e).value();
            let s = v.get_string().expect("a string error");
            assert_eq!(s.as_bytes(), b"stack overflow");
        }),
        other => panic!("expected a Lua error, got {other:?}"),
    }
}

/// `deep(n, f)` calls `f` under `n` non-tail frames, at least 16000 slots up.
const DEEP: &str = "local function deep(n, f) if n == 0 then return f() end \
                    local r = deep(n - 1, f) return r end ";

// Expected values from `lua` 5.5.1 on the same chunks scaled to its limit.
#[test]
fn resume_args_past_the_limit() {
    let co = "local co = coroutine.create(function() deep(8000, coroutine.yield) end) \
              coroutine.resume(co) ";
    assert_eq!(
        ok(&format!(
            "{DEEP}{co} local ok, e = coroutine.resume(co, table.unpack({{}}, 1, 60000)) \
             return cat(ok, e, coroutine.status(co))"
        )),
        "false too many arguments to resume suspended"
    );
    let w = "local w = coroutine.wrap(function() deep(8000, coroutine.yield) end) w() ";
    assert_eq!(
        ok(&format!(
            "{DEEP}{w} return cat(pcall(w, table.unpack({{}}, 1, 60000)))"
        )),
        "false too many arguments to resume"
    );
}

#[test]
fn resume_results_past_the_limit() {
    let co = "local co = coroutine.create(function() \
              coroutine.yield(table.unpack({}, 1, 60000)) return 1 end) ";
    assert_eq!(
        ok(&format!(
            "{DEEP}{co} local r = deep(8000, function() return cat(coroutine.resume(co)) end) \
             return cat(r, coroutine.status(co), coroutine.resume(co))"
        )),
        "false too many results to resume suspended true 1"
    );
    let w = "local w = coroutine.wrap(function() return table.unpack({}, 1, 60000) end) ";
    assert_eq!(
        err(&format!("{DEEP}{w} return deep(8000, w)")),
        "c:1: too many results to resume"
    );
}
