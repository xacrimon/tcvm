//! Calls past the stack limit raise "stack overflow" instead of growing the
//! stack without bound; a message handler still runs in the headroom above
//! it. Expected strings come from `lua` 5.5.1 on the same chunks named `=t`.
//! Each chunk hands its result to the host with `error(v, 0)`.

use tcvm::{Executor, LoadError, Lua, RuntimeError};

fn raised(src: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    match lua.execute::<()>(&ex) {
        Err(RuntimeError::Lua(stashed)) => lua.enter(|ctx| {
            let s = ctx.fetch(&stashed).value().get_string().expect("string");
            String::from_utf8_lossy(s.as_bytes()).into_owned()
        }),
        other => panic!("expected a Lua error, got {other:?}"),
    }
}

const RECURSE: &str = "local function f() return 1 + f() end ";

#[test]
fn unbounded_recursion_overflows() {
    let src = format!("{RECURSE} error(select(2, pcall(f)), 0)");
    assert_eq!(raised(&src), "t:1: stack overflow");
}

#[test]
fn vararg_recursion_overflows() {
    let src = "local function f(...) return 1 + f(1, 2, 3, ...) end error(select(2, pcall(f)), 0)";
    assert_eq!(raised(src), "t:1: stack overflow");
}

#[test]
fn overflow_in_a_coroutine() {
    let src = format!("{RECURSE} error(select(2, coroutine.resume(coroutine.create(f))), 0)");
    assert_eq!(raised(&src), "t:1: stack overflow");
}

#[test]
fn stack_is_usable_again_after_an_overflow() {
    let src = format!(
        "{RECURSE} pcall(f) \
         local function d(n) if n == 0 then return 0 end return 1 + d(n - 1) end \
         error(tostring(d(10000)) .. ' ' .. select(2, pcall(f)), 0)"
    );
    assert_eq!(raised(&src), "10000 t:1: stack overflow");
}

#[test]
fn message_handler_runs_after_an_overflow() {
    let src = format!(
        "{RECURSE} error(select(2, xpcall(f, function(m) return 'handled: ' .. m end)), 0)"
    );
    assert_eq!(raised(&src), "handled: t:1: stack overflow");
}

#[test]
fn overflow_inside_the_handler() {
    let src = format!(
        "{RECURSE} local function h(m) return 1 + h(m) end error(select(2, xpcall(f, h)), 0)"
    );
    assert_eq!(raised(&src), "error in error handling");
    // A `pcall` inside the handler catches the handler's own overflow.
    let src = format!(
        "{RECURSE} error(select(2, xpcall(f, function(m) return 'inner: ' .. select(2, pcall(f)) end)), 0)"
    );
    assert_eq!(raised(&src), "inner: error in error handling");
}

// The reference recurses on the C stack through metamethods and `pcall` and
// reports "C stack overflow"; here the Lua stack limit is what trips.
#[test]
fn recursion_through_metamethods_and_pcall() {
    let src = "local t = setmetatable({}, {}) \
               getmetatable(t).__index = function(t, k) return t[k] end \
               error(select(2, pcall(function() return t.x end)), 0)";
    assert_eq!(raised(src), "t:1: stack overflow");
    let src = "local function f() local ok, e = pcall(f) return e end error(f(), 0)";
    assert_eq!(raised(src), "stack overflow");
}
