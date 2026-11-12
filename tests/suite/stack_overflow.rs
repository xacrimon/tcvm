//! Calls past the stack limit raise "stack overflow" instead of growing the
//! stack without bound; a message handler still runs in the headroom above
//! it. Expected strings come from `lua` 5.5.1 on the same chunks named `=t`.
//! Each chunk hands its result to the host with `error(v, 0)`.

use tcvm::env::{Error, Function, LuaString, NativeClosure, Stack, Value};
use tcvm::vm::async_native::Spawned;
use tcvm::{Context, Executor, LoadError, Lua, RuntimeError, StepResult};

fn raised(src: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    raised_in(&mut lua, src)
}

fn raised_in(lua: &mut Lua, src: &str) -> String {
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

// The spread copies the extras above the frame, which can cross the limit
// even when the call itself fit.
#[test]
fn vararg_spread_overflows() {
    let src = "local function g(...) return ... end \
               local function h(...) local t = {...} return #t end \
               local n = 40000 \
               error(select(2, pcall(g, table.unpack({}, 1, n))) .. ' ' .. \
                     select(2, pcall(h, table.unpack({}, 1, n))) .. ' ' .. \
                     select('#', g(table.unpack({}, 1, n // 3))), 0)";
    assert_eq!(raised(src), "t:1: stack overflow t:1: stack overflow 13333");
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

// Each nested resume runs on a new thread with a stack limit of its own, so
// only a cap on nesting stops these. The `n > 10000` guards keep the chunks
// finite on a build without it.
const WRAP_RECURSE: &str = "local n = 0 local function f() n = n + 1 \
                            if n > 10000 then return 'unbounded' end \
                            return coroutine.wrap(f)() end ";

#[test]
fn nested_wraps_overflow() {
    let src = format!(
        "{WRAP_RECURSE} local ok, e = pcall(f) \
         error(tostring(ok) .. ' ' .. string.match(e, 'C stack overflow$'), 0)"
    );
    assert_eq!(raised(&src), "false C stack overflow");
}

#[test]
fn nested_resumes_overflow() {
    let src = "local n = 0 local function g() n = n + 1 \
               if n > 10000 then return 'unbounded' end \
               local ok, e = coroutine.resume(coroutine.create(g)) \
               return ok and e or 'failed: ' .. e end \
               error(g(), 0)";
    assert_eq!(raised(src), "failed: C stack overflow");
}

#[test]
fn nesting_below_the_resume_limit() {
    let src = "local function nest(k) if k == 0 then return 'bottom' end \
               return coroutine.wrap(nest)(k - 1) end \
               error(nest(150), 0)";
    assert_eq!(raised(src), "bottom");
}

#[test]
fn coroutines_work_after_a_resume_overflow() {
    let src = format!(
        "{WRAP_RECURSE} pcall(f) \
         local co = coroutine.wrap(function(a) local b = coroutine.yield(a + 1) return b * 2 end) \
         error(co(1) .. ' ' .. co(21), 0)"
    );
    assert_eq!(raised(&src), "2 42");
}

fn yielder<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    Ok(stack.spawn(|cx| async move {
        cx.yield_(0).await;
        Ok(())
    }))
}

// A handler that yields to the host leaves the thread in handler mode; a
// fresh `Executor::start` on the same thread must restore the normal limit.
#[test]
fn restart_after_a_handler_yields_to_the_host() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let y = Function::new_async(ctx.mutation(), yielder, &[]);
            let key = Value::string(LuaString::new(ctx, b"yielder"));
            ctx.globals().raw_set(ctx, key, Value::function(y));
            let src = format!("{RECURSE} xpcall(f, function(m) yielder() return m end)");
            let chunk = ctx.load(&src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    // Step in separate `enter`s past any `Pending` from GC exits.
    let yielded = loop {
        let r = lua.try_enter(|ctx| -> Result<Option<bool>, RuntimeError> {
            Ok(match ctx.fetch(&ex).step(ctx)? {
                StepResult::Pending => None,
                r => Some(matches!(r, StepResult::Yielded(_))),
            })
        });
        if let Some(y) = r.expect("step") {
            break y;
        }
    };
    assert!(yielded);

    let src = format!(
        "{RECURSE} local function id(x) return x end \
         error(select(2, xpcall(f, function(m) return 'handled: ' .. id(m) end)), 0)"
    );
    assert_eq!(raised_in(&mut lua, &src), "handled: t:1: stack overflow");
}

fn select_count(lua: &mut Lua, n: usize) -> Result<i64, String> {
    let ex = lua.enter(|ctx| {
        let select = ctx
            .globals()
            .raw_get(Value::string(LuaString::new(ctx, b"select")));
        let mut args = vec![Value::nil(); n];
        args[0] = Value::string(LuaString::new(ctx, b"#"));
        ctx.stash(Executor::start(ctx, select, args.as_slice()))
    });
    let r = lua.execute::<i64>(&ex).map_err(|e| match e {
        RuntimeError::Lua(stashed) => lua.enter(|ctx| {
            String::from_utf8_lossy(ctx.fetch(&stashed).message(ctx).as_bytes()).into_owned()
        }),
        e => panic!("expected a Lua error, got {e:?}"),
    });
    // Spent either way, as after any uncaught error.
    let again = lua.try_enter(|ctx| ctx.fetch(&ex).step(ctx).map(|_| ()));
    assert!(matches!(again, Err(RuntimeError::BadMode)), "{again:?}");
    r
}

// Arguments that would not fit the stack mapping used to panic.
#[test]
fn start_with_too_many_arguments() {
    let mut lua = Lua::new();
    lua.load_all();
    assert_eq!(select_count(&mut lua, 60_000), Ok(59_999));
    assert_eq!(select_count(&mut lua, 70_000), Err("stack overflow".into()));
}

#[test]
fn resume_with_too_many_arguments() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load("return coroutine.yield()", Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let yielded = loop {
        let r = lua.try_enter(|ctx| -> Result<Option<bool>, RuntimeError> {
            Ok(match ctx.fetch(&ex).step(ctx)? {
                StepResult::Pending => None,
                r => Some(matches!(r, StepResult::Yielded(_))),
            })
        });
        if let Some(y) = r.expect("step") {
            break y;
        }
    };
    assert!(yielded);
    let err = lua.enter(|ctx| {
        match ctx
            .fetch(&ex)
            .resume(ctx, vec![Value::nil(); 70_000].as_slice())
        {
            Err(RuntimeError::Lua(e)) => {
                String::from_utf8_lossy(ctx.fetch(&e).message(ctx).as_bytes()).into_owned()
            }
            r => panic!("expected a Lua error, got {r:?}"),
        }
    });
    assert_eq!(err, "too many arguments to resume");
    // The executor is still yielded and takes a resume that fits.
    lua.resume(&ex, (1i64, 2i64)).expect("resume");
    let r = lua.try_enter(|ctx| ctx.fetch(&ex).take_result::<(i64, i64)>(ctx));
    assert_eq!(r.expect("result"), (1, 2));
}

// A handler yields to the host from the headroom above the normal limit,
// where its resume still has room.
#[test]
fn resume_a_handler_yielded_in_the_headroom() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let src = format!(
                "{RECURSE} error(select(2, xpcall(f, function(m) \
                 return m .. ' ' .. coroutine.yield() end)), 0)"
            );
            let chunk = ctx.load(&src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let yielded = loop {
        let r = lua.try_enter(|ctx| -> Result<Option<bool>, RuntimeError> {
            Ok(match ctx.fetch(&ex).step(ctx)? {
                StepResult::Pending => None,
                r => Some(matches!(r, StepResult::Yielded(_))),
            })
        });
        if let Some(y) = r.expect("step") {
            break y;
        }
    };
    assert!(yielded);
    match lua.resume(&ex, (7i64,)) {
        Err(RuntimeError::Lua(e)) => lua.enter(|ctx| {
            assert_eq!(
                ctx.fetch(&e).message(ctx).as_bytes(),
                b"t:1: stack overflow 7"
            );
        }),
        r => panic!("expected a Lua error, got {r:?}"),
    }
}
