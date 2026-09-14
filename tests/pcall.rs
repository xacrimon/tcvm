//! `pcall` / `xpcall`. Expected values come from `lua` 5.5.1 running the
//! same snippets (chunk name `=t`); the message handler runs *before* the
//! stack unwinds, which the native-handler test observes directly.

use tcvm::env::thread::Frame;
use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::vm::sequence::CallbackAction;
use tcvm::{Executor, LoadError, Lua};

fn run_with<T: for<'gc> tcvm::FromMultiValue<'gc>>(
    src: &str,
    setup: impl for<'gc> FnOnce(tcvm::Context<'gc>),
) -> T {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            setup(ctx);
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
}

fn run_i64(src: &str) -> i64 {
    run_with(src, |_| {})
}

/// `return (cond) and 1 or 0` for a chunk that compares against expected
/// strings inline, so failures point at the Lua expression.
fn check(src: &str) {
    assert_eq!(run_i64(src), 1, "{src}");
}

#[test]
fn pcall_results() {
    check(
        "local ok, a, b, c = pcall(function() return 1, 2, 3 end) return (ok and a == 1 and b == 2 and c == 3) and 1 or 0",
    );
    check("return select('#', pcall(function() return 1, 2, 3 end)) == 4 and 1 or 0");
    check("local ok, e = pcall(error, 'm') return (not ok and e == 'm') and 1 or 0");
    check(
        "local ok, e = pcall(function() error('m') end) return (not ok and e == 't:1: m') and 1 or 0",
    );
    check(
        "local ok, e = pcall(function() local x return x.y end) return (not ok and e == 't:1: attempt to index a nil value') and 1 or 0",
    );
    check("local ok, e = pcall(error, {code = 7}) return (not ok and e.code == 7) and 1 or 0");
    check("local ok, e = pcall(error) return (not ok and e == '<no error object>') and 1 or 0");
}

#[test]
fn pcall_of_non_functions() {
    check(
        "local ok, e = pcall(5) return (not ok and e == 'attempt to call a number value') and 1 or 0",
    );
    check(
        "local t = setmetatable({}, {__call = function(self, a) return a * 2 end}) local ok, v = pcall(t, 21) return (ok and v == 42) and 1 or 0",
    );
    // A `__call` that is itself a callable table resolves through the chain.
    check(
        "local inner = setmetatable({}, {__call = function(a, b, c) return type(a) .. type(b) .. c end}) local t = setmetatable({}, {__call = inner}) local ok, v = pcall(t, 9) return (ok and v == 'tabletable9') and 1 or 0",
    );
    check(
        "local ok, e = pcall(setmetatable({}, {__name = 'Thing'})) return (not ok and e == 'attempt to call a Thing value') and 1 or 0",
    );
}

#[test]
fn pcall_without_arguments_is_an_error() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load("pcall()", Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let err = lua.execute::<()>(&ex).expect_err("pcall() must raise");
    let tcvm::RuntimeError::Lua(stashed) = err else {
        panic!("{err:?}")
    };
    let msg = lua.enter(|ctx| {
        let s = ctx.fetch(&stashed).value().get_string().unwrap();
        String::from_utf8_lossy(s.as_bytes()).into_owned()
    });
    assert_eq!(msg, "t:1: bad argument #1 to 'pcall' (value expected)");
}

#[test]
fn xpcall_handler_transforms_the_error() {
    check(
        "local ok, e = xpcall(function() error('orig') end, function(e) return 'H:' .. e end) return (not ok and e == 'H:t:1: orig') and 1 or 0",
    );
    // Only the handler's first result is used.
    check(
        "local ok, e, extra = xpcall(function() error('orig') end, function(e) return e, 'extra' end) return (not ok and e == 't:1: orig' and extra == nil) and 1 or 0",
    );
    check(
        "local ok, t = xpcall(function() error({}) end, function(e) return type(e) end) return (not ok and t == 'table') and 1 or 0",
    );
    check(
        "local ok, e = xpcall(function() error('orig') end, function(e) return nil end) return (not ok and e == '<no error object>') and 1 or 0",
    );
    // Extra arguments are passed to the function; success returns its results.
    check(
        "local ok, v = xpcall(function(a, b) return a + b end, print, 3, 4) return (ok and v == 7) and 1 or 0",
    );
}

#[test]
fn xpcall_handler_errors_and_nesting() {
    check(
        "local ok, e = xpcall(function() error('orig') end, function(e) error('in handler') end) return (not ok and e == 'error in error handling') and 1 or 0",
    );
    // An inner pcall shadows the outer handler for errors it catches.
    check(
        "local ok, v = xpcall(function() pcall(error, 'inner') return 'fine' end, function(e) return 'OUTER:' .. e end) return (ok and v == 'fine') and 1 or 0",
    );
    // A handler may itself use xpcall; its first result (false) becomes the error value.
    check(
        "local ok, e = xpcall(function() error('orig') end, function(e) return xpcall(function() error('nested') end, function(e2) return 'N:' .. e2 end) end) return (not ok and e == false) and 1 or 0",
    );
}

#[test]
fn xpcall_of_a_native_that_fails_immediately() {
    // The catcher is the top frame when the handler runs; the calling Lua
    // frame's registers below it must survive.
    check(
        "local a, b, c = 1, 2, 3 local ok, e = xpcall(string.rep, function(e) return 'H:' .. e end) return (not ok and string.sub(e, 1, 2) == 'H:' and a == 1 and b == 2 and c == 3) and 1 or 0",
    );
}

#[test]
fn xpcall_argument_checks() {
    check(
        "local ok, e = pcall(xpcall, function() end) return (not ok and e == \"bad argument #2 to 'xpcall' (function expected, got no value)\") and 1 or 0",
    );
    check(
        "local ok, e = pcall(xpcall, function() end, 5) return (not ok and e == \"bad argument #2 to 'xpcall' (function expected, got number)\") and 1 or 0",
    );
}

#[test]
fn xpcall_across_a_yield() {
    check(
        "local co = coroutine.wrap(function()\n\
           return xpcall(function() coroutine.yield(1) error('after yield') end, function(e) return 'HY:' .. e end)\n\
         end)\n\
         local first = co()\n\
         local ok, e = co()\n\
         return (first == 1 and not ok and e == 'HY:t:2: after yield') and 1 or 0",
    );
}

/// Counts the Lua frames on the running thread; as an `xpcall` handler it
/// sees the failing frames only if the handler runs before unwinding.
fn lua_frame_count<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let n = nctx
        .exec
        .frames()
        .iter()
        .filter(|f| matches!(f, Frame::Lua(_)))
        .count();
    stack.replace(&[Value::integer(n as i64)]);
    Ok(CallbackAction::Return)
}

#[test]
fn handler_runs_before_the_stack_unwinds() {
    // Main chunk, `outer`, and `deep` are live when the handler runs; after
    // unwinding only the main chunk would remain.
    let frames: i64 = run_with(
        "local function deep() error('x') end\n\
         local function outer() deep() end\n\
         local ok, n = xpcall(outer, lua_frames)\n\
         return n",
        |ctx| {
            let f = Function::new_native(ctx.mutation(), lua_frame_count as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"lua_frames"));
            ctx.globals().raw_set(ctx, key, Value::function(f));
        },
    );
    assert_eq!(frames, 3);
}
