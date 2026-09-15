//! `pcall` / `xpcall`. Expected values come from `lua` 5.5.1 running the
//! same snippets (chunk name `=t`); the message handler runs *before* the
//! stack unwinds, which the native-handler test observes directly.

use std::pin::Pin;

use tcvm::env::thread::Frame;
use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::vm::sequence::{BoxSequence, CallbackAction, Execution, Sequence, SequencePoll};
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

/// The message of the error `src` lets escape to the host.
fn run_err(src: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let tcvm::RuntimeError::Lua(stashed) = lua.execute::<()>(&ex).expect_err("must raise") else {
        panic!("not a Lua error")
    };
    lua.enter(|ctx| {
        let s = ctx.fetch(&stashed).value().get_string().unwrap();
        String::from_utf8_lossy(s.as_bytes()).into_owned()
    })
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
    // A non-callable target is raised inside the protected call, so the
    // xpcall handler sees it too.
    check(
        "local ok, e = xpcall(5, function(e) return 'H:' .. e end) return (not ok and e == 'H:attempt to call a number value') and 1 or 0",
    );
    // The chain bound is on hops, not on how many arguments ride along.
    check(
        "local t = setmetatable({}, {__call = function(self, ...) return select('#', ...) end}) local args = {} for i = 1, 300 do args[i] = i end local ok, n = pcall(t, table.unpack(args)) return (ok and n == 300) and 1 or 0",
    );
}

#[test]
fn pcall_without_arguments_is_an_error() {
    assert_eq!(
        run_err("pcall()"),
        "t:1: bad argument #1 to 'pcall' (value expected)"
    );
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
    // A nil error object reaches the handler as nil; only the catcher
    // substitutes the placeholder message.
    check(
        "local ok, e = xpcall(function() error() end, function(e) return 'H:' .. tostring(e) end) return (not ok and e == 'H:nil') and 1 or 0",
    );
    // Extra arguments are passed to the function; success returns its results.
    check(
        "local ok, v = xpcall(function(a, b) return a + b end, print, 3, 4) return (ok and v == 7) and 1 or 0",
    );
}

#[test]
fn xpcall_handler_errors_and_nesting() {
    // An error inside the handler calls the handler again with it (manual
    // §2.3); the loop is cut after a fixed depth. The reference gives up
    // when its C stack runs out (215 calls with LUAI_MAXCCALLS = 200); we
    // stop at exactly 1 + 200.
    check(
        "local n = 0 local ok, e = xpcall(function() error('orig') end, function(e) n = n + 1 error('in handler') end) return (not ok and e == 'error in error handling' and n == 201) and 1 or 0",
    );
    check(
        "local m = 0 local ok, e = xpcall(function() error('orig') end, function(e) m = m + 1 if m == 1 then error('once') end return 'H' .. m .. ':' .. e end) return (not ok and e == 'H2:t:1: once') and 1 or 0",
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
fn pcall_of_a_yielding_native() {
    // The resume args are the yield's results and land at the callee slot,
    // so pcall sees them (and not the stale `coroutine.yield`) as f's results.
    check(
        "local co = coroutine.wrap(function() return pcall(coroutine.yield, 1) end)\n\
         local first = co()\n\
         local ok, a, b = co('a', 'b')\n\
         return (first == 1 and ok == true and a == 'a' and b == 'b') and 1 or 0",
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
    let n = stack
        .frames()
        .iter()
        .filter(|f| matches!(f, Frame::Lua(_)))
        .count();
    stack.replace(&[Value::integer(nctx.ctx.mutation(), n as i64)]);
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

#[test]
fn handler_may_yield() {
    // Resumable everywhere, like LuaJIT: the reference's "attempt to yield
    // across a C-call boundary" is a limitation of its C-stack handler call.
    check(
        "local co = coroutine.create(function()\n\
           return xpcall(function() error('orig') end, function(e) return 'H:' .. coroutine.yield('mid') .. e end)\n\
         end)\n\
         local ok1, v1 = coroutine.resume(co)\n\
         local ok2, ok, e = coroutine.resume(co, 'R:')\n\
         return (ok1 and v1 == 'mid' and ok2 and not ok and e == 'H:R:t:2: orig') and 1 or 0",
    );
}

#[test]
fn retried_handler_still_sees_the_failing_frames() {
    // The first invocation fails; the retry runs with only that invocation
    // unwound, so it counts main, `outer`, `deep`, and itself.
    let frames: i64 = run_with(
        "local function deep() error('x') end\n\
         local function outer() deep() end\n\
         local first = true\n\
         local ok, n = xpcall(outer, function(e)\n\
           if first then first = false error('again') end\n\
           return lua_frames()\n\
         end)\n\
         return n",
        install_through,
    );
    assert_eq!(frames, 4);
}

/// `through(f, ...)`: call `f` under a sequence that keeps the default
/// `Catch::Pass`, i.e. the kind of native-with-callback (`sort`, `gsub`)
/// that must not shadow an enclosing `xpcall` handler.
fn lua_through<'gc>(
    nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    struct PassThrough;
    unsafe impl<'gc> tcvm::dmm::Collect<'gc> for PassThrough {
        const NEEDS_TRACE: bool = false;
    }
    impl<'gc> Sequence<'gc> for PassThrough {
        fn trace_pointers(&self, _cc: &mut dyn tcvm::dmm::Trace<'gc>) {}
        fn poll(
            self: Pin<&mut Self>,
            _ctx: tcvm::Context<'gc>,
            _exec: Execution<'gc>,
            _stack: Stack<'gc, '_>,
        ) -> Result<SequencePoll<'gc>, Error<'gc>> {
            Ok(SequencePoll::Return)
        }
    }
    let then = BoxSequence::new(nctx.ctx.mutation(), PassThrough);
    Ok(CallbackAction::call(Some(then)))
}

fn install_through(ctx: tcvm::Context<'_>) {
    for (name, f) in [
        ("lua_frames", lua_frame_count as NativeFn),
        ("through", lua_through as NativeFn),
    ] {
        let f = Function::new_native(ctx.mutation(), f, Box::new([]));
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        ctx.globals().raw_set(ctx, key, Value::function(f));
    }
}

#[test]
fn pass_through_sequence_is_transparent_to_errors() {
    // Success path returns the callee's results; an error passes through
    // it to the enclosing pcall.
    let v: i64 = run_with(
        "local ok, e = pcall(function() through(function() error('x') end) end)\n\
         local a, b = through(function() return 1, 2 end)\n\
         return (not ok and e == 't:1: x' and a == 1 and b == 2) and 1 or 0",
        install_through,
    );
    assert_eq!(v, 1);
}

#[test]
fn handler_looks_through_a_pass_through_sequence() {
    // The nearest sequence is `through`'s, but it isn't a catch point, so
    // the xpcall handler still runs with `outer` and `deep` intact.
    let frames: i64 = run_with(
        "local function deep() error('x') end\n\
         local function outer() through(deep) end\n\
         local ok, n = xpcall(outer, lua_frames)\n\
         return n",
        install_through,
    );
    assert_eq!(frames, 3);
}
