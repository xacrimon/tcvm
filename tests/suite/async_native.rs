//! Async natives: futures polled from the native's frame inside dispatch.

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context as TaskCx, Poll, Wake, Waker};

use tcvm::env::{Error, Function, LuaString, NativeClosure, Stack, Value};
use tcvm::vm::async_native::{AsyncFn, Spawned};
use tcvm::{Context, Executor, LoadError, Lua, RuntimeError, StashedExecutor, StepResult};

/// `each(n, f)`: the sum of `f(i)` for `i` in `1..=n`.
fn each<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    let Some(n) = stack.get(0).get_integer() else {
        return Err(Error::from_str(ctx, "each: n must be an integer"));
    };
    let f = stack.local(stack.get(1));
    Ok(stack.spawn(move |cx| async move {
        let mut sum = 0;
        for i in 1..=n {
            cx.enter(|ctx, mut stack| {
                stack.clear();
                let f = stack.get_local(f);
                stack.push(f);
                stack.push(Value::integer(ctx.mutation(), i));
            });
            cx.call(0).await;
            sum += cx.enter(|_, stack| stack.get(0).get_integer().unwrap_or(0));
        }
        cx.enter(|ctx, mut stack| stack.replace(&[Value::integer(ctx.mutation(), sum)]));
        Ok(())
    }))
}

/// `try(f, ...)`: `true` and `f(...)`'s results, or `false` and its error.
fn try_<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    Ok(stack.spawn(|cx| async move {
        match cx.pcall(0).await {
            Ok(()) => cx.enter(|_, mut stack| stack.insert(0, Value::boolean(true))),
            Err(e) => cx.enter(|_, mut stack| {
                let v = stack.get_local(e.value());
                stack.replace(&[Value::boolean(false), v]);
            }),
        }
        Ok(())
    }))
}

/// `ywrap(...)`: yields its arguments, then returns what it was resumed with.
fn ywrap<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    Ok(stack.spawn(|cx| async move {
        cx.yield_(0).await;
        Ok(())
    }))
}

/// `fail(msg)`: raises `msg` from inside its future.
fn fail<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    Ok(stack.spawn(|cx| async move {
        let msg = cx.enter(|_, stack| {
            String::from_utf8_lossy(stack.get(0).get_string().unwrap().as_bytes()).into_owned()
        });
        Err(cx.error(&msg))
    }))
}

thread_local! {
    static DROPS: Cell<u32> = const { Cell::new(0) };
}

struct CountDrop;

impl Drop for CountDrop {
    fn drop(&mut self) {
        DROPS.with(|d| d.set(d.get() + 1));
    }
}

/// `guarded(f)`: calls `f()` unprotected, holding a value whose drop is
/// counted.
fn guarded<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    Ok(stack.spawn(|cx| async move {
        let _guard = CountDrop;
        cx.call(0).await;
        Ok(())
    }))
}

/// A host future: pending until `ready` is set, then ready.
struct HostWait {
    ready: Rc<Cell<bool>>,
    waker: Rc<Cell<Option<Waker>>>,
}

impl Future for HostWait {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut TaskCx<'_>) -> Poll<()> {
        if self.ready.get() {
            Poll::Ready(())
        } else {
            self.waker.set(Some(cx.waker().clone()));
            Poll::Pending
        }
    }
}

thread_local! {
    static HOST: (Rc<Cell<bool>>, Rc<Cell<Option<Waker>>>) =
        (Rc::new(Cell::new(false)), Rc::new(Cell::new(None)));
}

/// `wait(v)`: waits on the host, then returns `v`, kept across the wait.
fn wait<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    let v = stack.local(stack.get(0));
    Ok(stack.spawn(move |cx| async move {
        let (ready, waker) = HOST.with(|h| (h.0.clone(), h.1.clone()));
        HostWait { ready, waker }.await;
        cx.enter(|_, mut stack| {
            let v = stack.get_local(v);
            stack.replace(&[v]);
        });
        Ok(())
    }))
}

fn lua_with_natives() -> Lua {
    let mut lua = Lua::new();
    lua.load_all();
    lua.enter(|ctx| {
        let natives: [(&str, AsyncFn); 6] = [
            ("each", each),
            ("try", try_),
            ("ywrap", ywrap),
            ("fail", fail),
            ("guarded", guarded),
            ("wait", wait),
        ];
        for (name, f) in natives {
            let f = Function::new_async(ctx.mutation(), f, &[]);
            let key = Value::string(LuaString::new(ctx, name.as_bytes()));
            ctx.globals().raw_set(ctx, key, Value::function(f));
        }
    });
    lua
}

fn start(lua: &mut Lua, src: &str) -> StashedExecutor {
    lua.try_enter(|ctx| -> Result<_, LoadError> {
        let chunk = ctx.load(src, Some("=t"))?;
        Ok(ctx.stash(Executor::start(ctx, chunk, ())))
    })
    .expect("load")
}

fn run(src: &str) -> Result<String, String> {
    let mut lua = lua_with_natives();
    let ex = start(&mut lua, src);
    match lua.finish(&ex) {
        Ok(()) => Ok(lua.enter(|ctx| text(ctx.fetch(&ex).take_result::<Value>(ctx).unwrap()))),
        Err(RuntimeError::Lua(e)) => Err(lua.enter(|ctx| text(ctx.fetch(&e).value()))),
        Err(e) => panic!("{e:?}"),
    }
}

fn text(v: Value<'_>) -> String {
    match v.get_string() {
        Some(s) => String::from_utf8_lossy(s.as_bytes()).into_owned(),
        None => v.type_name().to_owned(),
    }
}

#[test]
fn calls_lua_in_a_loop() {
    let src = "return tostring(each(100, function(i) return i * 2 end))";
    assert_eq!(run(src), Ok("10100".into()));
}

#[test]
fn calls_natives_and_async_natives() {
    let src = "return tostring(each(3, function(i) return each(i, math.abs) end))";
    assert_eq!(run(src), Ok("10".into()));
}

#[test]
fn tail_called() {
    let src = "local function f() return each(4, function(i) return i end) end \
               return tostring(f())";
    assert_eq!(run(src), Ok("10".into()));
}

#[test]
fn pcall_catches() {
    let src = "local a, b = try(function(x) return x + 1 end, 41) \
               local c, d = try(function() error('boom') end) \
               return tostring(a) .. ' ' .. b .. ' ' .. tostring(c) .. ' ' .. d";
    assert_eq!(run(src), Ok("true 42 false t:1: boom".into()));
}

#[test]
fn unprotected_error_drops_the_future() {
    DROPS.with(|d| d.set(0));
    let src = "local ok, e = pcall(guarded, function() error('x', 0) end) \
               return tostring(ok) .. e";
    assert_eq!(run(src), Ok("falsex".into()));
    assert_eq!(DROPS.with(Cell::get), 1);
}

#[test]
fn error_from_the_future() {
    assert_eq!(run("fail('nope')"), Err("t:1: nope".into()));
    let src = "local ok, e = pcall(fail, 'caught') return e";
    assert_eq!(run(src), Ok("caught".into()));
}

#[test]
fn error_before_spawning() {
    // Called by `pcall`, a native, it gets no position, as in the reference.
    let src = "return select(2, pcall(each, 'x'))";
    assert_eq!(run(src), Ok("each: n must be an integer".into()));
    let src = "return select(2, pcall(function() return each('x') end))";
    assert_eq!(run(src), Ok("t:1: each: n must be an integer".into()));
}

#[test]
fn yields_from_a_coroutine() {
    let src = "local co = coroutine.wrap(function(a) local r = ywrap(a, a + 1) return r * 10 end) \
               local x, y = co(1) \
               return x .. y .. co(5)";
    assert_eq!(run(src), Ok("1250".into()));
}

#[test]
fn yields_to_the_host() {
    let mut lua = lua_with_natives();
    let ex = start(&mut lua, "return ywrap(7) + 1");
    let yielded = lua.enter(|ctx| match ctx.fetch(&ex).step(ctx).unwrap() {
        StepResult::Yielded(v) => v[0].get_integer(),
        _ => None,
    });
    assert_eq!(yielded, Some(7));
    lua.resume(&ex, (41i64,)).unwrap();
    let r = lua.enter(|ctx| ctx.fetch(&ex).take_result::<i64>(ctx).unwrap());
    assert_eq!(r, 42);
}

#[test]
fn nested_deeply() {
    let src = "local function r(n) if n == 0 then return 1 end \
               return each(1, function() return r(n - 1) end) end \
               return tostring(r(150))";
    assert_eq!(run(src), Ok("1".into()));
}

struct NoWake;

impl Wake for NoWake {
    fn wake(self: std::sync::Arc<Self>) {}
}

#[test]
fn waits_on_the_host_across_a_collection() {
    let mut lua = lua_with_natives();
    let ex = start(&mut lua, "return wait({41})[1] + 1");
    let waker = Waker::from(std::sync::Arc::new(NoWake));
    HOST.with(|h| h.0.set(false));
    let pending = lua.enter(|ctx| {
        matches!(
            ctx.fetch(&ex).step_waker(ctx, &waker).unwrap(),
            StepResult::Pending
        )
    });
    assert!(pending);
    // The kept table survives a full collection while the native waits.
    lua.collect_all();
    let registered = HOST.with(|h| {
        let w = h.1.take();
        h.1.set(w.clone());
        w.is_some_and(|w| w.will_wake(&waker))
    });
    assert!(registered);
    HOST.with(|h| h.0.set(true));
    assert_eq!(lua.execute::<i64>(&ex).unwrap(), 42);
}

/// `yguard()`: yields while holding a value whose drop is counted.
fn yguard<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    Ok(stack.spawn(|cx| async move {
        let _guard = CountDrop;
        cx.yield_(0).await;
        Ok(())
    }))
}

#[test]
fn closing_a_coroutine_drops_the_future() {
    DROPS.with(|d| d.set(0));
    let mut lua = lua_with_natives();
    lua.enter(|ctx| {
        let f = Function::new_async(ctx.mutation(), yguard, &[]);
        let key = Value::string(LuaString::new(ctx, b"yguard"));
        ctx.globals().raw_set(ctx, key, Value::function(f));
    });
    let ex = start(
        &mut lua,
        "local co = coroutine.create(function() yguard() end) \
         coroutine.resume(co) local before = coroutine.status(co) \
         return before .. tostring(coroutine.close(co))",
    );
    lua.finish(&ex).unwrap();
    let r = lua.enter(|ctx| text(ctx.fetch(&ex).take_result::<Value>(ctx).unwrap()));
    assert_eq!(r, "suspendedtrue");
    assert_eq!(DROPS.with(Cell::get), 1);
}

/// `leak()`: returns nothing, but keeps its local for `stale()`.
fn leak<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    let l = stack.local(Value::boolean(true));
    STALE.with(|s| s.set(Some(l)));
    Ok(stack.spawn(|_| async { Ok(()) }))
}

thread_local! {
    static STALE: Cell<Option<tcvm::vm::async_native::Local>> = const { Cell::new(None) };
}

fn stale<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    let _ = stack.local(Value::boolean(false));
    let l = STALE.with(Cell::get).unwrap();
    let _ = stack.get_local(l);
    Ok(stack.spawn(|_| async { Ok(()) }))
}

#[test]
#[should_panic(expected = "a Local used after its native returned")]
fn a_stale_local_panics() {
    let mut lua = lua_with_natives();
    lua.enter(|ctx| {
        for (name, f) in [("leak", leak as AsyncFn), ("stale", stale)] {
            let f = Function::new_async(ctx.mutation(), f, &[]);
            let key = Value::string(LuaString::new(ctx, name.as_bytes()));
            ctx.globals().raw_set(ctx, key, Value::function(f));
        }
    });
    let ex = start(&mut lua, "leak() stale()");
    let _ = lua.finish(&ex);
}

/// Ready once another OS thread, started on the first poll, says so.
struct Timer {
    done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    started: bool,
}

impl Future for Timer {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut TaskCx<'_>) -> Poll<()> {
        use std::sync::atomic::Ordering;
        if self.done.load(Ordering::Acquire) {
            return Poll::Ready(());
        }
        if !self.started {
            self.started = true;
            let (done, waker) = (self.done.clone(), cx.waker().clone());
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(5));
                done.store(true, Ordering::Release);
                waker.wake();
            });
        }
        Poll::Pending
    }
}

/// `sleep(v)`: returns `v` once a timer on another thread fires.
fn sleep<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    let v = stack.local(stack.get(0));
    Ok(stack.spawn(move |cx| async move {
        let done = Default::default();
        Timer {
            done,
            started: false,
        }
        .await;
        cx.enter(|_, mut stack| {
            let v = stack.get_local(v);
            stack.replace(&[v]);
        });
        Ok(())
    }))
}

/// Run `fut` to completion on this thread, parking while it is pending.
fn block_on<F: Future>(fut: F) -> F::Output {
    struct Unpark(std::thread::Thread);
    impl Wake for Unpark {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(std::sync::Arc::new(Unpark(std::thread::current())));
    let mut cx = TaskCx::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        if let Poll::Ready(r) = fut.as_mut().poll(&mut cx) {
            return r;
        }
        std::thread::park();
    }
}

#[test]
fn finish_async_waits_for_the_host() {
    let mut lua = lua_with_natives();
    lua.enter(|ctx| {
        let f = Function::new_async(ctx.mutation(), sleep, &[]);
        let key = Value::string(LuaString::new(ctx, b"sleep"));
        ctx.globals().raw_set(ctx, key, Value::function(f));
    });
    let ex = start(
        &mut lua,
        "local t = {} for i = 1, 3 do t[i] = sleep(i) end return t[1] + t[2] + t[3]",
    );
    block_on(lua.finish_async(&ex)).unwrap();
    let r = lua.enter(|ctx| ctx.fetch(&ex).take_result::<i64>(ctx).unwrap());
    assert_eq!(r, 6);
}
