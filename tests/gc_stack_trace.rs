//! Tests for `ThreadState`'s hand-written `Collect` (issue #43), which traces
//! only the live stack high-water (`stack[0..live_top]`) instead of the whole
//! grown-not-shrunk vec. It has to get two opposite things right.
//!
//! *Under*-tracing is the unsound direction: if `live_top` is too low, a value
//! reachable only through a (suspended) thread's stack registers is left
//! unmarked and swept — and the post-resume read sees freed memory. The
//! `*_survives_forced_gc` tests park live tables on a suspended stack, force a
//! FULL mark/sweep via `Lua::collect_all`, allocate churn to reuse any freed
//! slots, then resume and assert the values are intact.
//!
//! *Over*-tracing is the bug #43 actually reports: dead slots above the live
//! region pin whatever a since-returned callee left in its registers, and the
//! memory never comes back. `dead_stack_slots_are_reclaimed` measures that
//! directly.

use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::vm::sequence::CallbackAction;
use tcvm::{Executor, LoadError, Lua, RuntimeError};

/// Native that yields to its resumer (the host, when called on the main thread).
fn yielder<'gc>(
    _nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Ok(CallbackAction::Yield { then: None })
}

fn setup(src: &str) -> (Lua, tcvm::StashedExecutor) {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let y = Function::new_native(ctx.mutation(), yielder as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"yielder"));
            ctx.globals().raw_set(ctx, key, Value::function(y));
            let chunk = ctx.load(src, Some("gc_stack"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    (lua, ex)
}

/// Allocate a lot of garbage so any slot freed by a buggy under-trace gets
/// overwritten — turning a latent use-after-free into an observable wrong value.
fn churn(lua: &mut Lua) {
    lua.enter(|ctx| {
        for i in 0..2000 {
            let t = tcvm::env::Table::new(ctx);
            t.raw_set(ctx, Value::integer(i), Value::integer(i * 7));
        }
    });
    lua.collect_all();
}

fn finish_i64(lua: &mut Lua, ex: &tcvm::StashedExecutor) -> i64 {
    lua.resume(ex, ()).expect("resume to completion");
    lua.try_enter(|ctx| {
        let ex = ctx.fetch(ex);
        ex.take_result::<i64>(ctx)
    })
    .expect("take result")
}

#[test]
fn live_main_stack_survives_forced_gc() {
    // a, b, c are live in the main frame's registers across the host yield.
    let (mut lua, ex) = setup(
        "local a, b, c = {x=11}, {x=22}, {x=33}\n\
         local function deep()\n\
           local d, e = {x=44}, {x=55}\n\
           yielder()\n\
           return a.x + b.x + c.x + d.x + e.x\n\
         end\n\
         return deep()",
    );
    let err = lua.finish(&ex).expect_err("main should yield");
    assert!(matches!(err, RuntimeError::MainYielded), "got {err:?}");

    lua.collect_all(); // full GC while a..e live only on the suspended main stack
    churn(&mut lua); // reuse any wrongly-freed memory

    assert_eq!(finish_i64(&mut lua, &ex), 11 + 22 + 33 + 44 + 55);
}

/// Builds one of three variants of the same program. `alloc` is spliced in
/// where `inner` would build its big table, `ret` is what `inner` hands back.
///
/// The ten padding locals are load-bearing, not decoration. `big` has to end up
/// in a register that (a) sits above main's `base + max_stack_size` and (b) is
/// never reused afterwards — that is the only position from which a #43
/// over-trace is observable. A dead slot *inside* a live frame's register
/// window is traced either way, and a dead slot in a *low* callee register just
/// gets overwritten by whatever the caller computes next, so the value would
/// become unreachable on its own and prove nothing. Pushing `big` up to
/// register 10 of a callee that main never re-enters parks it above everything
/// still live, where only the `Collect` bound decides its fate.
fn program(alloc: &str, ret: &str) -> String {
    format!(
        "local function inner()\n\
         \x20 local p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10\n\
         \x20 {alloc}\n\
         \x20 return {ret}\n\
         end\n\
         local kept = inner()\n\
         yielder()\n\
         return 0"
    )
}

const BUILD_BIG: &str = "local big = {} for i = 1, 2000 do big[i] = { i, i + 1 } end";

/// Run to the host yield and report live bytes after a full collection.
fn live_bytes_at_yield(src: &str) -> usize {
    let (mut lua, ex) = setup(src);
    let err = lua.finish(&ex).expect_err("main should yield");
    assert!(matches!(err, RuntimeError::MainYielded), "got {err:?}");
    lua.collect_all();
    lua.live_bytes()
}

#[test]
fn dead_stack_slots_are_reclaimed() {
    // Same program three ways; the only variable is the fate of the table.
    //
    //   baseline — never builds it.
    //   dead     — builds it, then abandons it in `inner`'s register 10.
    //   live     — builds it and hands it back to a live local in main.
    let baseline = live_bytes_at_yield(&program("", "p1 + p10"));
    let dead = live_bytes_at_yield(&program(BUILD_BIG, "p1 + p10"));
    let live = live_bytes_at_yield(&program(BUILD_BIG, "big"));

    // Guard against a vacuous pass: if the table isn't big enough to move the
    // needle, "it got reclaimed" would be indistinguishable from noise.
    let payload = live.saturating_sub(baseline);
    assert!(
        payload > 50_000,
        "payload too small to be conclusive: baseline={baseline} live={live}"
    );

    // The dead variant must give essentially all of it back. Under the old
    // derived `Collect` the whole vec was traced, `inner`'s abandoned register
    // stayed a GC root, and `dead` sat right on top of `live`.
    let retained = dead.saturating_sub(baseline);
    assert!(
        retained < payload / 10,
        "abandoned table not reclaimed: baseline={baseline} dead={dead} live={live} \
         (retained {retained} of {payload} payload bytes)"
    );
}

#[test]
fn suspended_coroutine_stack_survives_forced_gc() {
    // p, q live on a *coroutine's* stack while it is suspended at a yield.
    let (mut lua, ex) = setup(
        "local co = coroutine.create(function()\n\
           local p, q = {v=7}, {v=8}\n\
           coroutine.yield()\n\
           return p.v + q.v\n\
         end)\n\
         coroutine.resume(co)   -- run co to its yield; p,q live on co's stack\n\
         yielder()              -- main yields to host\n\
         local ok, s = coroutine.resume(co)\n\
         return s",
    );
    let err = lua.finish(&ex).expect_err("main should yield");
    assert!(matches!(err, RuntimeError::MainYielded), "got {err:?}");

    lua.collect_all(); // full GC while p,q live only on the suspended coroutine stack
    churn(&mut lua);

    assert_eq!(finish_i64(&mut lua, &ex), 7 + 8);
}
