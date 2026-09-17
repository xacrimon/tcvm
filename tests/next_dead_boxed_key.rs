//! `next`'s dead-key resume path (`hash_part::position`) must never dereference a
//! deleted entry's key: a deleted boxed integer (`i64` outside `i32`, heap-allocated
//! per `Value::integer`) is untraced once dead and its box can be swept, so comparing
//! it by value would read freed memory. `Key::same_dead` compares dead entries by bit
//! identity instead — see `src/env/table/hash_part.rs`.
//!
//! `dead_boxed_key_is_rejected_not_dereferenced` parks the freed box's address under
//! GC churn and confirms `next` reports "invalid key" rather than reading through it.
//! `live_boxed_key_still_resumes_by_value` is the control: a *live* entry must still
//! resume by value, since a differently-boxed but numerically equal key is exactly
//! what `for k, v in pairs(t) do t[k] = nil end` yields once other entries also use
//! boxed integers.

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
            let chunk = ctx.load(src, Some("t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
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

fn finish_str(lua: &mut Lua, ex: &tcvm::StashedExecutor) -> String {
    lua.resume(ex, ()).expect("resume to completion");
    lua.try_enter(|ctx| {
        let r = ctx.fetch(ex).take_result::<LuaString>(ctx)?;
        Ok::<_, RuntimeError>(String::from_utf8_lossy(r.as_bytes()).into_owned())
    })
    .expect("take result")
}

#[test]
fn dead_boxed_key_is_rejected_not_dereferenced() {
    // Deletes a boxed-int key (`t[-(big+1)] = nil`), forces a full GC while it's
    // only reachable as a dead hash entry, then churns boxed-int allocations to
    // recycle the freed box's slot before resuming past the yield and calling
    // `next` with a freshly evaluated (differently-boxed, numerically equal) key.
    let (mut lua, ex) = setup(
        "local t = {}\n\
         local big = 1 << 40\n\
         t[-(big+1)] = 1\n\
         t[-(big+2)] = 2\n\
         t[-(big+1)] = nil\n\
         yielder()\n\
         local ok, k, v = pcall(next, t, -(big+1))\n\
         return tostring(ok) .. ' ' .. tostring(k) .. ' ' .. tostring(v)",
    );
    let err = lua.finish(&ex).expect_err("main should yield");
    assert!(matches!(err, RuntimeError::MainYielded), "got {err:?}");

    lua.collect_all(); // sweeps the dead entry's now-unreachable box
    churn_boxed_ints(&mut lua); // recycle its freed slot with different i64s

    // A distinct box with the same value no longer matches a dead entry, so this
    // is "invalid key", not the pre-fix behavior of dereferencing the freed box
    // (which produced whatever value churn happened to leave there).
    assert_eq!(
        finish_str(&mut lua, &ex),
        "false invalid key to 'next' nil"
    );
}

#[test]
fn live_boxed_key_still_resumes_by_value() {
    // The ordinary `pairs`-delete idiom: every key is a boxed int, and resuming
    // from one must still find its *live* entry by value even though the `next`
    // call below constructs its key expression fresh rather than reusing a Lua
    // reference to the entry's own box.
    // Negative keys: `array_index` (table/mod.rs) treats any *positive* integer
    // key as an array index with no upper bound, so a positive `1<<40`-scale key
    // would try to grow the array part to trillions of slots — unrelated to what
    // this test checks, so it's avoided here.
    let (mut lua, ex) = setup(
        "local t = {}\n\
         local big = 1 << 40\n\
         for i = 1, 5 do t[-(big + i)] = i end\n\
         local k, v = next(t, -(big + 2))\n\
         return tostring(k) .. '=' .. tostring(v)",
    );
    lua.finish(&ex).expect("run to completion");
    let s = lua
        .try_enter(|ctx| {
            let r = ctx.fetch(&ex).take_result::<LuaString>(ctx)?;
            Ok::<_, RuntimeError>(String::from_utf8_lossy(r.as_bytes()).into_owned())
        })
        .expect("take result");
    // Verified against lua 5.5.1 on this exact snippet.
    assert_eq!(s, "-1099511627777=1");
}
