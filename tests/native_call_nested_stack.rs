//! Regression: a native call made from inside a *nested* function truncated the
//! shared thread stack down to the native's argument window, and the return
//! path back into the caller (`frame_return`'s `Caller` branch) did not restore
//! the caller's full register window. The caller then resumed with
//! `stack.len() <` its window, so register writes through the raw `registers`
//! pointer landed in the Vec's spare capacity (past `len`) and the next native
//! call's `resize` nilled them — silently clobbering the caller's locals (and
//! globals loaded into registers) above the nested-call slot.
//!
//! Every expectation is checked against `lua` 5.5.0.

use tcvm::{Executor, LoadError, Lua};

fn eval_bool(src: &str) -> bool {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("native_call_nested_stack"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute::<bool>(&ex).expect("run")
}

#[test]
fn nested_native_call_preserves_caller_locals() {
    // `up()` calls a native (`type`); `w`'s locals declared before `up` must
    // survive the call. Pre-fix, `lc` came back nil.
    assert!(eval_bool(
        "local type = type\n\
         local function w()\n\
           local la, lb, lc = 11, 22, 33\n\
           local function up() local _ = type(nil) end\n\
           up()\n\
           return la == 11 and lb == 22 and lc == 33\n\
         end\n\
         return w()"
    ));
}

#[test]
fn four_locals_survive_nested_native_call() {
    assert!(eval_bool(
        "local type = type\n\
         local function w()\n\
           local a, b, c, d = 1, 2, 3, 4\n\
           local function up() local _ = type(nil) end\n\
           up()\n\
           return a == 1 and b == 2 and c == 3 and d == 4\n\
         end\n\
         return w()"
    ));
}

#[test]
fn caller_survives_when_native_called_before_and_after_grow() {
    // Native call inside `up` both before and after it grows its own window;
    // exercises a couple of truncate/restore round-trips before `w` resumes.
    assert!(eval_bool(
        "local type = type\n\
         local function w()\n\
           local a, b, c, d, e = 5, 6, 7, 8, 9\n\
           local function up() local _ = type(nil); local _ = type(type(1)) end\n\
           up()\n\
           return a == 5 and b == 6 and c == 7 and d == 8 and e == 9\n\
         end\n\
         return w()"
    ));
}

#[test]
fn nested_global_decl_then_native_keeps_outer_globals() {
    // The symptom this was first noticed through: a nested function declares new
    // globals and makes a native call; the outer function's earlier globals must
    // still read back correctly afterward.
    assert!(eval_bool(
        "global print\n\
         local function w()\n\
           global a, b = 1, 2\n\
           global c, d = 3, 4\n\
           global e, g = 5, 6\n\
           local function up() global oo, pp = 77, 88; print(oo, pp) end\n\
           up()\n\
           return a == 1 and b == 2 and c == 3 and d == 4 and e == 5 and g == 6\n\
         end\n\
         return w()"
    ));
}
