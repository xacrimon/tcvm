//! The JIT, on: whole Lua programs run through the executor with the compile
//! hook live, checked against the answer the interpreter alone would give.
//!
//! These are the first tests where nothing hand-builds a frame. The interpreter
//! decides when to compile, hands a real Lua frame to native code, and takes back
//! whatever it returns — so a broken calling convention, a wrong resume pc, or a
//! region that the collector fails to keep alive shows up here as a wrong number
//! or a crash, which is the point.

use std::fs;

use crate::env::function::Function;
use crate::env::value::Value;
use crate::jit::region::HOT_CALL;
use crate::{Executor, Lua};

/// What a run tells us: the chunk's answer, and what the region it compiled
/// actually did.
struct Run {
    total: i64,
    calls: u32,
    entries: u64,
    deopts: u64,
}

/// Run a chunk that returns `(total, f)`, then read `f`'s region.
fn run(path: &str) -> Run {
    let source = fs::read_to_string(path).unwrap();
    let mut lua = Lua::new();
    lua.load_all();

    let ex = lua
        .try_enter(|ctx| {
            let chunk = ctx.load(&source, Some("jit")).expect("compile");
            Ok::<_, crate::RuntimeError>(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("start");
    lua.finish(&ex).expect("run");

    lua.enter(|ctx| {
        let (total, f) = ctx
            .fetch(&ex)
            .take_result::<(Value, Function)>(ctx)
            .expect("chunk returns (total, f)");
        let proto = f.as_lua().expect("a Lua closure").proto;
        let region = proto.jit.borrow().expect("the function never compiled");
        Run {
            total: total.get_integer().expect("an integer total"),
            calls: proto.jit_calls.get(),
            entries: region.entries.get(),
            deopts: region.deopts.get(),
        }
    })
}

/// Calls after the threshold enter native code. The one that crosses it compiles
/// and then runs what it just built, so the native call count is off by one from
/// the obvious arithmetic — worth pinning, since an off-by-one here would mean a
/// region was compiled and then never entered.
const NATIVE_FROM: u32 = HOT_CALL;

/// 200 calls of `sum_field(t, 10)`, each summing `t.x == 7` ten times.
///
/// The first `HOT_CALL - 1` are interpreted, the rest are native, and the total
/// says the two agree. Asserting on `entries` is what makes this a JIT test at
/// all: the entry check could reject every call and the answer would still be
/// right.
#[test]
fn a_hot_function_runs_as_native_code() {
    let r = run("test-files/jit_hot_call.lua");

    assert_eq!(r.calls, HOT_CALL, "the counter should pin once compiled");
    assert_eq!(
        r.entries,
        (200 - NATIVE_FROM + 1) as u64,
        "calls past the threshold should all have run natively"
    );
    assert_eq!(r.deopts, 0, "nothing here should fail a guard");
    assert_eq!(r.total, 200 * 10 * 7);
}

/// The same program, then 50 more calls with a table of a *different* shape.
///
/// Those enter native code — the entry check only looks at types, and a table is
/// a table — fail the shape guard on the first loop iteration, and deopt. The
/// interpreter then has to finish a loop it never started, from a frame it never
/// built, and the total is what says it could.
#[test]
fn a_failed_guard_finishes_in_the_interpreter() {
    let r = run("test-files/jit_hot_call_deopt.lua");

    assert_eq!(r.entries, (250 - NATIVE_FROM + 1) as u64);
    assert_eq!(r.deopts, 50, "every call on the foreign shape should deopt");
    assert_eq!(r.total, 250 * 10 * 7);
}
