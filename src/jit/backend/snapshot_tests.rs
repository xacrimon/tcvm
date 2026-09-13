//! Backend snapshots, taken over the same functions the frontend snapshots pin.
//!
//! These exist to make the machine-level consequences of a frontend decision
//! visible: whether a boxed value cost a tag register, whether a pack became an
//! instruction or a rename, and exactly what each exit stub has to write back.

use std::fs;

use insta::assert_snapshot;

use super::isel::select;
use super::mach::print_mfunc;
use crate::jit::frontend::lower::lower;
use crate::jit::ir::ty::{Rep, Ty, TypeSet};
use crate::{Executor, Lua};

const INT: Ty = Ty::new(Rep::Val, TypeSet::INT);
const TAB: Ty = Ty::new(Rep::Val, TypeSet::TAB);

/// `sum_field(t, n)` with warm inline caches — the running example, all the way
/// down to machine IR.
///
/// The things worth reading in the output: the `add.i64` accumulator never
/// touches a tag, the `guard.shape` is a load-plus-compare with no type check in
/// front of it, and each exit stub names exactly the registers the interpreter
/// needs — which is also what keeps them alive through the allocator.
#[test]
fn test_mir_loop_warm_ic() {
    let source = fs::read_to_string("test-files/jit_loop_warm.lua").unwrap();
    let mut lua = Lua::new();
    lua.load_all();

    let ex = lua.try_enter(|ctx| {
        let chunk = ctx.load(&source, Some("test")).expect("compile");
        Ok::<_, crate::RuntimeError>(ctx.stash(Executor::start(ctx, chunk, ())))
    });
    let ex = ex.expect("start");
    lua.finish(&ex).expect("run");
    let f = lua.enter(|ctx| {
        let v = ctx
            .fetch(&ex)
            .take_result::<crate::env::value::Value>(ctx)
            .expect("chunk produced a result");
        ctx.stash(v.get_function().expect("chunk returns a function"))
    });

    let out = lua.enter(|ctx| {
        let closure = ctx.fetch(&f).as_lua().expect("Lua closure");
        let func = lower(closure.proto, 0, vec![TAB, INT]).expect("lower");
        match select(&func) {
            Ok(m) => print_mfunc(&m),
            Err(e) => format!("declined: {e:?}\n"),
        }
    });
    assert_snapshot!(out);
}
