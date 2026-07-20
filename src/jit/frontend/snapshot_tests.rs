//! Frontend snapshots, taken over bytecode the compiler actually emits rather
//! than bytecode hand-written to match the frontend's assumptions.
//!
//! Two families. The **cfg** snapshots pin block boundaries (in particular that
//! a compare and its trailing `JMP` stay *one* two-way branch), the live-in sets
//! that become block parameters, and which regions we decline to compile at all.
//! The **ir** snapshots pin what lowering makes of them.

use std::fs;

use insta::assert_snapshot;
use paste::paste;

use super::lower::lower;
use super::print::format_cfgs;
use crate::jit::ir::print::print_func;
use crate::jit::ir::ty::{Ty, TypeSet};
use crate::jit::ir::{Func, verify};
use crate::{Executor, Lua};

/// Print a lowered function, but only after it verifies. Every IR snapshot goes
/// through here, so no snapshot can be accepted for IR that violates an
/// invariant — the failure is a test failure, not a puzzling diff.
fn checked(func: &Func<'_>) -> String {
    let text = print_func(func);
    if let Err(e) = verify::verify(func) {
        panic!("IR failed verification:\n{e}\n--- ir ---\n{text}");
    }
    text
}

fn cfgs_of(source: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    lua.enter(|ctx| {
        let chunk = ctx.load(source, Some("test")).expect("compile");
        let closure = chunk.as_lua().expect("chunk is a Lua closure");
        format_cfgs(&closure.proto)
    })
}

/// Lower the chunk's first sub-prototype from pc 0, with `entry` standing in for
/// the register types the compiler would have read off the live stack at the
/// moment compilation was triggered.
fn ir_of(source: &str, entry: &[Ty]) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    lua.enter(|ctx| {
        let chunk = ctx.load(source, Some("test")).expect("compile");
        let closure = chunk.as_lua().expect("chunk is a Lua closure");
        let inner = closure.proto.prototypes[0];
        match lower(inner, 0, entry.to_vec()) {
            Ok(func) => checked(&func),
            Err(e) => format!("declined: {e:?}\n"),
        }
    })
}

macro_rules! cfg_test {
    ($name:ident, $path:literal) => {
        paste! {
            #[test]
            fn [<test_cfg_ $name>]() {
                let source = fs::read_to_string($path).unwrap();
                assert_snapshot!(cfgs_of(&source));
            }
        }
    };
}

cfg_test!(loop_, "test-files/jit_loop.lua");
cfg_test!(branch, "test-files/jit_branch.lua");
cfg_test!(multret, "test-files/jit_multret.lua");
cfg_test!(primes, "test-files/primes.lua");
cfg_test!(nbody, "test-files/nbody.lua");
// The only `TESTSET` in the corpus, and so the only `edge_params` that come
// from anything but a numeric `for`.
cfg_test!(testset, "test-files/jit_testset.lua");

const INT: Ty = Ty::new(crate::jit::ir::ty::Rep::Val, TypeSet::INT);
const TAB: Ty = Ty::new(crate::jit::ir::ty::Rep::Val, TypeSet::TAB);

/// `sum_field(t, n)` — the running example. The loop counter should end up as an
/// unboxed `i64` block parameter, and the field read should be a bare
/// `slot.get` once the IC has been filled... except it hasn't been, here: this
/// prototype has never run, so its inline caches are empty and `t.x` must stay
/// generic. That is the honest output, and worth pinning: it shows the IC is a
/// *runtime* feedback source, and lowering a never-executed function gets no
/// shape information.
#[test]
fn test_ir_loop_cold_ic() {
    let source = fs::read_to_string("test-files/jit_loop.lua").unwrap();
    assert_snapshot!(ir_of(&source, &[TAB, INT]));
}

/// `classify(a, b)` — if/else, then a `while`. Exercises compare-driven
/// branching and the merge that follows.
#[test]
fn test_ir_branch() {
    let source = fs::read_to_string("test-files/jit_branch.lua").unwrap();
    assert_snapshot!(ir_of(&source, &[INT, INT]));
}

/// `is_prime(x)` — a numeric `for` whose body has an early return, so the loop
/// body is a diamond that never rejoins.
#[test]
fn test_ir_primes() {
    let source = fs::read_to_string("test-files/primes.lua").unwrap();
    assert_snapshot!(ir_of(&source, &[INT]));
}

/// The payoff case. Same `sum_field` as above, but the chunk *runs* it first, so
/// the interpreter has filled the `GETFIELD` inline cache with the receiver's
/// shape. Lowering picks that up as feedback and the field read collapses from a
/// `lua.getindex` call into a shape guard plus a constant-offset `slot.get` —
/// which, being loop-invariant, is what LICM will hoist out of the loop
/// entirely.
#[test]
fn test_ir_loop_warm_ic() {
    let source = fs::read_to_string("test-files/jit_loop_warm.lua").unwrap();
    let mut lua = Lua::new();
    lua.load_all();

    // Execute the chunk so the inline caches are warm, and keep the returned
    // closure.
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
        match lower(closure.proto, 0, vec![TAB, INT]) {
            Ok(func) => checked(&func),
            Err(e) => format!("declined: {e:?}\n"),
        }
    });
    assert_snapshot!(out);
}
