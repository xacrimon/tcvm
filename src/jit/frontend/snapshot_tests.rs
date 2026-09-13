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

/// Every prototype in the corpus that lowers at all, run through the verifier.
///
/// The snapshots above verify what they print, but they cover four functions.
/// On-the-fly SSA construction fails by placing *too few* parameters — a use
/// that its definition no longer dominates — and that is exactly what
/// `verify` checks, so pointing it at the whole corpus is the cheapest coverage
/// available for the one failure mode that is a miscompile rather than a
/// slowdown. Declining to compile is a legal answer and not counted.
#[test]
fn every_lowered_prototype_verifies() {
    let mut checked = 0;
    for entry in fs::read_dir("test-files").unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("lua") {
            continue;
        }
        let Ok(source) = fs::read_to_string(&path) else {
            continue;
        };
        let mut lua = Lua::new();
        lua.load_all();
        lua.enter(|ctx| {
            let Ok(chunk) = ctx.load(&source, Some("t")) else {
                return;
            };
            let Some(closure) = chunk.as_lua() else {
                return;
            };
            for proto in closure.proto.prototypes.iter() {
                // Entry types the compile trigger would have read off the live
                // stack; `any` is the honest stand-in and the widest case.
                let entry = vec![Ty::ANY; proto.num_params as usize];
                if let Ok(func) = lower(*proto, 0, entry) {
                    if let Err(e) = verify::verify(&func) {
                        panic!(
                            "{} prototype failed verification:\n{e}\n--- ir ---\n{}",
                            path.display(),
                            print_func(&func)
                        );
                    }
                    checked += 1;
                }
            }
        });
    }
    assert!(
        checked > 20,
        "only {checked} prototypes lowered; corpus lost?"
    );
}

/// `simplify_params` is meant to be re-run after later passes, so running it on
/// freshly-built IR must be sound and must change nothing: construction already
/// left no redundant parameter behind. A second run that fired would mean the
/// seeded call had missed something; one that broke verification would mean the
/// pass cannot be re-run at all.
#[test]
fn simplify_params_is_idempotent_after_construction() {
    for name in ["mix", "mix2", "primes", "nbody", "jit_loop", "jit_branch"] {
        let source = fs::read_to_string(format!("test-files/{name}.lua")).unwrap();
        let mut lua = Lua::new();
        lua.load_all();
        lua.enter(|ctx| {
            let chunk = ctx.load(&source, Some("t")).expect("compile");
            let inner = chunk.as_lua().unwrap().proto.prototypes[0];
            let entry = vec![Ty::ANY; inner.num_params as usize];
            let Ok(mut func) = lower(inner, 0, entry) else {
                return;
            };
            let before = print_func(&func);
            func.simplify_params();
            let after = print_func(&func);
            assert_eq!(before, after, "{name}: a second run changed the IR");
            if let Err(e) = verify::verify(&func) {
                panic!("{name}: IR failed verification after a second run:\n{e}");
            }
        });
    }
}
