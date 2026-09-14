use std::fs;

use cstree::build::NodeCache;
use insta::assert_snapshot;
use paste::paste;

use super::compile_chunk;
use super::format::format_prototype;
use crate::Lua;
use crate::parser::{self, syntax::Root};

fn compile_and_format(source: &str) -> String {
    let mut cache = NodeCache::new();
    let (syntax_tree, reports) = parser::parse(&mut cache, source);
    assert!(reports.is_empty(), "parse errors: {}", reports.len());
    let root = Root::new(syntax_tree).expect("not a root node");
    let interner = cache.interner();

    let mut lua = Lua::new();
    lua.enter(|ctx| {
        let proto = compile_chunk(ctx, &root, interner).unwrap();
        format_prototype(&proto)
    })
}

fn compile_err_and_format(source: &str) -> String {
    let mut cache = NodeCache::new();
    let (syntax_tree, reports) = parser::parse(&mut cache, source);
    assert!(reports.is_empty(), "parse errors: {}", reports.len());
    let root = Root::new(syntax_tree).expect("not a root node");
    let interner = cache.interner();

    let mut lua = Lua::new();
    lua.enter(|ctx| match compile_chunk(ctx, &root, interner) {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("expected compile error, got success"),
    })
}

macro_rules! test {
    ($name:ident, $path:literal) => {
        paste! {
            #[test]
            fn [<test_compile_ $name>]() {
                let source = fs::read_to_string($path).unwrap();
                let output = compile_and_format(&source);
                assert_snapshot!(output);
            }
        }
    };
}

macro_rules! test_err {
    ($name:ident, $path:literal) => {
        paste! {
            #[test]
            fn [<test_compile_ $name>]() {
                let source = fs::read_to_string($path).unwrap();
                let output = compile_err_and_format(&source);
                assert_snapshot!(output);
            }
        }
    };
}

test!(comment, "test-files/comment.lua");
test!(declare, "test-files/declare.lua");
test!(function, "test-files/function.lua");
test!(hello, "test-files/hello.lua");
test!(if, "test-files/if.lua");
test!(
    issue_65_conditional_return,
    "test-files/issue_65_conditional_return.lua"
);
test!(jens, "test-files/jens.lua");
test!(logic, "test-files/logic.lua");
test!(literal, "test-files/literal.lua");
test!(nbody, "test-files/nbody.lua");
test!(op_prec, "test-files/op_prec.lua");
test!(primes, "test-files/primes.lua");
test!(jens2, "test-files/jens2.lua");
test!(nested_call, "test-files/nested_call.lua");
test!(freereg_chain, "test-files/freereg_chain.lua");
test!(
    freereg_call_arg_reclaim,
    "test-files/freereg_call_arg_reclaim.lua"
);
test!(freereg_nested_call, "test-files/freereg_nested_call.lua");
test!(
    freereg_discarded_call,
    "test-files/freereg_discarded_call.lua"
);
test!(freereg_and_call, "test-files/freereg_and_call.lua");
test!(
    freereg_for_num_temps,
    "test-files/freereg_for_num_temps.lua"
);
test!(multi_assign_swap, "test-files/multi_assign_swap.lua");
test!(multi_assign_cycle, "test-files/multi_assign_cycle.lua");
test!(
    multi_assign_index_conflict,
    "test-files/multi_assign_index_conflict.lua"
);
test!(
    multi_assign_property_conflict,
    "test-files/multi_assign_property_conflict.lua"
);
test!(
    multi_assign_no_conflict,
    "test-files/multi_assign_no_conflict.lua"
);
test!(
    short_circuit_local_clobber,
    "test-files/short_circuit_local_clobber.lua"
);
test!(global_decl, "test-files/global_decl.lua");
test!(global_star, "test-files/global_star.lua");
test!(errnnil_runtime, "test-files/errnnil_runtime.lua");
test_err!(
    global_const_assign_err,
    "test-files/global_const_assign_err.lua"
);
test_err!(
    global_undeclared_err,
    "test-files/global_undeclared_err.lua"
);
test_err!(
    for_counter_readonly_err,
    "test-files/for_counter_readonly_err.lua"
);
test_err!(
    global_nested_propagation,
    "test-files/global_nested_propagation.lua"
);
test_err!(global_star_nested, "test-files/global_star_nested.lua");
test!(global_const_star, "test-files/global_const_star.lua");
test!(global_shadows_local, "test-files/global_shadows_local.lua");
test_err!(global_self_init_err, "test-files/global_self_init_err.lua");
test_err!(multiple_close_err, "test-files/multiple_close_err.lua");
test!(global_init_shadow, "test-files/global_init_shadow.lua");
test!(const_fold, "test-files/const_fold.lua");
test!(const_no_fold, "test-files/const_no_fold.lua");
test!(const_fold_branch, "test-files/const_fold_branch.lua");
test!(op_prec_runtime, "test-files/op_prec_runtime.lua");
test!(const_local_fold, "test-files/const_local_fold.lua");
test!(const_local_no_fold, "test-files/const_local_no_fold.lua");
test!(
    const_local_outer_fold,
    "test-files/const_local_outer_fold.lua"
);
test!(
    const_local_multi_level,
    "test-files/const_local_multi_level.lua"
);
test!(method_call, "test-files/method_call.lua");
test!(method_def, "test-files/method_def.lua");
test!(loop_close, "test-files/loop_close.lua");
test!(goto_close, "test-files/goto_close.lua");
test!(not_andor, "test-files/not_andor.lua");
test!(jmp_elim, "test-files/jmp_elim.lua");
test!(if_empty_nested, "test-files/if_empty_nested.lua");
test!(if_empty_goto, "test-files/if_empty_goto.lua");
test!(paren_adjust, "test-files/paren_adjust.lua");
test!(paren_prefix, "test-files/paren_prefix.lua");
test!(call_sugar, "test-files/call_sugar.lua");
test!(
    global_multiret_expand,
    "test-files/global_multiret_expand.lua"
);
// A named vararg *following* named parameters. Nothing compiled this file, and
// the combination tripped an assertion in `adjust_locals`.
test!(vararg_param, "test-files/vararg_param.lua");

/// Register-ceiling shapes that used to wrap a `u8` and panic the VM (#11).
/// Each must surface as a compile error instead.
#[test]
fn test_register_limit_is_a_compile_error() {
    let list = |n: usize| (0..n).map(|i| i.to_string()).collect::<Vec<_>>().join(", ");
    let names = (0..255)
        .map(|i| format!("v{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sources = [
        // func at R1 plus 254 args: the last lands on R255.
        format!("local function f(...) end f({})", list(254)),
        // 255 return values: `count = n + 1` wraps to MULTRET.
        format!("local function f() return {} end", list(255)),
        // 255 targets from one call: `returns = n + 1` wraps.
        format!("local function g() end local {names} = g()"),
        // Results reserved past R255.
        format!("local {names} local function g() end {names} = g()"),
    ];
    for src in sources {
        assert_eq!(
            compile_err_and_format(&src),
            "compiler error at line 1: insufficient available registers"
        );
    }
}
