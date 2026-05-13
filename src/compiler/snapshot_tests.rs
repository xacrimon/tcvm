use std::fs;

use cstree::build::NodeCache;
use insta::assert_snapshot;
use paste::paste;

use crate::Lua;
use crate::parser::{self, syntax::Root};

use super::compile_chunk;
use super::format::format_prototype;

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
test!(
    global_nested_propagation,
    "test-files/global_nested_propagation.lua"
);
test!(global_star_nested, "test-files/global_star_nested.lua");
test!(global_const_star, "test-files/global_const_star.lua");
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
