use std::fs;

use cstree::build::NodeCache;
use insta::assert_snapshot;
use paste::paste;

use crate::dmm::{Arena, Static};
use crate::parser::{self, syntax::Root};

use super::compile_chunk;
use super::format::format_prototype;

fn compile_and_format(source: &str) -> String {
    let mut cache = NodeCache::new();
    let (syntax_tree, reports) = parser::parse(&mut cache, source);
    assert!(reports.is_empty(), "parse errors: {}", reports.len());
    let root = Root::new(syntax_tree).expect("not a root node");
    let interner = cache.interner();

    let arena = Arena::<Static<()>>::new(|_mc| Static(()));
    arena.mutate(|mc, _| {
        let proto = compile_chunk(mc, &root, interner).unwrap();
        format_prototype(&proto)
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

test!(comment, "test-files/comment.lua");
test!(declare, "test-files/declare.lua");
test!(function, "test-files/function.lua");
test!(hello, "test-files/hello.lua");
test!(if, "test-files/if.lua");
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
