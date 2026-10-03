//! A goto reaches only the labels visible from it: those of its own block
//! and the enclosing ones, never a closed or nested block's. Expected
//! strings from lua 5.5.1.

use crate::common::ok;

#[test]
fn closed_block_label_is_invisible() {
    // #232: the goto used to jump back into the first `do` block.
    assert_eq!(
        ok("local n = 0
            do ::l1:: n = n + 1; if n > 3 then error('back in a closed block') end end
            do goto l1; ::l1:: end
            return cat(n)"),
        "1"
    );
}

#[test]
fn nested_block_label_is_invisible() {
    // #232: the goto used to bind the `else` block's label.
    assert_eq!(
        ok("local function f(a)
              if a == 1 then goto l1 else ::l1:: return 'inner' end
              ::l1:: return 'outer'
            end
            return cat(f(1), f(2))"),
        "outer inner"
    );
    // A pending goto isn't resolved by a nested block's label, even one a
    // goto inside that block resolves to.
    assert_eq!(
        ok("goto x
            do goto x; ::x:: end
            do return 'nested' end
            ::x:: return cat('outer')"),
        "outer"
    );
}

#[test]
fn enclosing_block_labels() {
    // Forward out of two blocks, closing a captured local on the way.
    assert_eq!(
        ok("local fs = {}
            for i = 1, 2 do
              do local v = i; fs[i] = function() return v end; if i == 1 then goto cont end end
              ::cont::
            end
            return cat(fs[1](), fs[2]())"),
        "1 2"
    );
    // Backward out of nested blocks.
    assert_eq!(
        ok("local k = 0
            ::top::
            k = k + 1
            do do if k < 3 then goto top end end end
            return cat(k)"),
        "3"
    );
}
