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

/// `load`'s error for `src`, or `nil` if it loads.
fn load_err(src: &str) -> String {
    ok(&format!("return cat((select(2, load({src:?}, '=c'))))"))
}

#[test]
fn compile_errors() {
    // #222: these all used to load (`goto nowhere` then looped forever).
    for (src, msg) in [
        (
            "goto nowhere",
            "no visible label 'nowhere' for <goto> at line 1",
        ),
        (
            "do goto l1 end",
            "no visible label 'l1' for <goto> at line 1",
        ),
        (
            "goto l1; do ::l1:: end",
            "no visible label 'l1' for <goto> at line 1",
        ),
        (
            "goto f; local x; ::f:: print(x)",
            "<goto f> at line 1 jumps into the scope of 'x'",
        ),
        (
            "goto f; do local x; ::f:: end",
            "no visible label 'f' for <goto> at line 1",
        ),
        (
            "local a; do goto f end; local x; ::f:: print(x)",
            "<goto f> at line 1 jumps into the scope of 'x'",
        ),
        (
            "goto f; local x <const> = 1; ::f:: print(x)",
            "<goto f> at line 1 jumps into the scope of 'x'",
        ),
        (
            "goto f; global y; ::f:: y = 1",
            "<goto f> at line 1 jumps into the scope of 'y'",
        ),
        (
            "goto f; global *; ::f:: y = 1",
            "<goto f> at line 1 jumps into the scope of '*'",
        ),
        // `until` doesn't end the block: its condition sees `x`.
        (
            "repeat goto f; local x; ::f:: until x",
            "<goto f> at line 1 jumps into the scope of 'x'",
        ),
        ("::a:: ::a::", "label 'a' already defined on line 1"),
        ("::a:: do ::a:: end", "label 'a' already defined on line 1"),
    ] {
        assert_eq!(load_err(src), format!("c:1: {msg}"), "{src}");
    }
}

#[test]
fn error_lines() {
    // Not luac's lines, which are wherever its single pass noticed: an
    // undefined goto at its own line, a jump at the label, a duplicate at
    // the later label.
    assert_eq!(
        load_err("\n\ngoto nowhere\n\nlocal y\n\n"),
        "c:3: no visible label 'nowhere' for <goto> at line 3"
    );
    assert_eq!(
        load_err("goto f\nlocal x\n::f::\nprint(x)\n"),
        "c:3: <goto f> at line 1 jumps into the scope of 'x'"
    );
    assert_eq!(
        load_err("::a::\n;\n::b::\n::a::"),
        "c:4: label 'a' already defined on line 1"
    );
}

#[test]
fn allowed_jumps() {
    // A label that ends its block (only labels and `;` follow) is past the
    // block's locals, and labels in closed blocks or other functions don't
    // clash.
    for src in [
        "do goto f; local x; ::f:: end",
        "do goto f; local x; ::f:: ; ::g:: ; end",
        "for i = 1, 2 do goto continue; local x = i; ::continue:: end",
        "do goto f; local x; ::f:: end ::f::",
        "::a:: do ::b:: end ::b:: local function f() ::a:: end",
    ] {
        assert_eq!(load_err(src), "nil", "{src}");
    }
}
