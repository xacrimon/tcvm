//! Globals resolve through the innermost `_ENV` in scope, which may be a
//! local or parameter of the same function. Expected strings from lua 5.5.1.

use crate::common::ok;

#[test]
fn same_function_access() {
    // #194: the closure was right, the direct access read the chunk's `_ENV`.
    assert_eq!(
        ok("local _ENV = {y = 5}; local function g() return y end; return cat(g(), y)"),
        "5 5"
    );
    assert_eq!(
        ok("local t = {}
            do local _ENV = t; x = 1; function f() end; a, b = 2, 3 end
            return cat(t.x, type(t.f), t.a, t.b, x)"),
        "1 function 2 3 nil"
    );
    assert_eq!(
        ok(
            "local _ENV <close> = setmetatable({y = 1}, {__close = function() end})
            return cat(y)"
        ),
        "1"
    );
}

#[test]
fn global_statements() {
    assert_eq!(
        ok(r#"local t = {}
              do local _ENV = t; global function gf() end; global y = 2 end
              return cat(type(t.gf), t.y,
                  pcall(load("local _ENV = {AA = false}; global AA = 10", "=g")))"#),
        "function 2 false g:1: global 'AA' already defined"
    );
}

#[test]
fn parameters() {
    assert_eq!(
        ok("local t = {}; local function f(_ENV) n = 7 end; f(t)
            local function v(..._ENV) global n, x; n = 8; return _ENV, x end
            local e, x = v(1)
            return cat(t.n, e.n, e[1], x, n)"),
        "7 8 1 nil nil"
    );
}

#[test]
fn assigned_alongside_globals() {
    // The other target and the read both use the `_ENV` from before the
    // assignment.
    assert_eq!(
        ok("local t = {x = 5}; local _ENV = t; _ENV, a = {}, x; return cat(t.a, a, x)"),
        "5 nil nil"
    );
    // Also when `_ENV` is an upvalue, in either target order.
    assert_eq!(
        ok("local function run(env_first)
              local old, new = {}, {}
              local _ENV = old
              local function f()
                if env_first then _ENV, a = new, 1 else a, _ENV = 2, new end
              end
              f()
              return old.a, new.a
            end
            local o1, n1 = run(true)
            local o2, n2 = run(false)
            return cat(o1, n1, o2, n2)"),
        "1 nil 2 nil"
    );
}

#[test]
fn constant() {
    // #278: a folded `<const>` `_ENV` is loaded into a register and indexed,
    // here or from a nested function.
    for (src, value) in [
        (
            "local _ENV <const> = nil; local function f() return x end; return f()",
            "nil",
        ),
        ("local _ENV <const> = 11; X = 'hi'", "number"),
        (
            "local _ENV <const> = true; local function f() global function g() end end; f()",
            "boolean",
        ),
        (
            "local _ENV <const> = 1.5; local function f() global y = 1 end; f()",
            "number",
        ),
        (
            "local _ENV <const> = false; local function f() q, r = 3, w end; f()",
            "boolean",
        ),
    ] {
        assert_eq!(
            ok(&format!("return select(2, pcall(load({src:?}, '=c')))")),
            format!("c:1: attempt to index a {value} value"),
            "{src}"
        );
    }
}
