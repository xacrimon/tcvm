//! `load`. Expected strings come from `lua` 5.5.1 running the same chunk,
//! except where tcvm diverges on purpose: syntax errors are rendered ariadne
//! reports, and only UTF-8 text chunks load.

use crate::common::ok;

#[test]
fn string_chunk() {
    assert_eq!(
        ok(
            r#"local f = load("local a, b = ... return x, a + b, _ENV ~= nil", "=env", "t", {x = "env"})
               return cat(f(1, 2))"#
        ),
        "env 3 true"
    );
    // An explicit nil `env` still replaces `_ENV`.
    assert_eq!(
        ok(r#"return cat(load("return _ENV", "=e", "t", nil)())"#),
        "nil"
    );
    assert_eq!(ok(r#"return cat(load("return x")())"#), "nil");
}

#[test]
fn reader_function() {
    // Pieces are concatenated (numbers converted) up to the empty string.
    assert_eq!(
        ok(r#"local parts, i = {"return ", 4, "2", "", "error()"}, 0
               return cat(load(function() i = i + 1 return parts[i] end)())"#),
        "42"
    );
    assert_eq!(
        ok("return cat(load(function() return true end))"),
        "nil c:1: reader function must return a string"
    );
    assert_eq!(
        ok(r#"return cat(load(function() error("boom", 0) end))"#),
        "nil boom"
    );
    // An unnamed reader chunk is `=(load)`.
    assert_eq!(
        ok(r#"local done
              return cat(load(function()
                  if done then return nil end
                  done = true
                  return "local x <const> = 1\nx = 2"
              end))"#),
        "nil (load):2: attempt to assign to const variable 'x'"
    );
}

#[test]
fn modes() {
    // A mode without `t` is rejected up front: the reader never runs.
    assert_eq!(
        ok(r#"return cat(pcall(load, "return 1", "=m", "b"))"#),
        "false bad argument #3 to 'load' (binary chunks are not supported)"
    );
    assert_eq!(
        ok(r#"local n = 0
               local ok = pcall(load, function() n = n + 1 end, "=m", "b")
               return cat(ok, n)"#),
        "false 0"
    );
    assert_eq!(ok(r#"return cat(load("return 1", "=m", "bt")())"#), "1");
    assert_eq!(
        ok(r#"return cat(pcall(load, "return 1", nil, "tB"))"#),
        "false bad argument #3 to 'load' (invalid mode)"
    );
}

#[test]
fn bad_arguments() {
    assert_eq!(
        ok("return cat(pcall(load, {}))"),
        "false bad argument #1 to 'load' (function expected, got table)"
    );
    assert_eq!(
        ok(r#"return cat(pcall(load, "x", {}))"#),
        "false bad argument #2 to 'load' (string expected, got table)"
    );
}

#[test]
fn errors() {
    assert_eq!(
        ok(r#"return cat(load("local x <const> = 1\nx = 2"))"#),
        "nil [string \"local x <const> = 1...\"]:2: attempt to assign to const variable 'x'"
    );
    assert_eq!(
        ok(r#"return cat(load("x = = 1"))"#),
        "nil Error: expected a statement\n   \
         ╭─[ [string \"x = = 1\"]:1:5 ]\n   \
         │\n \
         1 │ x = = 1\n   \
         │     ┬  \n   \
         │     ╰── expected a statement but got \"=\"\n\
         ───╯"
    );
    assert_eq!(
        ok(r#"return cat(load("return '\255'"))"#),
        "nil [string \"return '\u{FFFD}'\"]: chunk is not valid UTF-8"
    );
}

#[test]
fn comments_and_long_brackets() {
    // A comment at the end of the chunk, CRLF or a lone CR after a comment,
    // leveled and non-ASCII long brackets (#233).
    assert_eq!(
        ok(r#"return cat(type(load("--c")), type(load("x = 1 --c")),
              load("--ab\r\nreturn 1")(), load("--\r\nreturn 2")(), load("--\rreturn 3")(),
              load("--x\n\rreturn 4")(), load("--[=[ a\n]] b ]=] return 5")(),
              load("--[==[\n]=]\n]==] return 6")(), load("--[= x\nreturn 7")(),
              load("--[\nreturn 8")(), load("--[[aé]]return 9")(), load("return #[[aé]]")(),
              load("return [=[x]]y]=]")())"#),
        "function function 1 2 3 4 5 6 7 8 9 3 x]]y"
    );
    // Unfinished long brackets are syntax errors.
    assert_eq!(
        ok(r#"return cat((load("return 1 --[[")), (load("return [==[x]=]")))"#),
        "nil nil"
    );
}

#[test]
fn readonly_variables() {
    // `<close>` variables and named vararg parameters are read-only like
    // `<const>` ones (#242, #247), also through upvalues, and so is a
    // `<const>` whose value isn't a compile-time constant. Of a generic
    // for's variables, only the first is.
    for (src, name) in [
        ("return function (... t) t = 10 end", "t"),
        (
            "local function f(...e) return function () return function () e = 1 end end end",
            "e",
        ),
        ("local x <close> = nil; x = 1", "x"),
        ("local x <close> = nil; local function f() x = 1 end", "x"),
        ("local k <const> = f(); local function g() k = 1 end", "k"),
        (
            "local k <const> = f(); local function g() local a = k; return function() k = 1 end end",
            "k",
        ),
        (
            "for i, v in pairs({}) do local function f() i = 1 end end",
            "i",
        ),
        (
            "local k <const> = f(); local function g() function k() end end",
            "k",
        ),
    ] {
        assert_eq!(
            ok(&format!("return select(2, load({src:?}, '=c'))")),
            format!("c:1: attempt to assign to const variable '{name}'"),
            "{src}"
        );
    }
    for src in [
        "local function f(...t) t[1] = 5; t.n = 1; return function() t[2] = 1 end end",
        "local x <close> = nil; local function f() return x end",
        "local k <const> = f(); local function g() local k = 1; return function() k = 2 end end",
        "for i, v in pairs({}) do v = 1; local function f() v = 2 end end",
    ] {
        assert_eq!(
            ok(&format!("return type(load({src:?}))")),
            "function",
            "{src}"
        );
    }
}

#[test]
fn global_env() {
    // A global access while `global _ENV` is in scope, including from a
    // nested function (#235). Expected messages from lua 5.5.1.
    for (src, name) in [
        ("global _ENV, a; a = 10", "a"),
        ("global _ENV; return _ENV", "_ENV"),
        ("global *; global _ENV; return w", "w"),
        (
            "global *; global _ENV; local function f() return w end",
            "w",
        ),
        (
            "global *; local function f() local a = q; do global _ENV; return w end end",
            "w",
        ),
        ("global *; global _ENV; global function gf() end", "gf"),
    ] {
        assert_eq!(
            ok(&format!("return select(2, load({src:?}, '=c'))")),
            format!("c:1: _ENV is global when accessing variable '{name}'"),
            "{src}"
        );
    }
    // The undeclared check comes first, and a declaration's initializers
    // run before it takes effect.
    assert_eq!(
        ok("return select(2, load('global _ENV; print(1)', '=c'))"),
        "c:1: variable 'print' not declared"
    );
    assert_eq!(
        ok(
            "return cat(type(load('global _ENV; return 1')), type(load('global *; global _ENV, b = 1, 2')))"
        ),
        "function function"
    );
}

#[test]
fn function_statement_line() {
    // A `function` or `global function` statement's store, and its error,
    // are on its first line, not the body's `end` (#236). Expected from lua
    // 5.5.1.
    assert_eq!(
        ok(
            r#"return select(2, load("local foo <const> = 1\nfunction foo (x)\n  return\nend\n", "=c"))"#
        ),
        "c:2: attempt to assign to const variable 'foo'"
    );
    assert_eq!(
        ok(
            r#"return select(2, load("global foo <const>\nfunction foo (x)\n  return\nend\n", "=c"))"#
        ),
        "c:2: attempt to assign to const variable 'foo'"
    );
    assert_eq!(
        ok(r#"local mt = {__newindex = function() error('ni', 2) end}
              return cat(pcall(load("local t = setmetatable({}, ...)\nfunction t.m ()\n  return\nend\n", "=c"), mt))"#),
        "false c:2: ni"
    );
    assert_eq!(
        ok(
            r#"f = 1 return select(2, pcall(load("local x = 1\nglobal function f ()\n  return\nend\n", "=c")))"#
        ),
        "c:2: global 'f' already defined"
    );
}
