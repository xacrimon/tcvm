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
