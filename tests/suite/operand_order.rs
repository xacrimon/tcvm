//! Assignments and `..` read and write locals in luac's and LuaJIT's order,
//! even when a call or metamethod in the same statement touches them (#275).
//! Expected strings come from `lua` 5.5.1.

use crate::common::ok;

#[test]
fn assignment_reads_a_local_before_later_values() {
    assert_eq!(
        ok("local c = 0
            local function bump() c = 99 return 5 end
            local b
            b, c = c, bump()
            return cat(b, c)"),
        "0 5"
    );
    // The value would otherwise be written into its own target directly.
    assert_eq!(
        ok("local c = 0
            local function bump() c = 99 return 5 end
            local b
            c, b = c, bump()
            return cat(c, b)"),
        "0 5"
    );
    assert_eq!(
        ok("local c = 0
            local t = setmetatable({}, {__index = function() c = 99 return 5 end})
            local b
            b, c = c, t.x
            return cat(b, c)"),
        "0 5"
    );
}

#[test]
fn assignment_writes_a_local_after_later_values() {
    assert_eq!(
        ok("local a = 1
            local function g() return a end
            local b
            a, b = 2, g()
            return cat(a, b)"),
        "2 1"
    );
    // A global in `g` reads the captured `_ENV`.
    assert_eq!(
        ok("local P = cat
            local _ENV = {x = 1}
            local function g() return x end
            local b
            _ENV, b = 5, g()
            return P(b)"),
        "1"
    );
}

#[test]
fn assignment_reads_a_local_before_writing_it() {
    assert_eq!(
        ok("local c, d = 0
            d, c = c, 1
            return cat(d, c)"),
        "0 1"
    );
    assert_eq!(
        ok("local function f(...)
              local c, d = 0
              d, c = c, ...
              return cat(d, c)
            end
            return f(7)"),
        "0 7"
    );
    assert_eq!(
        ok("local y, z = 1, 2
            y, z = z
            return cat(y, z)"),
        "2 nil"
    );
}

#[test]
fn assignment_stores_right_to_left() {
    // A store's `__newindex` sees the locals stored after it unchanged.
    assert_eq!(
        ok("local c, seen = 0
            local t = setmetatable({}, {__newindex = function() seen = c end})
            c, t.x = 1, 2
            return cat(seen, c)"),
        "0 1"
    );
    assert_eq!(
        ok("local c, u = 0, {}
            local t = setmetatable({}, {__newindex = function() c = 99 end})
            t.x, u.y = 2, c
            return cat(u.y, c)"),
        "0 99"
    );
    assert_eq!(
        ok("local x = 1
            x, x = 2, 3
            return cat(x)"),
        "2"
    );
}

#[test]
fn concat_reads_its_left_operand_first() {
    assert_eq!(
        ok(r#"local c = 0
              local function bump() c = 99 return 5 end
              return cat(c .. bump(), c .. "x" .. bump())"#),
        "05 99x5"
    );
    assert_eq!(
        ok("local c = 0
            local t = setmetatable({}, {__index = function() c = 99 return 5 end})
            return cat(c .. t.x)"),
        "05"
    );
}
