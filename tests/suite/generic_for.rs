//! Execution coverage for the generic `for ... in ... do` loop: the control
//! register layout (TFORCALL/TFORLOOP) and multi-value iterator adjustment.

use crate::common::{err, eval};

fn run(src: &str) -> i64 {
    eval(src)
}

#[test]
fn explicit_three_value_iterator() {
    // Explicit `f, s, control` form — exercises the TFORCALL/TFORLOOP layout.
    assert_eq!(
        run(
            "local function iter(_, c) if c < 3 then return c + 1 end end\n\
             local sum = 0\n\
             for x in iter, nil, 0 do sum = sum + x end\n\
             return sum"
        ),
        6
    );
}

#[test]
fn multi_value_iterator_call() {
    // A single call returning (iter, state, control) must spread into the
    // three control slots (the multires adjustment).
    assert_eq!(
        run(
            "local function iter(_, c) if c < 3 then return c + 1 end end\n\
             local function mk() return iter, nil, 0 end\n\
             local sum = 0\n\
             for x in mk() do sum = sum + x end\n\
             return sum"
        ),
        6
    );
}

#[test]
fn two_loop_variables() {
    // Iterator returning two values per step (key-like, value-like).
    assert_eq!(
        run("local function iter(_, c)\n\
             \x20 if c < 3 then return c + 1, (c + 1) * 10 end\n\
             end\n\
             local sum = 0\n\
             for k, v in iter, nil, 0 do sum = sum + k + v end\n\
             return sum"),
        // k: 1+2+3=6, v: 10+20+30=60
        66
    );
}

#[test]
fn empty_iteration() {
    // Iterator returns nil immediately: body never runs.
    assert_eq!(
        run("local function iter() return nil end\n\
             local n = 0\n\
             for x in iter, nil, 0 do n = n + 1 end\n\
             return n"),
        0
    );
}

#[test]
fn pairs_style_over_table() {
    // Hand-rolled stateful iterator over an array-like table, returned as a
    // single multi-value call — the realistic `for k,v in pairs(t)` shape.
    assert_eq!(
        run("local t = { 10, 20, 30, 40 }\n\
             local function inext(tbl, i)\n\
             \x20 i = i + 1\n\
             \x20 local v = tbl[i]\n\
             \x20 if v ~= nil then return i, v end\n\
             end\n\
             local function each(tbl) return inext, tbl, 0 end\n\
             local sum = 0\n\
             for i, v in each(t) do sum = sum + i + v end\n\
             return sum"),
        // i: 1+2+3+4=10, v: 10+20+30+40=100
        110
    );
}

#[test]
fn iterator_call_line() {
    // Calling the iterator is on the line the explist starts, past comments
    // and into parentheses, not the `for` line (#253). Lines from lua 5.5.1,
    // which also appends `(for iterator 'for iterator')` (#204).
    assert_eq!(
        err("\n\n for k,v in \n 3 \n do \n end"),
        "c:4: attempt to call a number value"
    );
    assert_eq!(
        err("for k in\n-- c\n--[[\n]]\n 3, 4 do end"),
        "c:5: attempt to call a number value"
    );
    assert_eq!(
        err("for k in\n(\n 3) do end"),
        "c:2: attempt to call a number value"
    );
}
