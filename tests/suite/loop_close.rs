//! Leaving a scope whose locals were captured must CLOSE them, so each loop
//! iteration's closure gets a fresh upvalue and a later local reusing the
//! register doesn't alias an open one. Regression for #12.

use crate::common::{eval, ok};

fn run(src: &str) -> i64 {
    eval(src)
}

#[test]
fn numeric_for_counter() {
    assert_eq!(
        run("local t = {}\n\
             for i = 1, 3 do t[i] = function() return i end end\n\
             return tonumber(t[1]() .. t[2]() .. t[3]())"),
        123
    );
}

#[test]
fn generic_for_variables() {
    assert_eq!(
        run("local t = {}\n\
             for k, v in ipairs({7, 8}) do t[k] = function() return k * 10 + v end end\n\
             return t[1]() * 100 + t[2]()"),
        1728
    );
}

#[test]
fn body_local_is_per_iteration_and_mutable() {
    assert_eq!(
        run("local fns = {}\n\
             for i = 1, 2 do\n\
               local x = i * 10\n\
               fns[i] = function() x = x + 1; return x end\n\
             end\n\
             return tonumber(fns[1]() .. fns[1]() .. fns[2]())"),
        111221
    );
}

#[test]
fn while_and_repeat() {
    assert_eq!(
        run("local w, r, k = {}, {}, 0\n\
             while k < 3 do k = k + 1; local x = k; w[k] = function() return x end end\n\
             repeat local y = k; r[k] = function() return y end; k = k - 1 until y <= 1\n\
             return tonumber(w[1]() .. w[2]() .. w[3]() .. r[3]() .. r[2]() .. r[1]())"),
        123321
    );
}

#[test]
fn break_closes_before_leaving() {
    // A later local reuses the counter's register; the closure must not
    // see it through a still-open upvalue.
    assert_eq!(
        run("local fns = {}\n\
             for i = 1, 5 do\n\
               fns[i] = function() return i end\n\
               if i == 2 then break end\n\
             end\n\
             local a, b, c, d, e = 9, 9, 9, 9, 9\n\
             return fns[1]() * 10 + fns[2]()"),
        12
    );
}

#[test]
fn do_block_closes() {
    assert_eq!(
        run("local f\n\
             do local x = 1; f = function() return x end end\n\
             local y = 2\n\
             return f() * 10 + y"),
        12
    );
}

/// Upvalues captured out of slot order, then closed partway: a `CLOSE` must
/// close exactly the ones at or above its level, and a later capture must
/// find the one still open (`ThreadState::open_upvalues` is sorted by slot).
#[test]
fn upvalues_captured_out_of_order() {
    let src = r#"
local fs = {}
do
  local a = 1
  do
    local b = 2
    fs[1] = function() return b end
    fs[2] = function() return a end
    b = 20
  end
  local c = 99
  a = 10
  fs[3] = function() a = a + 1 return a end
  a = 100
end
local co = coroutine.wrap(function(x)
  local gs = {}
  for i = 3, 1, -1 do local v = x * i gs[i] = function() return v end end
  coroutine.yield(gs[1]() + gs[2]() + gs[3]())
  return gs[3]()
end)
return fs[1]() .. ' ' .. fs[2]() .. ' ' .. fs[3]() .. ' ' .. fs[2]() .. ' ' .. co(5) .. ' ' .. co()
"#;
    assert_eq!(ok(src), "20 100 101 101 30 15");
}
