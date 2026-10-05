//! Upvalues: open ones follow their slot when the stack moves, and captures
//! of variables nothing reassigns are copied into the closure. Expected
//! values from Lua 5.5.1.

use crate::common::ok;

/// Pushes about 10 slots per level, enough to move a stack of a few hundred.
const DEEP: &str = "local function deep(n) if n == 0 then return 0 end \
    local a, b, c, d, e, f, g, h = 1, 2, 3, 4, 5, 6, 7, 8 return 1 + deep(n - 1) end ";

#[test]
fn open_upvalue_survives_stack_growth() {
    let src = format!(
        "{DEEP} local x = 1
         local get = function() return x end
         local set = function(v) x = v end
         deep(3000)
         set(42)
         local a, b = x, get()
         x = 7
         return cat(a, b, get())"
    );
    assert_eq!(ok(&src), "42 42 7");
}

#[test]
fn open_upvalues_of_many_frames_survive_growth() {
    let src = format!(
        "{DEEP} local fs = {{}}
         local function nest(n)
           local v = n
           fs[#fs + 1] = function(w) if w then v = w end return v end
           if n > 0 then nest(n - 1) else deep(3000) end
         end
         nest(20)
         fs[1](100)
         return cat(fs[1](), fs[11](), fs[21]())"
    );
    assert_eq!(ok(&src), "100 10 0");
}

#[test]
fn coroutine_open_upvalue_survives_its_stack_growing() {
    let src = format!(
        "{DEEP} local co = coroutine.wrap(function()
           local y = 10
           local f = function(v) if v then y = v end return y end
           coroutine.yield(f)
           deep(3000)
           local seen = y
           coroutine.yield(seen)
           return y
         end)
         local f = co()
         f(11)
         local seen = co()
         f(12)
         return cat(seen, co())"
    );
    assert_eq!(ok(&src), "11 12");
}

#[test]
fn assignment_after_capture_is_seen() {
    assert_eq!(
        ok("local x = 1 local f = function() return x end x = 2
            local y = 1 local g = function() return y end
            local function h() return function() y = 7 end end h()()
            return cat(f(), g())"),
        "2 7"
    );
}

#[test]
fn local_function_captures_itself() {
    assert_eq!(
        ok(
            "local function fact(n) if n <= 1 then return 1 end return n * fact(n - 1) end
            local function f() return function() return f end end
            local function r() return r end local first = r r = 42
            return cat(fact(10), f()() == f, first())"
        ),
        "3628800 true 42"
    );
}

#[test]
fn loop_captures_take_each_iteration_value() {
    assert_eq!(
        ok("local a, b, c = {}, {}, {}
            for k, v in ipairs({10, 20, 30}) do a[k] = function() return v end end
            for k, v in ipairs({10, 20, 30}) do b[k] = function() return v end v = v + 1 end
            for i = 1, 3 do local x = i c[i] = function() return x end x = x * 10 end
            return cat(a[2](), b[2](), c[2]())"),
        "20 21 20"
    );
}

#[test]
fn assigned_env_is_shared() {
    assert_eq!(
        ok("local f = load([[
              local f = function() return y end
              local g = function() _ENV = {y = 9} end
              _ENV = {y = 3}
              local a = f()
              g()
              return a, f()
            ]], 'c', 't', {})
            return cat(f())"),
        "3 9"
    );
}
