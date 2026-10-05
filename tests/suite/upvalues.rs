//! Upvalues: open ones follow their slot when the stack moves. Expected
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
