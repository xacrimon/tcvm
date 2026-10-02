//! A function that tail-calls a native which suspends (`coroutine.yield`,
//! `pcall`) hands its caller's expectations to the suspension: the
//! metamethod/iterator continuation it carried, and its own function slot,
//! which a vararg frame keeps below its extra arguments. Expected strings
//! come from `lua` 5.5.1 on the same chunks; each chunk hands its result to
//! the host with `error(v, 0)`.

use crate::common::err;

#[test]
fn comparison_metamethod_yields_its_result() {
    let src = "local y = setmetatable({}, { __lt = function() return coroutine.yield() end }) \
               local function cmp(v) \
                 local co = coroutine.wrap(function() local r = y < y return r end) \
                 co() return tostring(co(v)) end \
               local function iff(v) \
                 local co = coroutine.wrap(function() if y < y then return 'then' end return 'else' end) \
                 co() return co(v) end \
               error(table.concat({ cmp(false), cmp(true), cmp(nil), cmp(0), iff(false), iff(1) }, ' '), 0)";
    assert_eq!(err(src), "false true false true else then");
}

#[test]
fn index_metamethod_tail_calls_yield_and_pcall() {
    let src = "local t = setmetatable({}, { __index = function(t, k) return coroutine.yield(k) end }) \
               local co = coroutine.wrap(function() local r = t.key return r .. ' after' end) \
               co() \
               local p = setmetatable({}, { __index = function(t, k) return pcall(function() return k end) end }) \
               error(co('val') .. ' ' .. tostring(p.x), 0)";
    assert_eq!(err(src), "val after true");
}

#[test]
fn iterator_tail_calls_yield() {
    let src = "local co = coroutine.wrap(function() \
                 local got = {} \
                 for a, b in function() return coroutine.yield() end do \
                   got[#got + 1] = a .. b if #got == 2 then break end \
                 end \
                 return table.concat(got, ',') end) \
               co() co('p', 'q') error(co('r', 's'), 0)";
    assert_eq!(err(src), "pq,rs");
}

#[test]
fn vararg_function_tail_calls_yield() {
    let src = "local function v(...) return coroutine.yield() end \
               local co = coroutine.wrap(function() local a, b = v(1, 2, 3) return a .. b end) \
               co() error(co('a', 'b'), 0)";
    assert_eq!(err(src), "ab");
}
