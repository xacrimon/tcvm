//! A frame a tail-called native takes over (its action converts the frame
//! into the native's) loses its vararg count, which neither the `xpcall`
//! handler nor a RETURN's `__close` continuation may need. Expected values
//! from Lua 5.5.1.

use crate::common::ok;

#[test]
fn xpcall_handler_survives_vararg_frame_taken_by_wrap() {
    let src = "local function h(e) return 'handled' end
        local ok, msg = xpcall(function(...)
          return coroutine.wrap(function() error('x') end)()
        end, h, 1)
        return cat(ok, msg)";
    assert_eq!(ok(src), "false handled");
}

#[test]
fn close_of_vararg_metamethod_yielding_from_a_tail_call() {
    let src = "local log = {}
        local co = coroutine.wrap(function()
          do
            local x <close> = setmetatable({}, {
              __close = function(...)
                log[#log + 1] = select('#', ...)
                return coroutine.yield('in close')
              end,
            })
            return 'ret'
          end
        end)
        local a = co()
        local b = co()
        return cat(a, b, log[1])";
    assert_eq!(ok(src), "in close ret 1");
}

#[test]
fn tail_called_pcall_shadows_the_xpcall_handler() {
    let src = "local function h(e) return 'handled' end
        local a, b, c = xpcall(function() return pcall(error, 'x', 0) end, h)
        local d, e, f = xpcall(function(...) return pcall(error, 'y', 0) end, h, 1)
        return cat(a, b, c, d, e, f)";
    assert_eq!(ok(src), "true false x true false y");
}
