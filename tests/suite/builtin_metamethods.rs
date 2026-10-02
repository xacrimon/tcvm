//! A builtin that calls Lua through a follow-up sequence (`pcall`) works as a
//! metamethod or generic-for iterator: the sequence's results reach the
//! caller's continuation. Expected strings come from `lua` 5.5.1 on the same
//! chunk, which hands its result to the host with `error(v, 0)`.

use crate::common::err;

/// One case per continuation kind: `__index` stores the first result,
/// `__newindex` drops it, the iterator fills the loop variables and `__lt`
/// branches on it.
#[test]
fn pcall_as_metamethod_and_iterator() {
    let src = r#"
        local out, log = {}, nil
        local t = setmetatable({}, {
          __index = pcall,
          __newindex = pcall,
          __call = function(self, k, v) if v then log = k .. "=" .. v end return k .. "!" end,
        })
        out[#out + 1] = tostring(t.x)
        t.y = 2
        out[#out + 1] = log
        for a, b in pcall, function() return 1 end do out[#out + 1] = tostring(a) .. " " .. tostring(b) break end
        local u = setmetatable({}, {__lt = pcall})
        out[#out + 1] = tostring(u < u)
        error(table.concat(out, ", "), 0)
    "#;
    assert_eq!(err(src), "true, y=2, true 1, false");
}
