//! `math.min` and `math.max` order their arguments with `<`, `__lt` included
//! (#313). The expected string is `lua` 5.5.1's result for the same chunk.

use crate::common::{err, ok};

/// Each probe's pick and the `__lt` calls it made, in order: the first of
/// equals stays, the method may come from either operand, its first result
/// decides by truthiness, errors are the `<` operator's, and the method is
/// looked up again for every comparison.
#[test]
fn lt_metamethod() {
    let src = r##"
        local out, log = {}, {}
        local mt = {__lt = function(a, b)
          local x = type(a) == "table" and a.n or a
          local y = type(b) == "table" and b.n or b
          log[#log + 1] = x .. "<" .. y
          return x < y
        end}
        local function o(n) return setmetatable({n = n}, mt) end
        local function show(v) return type(v) == "table" and "o" .. tostring(v.n) or tostring(v) end
        local function probe(f, ...)
          log = {}
          local ok, r = pcall(f, ...)
          out[#out + 1] = (ok and show(r) or r) .. "[" .. table.concat(log, ",") .. "]"
        end
        probe(math.min, o(2), o(1), o(3))
        probe(math.max, o(2), o(1), o(3))
        local e1, e2 = o(5), o(5)
        out[#out + 1] = tostring(math.min(e1, e2) == e1 and math.max(e1, e2) == e1)
        probe(math.min, 4, o(2))
        probe(math.max, 1, 2, 3, 4, 5, o(9), 6, o(0))
        probe(math.min, 3, 4, o(2), 1, 5.5, o(0), 7)
        local tmt = {__lt = function(a, b) return a.r, true end}
        local p, q = setmetatable({r = 0}, tmt), setmetatable({r = false}, tmt)
        local z = setmetatable({}, {__lt = function() end})
        out[#out + 1] = tostring(math.min(q, p) == p and math.max(p, q) == q and math.min(z, z) == z)
        probe(math.min, {}, {})
        probe(math.min, 1, {})
        probe(math.max, setmetatable({}, {__name = "Point"}), setmetatable({}, {__name = "Point"}))
        probe(math.min, setmetatable({}, {__lt = function() error("boom", 0) end}), {})
        local cm = {}
        cm.__lt = function() cm.__lt = nil return true end
        probe(math.min, setmetatable({n = 1}, cm), setmetatable({n = 2}, cm), setmetatable({n = 3}, cm))
        local gm = {__lt = function(a, b) collectgarbage() return a.n < b.n end}
        local args = {}
        for i = 1, 200 do args[i] = setmetatable({n = (i * 7919) % 1000}, gm) end
        out[#out + 1] = math.min(table.unpack(args)).n .. "/" .. math.max(table.unpack(args)).n
        local function tail(...) return math.min(...) end
        out[#out + 1] = show(tail(o(3), o(1), o(2))) .. select("#", math.max(o(1), o(2)))
        return table.concat(out, " ")
    "##;
    assert_eq!(
        ok(src),
        "o1[1<2,3<1] o3[2<1,2<3] true o2[2<4] o9[5<9,9<6,9<0] o0[2<3,1<2,0<1,7<0] true \
         attempt to compare two table values[] attempt to compare table with number[] \
         attempt to compare two Point values[] boom[] attempt to compare two table values[] \
         3/987 o11"
    );
}

/// The compare error is raised in the builtin, so it has no position, as
/// in `table.sort`.
#[test]
fn compare_error_position() {
    let src = "local ok, e = pcall(function() return math.max(1, {}) end) error(e, 0)";
    assert_eq!(err(src), "attempt to compare number with table");
}
