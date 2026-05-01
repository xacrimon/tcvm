-- `<const>` declared in an outer scope is inlined inside an inner
-- function. The inner function must NOT register an upvalue for the
-- const-only references — its prototype's upvalues list should be empty.
-- Compare with luac: function f reports `0 upvalues` for the same source.

local k <const> = 5
local m <const> = 3.14

local function f(x)
    return k + 3, m * 2, k * x
end

return f
