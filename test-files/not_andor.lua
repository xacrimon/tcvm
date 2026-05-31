-- Regression for issue #35: `not` of a parenthesized and/or must carry
-- (downgrade + swap) the operand's short-circuit jump lists, matching
-- Lua's `codenot`. Before the fix these emitted an unpatched
-- `TESTSET <NO_REG>` / `JMP +0` with no boolean-materialization tail.
local a, b, c = 1, 2, 3

-- value context
local p = not (a and b)
local q = not (a or b)
local r = not (a and b and c)
local s = not (a or b or c)
local t = not (a and 5)

-- branch context
if not (a and b) then
    return 0
end

return p, q, r, s, t
