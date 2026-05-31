-- Followup to issue #35: nested-const-fold and/or used to leave a dead
-- `TEST/TESTSET; JMP +0` pair (a value-preserving short-circuit jump patched
-- to its own successor). `patch_to_here` now elides such trailing no-op pairs,
-- so e.g. `(a or 5) and b` compiles to a single MOVE. (CMP-controlled jumps
-- are kept — the comparison may run a metamethod.)
local a, b, c = 1, 2, 3

local p = (a or 5) and b
local q = (a and false) or a
local r = (b or true) and a
if (a or 5) and b then
    c = 1
end

return p, q, r, c
