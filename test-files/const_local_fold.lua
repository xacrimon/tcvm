-- Same-function `<const>` participation in folding. Each global
-- assignment should compile to a single LOAD K with the folded value;
-- the const local's register slot is still allocated but the value
-- never has to be loaded from it.

local k <const> = 5
local m <const> = 1 + 2 * 3      -- RHS folds to 7
local n <const> = true
local p <const> = nil

a = k + 3                         -- 8
b = m * 2                         -- 14
c = -k                            -- -5
d = not n                         -- false
e = not p                         -- true
f = k * m + 1                     -- 36 (5*7+1)
