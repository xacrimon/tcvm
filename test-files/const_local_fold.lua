-- Same-function `<const>` participation in folding. Each global
-- assignment should compile to a single LOAD K with the folded value,
-- and the const locals own no register.

local k <const> = 5
local m <const> = 1 + 2 * 3      -- RHS folds to 7
local n <const> = true
local p <const> = nil
local s <const> = "hello"

a = k + 3                         -- 8
b = m * 2                         -- 14
c = -k                            -- -5
d = not n                         -- false
e = not p                         -- true
f = k * m + 1                     -- 36 (5*7+1)
g = s                             -- "hello"
