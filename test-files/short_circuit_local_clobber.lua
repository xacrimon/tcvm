local a = false
local b = 5
local x = a and b

local c = 7
local d = 5
local y = c or d

local e = 7
local z = (e == 5) and e

return b, d, e, x, y, z
