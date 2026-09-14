local f = function() return {} end
local a = {}
(a).b = 1
(f()).x, (a)[1] = 2, 3
local n = 1 + 2
(a).c = n
local m = -a.b ^ 2 + (a)[1] * #a
