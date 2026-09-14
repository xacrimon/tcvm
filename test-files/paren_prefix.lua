local f = function() return {} end;
local a = {};
(a).b = 1;
(f()).x, (a)[1] = 2, 3;
local n = (a).b + (f())[1]
