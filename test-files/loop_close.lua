local t = {}
for i = 1, 3 do t[i] = function() return i end end
local k = 0
while k < 3 do k = k + 1; local x = k; t[k] = function() return x end end
repeat local y = k; t[k] = function() return y end; k = k - 1 until y <= 1
for _, v in ipairs(t) do t[v] = function() return v end end
do local z = 1; t.f = function() return z end end
for i = 1, 3 do t[i] = function() return i end if i == 2 then break end end
