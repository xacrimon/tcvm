local function f() end
f(); f()
local x, y = 1, 2
if not x then return 1 end
return x + y
