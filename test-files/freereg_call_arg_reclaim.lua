local function f(v) return v end
local t = {f(1), f(2), f(3)}
return t[1] + t[2] + t[3]
