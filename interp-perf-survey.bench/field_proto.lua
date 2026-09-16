local Class = {}
Class.__index = Class
function Class.area(s) return s.w * s.h end
local obj = setmetatable({w = 2, h = 3, id = 7}, Class)
local acc = 0
for i = 1, 30000000 do local f = obj.area; if f then acc = acc + 1 end end
print(acc)
