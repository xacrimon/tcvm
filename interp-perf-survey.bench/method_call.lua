local Class = {}
Class.__index = Class
function Class.get(s) return s.w end
local obj = setmetatable({w = 2, h = 3, id = 7}, Class)
local acc = 0
for i = 1, 10000000 do acc = acc + obj:get() end
print(acc)
