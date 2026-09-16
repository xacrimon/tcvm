local Base = {} Base.__index = Base
function Base.area(s) return s.w * s.h end
local Derived = setmetatable({}, Base) Derived.__index = Derived
function Derived.perim(s) return 2*(s.w + s.h) end
local obj = setmetatable({w = 2, h = 3, id = 7}, Derived)
local acc = 0
for i = 1, 30000000 do local f = obj.area; if f then acc = acc + 1 end end
print(acc)
