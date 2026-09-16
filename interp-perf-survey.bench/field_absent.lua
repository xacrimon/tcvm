local p = {x = 1, y = 2}
local acc = 0
for i = 1, 30000000 do if p.z == nil then acc = acc + 1 end end
print(acc)
