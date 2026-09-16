local p = {x = 1.5, y = 2.5, z = 3.5, w = 4.5}
local acc = 0
for i = 1, 30000000 do acc = acc + p.x + p.y end
print(acc)
