local objs = { {x = 1, y = 2}, {y = 2, x = 1}, {x = 1, z = 2}, {z = 2, x = 1} }
local acc = 0
for i = 1, 30000000 do local o = objs[(i % 4) + 1]; acc = acc + o.x end
print(acc)
