local objs = { {x = 1, y = 2}, {x = 1, y = 2} }
local acc = 0
for i = 1, 30000000 do local o = objs[(i % 2) + 1]; acc = acc + o.x end
print(acc)
