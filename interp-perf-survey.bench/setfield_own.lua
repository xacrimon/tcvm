local p = {x = 0, y = 0}
for i = 1, 30000000 do p.x = i; p.y = i end
print(p.x + p.y)
