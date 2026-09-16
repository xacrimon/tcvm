local t = {}
for i = 1, 1000 do t[i] = i end
local acc = 0
for r = 1, 30000 do
  for i = 1, 1000 do acc = acc + t[i] end
end
print(acc)
