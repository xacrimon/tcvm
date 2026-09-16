local t = {}
for i = 1, 1000 do t[i] = i end
local acc = 0
for r = 1, 20000 do
  for i, v in ipairs(t) do acc = acc + v end
end
print(acc)
