local acc = 0.0
for i = 1, 5000000 do
  local v = {x = i, y = i + 1, z = i + 2}
  acc = acc + v.z
end
print(acc)
