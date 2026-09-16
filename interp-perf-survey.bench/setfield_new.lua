local acc = 0
for i = 1, 5000000 do
  local p = {}
  p.x = i; p.y = i; p.z = i
  acc = acc + p.z
end
print(acc)
