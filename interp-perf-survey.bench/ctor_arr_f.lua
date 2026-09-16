local acc = 0.0
for i = 1, 5000000 do
  local v = {i, i + 1, i + 2}
  acc = acc + v[3]
end
print(acc)
