local a, b, c, d = "alpha", "beta", "gamma", "delta"
local n = 0
for i = 1, 3000000 do
  local s = a .. b .. c .. d
  n = n + #s
end
print(n)
