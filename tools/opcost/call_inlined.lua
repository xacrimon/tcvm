local function f(x) return x end
local r
for i = 1, N do
  r = f(i)
end
if r == nil then print(r) end
