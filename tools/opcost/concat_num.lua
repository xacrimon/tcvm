-- n: 1000000
local function run(s)
  local r
  for i = 1, N do r = s .. i end
  return r
end
if run("abc") == nil then print("wrong") end
