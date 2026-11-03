-- n: 1000000
local function run(s1, s2)
  local r
  for _ = 1, N do r = s1 .. s2 end
  return r
end
if run("abc", "def") ~= "abcdef" then print("wrong") end
