-- String building: string.format, `..`, table.concat and gsub with a
-- function, N rounds.
local N = tonumber(arg and arg[1]) or 2000
local format, concat, gsub = string.format, table.concat, string.gsub

local total = 0
for _ = 1, N do
  local parts = {}
  for i = 1, 400 do
    parts[#parts + 1] = format("%5d|%-8s|%8.3f|%x", i, "row" .. i, i / 7, i * 31)
  end
  local s = concat(parts, "\n")
  local t = gsub(s, "(%d+)", function(d) return "<" .. d .. ">" end)
  local u = gsub(t, "row(%d+)", function(d)
    if tonumber(d) % 3 == 0 then return "R" .. d end
  end)
  local line = ""
  for i = 1, 100 do line = line .. i .. "," end
  total = total + #t + #u + #line
end
print(total)
