-- rxi's json.lua (lib/rxi_json.lua, run from this directory): encode a
-- generated document and decode it again, N times.
local json = dofile("lib/rxi_json.lua")
local N = tonumber(arg and arg[1]) or 200

local function doc(seed)
  local items = {}
  for i = 1, 300 do
    items[i] = {
      id = i,
      name = "item" .. (i + seed),
      price = i * 1.25 + seed,
      tags = {"a" .. i % 7, "b" .. i % 3, "c"},
      active = i % 2 == 0,
      nested = {x = i, y = -i, z = {1, 2, 3}},
      note = "line\n\"quoted\"\t" .. i,
    }
  end
  return {version = 1, seed = seed, items = items}
end

local total = 0
for k = 1, N do
  local s = json.encode(doc(k))
  local d = json.decode(s)
  total = total + #s + #d.items + d.items[7].nested.x
end
print(total)
