-- Runs bench/lualua under tcvm-cli: `tcvm-cli -f lualua-shim.lua LUALUA script args...`.
-- tcvm has no require (#226) and passes script args only in `arg` (#318).
package.path = package.path or "./?.lua"
package.loaded = package.loaded or {}
function require(name)
  local v = package.loaded[name]
  if v ~= nil then return v end
  local fname = name:gsub("%.", "/")
  for tpl in package.path:gmatch("[^;]+") do
    local p = tpl:gsub("%?", fname)
    local f = loadfile(p)
    if f then
      local r = f(name, p)
      if r == nil then r = package.loaded[name] or true end
      package.loaded[name] = r
      return r
    end
  end
  error("module '" .. name .. "' not found")
end
local a, n = { [0] = arg[1] }, 0
while arg[n + 2] ~= nil do n = n + 1; a[n] = arg[n + 1] end
arg = a
return assert(loadfile(a[0]))(table.unpack(a, 1, n))
