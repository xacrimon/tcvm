-- a call returned inside a `<close>` scope is not a tail call, and keeps all results
local function f(c, g)
  local a <close> = c
  return g(1)
end

local function m(c, o)
  local a <close> = c
  do return o:get() end
end

-- the scope has closed, so this one is
local function t(c, g)
  do local a <close> = c end
  return g(1)
end
