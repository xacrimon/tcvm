local function forward(g, h)
  return h(g())
end

local function pack_all(...)
  return { ... }
end

return forward, pack_all
