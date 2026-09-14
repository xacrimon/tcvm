-- backward goto leaving a block with a captured local
local fns, i = {}, 0
::top::
i = i + 1
do
  local x = i
  fns[i] = function() return x end
  if i < 3 then goto top end
end

-- backward goto in the same scope, past a captured local's declaration
local g, n = {}, 0
::again::
n = n + 1
local y = n
g[n] = function() return y end
if n < 3 then goto again end

-- forward goto out of a block with a captured local, then register reuse
local h
do
  local z = 42
  h = function() return z end
  goto out
end
::out::
local w = 7

-- continue idiom (forward goto within the loop body scope)
local c = {}
for k = 1, 4 do
  local v = k
  if k % 2 == 0 then goto continue end
  c[#c + 1] = function() return v end
  ::continue::
end

-- forward goto with no locals involved
goto skip

::skip::

