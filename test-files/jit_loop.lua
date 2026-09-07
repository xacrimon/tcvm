local function sum_field(t, n)
  local s = 0
  for i = 1, n do
    s = s + t.x
  end
  return s
end

return sum_field
