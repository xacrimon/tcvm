local function classify(a, b)
  local r = 0
  if a < b then
    r = 1
  else
    r = 2
  end
  while r < 10 do
    r = r + a
  end
  return r
end

return classify
