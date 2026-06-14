global print
local v = 1
do
  global v
  v = 2
end
print(v)
