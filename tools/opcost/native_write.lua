-- live: 0,4,8
local write = io.write
--@decl
for _ = 1, N do
  --@live
  write("x")
end
--@use
