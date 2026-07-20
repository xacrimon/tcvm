-- `and`/`or` in native code, which the compiler lowers to `TESTSET`.
--
-- `TESTSET` assigns its destination on only one of its two edges, and its skip
-- condition is the *opposite* of `TEST`'s for the same `inverted` flag:
-- `op_testset` skips when `truthy == inverted`, `op_test` when they differ.
-- Reading one as the other gets both polarities backwards, which shows up as a
-- wrong answer rather than a crash.
--
-- `a` is genuinely `false` every third iteration. That matters: an operand the
-- type lattice already knows is truthy folds the branch to a constant, so the
-- version where the edge choice is live would never be built. Both polarities
-- appear — `and` compiles to `inv=true`, `or` to `inv=false`.
local function pick(n)
  local last = 0
  for i = 1, n do
    local a = (i % 3 ~= 0) and i or false
    local t = a and (i * 2) -- inv=true; `false` when `a` is
    last = t or -1 -- inv=false
  end
  return last
end

-- Well past the compile threshold, so most of these run natively.
local h = 0
for _ = 1, 200 do
  h = (h + pick(50)) % 1000003
end
print(h)
print(pick(1000)) -- 1000 % 3 ~= 0, so `a` is truthy: 2000
print(pick(999)) -- 999 % 3 == 0, so `a` is false: -1
print(pick(3))

return h, pick
