-- FFT - Fast Fourier Transform, translated from r7rs-benchmarks fft.scm
-- (itself translated from "Numerical Recipes in C").
-- Reads count, input1 (vector size), input2 (fill), output from stdin,
-- like the Scheme benchmark harness. Vectors are 0-indexed tables to
-- match the Scheme code exactly.

local sin, floor = math.sin, math.floor

local function make_vector(n, fill)
  local v = {}
  for i = 0, n - 1 do v[i] = fill end
  return v
end

local function four1(data, n)
  local pi2 = 6.28318530717959 -- to compute the inverse, negate this value

  -- bit-reversal section
  local i, j = 0, 0
  while i < n do
    if i < j then
      data[i], data[j] = data[j], data[i]
      data[i + 1], data[j + 1] = data[j + 1], data[i + 1]
    end
    local m = floor(n / 2)
    while m >= 2 and j >= m do
      j = j - m
      m = floor(m / 2)
    end
    i = i + 2
    j = j + m
  end

  -- Danielson-Lanczos section
  local mmax = 2
  while mmax < n do
    local theta = pi2 / mmax
    local x = sin(0.5 * theta)
    local wpr = -2.0 * (x * x)
    local wpi = sin(theta)
    local wr, wi = 1.0, 0.0
    local m = 0
    while m < mmax do
      local i = m
      while i < n do
        local j = i + mmax
        local tempr = wr * data[j] - wi * data[j + 1]
        local tempi = wr * data[j + 1] + wi * data[j]
        data[j] = data[i] - tempr
        data[j + 1] = data[i + 1] - tempi
        data[i] = data[i] + tempr
        data[i + 1] = data[i + 1] + tempi
        i = j + mmax
      end
      -- RHS evaluated before assignment, so wi's update sees the old wr,
      -- matching the Scheme named-let semantics.
      wr, wi = (wr * wpr - wi * wpi) + wr, (wi * wpr + wr * wpi) + wi
      m = m + 2
    end
    mmax = mmax * 2
  end
end

local function run(data, n)
  four1(data, n)
  return data[0]
end

-- Returns x without making it too easy for compilers to tell it will be
-- returned (mirror of `hide` in common.scm).
local function hide(r, x)
  local v = { [0] = function(y) return y end, [1] = function(y) return y end }
  local i = (r < 100) and 0 or 1
  return v[i](x)
end

local function this_implementation_name()
  if type(jit) == "table" and jit.version then
    return "luajit"
  end
  return "lua"
end

local function run_r7rs_benchmark(name, count, thunk, ok)
  print("Running " .. name)
  io.flush()
  local t0 = os.clock()
  local result = nil
  for _ = 1, count do
    result = thunk()
  end
  local secs = os.clock() - t0
  if ok(result) then
    print(string.format("Elapsed time: %g seconds for %s", secs, name))
    print(string.format("+!CSVLINE!+%s,%s,%g", this_implementation_name(), name, secs))
  else
    print("ERROR: returned incorrect result: " .. tostring(result))
    print(string.format("+!CSVLINE!+%s,%s,INCORRECT", this_implementation_name(), name))
  end
  io.flush()
end

local count = io.read("*n")
local input1 = io.read("*n")
local input2 = io.read("*n")
local output = io.read("*n")

run_r7rs_benchmark(
  string.format("fft:%d:%d", input1, count),
  count,
  function()
    return run(hide(count, make_vector(input1, input2)), input1)
  end,
  function(result) return result == output end)
