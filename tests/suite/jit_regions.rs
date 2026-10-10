//! Compiled regions agree with the interpreter on programs that run long
//! enough to compile them.

use tcvm::env::Value;
use tcvm::{Lua, StepResult};

use crate::common::start_on;

/// `src`'s string result with the JIT on or off.
fn result(src: &str, jit: bool) -> String {
    let mut lua = Lua::new();
    lua.set_jit(jit);
    lua.load_all();
    let ex = start_on(&mut lua, src);
    lua.finish(&ex)
        .unwrap_or_else(|e| panic!("{src:?} failed: {e:?}"));
    lua.enter(|ctx| {
        let v: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        let s = v.get_string().expect("a string result");
        String::from_utf8_lossy(s.as_bytes()).into_owned()
    })
}

fn agrees(src: &str) {
    assert_eq!(result(src, true), result(src, false), "{src}");
}

/// A constant shared by code before and after a call is defined again after
/// it: the callee clobbers every register.
#[test]
fn constant_live_across_call() {
    agrees(
        "local sin, floor = math.sin, math.floor
        local function four1(data, n)
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
          local mmax = 2
          while mmax < n do
            local theta = 6.28318530717959 / mmax
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
              wr, wi = (wr * wpr - wi * wpi) + wr, (wi * wpr + wr * wpi) + wi
              m = m + 2
            end
            mmax = mmax * 2
          end
        end
        local data = {}
        for k = 1, 50 do
          for i = 0, 7 do data[i] = (i * 7 % 13) * 0.5 end
          four1(data, 8)
        end
        local s = 0
        for i = 0, 7 do s = s + data[i] * (i + 1) end
        return string.format('%.6f', s)",
    );
}

/// A region that boxes big integers outside any loop, entered from a native
/// that never checks the collector itself, still lets it run: the check at
/// the region's entry.
#[test]
fn boxes_outside_loops_are_collected() {
    let src = "local function f(c) return (c + 1) << 40 end
        local s = string.rep('x', 1000000)
        local r = s:gsub('.', function() return f(1) & 1 end)
        return tostring(#r)";
    let mut lua = Lua::new();
    lua.load_all();
    let ex = start_on(&mut lua, src);
    let mut collections = 0;
    loop {
        let done = lua.enter(|ctx| match ctx.fetch(&ex).step(ctx).expect("step") {
            StepResult::Done => true,
            StepResult::Pending => false,
            StepResult::Yielded(_) => panic!("yielded"),
        });
        if done {
            break;
        }
        collections += 1;
    }
    // About seventeen with the check, three without.
    assert!(collections > 8, "{collections} collections");
}

/// Compares of operands nothing types run inline in the handler's order,
/// then the helpers, which fail to the interpreter for metamethods and
/// errors.
#[test]
fn generic_compares() {
    agrees(
        r#"
        local function lt(a, b) if a < b then return 1 else return 0 end end
        local function le(a, b) if a <= b then return 1 else return 0 end end
        local function eq(a, b) if a == b then return 1 else return 0 end end
        local function ne(a, b) if a ~= b then return 1 else return 0 end end
        local function gti(a) if a > 3 then return 1 else return 0 end end
        local function lei(a) if a <= -2 then return 1 else return 0 end end
        local function eqi(a) if a == 5 then return 1 else return 0 end end
        local nan = 0/0
        local t1, t2 = {}, {}
        local mt = {__eq = function(x, y) return true end, __lt = function(x, y) return false end, __le = function() return true end}
        local u1, u2 = setmetatable({}, mt), setmetatable({}, mt)
        local big = math.maxinteger
        local vals = {1, 2, 2.0, 2.5, -3, nan, big, big - 1, math.mininteger, 1e300, -0.0, 0, "a", "b", "ab", t1, t2, u1, u2, true, false}
        local s = {}
        for rep = 1, 3 do
          for i = 1, #vals do
            local a = vals[i]
            s[#s + 1] = gti(type(a) == "number" and a or 7) .. lei(type(a) == "number" and a or 1) .. eqi(a)
            for j = 1, #vals do
              local b = vals[j]
              s[#s + 1] = eq(a, b) .. ne(a, b)
              local ok, r = pcall(lt, a, b)
              s[#s + 1] = ok and tostring(r) or "E"
              ok, r = pcall(le, a, b)
              s[#s + 1] = ok and tostring(r) or "E"
            end
          end
        end
        return table.concat(s)
"#,
    );
}

/// Arithmetic sites without a form run inline small-integer and number
/// cases, then `jit_arith`, whose failure (metamethods, strings, errors)
/// deopts; `UNM`, `BNOT`, `LEN` and `CONCAT` likewise.
#[test]
fn generic_arithmetic() {
    agrees(
        r#"
        local function mix(a, b) return a + b, a - b, a * b, a / b, a % b, a // b, a ^ 2 end
        local function bits(a, b) return a & b, a | b, a ~ b, a << 3, a >> 2, ~a, a & 2.0 end
        local function add(a, b) return a + b end
        local function un(v) return -v end
        local function ln(v) return #v end
        local function cc(a, b) return a .. b end
        local vals = {1, 2.5, -3, 7.25, 2^31, math.maxinteger, math.mininteger, 0.5, -2^40}
        local ints = {5, 1 << 40, 3, -7, 0x7fffffff, 1 << 62}
        local mt = setmetatable({}, {__add = function() return 42 end, __unm = function() return 7 end})
        local s = {}
        for round = 1, 300 do
          for i = 1, #vals do
            local a, b = vals[i], vals[(i % #vals) + 1]
            local r = {mix(a, b)}
            if round > 298 then s[#s + 1] = table.concat(r, ",") end
            s[#s + 1] = tostring(un(a))
          end
          for i = 1, #ints do
            local r = {bits(ints[i], 12)}
            if round > 298 then s[#s + 1] = table.concat(r, ",") end
          end
          local ok, e = pcall(mix, round, round % 50 == 0 and 0 or 3)
          s[#s + 1] = ok and "ok" or "E"
          s[#s + 1] = tostring(add(round % 7 == 0 and mt or round, 1.5))
          s[#s + 1] = tostring(mix(round % 3 == 0 and "10" or round, 2))
          s[#s + 1] = tostring(un(round % 9 == 0 and mt or round))
          s[#s + 1] = ln(round % 2 == 0 and "hello" or {1, 2, round}) .. cc(round % 3 == 0 and 1.5 or "x", round)
        end
        return table.concat(s, " ")
"#,
    );
}

/// Upvalues read and written through their cells, closures made in hot
/// code, `CLOSE`, and returns that close the frame's cells.
#[test]
fn upvalues_and_closures() {
    agrees(
        r#"
        local function counter()
          local n = 0
          return function(k) n = n + k; return n end, function() return n end
        end
        local inc, get = counter()
        for i = 1, 1000 do inc(i % 3) end
        local function make(x)
          local acc = x
          local f = function(d) acc = acc + d; return acc end
          for k = 1, 3 do acc = acc + k end
          return f, acc
        end
        local total = 0
        for i = 1, 400 do
          local f, a = make(i)
          total = total + f(1) + a
        end
        local last
        local function keep(v) last = v end
        local kept = {}
        for i = 1, 600 do
          keep(i % 2 == 0 and ("s" .. i) or {i})
          if i % 100 == 0 then kept[#kept + 1] = type(last) == "table" and last[1] or last end
          if i % 150 == 0 then collectgarbage() end
        end
        local function pair()
          local v = 1.5
          return function(n) local r = 0 for i = 1, n do r = r + v end return r end,
                 function(x) v = x end
        end
        local sum, set = pair()
        local r = 0
        for i = 1, 300 do
          r = r + sum(10)
          set(i % 2 == 0 and 2 or 0.25)
        end
        local fs = {}
        for i = 1, 500 do
          local j = i * 2
          fs[i] = function() return i + j end
        end
        local s = 0
        for i = 1, 500 do s = s + fs[i]() end
        return table.concat({get(), total, table.concat(kept, ","), r, s}, " ")
"#,
    );
}

/// Fixed-count VARARG and VARARGGET read the extras below the header, nil
/// past them; vararg tables, NEWTABLE and SETLIST go through helpers. A
/// definition guard on an extra waits for its instruction's other results,
/// and a missing extra (nil) does not get one from a use behind `or`.
#[test]
fn varargs_and_constructors() {
    agrees(
        r#"
        local function first3(...)
          local a, b, c = ...
          return (a or 0) + (b or 0) + (c or 0)
        end
        local function pick(k, ...t) return t[k], t.n end
        local function keep(...t) return t end
        local function mk(i)
          local t = {i, i * 2, i * 3, x = i, [10] = 5, "s" .. i}
          return t[1] + t[2] + t[3] + t.x + t[10] + #t
        end
        local s, acc = 0, {}
        for i = 1, 600 do
          if i % 3 == 0 then s = s + first3(i)
          elseif i % 3 == 1 then s = s + first3(i, 1)
          else s = s + first3(i, 2, 3, 4, 5) end
          local v, n = pick(i % 5, "a", "b", "c")
          local w = pick((i % 3) + 0.0, 10, 20)
          if i > 594 then acc[#acc + 1] = tostring(v) .. "/" .. n .. "/" .. tostring(w) end
          local t = keep(i, i + 1)
          s = s + t[2] + t.n + mk(i)
        end
        return s .. " " .. table.concat(acc, " ")
"#,
    );
}

/// Field sites compile from their inline caches (own slots, absent keys,
/// `__index` holders, transitions) and integer keys through the array
/// part; the rest, and a nil where a metatable may answer, go through the
/// index helpers. `SELF` passes its receiver on every path.
#[test]
fn tables_and_fields() {
    agrees(
        r#"
        local function step(p) p.x = p.x + p.vx; p.n = p.n + 1; return p.x end
        local p = {x = 0.0, vx = 0.5, n = 0}
        local s = 0
        for i = 1, 1000 do s = s + step(p) end
        local function make(i)
          local o = {}
          o.a = i; o.b = "b" .. i; o.c = {i}; o.d = i; o.e = i; o.f = i; o.g = i; o.h = i; o.i = i
          return o
        end
        for i = 1, 600 do local o = make(i); s = s + o.a + #o.b + o.c[1] + o.i end
        local C = {}
        for k = 1, 10 do C["m" .. k] = function(self) return k + self.v end end
        C.__index = C
        local inst = setmetatable({v = 1}, C)
        for _ = 1, 500 do s = s + inst:m1() + inst:m10() end
        C.m10 = function() return 0 end
        s = s + inst:m10()
        local arr = {}
        for i = 1, 100 do arr[i] = i * 3 end
        local function sum(a, n) local t = 0 for i = 0, n do t = t + (a[i] or 1) end return t end
        for r = 1, 50 do s = s + sum(arr, 120) end
        local function fill(a, n) for i = 1, n do a[i] = i end return #a end
        for r = 1, 50 do s = s + fill({}, r) end
        local log = 0
        local proxy = setmetatable({}, {__newindex = function() log = log + 1 end})
        local sink = {}
        local redirect = setmetatable({}, {__newindex = sink})
        local function put(t, v) t.slot = v end
        for i = 1, 300 do put(i % 3 == 0 and proxy or (i % 3 == 1 and redirect or {}), i) end
        local fnidx = setmetatable({}, {__index = function(t, k) return k .. "!" end})
        local function get(t) return t.name end
        local names = {}
        for i = 1, 300 do names[#names + 1] = get(i % 2 == 0 and fnidx or {name = "own"}) end
        local function up(x) return x:upper() end
        return table.concat({s, log, sink.slot, names[299], names[300], up("ab")}, " ")
"#,
    );
}

/// Builtins with intrinsics compile inline behind a guard on the callee,
/// in tail position too; an argument the inline code does not take, or a
/// replaced builtin, goes back to a call.
#[test]
fn intrinsics() {
    agrees(
        r#"
        local sqrt, sin, floor, ceil, abs = math.sqrt, math.sin, math.floor, math.ceil, math.abs
        local vals = {0, 1, 2.5, -3, -7.25, 2^31, 1e300, -0.0, 1/0, 0/0, math.mininteger, -2147483648, 4}
        local out = {}
        local function f(x)
          local a, b, c, d, e = sqrt(x >= 0 and x or -x), sin(x), floor(x), ceil(x), abs(x)
          return tostring(a) .. tostring(b) .. tostring(c) .. tostring(d) .. tostring(e)
        end
        for r = 1, 300 do
          for i = 1, #vals do
            local s = f(vals[i])
            if r == 300 then out[#out + 1] = s end
          end
        end
        local function h(x) return math.floor(x) end
        local hs = 0
        for i = 1, 500 do
          if i == 300 then math.floor = function() return 42 end end
          hs = hs + h(i + 0.5)
        end
        local function bad(x) return sqrt(x) end
        local errs = 0
        for i = 1, 300 do
          if not pcall(bad, i % 10 == 0 and {} or (i % 7 == 0 and "16" or i)) then errs = errs + 1 end
        end
        local function chk(x) assert(x, "nope") return assert(x) end
        for i = 1, 300 do if not pcall(chk, i % 13 ~= 0 and i or false) then errs = errs + 1 end end
        return table.concat(out, " ") .. hs .. " " .. errs
"#,
    );
}

/// A guard that exits when its test holds (`GuardFalse`) and fuses the test
/// exits on falsy values and on values in the set, not on the others.
#[test]
fn inverted_fused_tests() {
    agrees(
        r#"
        local function chk(x) assert(x, "nope") return 1 end
        local ok, bad = 0, 0
        for i = 1, 400 do
          if pcall(chk, i % 13 ~= 0 and i or false) then ok = ok + 1 else bad = bad + 1 end
        end
        return ok .. " " .. bad
"#,
    );
}
