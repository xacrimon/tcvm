//! Compiled regions agree with the interpreter on programs that run long
//! enough to compile them.

use tcvm::env::{Error, Function, LuaString, NativeClosure, NativeFn, Stack, Value};
use tcvm::{Context, Lua, RuntimeError, StepResult};

use crate::common::start_on;

/// The lib's `jit` cfg (`build.rs`): no JIT on stable or on other targets.
const JIT: bool = cfg!(all(
    feature = "jit",
    feature = "nightly",
    any(target_arch = "aarch64", target_arch = "x86_64")
));

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

/// `agrees`, every compile succeeding or refusing the code: none failing on
/// a verifier or backend error, which runs the code interpreted unseen.
fn compiles_cleanly(src: &str, inline: bool) {
    let mut lua = Lua::new();
    lua.set_jit_inline(inline);
    lua.load_all();
    let ex = start_on(&mut lua, src);
    lua.finish(&ex)
        .unwrap_or_else(|e| panic!("{src:?} failed: {e:?}"));
    let got = lua.enter(|ctx| {
        let v: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        String::from_utf8_lossy(v.get_string().expect("a string result").as_bytes()).into_owned()
    });
    assert_eq!(got, result(src, false), "{src}");
    assert_eq!(lua.jit_internal_errors(), 0, "{src}");
}

/// `compiles_cleanly` with every region compiled at its first count and at
/// least one compiled; the run's stats.
fn compiles_at_first_count(src: &str, inline: bool) -> String {
    compiles_hot(src, inline, 1, 1)
}

/// `compiles_at_first_count` at these hot counts (`Lua::set_jit_hot`).
fn compiles_hot(src: &str, inline: bool, call: u16, r#loop: u16) -> String {
    let mut lua = Lua::new();
    lua.set_jit_hot(call, r#loop);
    lua.set_jit_inline(inline);
    lua.set_stats(true);
    lua.load_all();
    let ex = start_on(&mut lua, src);
    lua.finish(&ex)
        .unwrap_or_else(|e| panic!("{src:?} failed: {e:?}"));
    let got = lua.enter(|ctx| {
        let v: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        String::from_utf8_lossy(v.get_string().expect("a string result").as_bytes()).into_owned()
    });
    assert_eq!(got, result(src, false), "{src}");
    assert_eq!(lua.jit_internal_errors(), 0, "{src}");
    let stats = lua.stats();
    assert!(!JIT || stats.contains("jit compiled "), "{src}\n{stats}");
    stats
}

/// `key`'s count in `stats`.
fn count(stats: &str, key: &str) -> u64 {
    (stats.lines())
        .find_map(|l| l.strip_prefix(key)?.strip_prefix(' ')?.parse().ok())
        .unwrap_or(0)
}

#[test]
fn hot_counts_of_one_compile_at_the_first_count() {
    let src = "local function f(a) return a + 1 end
        local s = 0 for i = 1, 3 do s = f(s) end return tostring(s)";
    let stats = compiles_at_first_count(src, false);
    if JIT {
        // The chunk and `f`.
        assert_eq!(count(&stats, "jit compiled FUNC"), 2, "{stats}");
        assert_eq!(count(&stats, "jit compiled FORLOOP_I"), 1, "{stats}");
    }
}

/// `agrees` with inlining off, for tests of real calls: what the inliner takes
/// changes as it grows.
fn agrees_calling(src: &str) {
    let mut lua = Lua::new();
    lua.set_jit_inline(false);
    lua.load_all();
    let ex = start_on(&mut lua, src);
    lua.finish(&ex)
        .unwrap_or_else(|e| panic!("{src:?} failed: {e:?}"));
    let got = lua.enter(|ctx| {
        let v: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        String::from_utf8_lossy(v.get_string().expect("a string result").as_bytes()).into_owned()
    });
    assert_eq!(got, result(src, false), "{src}");
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
/// the region's entry. (`f` takes varargs so that it is not inlined, which
/// would fold its box away.)
#[test]
fn boxes_outside_loops_are_collected() {
    let src = "local function f(c, ...) return (c + 1) << 40 end
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

/// Metamethods are called from compiled code once a site's feedback saw
/// one run (before that its helper's failure exits and records it):
/// arithmetic, unary, length, concatenation, compares, string arithmetic,
/// `__index` and `__newindex` functions; one may yield or raise.
#[test]
fn metamethod_calls() {
    agrees(
        r#"
        local V = {}
        V.__index = V
        local function vec(x, y) local v = setmetatable({x = x, y = y}, V) return v end
        V.__add = function(a, b) return vec(a.x + b.x, a.y + b.y) end
        V.__unm = function(a) return vec(-a.x, -a.y) end
        V.__len = function(a) return a.x * a.x + a.y * a.y end
        V.__concat = function(a, b) return "v" .. (type(a) == "table" and a.x or a) .. (type(b) == "table" and b.y or b) end
        V.__eq = function(a, b) return a.x == b.x end
        V.__lt = function(a, b) return #a < #b end
        V.__le = function(a, b) return #a <= #b end
        local function step(p, q) return -(p + q) end
        local function cmp(a, b) local r = 0 if a == b then r = r + 1 end if a < b then r = r + 2 end if a <= b then r = r + 4 end return r end
        local function cat(a, b) return a .. b end
        local function len(a) return #a end
        local function sadd(a, b) return a + b end
        local s, p = 0, vec(0, 0)
        for i = 1, 600 do
          p = step(p, vec(i % 3, i % 5))
          s = s + cmp(vec(i % 3, 1), vec(1, i % 4)) + len(p) % 7 + #cat(p, i) + sadd(tostring(i), "1")
        end
        local hits = 0
        local P = setmetatable({}, {__index = function(t, k) hits = hits + 1 return #k end})
        local store = {}
        local W = setmetatable({}, {__newindex = function(t, k, v) store[k] = v * 2 end})
        local function rd(t) return t.abc end
        local function wr(t, v) t.slot = v end
        for i = 1, 600 do s = s + rd(P); wr(W, i) end
        local E = setmetatable({}, {__add = function() error("boom") end})
        local function addE(a) return a + 1 end
        local errs = 0
        for i = 1, 600 do if not pcall(addE, i % 50 == 0 and E or i) then errs = errs + 1 end end
        local Y = setmetatable({}, {__add = function(a, b) return coroutine.yield(b) end})
        local function addY(a, b) return a + b end
        local co = coroutine.wrap(function()
          local t = 0
          for i = 1, 600 do t = t + addY(i % 3 == 0 and Y or i, i) end
          return t
        end)
        local got = co()
        for _ = 1, 200 do got = co(got * 10) end
        return table.concat({s, p.x, p.y, hits, store.slot, errs, tostring(got)}, " ")
"#,
    );
}

/// Generic `for` loops: `pairs` and `ipairs` steps without calling the
/// iterator (`next`'s array part inline, past holes), other iterators
/// called, and the forms TFORPREP chose guarded.
#[test]
fn generic_for_loops() {
    agrees(
        r#"
        local arr, hash = {}, {}
        for i = 1, 50 do arr[i] = i * 2; hash["k" .. i] = i end
        local mixed = {10, 20, 30, a = 1, b = 2, [100] = 7, [2.5] = 3, [true] = 9}
        local holes = {1, nil, 3, nil, nil, 6, nil, x = 1}
        local idx = setmetatable({1, 2}, {__index = function(t, k) if k < 6 then return k * 10 end end})
        local function psum(t) local s, n = 0, 0 for k, v in pairs(t) do n = n + 1 if type(v) == "number" then s = s + v end end return s + n end
        local function isum(t) local s = 0 for i, v in ipairs(t) do s = s + v * i end return s end
        local function bnext(t) local s = 0 for k, v in next, t do if type(v) == "number" then s = s + v end end return s end
        local function range(n) local i = 0 return function() i = i + 1 if i <= n then return i end end end
        local function custom(n) local s = 0 for i in range(n) do s = s + i end return s end
        local s = 0
        for r = 1, 400 do
          s = s + psum(arr) + psum(hash) + psum(mixed) + isum(arr) + isum(idx) + bnext(mixed) + custom(r % 20)
          s = s + psum(holes) + bnext(holes)
        end
        return tostring(s)
"#,
    );
}

/// RETURN up to `top`, and TAILCALLs the fast arm does not take, leave
/// through the interpreter's handlers with the frame stored; CALL, SETLIST
/// and VARARG up to `top` compile.
#[test]
fn tail_calls_and_multret() {
    agrees(
        r##"
        local function count(n, acc) if n == 0 then return acc end return count(n - 1, acc + 1) end
        local callable = setmetatable({}, {__call = function(self, x) return x * 3 end})
        local function t1(x) return math.max(x, 5) end
        local function t3(x) return callable(x) end
        local function t4(...) return select("#", ...) end
        local function two(x) return x, x + 1 end
        local function many(n) local t = {} for i = 1, n do t[i] = i end return table.unpack(t) end
        local function pass(...) return ... end
        local function sum(...) local s = 0 for i = 1, select("#", ...) do s = s + (select(i, ...) or 0) end return s end
        local function wrap(...) return {...} end
        local function opened(...) local x = 1 local f = function() return x end x = 2 return f(), ... end
        local s = count(100000, 0)
        for i = 1, 600 do
          s = s + t1(i % 10) + t3(i) + t4(i, nil, i) + sum(two(i)) + sum(many(i % 7))
          s = s + #wrap(two(i), two(i)) + sum(pass(1, 2, 3)) + sum(opened(i, i))
        end
        return tostring(s)
"##,
    );
}

/// A tail-called builtin that returns nothing returns nothing from compiled
/// code: `table.insert` as its intrinsic, from a function region, from a body
/// inlined in a loop and from one inlined in a region frame's place;
/// `string.byte` out of range by the call.
#[test]
fn tail_called_builtins_without_results() {
    let mixed = r##"
    local t, u = {}, {}
    local function f(a, b) return a:byte(b) end
    local function g(a) return table.insert(t, a) end
    local function k(a) return g(a) end
    local function h(a, b) return (a:byte(b)) end
    local function w(a) local x = table.insert(u, a); return x end
    local n0, n1, m, mixed = 0, 0, 0, 0
    for i = 1, 3000 do
        if select("#", f("", -1)) == 0 then n0 = n0 + 1 else n1 = n1 + 1 end
        m = m + select("#", g(i)) + select("#", k(i)) + select("#", w(i)) + select("#", (g(i)))
        mixed = mixed + select("#", f("ab", i % 3)) + (f("ab", 1) or 0) + select("#", h("", 1))
    end
    return table.concat({n0, n1, m, #t, #u, mixed}, " ")
"##;
    let inserts = r##"
    local t = {}
    local function g(a) return table.insert(t, a) end
    local function k(a) return g(a) end
    local m = 0
    for i = 1, 3000 do
        m = m + select("#", g(i)) + select("#", k(i)) + select("#", (g(i)))
    end
    return m .. " " .. #t
"##;
    for inline in [true, false] {
        for (src, want) in [
            (mixed, "3000 0 6000 9000 3000 296000"),
            (inserts, "3000 9000"),
        ] {
            let mut lua = Lua::new();
            lua.set_jit_inline(inline);
            lua.set_stats(true);
            lua.load_all();
            let ex = start_on(&mut lua, src);
            lua.finish(&ex).unwrap();
            lua.enter(|ctx| {
                let v: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
                assert_eq!(
                    String::from_utf8_lossy(v.get_string().expect("a string").as_bytes()),
                    want,
                    "inline {inline}"
                );
            });
            // The intrinsic, not the interpreter's TAILCALL of the builtin
            // (the lib's `jit` cfg: no JIT on stable or on other targets).
            if src == inserts
                && cfg!(all(
                    feature = "jit",
                    feature = "nightly",
                    any(target_arch = "aarch64", target_arch = "x86_64")
                ))
            {
                let stats = lua.stats();
                let tailcalls: u64 = (stats.lines())
                    .find_map(|l| l.strip_prefix("op TAILCALL ")?.parse().ok())
                    .unwrap_or(0);
                assert!(tailcalls < 1000, "inline {inline}: {tailcalls}");
            }
        }
    }
}

/// TAILCALL's fast arm (a Lua callee without varargs, from a frame with
/// nothing to close) and the callees and frames that take the interpreter's.
#[test]
fn fast_tail_calls() {
    let locals: Vec<String> = (1..=150).map(|i| format!("l{i}")).collect();
    let src = format!(
        r##"
        local function three(a, b, c) return (a or 0) + (b or 10) + (c or 100) end
        local function one(x) return three(x) end
        local function extra(x) return three(x, 1, 2, 3, 4) end
        local function vararg(...) return select("#", ...) end
        local callable = setmetatable({{}}, {{__call = function(self, x) return x * 3 end}})
        local function via(f, x) return f(x) end
        local function fixed(a, b) return a + b end
        local function from_vararg(x, ...) return fixed(x, 1) end
        local function id(x) return x end
        local function closing(n) local x = n local f = function() return x end return id(f()) end
        local function big(n) local {} = n return l1 + (l150 or 1) end
        local function deep(n) if n == 0 then return big(1) end return 1 + deep(n - 1) end
        local s = 0
        for i = 1, 600 do
          s = s + one(i) + extra(i) + via(id, i) + via(math.abs, -i) + via(vararg, i)
          s = s + via(callable, i) + from_vararg(i, 1, 2, 3) + select("#", from_vararg(i, 5)) + closing(i)
        end
        for i = 1, 20 do s = s + deep(400 * i) end
        return tostring(s)
"##,
        locals.join(", "),
    );
    agrees(&src);
}

/// Builtins compiled through their helpers (`type`, `getmetatable`,
/// `setmetatable`, the `raw*` ones, `tostring`, `tonumber`, `string.len`,
/// `string.sub`, `string.byte`) and `select`, with the builtin called where
/// a helper declines.
#[test]
fn builtin_intrinsics() {
    agrees(
        r##"
        local out = {}
        local function emit(...) out[#out + 1] = table.concat({...}, " ") end
        local vals = {1, 2.5, "s", {}, print, true, nil, coroutine.create(function() end), 2^53}
        local function types(n)
          local t = {}
          for i = 1, n do t[#t + 1] = type(vals[i % 9 + 1]) end
          local x = 0
          for i = 1, 100 do x = x + (type(i) == "number" and 1 or 0) + (type("a") == "string" and 1 or 0) end
          return table.concat(t, ",") .. x
        end
        for r = 1, 300 do out.types = types(20) end
        emit(out.types)

        local P = {}; P.__index = P
        local Q = {__index = P, __metatable = "locked"}
        local function new(x) return setmetatable({x = x}, P) end
        local function newq(x) return setmetatable({x = x}, Q) end
        local function again(o) return setmetatable(o, P) end
        local s = 0
        for i = 1, 300 do
          local o = new(i); s = s + o.x
          local q = newq(i); s = s + (getmetatable(q) == "locked" and 1 or 0) + (getmetatable(o) == P and 1 or 0)
          local ok = pcall(setmetatable, q, P); s = s + (ok and 100 or 1)
          again(o); s = s + (getmetatable(o) == P and 1 or 0)
          s = s + (getmetatable("x") ~= nil and 1 or 0) + (getmetatable(1) == nil and 1 or 0)
        end
        emit("meta", s)

        local mt = {__index = function(t, k) return "idx" end, __newindex = function(t, k, v) rawset(t, k, v * 2) end, __len = function() return 99 end, __eq = function() return true end}
        local r = 0
        local log = {}
        for i = 1, 300 do
          local t = setmetatable({}, mt)
          t.a = i
          r = r + rawget(t, "a") + (rawget(t, "zz") == nil and 1 or 0) + rawlen(t) + rawlen("abc") + #t
          rawset(t, "b", i); r = r + rawget(t, "b")
          local u = setmetatable({}, mt)
          r = r + (rawequal(t, u) and 1000 or 1) + (t == u and 1 or 0) + (rawequal(1, 1.0) and 1 or 0) + (rawequal("a", "a") and 1 or 0)
          local ok = pcall(rawset, t, nil, 1); r = r + (ok and 0 or 7)
          ok = pcall(rawlen, 5); r = r + (ok and 0 or 3)
        end
        emit("raw", r)

        local T = setmetatable({}, {__tostring = function() return "custom" end})
        local N = setmetatable({}, {__name = "Named"})
        local st = {}
        for i = 1, 300 do
          st[1] = tostring(i) .. tostring(1.5) .. tostring(true) .. tostring(nil) .. tostring(T)
          st[2] = (tostring(N):match("^Named") or "?")
          st[3] = tostring(tonumber("12")) .. tostring(tonumber("0x10")) .. tostring(tonumber(" 1e2 ")) .. tostring(tonumber("z")) .. tostring(tonumber(3)) .. tostring(tonumber({}))
          st[4] = string.len("hello") + string.len("") + (pcall(string.len, {}) and 0 or 5)
        end
        emit("str", st[1], st[2], st[3], st[4])

        local function count(...) return select("#", ...) end
        local function second(...) return select(2, ...) end
        local function last(...) return (select(-1, ...)) end
        local function fixed(a, b, c) return select("#", a, b, c) + select(2, a, b, c) end
        local function bad(...) return pcall(select, 0, ...) end
        local c = 0
        for i = 1, 300 do
          c = c + count() + count(1) + count(1, nil, 3) + (second(1, 2, 3) or 0) + (second(1) or 100) + last(4, 5, 6) + fixed(1, 2, 3)
          c = c + (bad(1) and 0 or 9)
        end
        emit("select", c)

        local sb = {}
        for i = 1, 300 do
          local s = "hello world"
          sb[1] = string.sub(s, 2) .. string.sub(s, -5) .. string.sub(s, 3, 5) .. string.sub(s, 0) .. string.sub(s, 20) .. "|" .. string.sub(s, -100, 2) .. s:sub(i % 12, -2) .. string.sub(s, 2.0, 3)
          sb[2] = (string.byte(s) or 0) + (string.byte(s, 3) or 0) + (string.byte(s, -1) or 0) + (string.byte(s, 0) or 1000) + (string.byte(s, 99) or 7) + (s:byte(i % 11 + 1)) + (string.byte(s, 2.0) or 0)
          sb[3] = pcall(string.sub, {}, 1) and "x" or "e"
        end
        emit("strsub", sb[1], sb[2], sb[3])
        return table.concat(out, "\n")
"##,
    );
}

/// `assert()` without an argument raises in compiled code as well: there is
/// no first argument to test, in the frame or past its end.
#[test]
fn assert_without_arguments() {
    let src = r##"
    local function f()
        local t = {1, 2, 3, 4, 5, 6}
        local x = assert()
        return x
    end
    local function g() assert() end
    local function h() return assert() end
    local n, msg = 0
    for i = 1, 3000 do
        for _, fn in ipairs({f, g, h}) do
            local ok, e = pcall(fn)
            if ok then n = n + 1 else msg = e end
        end
    end
    return n .. " " .. msg
"##;
    for inline in [true, false] {
        let mut lua = Lua::new();
        lua.set_jit_inline(inline);
        lua.load_all();
        let ex = start_on(&mut lua, src);
        lua.finish(&ex).unwrap();
        lua.enter(|ctx| {
            let v: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
            assert_eq!(
                String::from_utf8_lossy(v.get_string().expect("a string").as_bytes()),
                "0 c:8: bad argument #1 to 'assert' (value expected)"
            );
        });
        assert_eq!(lua.jit_internal_errors(), 0);
    }
}

/// The metamethod forms (`ARITH_MM`, `_MM_R`, `_MMI` either side) call the
/// operand class's metamethod; a table without it and a number metatable
/// set after compiling take the handlers.
#[test]
fn metamethod_forms() {
    agrees(
        r#"
        local V = {}
        V.__index = V
        local function vec(x, y) return setmetatable({x = x, y = y}, V) end
        V.__add = function(a, b)
          if type(a) == "number" then return vec(a + b.x, a + b.y) end
          if type(b) == "number" then return vec(a.x + b, a.y + b) end
          return vec(a.x + b.x, a.y + b.y)
        end
        V.__mul = function(a, b)
          if type(a) == "number" then return vec(a * b.x, a * b.y) end
          if type(b) == "number" then return vec(a.x * b, a.y * b) end
          return vec(a.x * b.x, a.y * b.y)
        end
        V.__sub = function(a, b)
          if type(a) == "number" then return vec(a - b.x, a - b.y) end
          return vec(a.x - b, a.y - b)
        end
        local W = setmetatable({}, {__index = V})
        local function run(n)
          local acc = vec(0, 0)
          local s = 0
          for i = 1, n do
            local v = vec(i, -i)
            acc = acc + v * 2
            acc = 1 + acc
            acc = acc - 1
            local w = 3 - v
            local m = i * v
            s = s + w.x + m.y
          end
          return acc.x + acc.y + s
        end
        local out = {}
        for r = 1, 200 do out[1] = run(30) end
        -- A table without the metamethod at a compiled site, then the other side's.
        local plain = {x = 1, y = 2}
        out[2] = select(2, pcall(function() local a = vec(1, 1) return (plain + a).x end))
        -- Numbers get a metatable: its __add comes first where the number is on the left.
        debug.setmetatable(0, {__add = function(a, b) return "num" end, __sub = function(a, b) return "nsub" end})
        out[3] = tostring(select(2, pcall(run, 3)))
        debug.setmetatable(0, nil)
        out[4] = run(30)
        return table.concat(out, " ")
"#,
    );
}

/// A constant on the left of a commutative operator stays on the left for
/// the metamethod and for string coercion, at a site compiled on numbers.
#[test]
fn constant_left_operands_keep_source_order() {
    let src = r##"
    local function side(a, b) return type(a) == "table" and "TN" or "NT" end
    local t = setmetatable({}, {__add = side, __mul = side, __band = side, __bor = side,
                                __bxor = side, __sub = side, __shl = side, __div = side})
    local fs = {
        function(b) return 64 ~ b end, function(b) return b ~ 64 end,
        function(b) return 3 + b end, function(b) return b + 3 end,
        function(b) return 2.5 * b end, function(b) return b * 2.5 end,
        function(b) return 7 & b end, function(b) return 7 | b end,
        function(b) return 5 - b end, function(b) return b - 5 end,
        function(b) return 1 << b end, function(b) return 2 / b end,
    }
    local out = {}
    for i, f in ipairs(fs) do
        for j = 1, 300 do f(j) end
        local n = {NT = 0, TN = 0}
        for j = 1, 300 do local s = f(t); n[s] = n[s] + 1 end
        out[#out + 1] = n.NT .. "/" .. n.TN
    end
    local function g(b) return 10 + b end
    for j = 1, 300 do g(j) end
    local s = 0
    for j = 1, 300 do s = s + g("5") end
    out[#out + 1] = s
    return table.concat(out, " ")
"##;
    assert_eq!(
        result(src, true),
        "300/0 0/300 300/0 0/300 300/0 0/300 300/0 300/0 300/0 0/300 300/0 300/0 4500"
    );
}

/// Array slots holding nil: the result, or the store, inline unless the
/// table's class has `__index` or `__newindex` at the time.
#[test]
fn nil_array_slots() {
    agrees(
        r#"
        local function fill(t, n) for i = 1, n do if t[i] == nil then t[i] = i end end return t end
        local function sum(t, n) local s = 0 for i = 1, n do s = s + (t[i] or 0) end return s end
        local out = {}
        for r = 1, 300 do
          local t = {nil, nil, nil, nil, nil, nil, nil, nil}
          for i = 1, 8 do t[i] = false end
          for i = 1, 8, 2 do t[i] = nil end
          fill(t, 8)
          out[1] = sum(t, 8)
          local log = {}
          local p = setmetatable({1, nil, 3, nil}, {__index = function(_, k) return k * 100 end, __newindex = function(t, k, v) rawset(t, k, v * 10) end})
          out[2] = sum(p, 4)
          fill(p, 4)
          out[3] = sum(p, 4)
        end
        -- __index/__newindex added after the regions compiled
        local late = {1, nil, 3}
        for r = 1, 300 do sum(late, 3); fill({1, nil, 3}, 3) end
        setmetatable(late, {__index = function() return 7 end})
        out[4] = sum(late, 3)
        local q = {1, nil, 3}
        setmetatable(q, {__newindex = function(t, k, v) rawset(t, k, -v) end})
        fill(q, 3)
        out[5] = sum(q, 3)
        return table.concat(out, " ")
"#,
    );
}

/// `#`: strings inline, tables through the length helper unless their class
/// has `__len` at the time, which is called.
#[test]
fn length_operator() {
    agrees(
        r#"
        local L = setmetatable({}, {__len = function() return 42 end})
        local items = {"abc", {1, 2, 3}, "", L, {n = 1}, setmetatable({1, 2}, {__index = function() return 1 end})}
        local function len(x) return #x end
        local function strs(s) local n = 0 for i = 1, 10 do n = n + #s end return n end
        local function tabs(t) local n = 0 for i = 1, 10 do t[#t + 1] = i; n = n + #t end return n end
        local out = {}
        for r = 1, 300 do
          local s = 0
          for i = 1, #items do s = s + len(items[i]) end
          out[1] = s
          out[2] = strs("hello")
          out[3] = tabs({})
        end
        local late = {1, 2, 3}
        for r = 1, 300 do len(late) end
        setmetatable(late, {__len = function() return -1 end})
        out[4] = len(late)
        out[5] = select(2, pcall(len, 5))
        return table.concat(out, " ")
"#,
    );
}

/// String methods through the strings' metatable, its `__index` and the
/// library's slot, each guarded; replacing a method, or `__index` itself, after
/// compiling.
#[test]
fn string_methods() {
    agrees(
        r#"
        local function work(s, i)
          return s:upper() .. s:sub(2, 3) .. (s:byte(i % #s + 1) or 0) .. s:len() .. s:rep(2, "-") .. ("%d"):format(i)
        end
        local out = {}
        for r = 1, 300 do out[1] = work("hello", r) end
        local saved = string.sub
        string.sub = function(s, i, j) return "SUB" end
        for r = 1, 50 do out[2] = work("abc", r) end
        string.sub = saved
        local smt = getmetatable("")
        local lib = smt.__index
        smt.__index = {upper = function() return "hijacked" end, sub = lib.sub, byte = lib.byte, len = lib.len, rep = lib.rep, format = lib.format}
        for r = 1, 50 do out[3] = work("xyz", r) end
        smt.__index = lib
        for r = 1, 50 do out[4] = work("back", r) end
        return table.concat(out, " ")
"#,
    );
}

/// `next` over named slots inline: inline and spilled slots, nil ones, a
/// table keeping its metatable in a slot, stores to existing fields while
/// walking, and dict-mode tables through the helper.
#[test]
fn next_over_named_slots() {
    agrees(
        r#"
        local function keys(t) local n, s = 0, 0 for k, v in pairs(t) do n = n + 1; if type(v) == "number" then s = s + v end end return n * 1000 + s end
        local function cat(t) local o = {} for k, v in pairs(t) do o[#o + 1] = tostring(k) .. "=" .. tostring(v) end table.sort(o) return table.concat(o, ",") end
        local big = {}
        for i = 1, 40 do big["f" .. i] = i end
        local mixed = {1, 2, 3, a = 1, b = 2, c = 3, [10] = 5, [1.5] = 6}
        local holes = {x = 1, y = 2, z = 3}
        holes.y = nil
        local P = {}; P.__index = P
        local obj = setmetatable({u = 1, v = 2}, P)
        local poly = setmetatable({p = 1}, setmetatable({}, {__index = P}))
        local out = {}
        for r = 1, 300 do
          out[1] = keys(big); out[2] = keys(mixed); out[3] = keys(holes); out[4] = keys(obj); out[5] = keys(poly)
          out[6] = cat(mixed); out[7] = cat(holes)
        end
        -- assignment to existing fields while traversing
        local m = {a = 1, b = 2, c = 3}
        for r = 1, 300 do for k, v in pairs(m) do m[k] = v + 1 end end
        out[8] = cat(m)
        local dict = {}
        for i = 1, 300 do dict["d" .. i] = i end
        out[9] = keys(dict)
        return table.concat(out, " ")
"#,
    );
}

/// A field site that saw several shapes compiles a test per shape (own
/// slots, `__index` holders, absent keys); one that saw more takes the
/// helper.
#[test]
fn polymorphic_field_sites() {
    agrees(
        r#"
        local A = {}; A.__index = A
        function A.new(v) return setmetatable({v = v}, A) end
        function A:get() return self.v end
        local B = {}; B.__index = B
        function B.new(v) return setmetatable({w = 0, v = v * 2}, B) end
        function B:get() return self.v + 1 end
        local objs = {}
        for i = 1, 60 do objs[i] = i % 2 == 0 and A.new(i) or B.new(i) end
        local function sumget(t) local s = 0 for i = 1, #t do s = s + t[i]:get() + t[i].v end return s end
        local shapes, mega = {}, {}
        for i = 1, 40 do
          local k = i % 4
          shapes[i] = k == 0 and {x = i} or (k == 1 and {a = 0, x = i} or (k == 2 and {a = 0, b = 0, x = i} or {y = i}))
        end
        for i = 1, 50 do local t = {} for j = 1, i % 9 do t["f" .. j] = j end t.x = i mega[i] = t end
        local function bump(t) for i = 1, #t do t[i].x = (t[i].x or 0) + 1 end end
        local function total(t) local s = 0 for i = 1, #t do s = s + (t[i].x or -1) end return s end
        local s = 0
        for r = 1, 300 do s = s + sumget(objs); bump(shapes); s = s + total(shapes) + total(mega) end
        function A:get() return -self.v end
        for r = 1, 100 do s = s + sumget(objs) end
        return tostring(s)
"#,
    );
}

/// Calls from compiled code to compiled functions take the fast path
/// (`blr` to the callee's fast entry, `ret` back), and leave it for the
/// caller's slow path or the interpreter in every case the fast entry does
/// not cover: missing arguments, a stack window that needs growing deep in a
/// recursion, a callee with varargs, a yield, an error, a deopt, an
/// allocation, a tail call.
#[test]
fn fast_calls() {
    agrees(
        r##"
        local out = {}
        local function sum(n) if n == 0 then return 0 end return n + sum(n - 1) end
        local s = 0
        for i = 1, 300 do s = s + sum(i % 40) end
        out[1] = s .. " " .. sum(10000)
        local function opt(a, b) return a + (b or 100) end
        local function both(i) return opt(i) + opt(i, i) end
        s = 0
        for i = 1, 500 do s = s + both(i) end
        out[2] = s
        local function grow(n) local a, b, c, d, e, f, g = n, n, n, n, n, n, n
          if n == 0 then return 0 end return grow(n - 1) + a - b + c - d + e - f + g end
        out[3] = grow(3000)
        local function va(...) return select("#", ...) end
        local function callva(i) return va(i, i) + va() end
        s = 0
        for i = 1, 400 do s = s + callva(i) end
        out[4] = s
        local function inner(x) coroutine.yield(x) return x + 1 end
        local function outer(x) return inner(x) * 2 end
        local co = coroutine.wrap(function() local t = 0 for i = 1, 600 do t = t + outer(i) end return t end)
        local last
        for i = 1, 601 do last = co() end
        out[5] = last
        local function bad(x) if x % 97 == 0 then error("e" .. x) end return x end
        local function mid(x) return bad(x) + 1 end
        local ok, fails = 0, 0
        for i = 1, 700 do if pcall(mid, i) then ok = ok + 1 else fails = fails + 1 end end
        out[6] = ok .. "/" .. fails
        local function g(x) return x + 1 end
        local function f(x) return g(x) * 2 end
        s = 0
        for i = 1, 400 do s = s + f(i) end
        for i = 1, 400 do s = s + f(i + 0.5) end
        out[7] = s
        local function mk(i) return {i, i + 1} end
        local function use(i) local t = mk(i) return t[1] + t[2] end
        s = 0
        for i = 1, 20000 do s = s + use(i) end
        out[8] = s
        local function t2(x) return x * 3 end
        local function t1(x) return t2(x + 1) end
        local function tc(x) return t1(x) + 1 end
        s = 0
        for i = 1, 500 do s = s + tc(i) end
        out[9] = s
        local function three(x) return x, x + 1, x + 2 end
        local function take(i) local a, b, c = three(i) return a * b - c end
        s = 0
        for i = 1, 500 do s = s + take(i) end
        out[10] = s
        return table.concat(out, " ")
"##,
    );
}

/// The value pairs every operator site is warmed up on and then probed
/// with, as a chunk returning `fmt(ok, v)` for each probe.
const KINDS_PRELUDE: &str = r##"
    local mt = {}
    for _, k in ipairs({"add", "sub", "mul", "div", "mod", "pow", "unm", "idiv",
                        "band", "bor", "bxor", "shl", "shr", "bnot", "lt", "le", "eq"}) do
        mt["__" .. k] = function(a, b) return k end
    end
    local vals = {1, -7, 1 << 40, 1.5, -0.0, 2^53, (1 << 53) + 1, math.huge,
                  "10", "x", {}, true, setmetatable({}, mt), 0/0, math.mininteger, 2^31}
    local warm = {{1, 2}, {4, 5}, {1, 4}, {3, 1}, {9, 2}, {13, 1}, {7, 6}}
    local function fmt(ok, v)
        if not ok then return "E:" .. tostring(v) end
        return (math.type(v) or type(v)) .. ":" .. tostring(v)
    end
    local out = {}
    local function probe(expr)
        for _, w in ipairs(warm) do
            local g = load("return function(a, b) return " .. expr .. " end")()
            local a, b = vals[w[1]], vals[w[2]]
            for i = 1, 300 do pcall(g, a, b) end
            for _, x in ipairs(vals) do
                for _, y in ipairs(vals) do
                    out[#out + 1] = fmt(pcall(g, x, y))
                end
            end
        end
    end
"##;

/// Every arithmetic operator, register and immediate forms and the unary
/// ones, compiled after warm-ups of several kind mixes, then run on every
/// pair of values: compiled code exits or takes its slow arm where the
/// kinds differ, and the results equal the interpreter's.
#[test]
fn arith_kinds() {
    let src = format!(
        "{KINDS_PRELUDE}
        for _, e in ipairs({{\"a + b\", \"a - b\", \"a * b\", \"a / b\", \"a // b\", \"a % b\",
            \"a ^ b\", \"a & b\", \"a | b\", \"a ~ b\", \"a << b\", \"a >> b\", \"a + 1\",
            \"a * 2.5\", \"3 - a\", \"a // 2\", \"a % 3\", \"a ^ 2\", \"a ^ 0.5\", \"2 ^ a\",
            \"-a\", \"~a\", \"a & 7\", \"a << 1\", \"1.5 / a\"}}) do
            probe(e)
        end
        return table.concat(out, ' ')"
    );
    agrees(&src);
}

/// `math.min`/`math.max` compiled after warm-ups of several kind mixes, then
/// run on every pair of values: tables with `__lt` are ordered by it (#313),
/// the other kinds raise or compare as the interpreter does.
#[test]
fn min_max_kinds() {
    let src = r##"
    local mt = {__lt = function(a, b)
        local x = type(a) == "table" and a.n or a
        local y = type(b) == "table" and b.n or b
        return x < y
    end}
    local function o(n) return setmetatable({n = n}, mt) end
    local vals = {1, -7, 1 << 40, 1.5, 0/0, 2^53, "10", "x", o(2), o(-1), o(1.5), {}, true}
    local warm = {{1, 2}, {4, 3}, {1, 4}, {7, 8}, {9, 10}, {1, 9}}
    local function fmt(ok, v)
        if not ok then return "E:" .. tostring(v) end
        if type(v) == "table" then return "o:" .. tostring(v.n) end
        return (math.type(v) or type(v)) .. ":" .. tostring(v)
    end
    local out = {}
    for _, expr in ipairs({"math.min(a, b)", "math.max(a, b)", "math.min(a, 2, b)",
                           "math.max(b, a, 1.5)"}) do
        for _, w in ipairs(warm) do
            local g = load("return function(a, b) return " .. expr .. " end")()
            local a, b = vals[w[1]], vals[w[2]]
            for i = 1, 300 do pcall(g, a, b) end
            for _, x in ipairs(vals) do
                for _, y in ipairs(vals) do
                    out[#out + 1] = fmt(pcall(g, x, y))
                end
            end
        end
    end
    return table.concat(out, " ")
"##;
    let jit = result(src, true);
    assert!(jit.contains("o:-1"), "no pick by __lt");
    assert_eq!(jit, result(src, false));
}

/// Every compare, register and immediate forms, compiled after warm-ups of
/// several kind mixes, then run on every pair of values: numbers around
/// 2^53, NaNs and signed zeros, strings with zeros and high bytes, tables
/// with and without metamethods.
#[test]
fn cmp_kinds() {
    let src = r##"
    local mt = {__eq = function() return "eq" end, __lt = function() return "lt" end,
                __le = function() return false end}
    local vals = {1, -7, 1 << 40, 1.5, -0.0, 0.0, 2^53, (1 << 53) + 1, 9007199254740993,
                  math.huge, 0/0, -(0/0), math.mininteger, 2^31, "10", "a", "ab", "b",
                  "a\0b", "a\0c", "", "\xff", "a\xff", {}, true, false,
                  setmetatable({}, mt), setmetatable({}, mt)}
    local warm = {{1, 2}, {4, 5}, {1, 4}, {3, 1}, {16, 17}, {27, 28}, {8, 7}, {11, 12}, {24, 25}}
    local function fmt(ok, v)
        if not ok then return "E:" .. tostring(v) end
        return tostring(v)
    end
    local out = {}
    for _, expr in ipairs({"a < b", "a <= b", "a > b", "a >= b", "a == b", "a ~= b",
                           "a < 3", "a <= -2", "a > 3", "a >= 0", "3 < a", "a == 5",
                           "a ~= 0", "a == 'a'", "not (a < b)"}) do
        for _, w in ipairs(warm) do
            local g = load("return function(a, b) return " .. expr .. " end")()
            local a, b = vals[w[1]], vals[w[2]]
            for i = 1, 300 do pcall(g, a, b) end
            for _, x in ipairs(vals) do
                for _, y in ipairs(vals) do
                    out[#out + 1] = fmt(pcall(g, x, y))
                end
            end
        end
    end
    return table.concat(out, " ")
"##;
    agrees(src);
}

/// `==` compiled at a site that saw only tables compares the big integers
/// that arrive later by value, with each other and with floats.
#[test]
fn eq_of_big_integers_the_site_never_saw() {
    let src = r##"
    local function eq(a, b) return a == b end
    local x = 1 << 40
    local t = {}
    for i = 1, 3000 do eq(t, t) end
    local out = {eq(x, x + 0), eq(x, 2.0^40), eq(2.0^40, x),
                 eq(x, x + 1), eq(x, 2.0^41), eq(2.0^41, x), eq(x, t), eq(1.5, t), eq("a", x)}
    local n = 0
    for i = 1, 3000 do
        if eq(x, x + 0) then n = n + 1 end
        if eq(2.0^40, x) then n = n + 1 end
    end
    for i = 1, #out do out[i] = tostring(out[i]) end
    return table.concat(out, " ") .. " " .. n
"##;
    let want = "true true true false false false false false false 6000";
    assert_eq!(result(src, true), want);
    let mut lua = Lua::new();
    lua.set_jit_inline(false);
    lua.load_all();
    let ex = start_on(&mut lua, src);
    lua.finish(&ex).unwrap();
    lua.enter(|ctx| {
        let v: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        assert_eq!(
            v.get_string().expect("a string").as_bytes(),
            want.as_bytes()
        );
    });
}

/// Array reads and writes compiled without misses, then given misses:
/// holes, keys outside the array part (0, negative, huge, floats), and
/// classes with `__index` or `__newindex` arriving after the compile.
#[test]
fn array_access_misses() {
    agrees(
        r##"
        local out = {}
        local function sum(t, n)
            local s = 0
            for i = 1, n do s = s + (t[i] or 100) end
            return s
        end
        local function fill(t, n, v)
            for i = 1, n do t[i] = v end
        end
        local a = {}
        for i = 1, 50 do a[i] = i end
        for r = 1, 30 do out[#out + 1] = sum(a, 50) end
        -- Past the end, a hole, a sparse key.
        out[#out + 1] = sum(a, 60)
        a[25] = nil
        out[#out + 1] = sum(a, 50)
        local sparse = {}
        sparse[1000000] = 7
        for i = 1, 20 do out[#out + 1] = sum(sparse, 3) end
        -- A class with __index after the compile.
        local b = {}
        for i = 1, 40 do b[i] = i end
        for r = 1, 30 do out[#out + 1] = sum(b, 40) end
        setmetatable(b, {__index = function(t, k) return k * 1000 end})
        b[3] = nil
        out[#out + 1] = sum(b, 45)
        -- Keys 0, negative, float, string at a compiled site.
        local function get(t, k) return t[k] end
        local c = {10, 20, 30, [0] = "zero", [-1] = "neg", x = "str"}
        for i = 1, 300 do get(c, (i % 3) + 1) end
        for _, k in ipairs({0, -1, 1.0, 2.5, "x", 3, 4, 2^31, math.mininteger}) do
            out[#out + 1] = tostring(get(c, k))
        end
        -- Writes: appends, holes, __newindex after the compile.
        local d = {}
        for r = 1, 30 do fill(d, 20, r) end
        fill(d, 25, "x")
        out[#out + 1] = #d .. ":" .. tostring(d[25])
        local log = {}
        local e = setmetatable({}, {__newindex = function(t, k, v) log[#log + 1] = k; rawset(t, k, v) end})
        for i = 1, 10 do rawset(e, i, 0) end
        for r = 1, 30 do fill(e, 10, r) end
        e[5] = nil
        fill(e, 12, "y")
        out[#out + 1] = table.concat(log, ",") .. ":" .. tostring(e[5]) .. tostring(e[12])
        local function put(t, k, v) t[k] = v end
        local f = {1, 2, 3}
        for i = 1, 300 do put(f, (i % 3) + 1, i) end
        for _, k in ipairs({0, -1, 1.0, 2.5, "x", 4, 2^31}) do put(f, k, "w") end
        out[#out + 1] = tostring(f[0]) .. tostring(f[-1]) .. tostring(f[1]) .. tostring(f[2.5])
            .. tostring(f.x) .. tostring(f[4]) .. tostring(f[2^31]) .. #f
        return table.concat(out, " ")
"##,
    );
}

/// 64-bit shifts by counts at and past the width, `math.mininteger` among
/// them, whose negation is itself.
#[test]
fn shift_counts_at_the_extremes() {
    agrees(
        r##"
        local function sh(a, b) return a << b, a >> b end
        local out = {}
        for i = 1, 300 do sh(1 << 40, i % 70) end
        for _, c in ipairs({math.mininteger, -64, -63, 63, 64, math.maxinteger, -1}) do
            local l, r = sh(5, c)
            out[#out + 1] = l .. ":" .. r
        end
        return table.concat(out, " ")
"##,
    );
}

/// By-value upvalues of one prototype's closures: one that differs between
/// closures is loaded, one the same in all stays a constant; closures that
/// differ made after a region compiled the first one's values as constants.
#[test]
fn upvalues_that_vary_between_closures() {
    agrees(
        "local function mk(p, drag)
          return function(n)
            local s = 0
            for i = 1, n do s = s + p.x * drag + i end
            return s
          end
        end
        local first = mk({x = 1}, 0.5)
        local s = 0
        for _ = 1, 500 do s = s + first(10) end
        local others = {}
        for i = 1, 20 do others[i] = mk({x = i}, 0.5) end
        for _ = 1, 50 do for i = 1, 20 do s = s + others[i](10) end end
        local self_ref = {}
        for i = 1, 20 do
          local function f(n) if n == 0 then return i end return f(n - 1) end
          self_ref[i] = f
        end
        for _ = 1, 50 do for i = 1, 20 do s = s + self_ref[i](3) end end
        return tostring(s)",
    );
}

/// Values compiled as constants: by-value upvalues that differ between
/// closures of one prototype, a loop's function register reassigned inside
/// it, a global builtin replaced between runs, intrinsics given big
/// integers.
#[test]
fn constants_the_builder_sees() {
    agrees(
        r##"
        local out = {}
        local function adder(k) return function(x) return x + k end end
        local fs = {}
        for i = 1, 8 do fs[i] = adder(i * 10) end
        local s = 0
        for r = 1, 400 do for i = 1, 8 do s = s + fs[i](r) end end
        out[#out + 1] = s
        local f, t = math.sin, 0
        for i = 1, 600 do
            t = t + f(i)
            if i == 300 then f = math.cos end
        end
        out[#out + 1] = string.format("%.10f", t)
        local function check(n)
            local c = 0
            for i = 1, n do if assert(i > 0, "pos") then c = c + 1 end end
            return c
        end
        out[#out + 1] = check(500)
        local real = assert
        assert = function(v, m) return v and "mine" or m end
        out[#out + 1] = check(500)
        assert = real
        local big = math.maxinteger // 3
        local q = 0
        for i = 1, 400 do q = q + math.sqrt(big + i) + math.floor(big + i * 0.5) * 0 end
        out[#out + 1] = string.format("%.3f", q)
        return table.concat(out, " ")
"##,
    );
}

/// An accumulator that starts an integer and becomes a float keeps its
/// kinds per iteration; a counter overflowing i32 mid-loop feeds float
/// operations and intrinsics.
#[test]
fn kinds_that_change_in_a_loop() {
    agrees(
        r##"
        local out = {}
        local function acc(n, x)
            local a = 0
            for i = 1, n do a = a + x end
            return math.type(a) .. ":" .. tostring(a)
        end
        for r = 1, 300 do acc(10, 0.5) end
        out[#out + 1] = acc(0, 0.5) .. " " .. acc(1, 0.5) .. " " .. acc(7, 0.5) .. " " .. acc(3, 2)
        local s = 0
        local k = 2147483000
        for i = 1, 2000 do
            k = k + 1
            local k3 = k * k * k
            s = s + math.sin(k) + (2 / 3) ^ (k % 7) + k3 / 1e30 + math.sqrt(k)
        end
        out[#out + 1] = string.format("%.6f", s) .. " " .. math.type(k)
        return table.concat(out, " ")
"##,
    );
}

/// Values of every kind live across compiled calls stay right when the callee
/// returns straight (the frame kept), and when it leaves compiled code first: a
/// native, a yield, an error, a collection, deep recursion that grows the stack
/// and tags return addresses, all resumed through the header and the values
/// read back from their slots.
#[test]
fn values_live_across_calls() {
    agrees(
        r##"
        local out = {}
        local mode = 0
        local function leaf(n)
            if mode == 1 then return #tostring(n) end
            if mode == 2 then return (coroutine.yield(n)) end
            if mode == 3 and n % 7 == 3 then error("boom " .. n) end
            if mode == 4 then local s = "" for i = 1, 20 do s = s .. i end return #s end
            if mode == 5 then
                local function deep(k) if k == 0 then return 0 end return 1 + deep(k - 1) end
                return deep(400) - 400 + n % 3
            end
            return n % 5
        end
        local function work(n)
            local i, big, f, t, s, b = 0, math.maxinteger - 5, 0.5, {k = 1}, "str", true
            local acc = 0
            for j = 1, n do
                i = i + 1
                f = f * 1.0001 + 0.25
                big = big + 1
                local r = leaf(j)
                acc = acc + r + i + (b and 1 or 0) + t.k + #s
                b = not b
                if big > math.maxinteger - 3 then big = big - 7 end
            end
            return string.format("%d %d %.6f %s %s", acc, big, f, tostring(b), s)
        end
        for m = 0, 5 do
            mode = m
            if m == 2 then
                local co = coroutine.wrap(function() return work(300) end)
                local r = co()
                local k = 0
                while type(r) == "number" do k = k + 1; r = co(r * 2) end
                out[#out + 1] = r .. " " .. k
            elseif m == 3 then
                for rep = 1, 5 do
                    local ok, e = pcall(work, 300)
                    out[#out + 1] = tostring(ok) .. " " .. tostring(e)
                end
                mode = 0
                out[#out + 1] = work(300)
            else
                for rep = 1, 3 do out[#out + 1] = work(300) end
            end
        end
        return table.concat(out, "\n")
"##,
    );
}

/// Invariant i32 arithmetic on a path the loop does not always take moves
/// out of the loop; where it leaves i32 the loop runs in the interpreter.
#[test]
fn invariant_arithmetic_leaves_the_loop() {
    agrees(
        r##"
        local function count(t, c, n)
            local hits = 0
            for i = 1, #t do
                local v = t[i]
                if v == 0 then hits = hits + 1
                elseif v - i == c - n or v + i == c + n then hits = hits + 10 end
            end
            return hits
        end
        local t = {}
        for i = 1, 64 do t[i] = i % 5 end
        local out = {}
        for r = 1, 300 do out[1] = count(t, r % 7, 3) end
        out[2] = count(t, math.maxinteger, -5)
        out[3] = count(t, 2147483647, -1)
        out[4] = count(t, -2147483648, 1)
        out[5] = count(t, 4, 1)
        return table.concat(out, " ")
"##,
    );
}

/// Float `%` inline: signed zeros, infinities, NaN, huge and tiny quotients,
/// exact multiples and the doubles either side of them.
#[test]
fn float_modulo_edges() {
    agrees(
        r##"
        local function bits(n) return string.unpack("<d", string.pack("<i8", n)) end
        local function step(x, d)
            return bits(string.unpack("<i8", string.pack("<d", x)) + d)
        end
        local as = {0.0, -0.0, 1.0, -1.0, 5.5, -5.5, 4.0, -4.0, 7.0, -7.0, 0.3, -0.3,
            1e308, -1e308, 1e-300, -1e-300, 5e-324, -5e-324, 2.0^52, 2.0^53 + 2,
            -2.0^60, 1e15 + 0.3, 1/0, -1/0, 0/0}
        local bs = {2.0, -2.0, 3.0, -3.0, 0.1, -0.1, 1.0, 0.7, -0.7, 2.0^-30, 1e10,
            1e300, 1e-300, 5e-324, 0.0, -0.0, 1/0, -1/0, 0/0}
        for _, k in ipairs({3.0, 10.0, 1e6, 2.0^40, -3.0, -1e6}) do
            for _, b in ipairs({0.1, 0.7, 3.3}) do
                local m = k * b
                as[#as + 1] = m
                as[#as + 1] = step(m, 1)
                as[#as + 1] = step(m, -1)
            end
        end
        local function run()
            local out = {}
            for i = 1, #as do
                local a = as[i]
                for j = 1, #bs do
                    local r = a % bs[j]
                    out[#out + 1] = r ~= r and "nan" or string.format("%a", r)
                end
            end
            return out
        end
        local out
        for _ = 1, 30 do out = run() end
        return table.concat(out, " ")
"##,
    );
}

/// String keys of table accesses: counting into a dictionary, a small table,
/// `__index` tables and functions, `__newindex` from the start and gained
/// later, other keys and receivers at a string site.
#[test]
fn string_table_keys() {
    agrees(
        r##"
        local function get(t, k) return t[k] end
        local function put(t, k, v) t[k] = v end
        local function count(t, s, n)
          for i = 1, #s - n + 1 do
            local k = string.sub(s, i, i + n - 1)
            t[k] = (t[k] or 0) + 1
          end
        end
        local seq = string.rep("ACGTTGCAAGCT", 40)
        local freq = {}
        for r = 1, 50 do count(freq, seq, 3) end
        local keys = {}
        for k in pairs(freq) do keys[#keys + 1] = k end
        table.sort(keys)
        local out = {#keys, freq.ACG, freq.TGC}
        local small = {}
        for r = 1, 300 do put(small, "k" .. (r % 5), r); out[4] = get(small, "k3") end
        local base = {inherited = "yes"}
        local obj = setmetatable({}, {__index = base})
        for r = 1, 300 do out[5] = get(obj, "inherited"); out[6] = tostring(get(obj, "missing")) end
        local fn = setmetatable({}, {__index = function(t, k) return k .. "!" end})
        for r = 1, 300 do out[7] = get(fn, "hi") end
        local log = {}
        local guarded = setmetatable({}, {__newindex = function(t, k, v) log[#log + 1] = k; rawset(t, k, v) end})
        for r = 1, 300 do put(guarded, "a" .. (r % 3), r) end
        out[8] = #log
        local mt = {}
        local late = setmetatable({}, mt)
        for r = 1, 300 do put(late, "x", r) end
        mt.__newindex = function(t, k, v) log[#log + 1] = "late:" .. k end
        put(late, "y", 1); put(late, "x", 2)
        out[9] = #log .. ":" .. tostring(rawget(late, "y")) .. ":" .. late.x
        out[10] = tostring(get(freq, 1)); put(freq, 2, "two"); out[11] = freq[2]
        out[12] = tostring(select(2, pcall(get, nil, "x")) ~= nil)
        local s = setmetatable({}, {__index = string})
        out[13] = get(s, "upper")("ok")
        return table.concat(out, " ")
"##,
    );
}

/// Float keys of array accesses: integral floats as the integers' keys, then
/// fractions, -0.0, floats past i32 and NaN, read and written, at sites that
/// saw only integral floats before.
#[test]
fn float_array_keys() {
    agrees(
        r##"
        local function get(t, k) return t[k] end
        local function put(t, k, v) t[k] = v end
        local t = {}
        for i = 1, 64 do t[i] = i * 10 end
        local s = 0
        for r = 1, 300 do
          for i = 1, 64 do s = s + get(t, i / 1) end
          for i = 1, 64 do put(t, i * 1.0, i) end
        end
        -- fractions, -0.0, huge floats, nan, mixed keys
        local f = {}
        f[0.5] = "half"; f[2^40] = "big"; f[0] = "zero"
        for r = 1, 300 do s = s + (get(t, r % 64 + 1.0) or 0) end
        local out = {s, get(f, 0.5), get(f, 2^40), get(f, -0.0), tostring(get(t, 0.0/0.0)), get(t, 3.0), get(t, 3)}
        put(f, 1.5, "x"); put(f, -0.0, "negzero")
        out[#out + 1] = f[1.5]; out[#out + 1] = f[0]
        local ok, err = pcall(put, f, 0/0, 1)
        out[#out + 1] = tostring(ok)
        for r = 1, 300 do put(t, r % 64 + 0.5, r) end
        out[#out + 1] = t[1.5]
        local n = 0 for k in pairs(t) do if math.type(k) == "float" then n = n + 1 end end
        out[#out + 1] = n
        return table.concat(out, " ")
"##,
    );
}

/// Inlined callees that store through the table helpers, where each helper
/// fails over to the builtin or the interpreter: `__newindex` and `__len` under
/// `table.insert`/`table.remove`, a protected metatable, a nil `rawset` key,
/// and a store whose class gains `__newindex` later.
#[test]
fn inlined_table_helpers() {
    agrees(
        r##"
        local log = {}
        local mt = {}
        local keys = {"p", "q", "r"}
        local function put(t, k, v) return rawset(t, k, v) end
        local function push(t, v) table.insert(t, v) end
        local function pop(t) return table.remove(t) end
        local function mk(x, m) return setmetatable({x = x}, m) end
        local function set(t, k, v) t[k] = v end
        local function run(t, n, m)
            local s = 0
            for i = 1, n do
                put(t, "a", i); push(t, i); s = s + pop(t)
                local o = mk(i, m); set(o, keys[i % 3 + 1], i); s = s + o.x
            end
            return s
        end
        local out = {}
        for r = 1, 300 do out[1] = run({}, 10, mt) end
        local logged = setmetatable({}, {__newindex = function(t, k, v)
            log[#log + 1] = k
            rawset(t, k, v)
        end})
        out[2] = run(logged, 3, mt)
        local counted = setmetatable({}, {__len = function() return 0 end})
        out[3] = tostring(pcall(run, counted, 3, mt))
        out[4] = select(2, pcall(run, {}, 3, setmetatable({}, {__metatable = "no"})))
        local function remeta(t, m) return setmetatable(t, m) end
        local function remetas(n) for i = 1, n do remeta({}, mt) end end
        for r = 1, 300 do remetas(5) end
        out[5] = select(2, pcall(remeta, mk(0, {__metatable = "locked"}), mt))
        out[6] = select(2, pcall(put, {}, nil, 1))
        mt.__newindex = function(t, k, v) log[#log + 1] = k end
        out[7] = run({}, 4, mt)
        out[8] = #log .. ":" .. table.concat(log, ",")
        return table.concat(out, " ")
"##,
    );
}

/// Array elements under classes with `__index`/`__newindex`: present ones read
/// and written plainly, nil ones through the metamethod, for varying and
/// invariant tables, a class gaining one later, a nil under a use that fails on
/// it, and an `__index` table.
#[test]
fn array_misses_under_classes() {
    agrees(
        r##"
        local function total(rows, j)
            local s = 0
            for i = 1, #rows do s = s + rows[i][j] end
            return s
        end
        local function sum(t, n)
            local s = 0
            for i = 1, n do s = s + (t[i] or 0) end
            return s
        end
        local function count(t, n)
            local c = 0
            for i = 1, n do if t[i] then c = c + 1 end end
            return c
        end
        local function fill(t, n, v) for i = 1, n do t[i] = v end end
        local function fill_rows(rows, j, v)
            for i = 1, #rows do rows[i][j] = v end
        end
        local out = {}
        local rows = {}
        for i = 1, 50 do rows[i] = {1, 2, 3, 4} end
        for r = 1, 300 do out[1] = total(rows, r % 4 + 1) end
        setmetatable(rows[7], {__index = function() return 100 end})
        for r = 1, 300 do out[2] = total(rows, 2) end
        rows[7][2] = nil
        out[3] = total(rows, 2)
        setmetatable(rows[9], {__index = {[3] = 0.5}})
        rows[9][3] = nil
        out[4] = total(rows, 3)
        rows[11][4] = nil
        out[5] = select(2, pcall(total, rows, 4))
        local h = {1, 2, 3, 4, 5, 6, 7, 8}
        for r = 1, 300 do out[6] = sum(h, 8) + count(h, 8) end
        h[3] = nil
        out[7] = sum(h, 8) + count(h, 8)
        setmetatable(h, {__index = function(_, k) return k * 10 end})
        out[8] = sum(h, 8) + count(h, 8)
        local log = {}
        local p = {1, 2, 3, 4}
        for r = 1, 300 do fill(p, 4, r) end
        setmetatable(p, {__newindex = function(t, k, v)
            log[#log + 1] = k
            rawset(t, k, v)
        end})
        fill(p, 4, 7)
        out[9] = #log
        p[2] = nil
        fill(p, 4, 9)
        out[10] = #log .. ":" .. tostring(log[1]) .. ":" .. p[2]
        local q = {}
        for i = 1, 40 do q[i] = {0, 0, 0} end
        for r = 1, 300 do fill_rows(q, r % 3 + 1, r) end
        local seen = 0
        setmetatable(q[5], {__newindex = function(t, k, v) seen = seen + v end})
        fill_rows(q, 2, 1)
        q[5][2] = nil
        fill_rows(q, 2, 3)
        out[11] = seen .. ":" .. tostring(q[5][2]) .. ":" .. q[6][2]
        return table.concat(out, " ")
"##,
    );
}

/// Bounds checks against a loop's range: limits past the array, early breaks,
/// negative and non-unit steps, an array grown in the loop and empty loops.
#[test]
fn loop_range_bounds() {
    agrees(
        r##"
        local t = {}
        for i = 1, 100 do t[i] = i end
        local function sum(t, a, b, s)
            local x = 0
            for i = a, b, s do x = x + t[i] end
            return x
        end
        local function find(t, n, v)
            for i = 1, n do
                local y = t[i]
                if y == v or y == nil then return i end
            end
            return -1
        end
        local function grow(t, n)
            local u = {}
            for i = 1, n do u[i] = i; u[i] = u[i] + (t[i] or 0) end
            return #u
        end
        local function store(t, a, b)
            for i = a, b do t[i] = i * 2 end
            return t[b]
        end
        local out = {}
        for r = 1, 400 do
            out[1] = sum(t, 1, 100, 1)
            out[2] = sum(t, 100, 1, -1)
            out[3] = sum(t, 2, 99, 3)
            out[4] = sum(t, 5, 4, 1)
            out[5] = find(t, 100, r % 120)
            out[6] = find(t, 150, 50)
            out[7] = grow(t, 120)
            out[8] = store({}, 1, 10)
            out[9] = store(t, 1, 100)
        end
        out[10] = pcall(sum, t, 1, 101, 1)
        out[11] = find(t, 1000, -5)
        out[12] = pcall(sum, t, 0, 100, 1)
        out[13] = sum(t, 100, 1, -7)
        out[14] = select(2, pcall(sum, {1, 2, 3}, 1, 4, 1)) ~= nil
        return table.concat(out, " ", 1, 9) .. " " .. tostring(out[10]) .. " "
            .. out[11] .. " " .. tostring(out[12]) .. " " .. out[13] .. " " .. tostring(out[14])
"##,
    );
}

/// Float compares of converted integers become integer compares where the
/// constant converts back exactly.
#[test]
fn int_compares_against_float_constants() {
    agrees(
        r##"
        local function f(x, y)
            local n = 0
            if x < 2.0 then n = n + 1 end
            if x <= 2.5 then n = n + 2 end
            if x == -0.0 then n = n + 4 end
            if x < 2147483648.0 then n = n + 8 end
            if 3.0 >= x then n = n + 16 end
            if x + 0.0 < y + 0.0 then n = n + 32 end
            if x > -2147483648.0 then n = n + 64 end
            if x ~= 1e300 then n = n + 128 end
            return n
        end
        local out = {}
        for r = 1, 300 do
            local x = r % 9 - 4
            out[#out + 1] = f(x, 2 - x)
        end
        out[#out + 1] = f(2147483647, -2147483648)
        out[#out + 1] = f(-2147483648, 0)
        out[#out + 1] = f(math.maxinteger, 1)
        out[#out + 1] = f(2.5, 3)
        return table.concat(out, " ")
"##,
    );
}

/// A compare with a float literal hands `__lt`/`__le` a float, on the side
/// the source has it.
#[test]
fn float_immediates_reach_compare_metamethods() {
    let src = r##"
    local kinds = {}
    local function rec(a, b)
        local k = (math.type(a) or type(a)) .. "," .. (math.type(b) or type(b))
        kinds[k] = (kinds[k] or 0) + 1
        return true
    end
    local t = setmetatable({}, {__lt = rec, __le = rec})
    local seen = {}
    local function f(x)
        if x < 1.0 then seen.lt = (seen.lt or 0) + 1 end
        if x > 2.0 then seen.gt = (seen.gt or 0) + 1 end
        if x <= 3.0 then seen.le = (seen.le or 0) + 1 end
        if x >= 4 then seen.ge = (seen.ge or 0) + 1 end
    end
    for i = 1, 3000 do f(t) end
    local out = {}
    for k, v in pairs(kinds) do out[#out + 1] = k .. "=" .. v end
    table.sort(out)
    return table.concat(out, " ") .. " " .. seen.lt + seen.gt + seen.le + seen.ge
"##;
    assert_eq!(
        result(src, true),
        "float,table=3000 integer,table=3000 table,float=6000 12000"
    );
}

/// A loop prepared in code compiled when its site had seen only float loops
/// stays an integer loop when its init and step are integers.
#[test]
fn integer_loops_at_sites_that_saw_float_loops() {
    let src = r##"
    local function g(x, n, s)
        local c, last = 0
        for j = 1, n do
            for i = x, 3, s do last = i end
            if math.type(last) == "integer" then c = c + 1 end
        end
        return c .. ":" .. tostring(last)
    end
    local function h(a) local last; for i = a, 3000 do last = i end return last end
    local out = {g(1.0, 100000, 1), g(1, 1000, 1), g(1, 1000, 1.0), g(1.0, 1000, 1.0),
                 h(1.0), h(1), h(2990), math.type(h(1))}
    return table.concat(out, " ")
"##;
    assert_eq!(
        result(src, true),
        "0:3.0 1000:3 0:3.0 0:3.0 3000.0 3000 3000 integer"
    );
}

/// The same where the function compiles after its first call, a float loop.
#[test]
fn integer_loops_after_one_float_loop() {
    compiles_at_first_count(
        "local function f(b) local last; for i = b, 1 do last = i end; return last end
        return f(0.5) .. ' ' .. f(1) .. ' ' .. math.type(f(1))",
        true,
    );
}

/// A compiled float loop's entry is a JIT word, whose form no FORPREP
/// rewrites: integer loops there compile it as an integer loop, and float
/// loops after those leave it to the interpreter, neither exiting again
/// and again.
#[test]
fn loops_of_both_kinds_at_one_site_settle() {
    let src = "local function f(a, b, c) local s = 0 for i = a, b, c do s = s + i end return s end
        local r = 0
        for k = 1, 2000 do r = r + f(1.0, 10.0, 1.0) end
        for k = 1, 2000 do r = r + f(1, 10, 1) end
        for k = 1, 2000 do r = r + f(1.0, 10.0, 1.0) end
        return tostring(r)";
    let mut lua = Lua::new();
    lua.set_stats(true);
    lua.load_all();
    let ex = start_on(&mut lua, src);
    lua.finish(&ex).unwrap();
    let got = lua.enter(|ctx| {
        let v: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        String::from_utf8_lossy(v.get_string().expect("a string result").as_bytes()).into_owned()
    });
    assert_eq!(got, "330000.0");
    let stats = lua.stats();
    assert!(count(&stats, "jit exit Type") <= 20, "{stats}");
    assert!(count(&stats, "jit recompiled") <= 2, "{stats}");
}

/// Results of compiled callees in registers: none, one (stored, in its slot
/// already, a float, a constant), two and three into one and two wanted,
/// varargs, deep chains and returns through the interpreter.
#[test]
fn call_results() {
    agrees(
        r##"
        local function r0() end
        local function r1(x) return x end
        local function r1f(x) return x * 0.5 end
        local function r1k() return 42 end
        local function r2(x) return x, x + 1 end
        local function r3(x) return x, x + 1, x + 2 end
        local function rv(...) return ... end
        local function deep(n) if n == 0 then return "d" end return (deep(n - 1)) end
        local function odd(x) if x % 7 == 0 then return "s" .. x end return x end
        local out = {}
        for i = 1, 500 do
            local a = r0()
            local b = r1(i)
            local c = r1f(i)
            local d = r1k()
            local e, f = r2(i)
            local g, h = r1(i)
            local j, k = r3(i)
            local m = rv(i, 2)
            local n = deep(10)
            local o = odd(i)
            local p, q = r0()
            out[1] = tostring(a) .. b .. c .. d .. e .. f .. g .. tostring(h) .. j .. k
                .. m .. n .. o .. tostring(p) .. tostring(q)
        end
        return out[1]
"##,
    );
}

/// Headers a fast call leaves at word 0, read by everything after a leave:
/// tracebacks, error positions, a native call, an exit deep in a chain and a
/// yield across compiled frames.
#[test]
fn lazy_call_headers() {
    agrees(
        r##"
        local out = {}
        local function f3(x, mode)
            if mode == 1 then error("e" .. x) end
            if mode == 2 then return debug.traceback("t", 1) end
            if mode == 3 then return tostring(x) .. "n" end
            if mode == 4 then return coroutine.yield(x) end
            if mode == 5 then return "s" .. x end
            return x + 1
        end
        local function f2(x, mode) local r = f3(x, mode); return r end
        local function f1(x, mode) local r = f2(x + 1, mode); return r end
        for i = 1, 300 do
            local mode = i > 250 and (i % 6) or 0
            if mode == 4 then
                local co = coroutine.wrap(function() return f1(i, 4) end)
                out[#out + 1] = tostring(co()) .. tostring(co(7))
            elseif mode == 1 then
                local ok, e = pcall(f1, i, 1)
                out[#out + 1] = tostring(ok) .. e
            else
                out[#out + 1] = tostring(f1(i, mode))
            end
        end
        return table.concat(out, "|")
"##,
    );
}

/// Calls of a callee known at compile time: replaced through an upvalue and a
/// global, retired and recompiled while its caller's region runs, and mutual
/// recursion past the native depth limit.
#[test]
fn known_callees() {
    agrees(
        r##"
        local function inc(x) return x + 1 end
        local cur = inc
        local function viaup(x) return cur(x) end
        G_f = inc
        local function viaglobal(x) return G_f(x) end
        local function kinds(x)
            if x > 250 then return x .. "" end
            return x * 2
        end
        local function callkinds(x) return kinds(x) end
        local even, odd
        function even(n) if n == 0 then return true end return odd(n - 1) end
        function odd(n) if n == 0 then return false end return even(n - 1) end
        local out = {}
        for i = 1, 400 do
            if i == 200 then cur = function(x) return x - 1 end end
            if i == 300 then G_f = function(x) return -x end end
            out[1] = viaup(i)
            out[2] = viaglobal(i)
            out[3] = callkinds(i)
            out[4] = tostring(even(i % 20))
        end
        out[5] = tostring(even(30001))
        return table.concat(out, " ")
"##,
    );
}

/// Closures capturing registers of fast-called frames, some of which exit
/// and return in the interpreter: the closure sets `HAS_OPEN` in the
/// frame's header word 1, which the call zeroes and the walk keeps.
#[test]
fn closures_in_fast_called_frames() {
    agrees(
        r##"
        local function mk(i, v)
            local x = i
            local f = function() return x end
            local y = v + 1
            return f, y
        end
        local function fill(a, b, c, d) return a + b + c + d end
        local fs = {}
        for i = 1, 300 do
            local v = i
            if i % 50 == 0 then v = i + 0.5 end
            local f = mk(i, v)
            fill(-1, -2, -3, -4)
            fs[i] = f
        end
        local out = {}
        for i = 1, 300, 37 do out[#out + 1] = fs[i]() end
        return table.concat(out, " ")
"##,
    );
}

/// Inlined callees: exits inside the inlined frame (a kind change, an error),
/// results into one and two wanted, missing and extra arguments, nesting, loops
/// and upvalues in the callee, globals, a callee replaced at run time, and one
/// with a real call, which stays a call.
#[test]
fn inlined_calls() {
    agrees(
        r##"
        local K = 3
        local function sq(x) return x * x end
        local function two(x) return x, x + K end
        local function none(x) end
        local function opt(a, b) if b == nil then return a end return a + b end
        local function nest(x) return sq(x) + sq(x + 1) end
        local function deep(x) return nest(x) - nest(x - 1) end
        local function loopy(n) local s = 0 for i = 1, n do s = s + i end return s end
        local function glob(x) return math.floor(x / 2) end
        local function calls(x) return tostring(x) end
        local function field(t) return t.v + t[1] end
        local cur = sq
        local function viaup(x) return cur(x) end
        local out = {}
        for i = 1, 400 do
            local v = i
            if i == 350 then v = i + 0.5 end
            local a = sq(v)
            local b, c = two(i)
            local d = none(i)
            local e = opt(i) + opt(i, 2, 99)
            local g = deep(i)
            local h = loopy(i % 10)
            local m = glob(i)
            local n = calls(i)
            local p = field({v = i, i})
            if i == 300 then cur = function(x) return -x end end
            local q = viaup(i)
            local ok, err = pcall(sq, i < 390 and i or nil)
            out[1] = table.concat({a, b, c, tostring(d), e, g, h, m, n, p, q, tostring(ok)}, " ")
        end
        return out[1]
"##,
    );
}

/// Calls from compiled code into the interpreter run as levels on top of the
/// compiled frames: errors unwind across them, coroutines yield inside them,
/// the collector walks them, and a deopted fast-called frame that enters a loop
/// region again calls through a level of its own.
#[test]
fn levels() {
    agrees(
        r##"
        local out = {}
        local junk = load("return function(n) local t = {} for i = 1, n do t[i] = {i, tostring(i)} end return #t end")()
        local function work(x, n)
          local s = 0
          for i = 1, n do s = s + junk(10) + x end
          return s
        end
        local function caller(x) return work(x, 20) + 1 end
        local acc = 0
        for i = 1, 300 do acc = acc + caller(i) end
        for i = 1, 300 do acc = acc + caller(i + 0.5) end
        out[1] = acc
        local function thrower(x) if x % 5 == 0 then error("boom" .. x) end return x end
        local mid = load("local f = ... return function(x) return f(x) + 1 end")(thrower)
        local function top(x) local v = {x} return mid(v[1]) * 2 end
        local ok, fail, msg = 0, 0, nil
        for i = 1, 300 do
          local s, e = pcall(top, i)
          if s then ok = ok + e else fail = fail + 1 msg = e end
        end
        out[2] = ok .. " " .. fail .. " " .. msg
        local function lvl2() error("at level 2", 2) end
        local function lvlcaller() lvl2() end
        for i = 1, 300 do local _, e = pcall(lvlcaller) out[3] = e end
        local closed = 0
        local function withclose(x)
          local h <close> = setmetatable({}, {__close = function() closed = closed + 1 end})
          return thrower(x)
        end
        for i = 1, 300 do pcall(withclose, i) end
        out[4] = closed
        local r
        for i = 1, 300 do local _, m = xpcall(top, function(m) return "h:" .. m end, i * 5) r = m end
        out[5] = r
        local function inner(x) coroutine.yield(x) return x + 1 end
        local yields = load("local f = ... return function(x) return f(x) * 2 end")(inner)
        local co = coroutine.wrap(function() local t = 0 for i = 1, 500 do t = t + yields(i) end return t end)
        local last
        for i = 1, 501 do last = co() end
        out[6] = last
        local deaths = 0
        for i = 1, 300 do
          local c = coroutine.create(function(x) if x % 3 == 0 then error("dead") end return x end)
          if not coroutine.resume(c, i) then deaths = deaths + 1 end
        end
        out[7] = deaths
        local interp = load("return function(x) return x * 3 end")()
        local function tc(x) return interp(x + 1) end
        local function use(x) return tc(x) + 1 end
        local u = 0
        for i = 1, 500 do u = u + use(i) end
        out[8] = u
        return table.concat(out, " ")
"##,
    );
}

/// Recursion that starts a level per call (each `pcall` from compiled code is a
/// native that calls Lua) runs past the native budget, where compiled entries
/// refuse and the interpreter goes on, and unwinds back.
#[test]
fn levels_past_the_native_budget() {
    agrees(
        "local function f(n)
          if n == 0 then return 0 end
          local ok, r = pcall(f, n - 1)
          return r + 1
        end
        local s = 0
        for i = 1, 300 do s = s + f(20) end
        return s .. ' ' .. f(4000)",
    );
}

/// Coroutines parked inside compiled frames, below levels, while the
/// collector runs; a yield inside a `pcall` from compiled code.
#[test]
fn coroutines_parked_in_compiled_frames() {
    agrees(
        "local function inner(x) return coroutine.yield(x) end
        local mid = load('local f = ... return function(x) return f(x) + 1 end')(inner)
        local cos = {}
        for i = 1, 200 do
          cos[i] = coroutine.wrap(function(x)
            local keep = {x, tostring(x)}
            local t = 0
            for k = 1, 50 do t = t + mid(k) + keep[1] end
            return t + #keep[2]
          end)
          cos[i](i)
        end
        local junk = {}
        local s = 0
        for round = 1, 50 do
          for i = 1, 200 do
            for j = 1, 20 do junk[j] = {round, i, j} end
            local v = cos[i](round)
            if round == 50 then s = s + v end
          end
        end
        local function step(x) local r = coroutine.yield(x) if r < 0 then error('neg') end return r end
        local co = coroutine.wrap(function()
          local good, bad = 0, 0
          for i = 1, 400 do
            local ok, v = pcall(step, i)
            if ok then good = good + v else bad = bad + 1 end
          end
          return good .. ' ' .. bad
        end)
        co()
        local last
        for i = 1, 400 do last = co(i % 7 == 0 and -1 or i) end
        return s .. ' ' .. last",
    );
}

/// A fast callee whose window does not fit the committed stack misses to
/// its caller's slow path, which must run with the caller's CLOSURE: a
/// mutable upvalue the caller reads after the call is a cell found through
/// it.
#[test]
fn fast_callee_miss_keeps_the_callers_closure() {
    agrees(
        "local count, kt = 0, {3}
        local function f(n)
          local a, b, c, d, e, g, h, i, j, k, l, m, o, p, q, r = n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
          local s, t, u, v, w, x, y, z = n, n, n, n, n, n, n, n
          return #('x' .. n) + (a + z) % 3
        end
        local function h(n)
          local x = f(n)
          count = count + 1
          if n > 0 then x = x + h(n - 1) end
          return x + count % 7 + kt[1]
        end
        local s = 0
        for r = 1, 200 do s = s + h(50 + r * 20) end
        return tostring(s)",
    );
}

/// Compiled resumes, yields and `wrap` calls that switch in the stub
/// (`abi::coro_entry`): values both ways, first resumes, ends, errors,
/// refused resumes, nesting, closes and collections while parked.
#[test]
fn compiled_coroutine_switches() {
    agrees(
        "local resume, yield, wrap, create = coroutine.resume, coroutine.yield, coroutine.wrap, coroutine.create
        local status, running, close = coroutine.status, coroutine.running, coroutine.close
        local out, junk = {}, {}
        local echo = create(function(...)
          local a, b, c, d, e = ...
          while true do
            local k = a % 6
            if k == 0 then a, b, c, d, e = yield()
            elseif k == 1 then a, b, c, d, e = yield(a)
            elseif k == 2 then a, b, c, d, e = yield(a, b)
            elseif k == 3 then a, b, c, d, e = yield(a, b, c)
            else a, b, c, d, e = yield(a, b, c, d, e) end
            junk[a % 50 + 1] = {a, b}
          end
        end)
        local s = 0
        for i = 1, 4000 do
          local r = table.pack(resume(echo, i, i + 1, i + 2, i + 3, i + 4))
          s = s + r.n
          for j = 2, r.n do s = s + (r[j] or 0) end
        end
        out[#out + 1] = s
        local ended, died = 0, 0
        for i = 1, 3000 do
          local co = create(function(x)
            local y = yield(x + 1)
            if y % 7 == 0 then error('seven') end
            return x + y, 'done'
          end)
          local ok1, v1 = resume(co, i)
          local ok2, v2, v3 = resume(co, v1)
          if ok2 then ended = ended + v2 + #v3 else died = died + 1 end
          local ok3, m = resume(co)
          if not ok3 and status(co) == 'dead' then ended = ended + #m end
        end
        out[#out + 1] = ended .. '/' .. died
        local gen = wrap(function(n) for k = 1, n do yield(k) end return 'last' end)
        local g = gen(3000)
        for i = 1, 3000 do g = gen() end
        local ok, err = pcall(gen)
        out[#out + 1] = tostring(g) .. ' ' .. tostring(ok) .. ' ' .. tostring(err)
        local caught = 0
        for i = 1, 2000 do
          local f = wrap(function() if i % 3 == 0 then error({i}) end return i end)
          local ok, e = pcall(f)
          caught = caught + (ok and e or e[1] * 2)
        end
        out[#out + 1] = caught
        local refused = 0
        local self = create(function()
          for i = 1, 2000 do
            local ok, m = resume(running())
            if not ok then refused = refused + #m end
            local ok2, m2 = pcall(resume)
            if not ok2 then refused = refused + 1 end
            yield(i)
          end
        end)
        for i = 1, 2000 do resume(self) end
        out[#out + 1] = refused
        local inner = create(function() local t = 0 while true do t = t + yield(t) end end)
        local outer = create(function()
          local t = 0
          while true do
            local _, v = resume(inner, 1)
            t = t + yield(v + t)
          end
        end)
        local nest = 0
        for i = 1, 3000 do local _, v = resume(outer, i) nest = (nest + v) % 1000003 end
        out[#out + 1] = nest
        local closed = 0
        for i = 1, 300 do
          local co = create(function()
            local x <close> = setmetatable({}, {__close = function() closed = closed + 1 end})
            for k = 1, 1e9 do yield(k) end
          end)
          for k = 1, 10 do resume(co) end
          close(co)
        end
        out[#out + 1] = closed
        local p = create(function()
          local n = 0
          for i = 1, 3000 do
            local ok, v = pcall(yield, i)
            if ok then n = n + v end
          end
          return n
        end)
        local last
        for i = 1, 3001 do local _, v = resume(p, i) last = v end
        out[#out + 1] = last
        return table.concat(out, ' ')",
    );
}

fn panicker<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    if stack.get(0).get_integer() == Some(5000) {
        panic!("native panic below compiled frames");
    }
    Ok(())
}

/// Fast calls deep enough to go in tagged (`deep_path`), whose callees miss
/// as the value stack grows: each miss finds its caller's slow path through
/// the site its return address names (`abi::call_miss`), and every return,
/// tagged or not, lands in the resume block.
#[test]
fn deep_misses_find_their_sites() {
    let src = "local function depth(n) if n == 0 then return 0 end return depth(n - 1) + 1 end
        local function sum(n) if n == 0 then return 0 end local r = sum(n - 1) return r + n end
        local s = 0
        for i = 1, 20 do s = s + depth(i * 500) + sum(i * 400) end
        return tostring(s)";
    // Reference Lua 5.5 prints 229747000.
    assert_eq!(result(src, true), "229747000");
    agrees_calling(src);
}

/// More live values than the registers once allocatable, so x23 and x24 hold
/// some (no longer the runtime and closure), across a helper call and into an
/// exit, which restores them from their image slots.
#[test]
fn values_in_former_pinned_registers_survive_exits() {
    agrees(
        "local function f(n, t)
          local a, b, c, d, e, g, h, i, j, k = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
          local l, m, o, p, q, r, s, u, v, w = 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
          local acc = 0
          for x = 1, n do
            a = (a + x) % 997 b = (b + a) % 997 c = (c + b) % 997 d = (d + c) % 997
            e = (e + d) % 997 g = (g + e) % 997 h = (h + g) % 997 i = (i + h) % 997
            j = (j + i) % 997 k = (k + j) % 997 l = (l + k) % 997 m = (m + l) % 997
            o = (o + m) % 997 p = (p + o) % 997 q = (q + p) % 997 r = (r + q) % 997
            s = (s + r) % 997 u = (u + s) % 997 v = (v + u) % 997 w = (w + v) % 997
            acc = acc + #tostring(x) + t[x % #t + 1]
          end
          return table.concat({a, b, c, d, e, g, h, i, j, k, l, m, o, p, q, r, s, u, v, w, acc}, ' ')
        end
        local t = {}
        for x = 1, 50 do t[x] = x end
        local out = f(2000, t)
        t[50] = 0.5
        return out .. ' / ' .. f(2000, t)",
    );
}

/// Calls passing their arguments in registers: the parameters left out nil,
/// past the eighth in slots, more arguments than parameters; a callee whose
/// entry guard fails stores its parameters for the interpreter; one result
/// returned in x0 alone, landed for `c = 0` and for more results than it gives.
#[test]
fn register_arguments_and_results() {
    agrees_calling(
        "local function f3(a, b, c) return (a or 0) + (b or 0) * 2 + (c or 0) * 3 end
        local function f10(a, b, c, d, e, g, h, i, j, k)
          return a + b + c + d + e + g + h + i + j * 10 + k * 100
        end
        local function typed(x) return x + 1 end
        local function one(x) return x end
        local s, t = 0, {}
        for i = 1, 3000 do
          s = s + f3(i) + f3(i, i) + f3(i, i, i, i, i)
          s = s + f10(1, 2, 3, 4, 5, 6, 7, 8, i, i)
          s = s + typed(i < 2900 and i or i + 0.5)
          t = {one(i), one(i)}
          local a, b, c, d = one(i)
          s = s + #t + a + (b or 1) + (c or 2) + (d or 3)
        end
        return tostring(s) .. ' ' .. table.concat(t, ',') .. ' ' .. select('#', one(1))",
    );
}

/// A recursion going deeper each time, its inlined callee's frame past its
/// own window: the entry makes sure of both, growing the stack where the
/// interpreter's call grew it for the recursing frame alone.
#[test]
fn inlined_frames_window_grows() {
    agrees(
        "local function wide(x)
          local a, b, c, d, e, f, g, h, i, j = x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7, x + 8, x + 9
          local k, l, m, n, o, p, q, r, s, t = a + b, c + d, e + f, g + h, i + j, a * b, c * d, e * f, g * h, i * j
          return k + l + m + n + o + p + q + r + s + t
        end
        local function down(n)
          if n == 0 then return 0 end
          return wide(n) + down(n - 1)
        end
        local s = 0
        for i = 1, 300 do s = s + down(i) end
        return tostring(s)",
    );
}

/// Float constants reaching loop-carried floats that live in stack slots
/// across a call: the allocator computes each in the transport register,
/// as its word, and stores that (a float register took it, unstored).
#[test]
fn float_constants_into_stack_slots() {
    compiles_cleanly(
        "local n = 0
        local function h() n = n + 1 end
        local function f(m)
          local a, b, c, d, e, g = 0.5, 1.5, 2.5, 3.5, 4.5, 5.5
          local p, q, r, s, t, u = 6.5, 7.5, 8.5, 9.5, 10.5, 11.5
          for i = 1, m do
            h()
            if i % 3 == 0 then
              a, b, c, d, e, g = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            else
              a, b, c, d, e, g = a + 1.0, b + 2.0, c + 3.0, d + 4.0, e + 5.0, g + 6.0
            end
            p, q, r, s, t, u = p + a, q + b, r + c, s + d, t + e, u + g
          end
          return a + b + c + d + e + g + p + q + r + s + t + u
        end
        local z = 0.0
        for k = 1, 300 do z = z + f(300) end
        return tostring(z) .. ' ' .. n",
        false,
    );
}

/// A tail call in a callee a call passed its arguments in registers, to a
/// function that never compiles (an irreducible loop): its miss runs the
/// caller's slow path with the tail call's arguments in the slots, where
/// nothing may overwrite them with the registers the first call passed.
#[test]
fn tail_misses_keep_their_arguments() {
    agrees_calling(
        "local function g(m, h)
          local i = 0
          if m > 3 then goto b end
          ::a:: i = i + 1
          ::b:: i = i + 2
          if i < 5 then goto a end
          return m * 2 + h + i
        end
        local T = {} T.__index = T
        function T:run(m) local msg = m + 1 return g(msg, self.handle) end
        local objs = {}
        for k = 1, 4 do objs[k] = setmetatable({handle = k}, T) end
        local s = 0
        for r = 1, 4000 do s = s + objs[r % 4 + 1]:run(r) end
        return tostring(s)",
    );
}

/// A vararg function's compiled tail call to a function that never compiles
/// (an irreducible loop) keeps the frame's vararg count through the miss:
/// the callee's own tail call finds the frame's function slot by it, and
/// the stack stays where it was.
#[test]
fn tail_misses_keep_the_vararg_count() {
    agrees_calling(
        "local F
        local function g(k)
          local i = 0
          if k < 0 then goto b end
          ::a:: i = i + 1
          ::b:: i = i + 2
          if i < 5 then goto a end
          if k == 0 then return 'done' end
          return F(k - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        end
        F = function(k, ...) local a, b, c return g(k) end
        local function run(k) local r = F(k) return r end
        return run(100000)",
    );
}

/// The same miss with no room for the callee's window raises the overflow
/// from the callee's frame, at its first line: the tail caller's is gone.
#[test]
fn tail_miss_overflow_is_raised_in_the_callee() {
    let mut lua = Lua::new();
    lua.set_jit_inline(false);
    lua.load_all();
    let src = "local function g(m)
          local LOCALS
          local i = 0
          if m < 0 then goto b end
          ::a:: i = i + 1
          ::b:: i = i + 2
          if i < 5 then goto a end
          return m + i
        end
        local function F(m, ...) return g(m) end
        local function rec(n)
          local i = 0
          if n < 0 then goto b end
          ::a:: i = i + 1
          ::b:: i = i + 2
          if i < 5 then goto a end
          return F(n, 1, 2, 3) + rec(n + 1)
        end
        local ok, e = pcall(rec, 1)
        return tostring(ok) .. ' ' .. e .. ' ' .. F(1, 2, 3)"
        .replace(
            "LOCALS",
            &(0..150)
                .map(|k| format!("l{k}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
    let ex = start_on(&mut lua, &src);
    lua.finish(&ex)
        .unwrap_or_else(|e| panic!("{src:?} failed: {e:?}"));
    let got = lua.enter(|ctx| {
        let v: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        String::from_utf8_lossy(v.get_string().expect("a string result").as_bytes()).into_owned()
    });
    // Line 10 where the tail call ran interpreted (no JIT, or it is off).
    assert!(
        got == "false c:1: stack overflow 7" || got == "false c:10: stack overflow 7",
        "{got}"
    );
}

/// Float equality reads its operands' low halves only: on x86-64 a negation
/// and an `abs` are packed operations on literals next to each other in the
/// pool, which leave the neighbour's bits in the upper half of the result.
#[test]
fn float_equality_ignores_the_upper_lane() {
    let src = "local function f(x)
          local y, z = -x, math.abs(x)
          local n = 0
          if y == 0 then n = n + 1 end
          if z == 0 then n = n + 10 end
          if y ~= -z then n = n + 100 end
          if y == -z then n = n + 1000 end
          return n
        end
        local function g(x) local y = -x return y == 0 end
        local r = 0
        for i = 1, 3000 do
          r = r + f(i + 0.5)
          if g(1.5) then r = r + 1 end
        end
        return tostring(r)";
    agrees(src);
    agrees_calling(src);
}

/// A region a dependency invalidated, called next through its register
/// entry by a compiled caller (`h` changes the method `g` folded): its dead
/// entry stores the parameters before the interpreter runs the function
/// from its slots.
#[test]
fn invalidated_entries_store_register_parameters() {
    agrees_calling(
        "local C = {} C.__index = C
        function C.val(self) return self.v end
        local o = setmetatable({v = 2}, C)
        local function g(t, k, m) return t:val() * k + m end
        local function h(i) if i == 1500 then C.val = function(self) return self.v + 1 end end end
        local function f(n) local s = 0 for i = 1, n do h(i) s = s + g(o, i, 3) end return s end
        return tostring(f(3000))",
    );
}

/// Parameters are pure in all but where they live, and nothing may move or
/// copy one: a call's boundary recomputing values after it would have read
/// a parameter no register holds there, and this region (`depth - 1`
/// across the recursive call) failed to compile, running interpreted.
#[test]
fn parameters_stay_at_the_entry() {
    compiles_cleanly(
        "local S = {} S.__index = S
        function S:build(depth)
          self.count = self.count + 1
          if depth == 1 then return {n = 1} end
          local arr = {n = 4}
          for i = 1, 4 do arr[i] = self:build(depth - 1) end
          return arr
        end
        local s = setmetatable({count = 0}, S)
        for r = 1, 50 do s:build(6) end
        return tostring(s.count)",
        true,
    );
}

/// Calls from inlined bodies: the frame walk shows the frames inlined around a
/// call, so error levels count them and positions are theirs. `pcall(error,
/// msg, k)` is a native call wherever it is: level 2 is `inner`, level 3 its
/// caller `mid`, both inlined into `run`'s loop; `fs[1]` is one of two closures
/// of a prototype, inlined by it, whose closure the walk reads from the call
/// site.
#[test]
fn errors_locate_inlined_frames() {
    agrees(
        "local function inner()
          local ok, e = pcall(error, 'm', 2)
          local ok3, e3 = pcall(error, 'n', 3)
          return e, e3
        end
        local function mid()
          local a, b = inner()
          return a, b
        end
        local function mk(k)
          return function()
            local ok, e = pcall(error, 'd', 2)
            return e
          end
        end
        local fs = {mk(1), mk(2)}
        local function run(n)
          local x, y, z
          for i = 1, n do
            x, y = mid()
            z = fs[1]()
          end
          return x .. ' ' .. y .. ' ' .. z
        end
        return run(3000)",
    );
}

/// An error in a native called from an inlined body names the body's line
/// (level 1 is the inlined frame) and unwinds through the region to a
/// `pcall` below it, which then runs the region again.
#[test]
fn errors_unwind_through_bodies() {
    agrees(
        "local args = {}
        for i = 1, 3000 do args[i] = 3 end
        args[2999] = 'x'
        local function body(i)
          return #string.rep('a', args[i])
        end
        local function run(n)
          local s = 0
          for i = 1, n do s = s + body(i) end
          return s
        end
        local ok, e = pcall(run, 3000)
        local ok2, s = pcall(run, 2000)
        return tostring(ok) .. ' ' .. e .. ' ' .. tostring(ok2) .. ' ' .. s",
    );
}

/// Collections during calls from an inlined body: what lives across them
/// is the region's, in slots counted from its base. `p` and `q`, results
/// landed in the body's window and not yet read, stay in their home slots
/// across the allocating call; `t` in the frame's spill slots.
#[test]
fn bodies_keep_values_across_collections() {
    agrees(
        "local function pair(i) return {i}, {i + 1} end
        local function body(t, i)
          local ok, p, q = pcall(pair, i)
          local b = string.rep('y', 300)
          return p[1] + q[1] + #b + t[1]
        end
        local function run(n)
          local s = 0
          for i = 1, n do
            local t = {i}
            s = s + body(t, i) + t[1]
          end
          return tostring(s)
        end
        return run(20000)",
    );
}

/// A call in an inlined body may write its callers' captured variables:
/// they are stored before the body and read back after it.
#[test]
fn bodies_calls_write_captured_variables() {
    agrees(
        "local function body(f) pcall(f) return 1 end
        local function run(n)
          local count, s = 0, 0
          local function bump() count = count + 1 end
          for i = 1, n do
            s = s + body(bump) + count
          end
          return s .. ' ' .. count
        end
        return run(3000)",
    );
}

/// A call keeps what the code after it uses and nothing can compute again
/// (a `^` LICM moved above it) without a register naming it: in `kept`'s
/// own region every register is live or the call's, and so in its body
/// inlined into the loop below. `g` never compiles (an irreducible loop).
#[test]
fn kept_values_need_no_dead_register() {
    let src = "local function g(m)
          local i = 0
          if m > 3 then goto b end
          ::a:: i = i + 1
          ::b:: i = i + 2
          if i < 5 then goto a end
          return m + i
        end
        local function kept(h)
          local x0 = 0.5
          local s = 0
          for i = 1, 100 do
            s = s + h(i) * (x0 + 1) ^ x0
          end
          return s
        end
        local t = 0
        for k = 1, 300 do t = t + kept(g) end
        return string.format('%.6f', t)";
    compiles_cleanly(src, false);
    compiles_cleanly(src, true);
}

/// Constructors and concatenation in inlined bodies: a table from its
/// prototype's template, positional items from the body's slots, a list ending
/// in a call's results (to `top`), and `..`.
#[test]
fn bodies_build_tables_and_strings() {
    agrees(
        "local function point(x, y) return {x = x, y = y, tag = 'p'} end
        local function triple(a) return {a, a + 1, a + 2} end
        local function copy(t) return {0, table.unpack(t)} end
        local function name(i) return 'n' .. i end
        local src = {7, 8, 9}
        local s, n, l = 0, 0, 0
        for i = 1, 3000 do
          local p = point(i, i * 2)
          local t = triple(i)
          local c = copy(src)
          s = s + p.x + p.y + t[1] + t[3] + #t + c[2] + c[4] + #c
          if p.tag == 'p' then n = n + 1 end
          l = l + #name(i)
        end
        return s .. ' ' .. n .. ' ' .. l",
    );
}

/// Closures made in an inlined body: `make`'s `get` and `inc` share `n`, open
/// on the body's slot while it runs (`inc` writes it from a real call there)
/// and closed when it returns: every `get` kept in `gets` reads its own `n`
/// after the stack is reused. `bads[2500]` takes an arm that never ran, exiting
/// inside the body with `n` open (the frame the exit writes must close it); a
/// `false` raises inside `inc`, an error that unwinds through the body to the
/// `pcall`, `saved` keeping `n`.
#[test]
fn bodies_make_closures() {
    agrees(
        "local saved
        local gets = {}
        local function make(k, bad)
          local n = k
          local function get() return n end
          local function inc(d) n = n + d end
          saved = get
          inc(1)
          inc(bad)
          if bad == 2 then n = n + 10 end
          return get, inc
        end
        local function use(i, bad)
          local get, inc = make(i, bad)
          inc(3)
          gets[i] = get
          return get()
        end
        local bads = {}
        for i = 1, 3000 do bads[i] = 1 end
        bads[2500] = 2
        bads[2999] = false
        local function run()
          local s = 0
          for i = 1, 3000 do s = s + use(i, bads[i]) end
          return s
        end
        local ok, e = pcall(run)
        local t = 0
        for i = 1, 2998 do t = t + gets[i]() end
        return tostring(ok) .. ' ' .. e .. ' ' .. saved() .. ' ' .. t",
    );
}

/// An inlined body's cell store may write a slot of the frame it was
/// inlined into: `bump` writes `count`, which `run` stored before it and
/// reads back after, nothing forwarding the store across the cell's.
#[test]
fn cell_stores_in_bodies_write_their_slots() {
    agrees(
        "local function run(n)
          local count, s = 0, 0
          local function bump() count = count + 1 end
          for i = 1, n do
            count = count + i
            bump()
            s = s + count
          end
          return s .. ' ' .. count
        end
        local r
        for k = 1, 30 do r = run(300) end
        return r",
    );
}

/// Generic `for` and metamethod calls in inlined bodies: the `next` and
/// `ipairs` steps at the body's slots, another iterator (a closure made in the
/// body) and `__add` as calls from a header past the body's registers, where
/// they overwrite nothing: `p`, `q` and `r` wait in the body's slots across the
/// metamethod call.
#[test]
fn bodies_iterate_and_call_metamethods() {
    agrees(
        "local function sum_pairs(t) local s = 0 for k, v in pairs(t) do s = s + v end return s end
        local function sum_ipairs(t) local s = 0 for i, v in ipairs(t) do s = s + i * v end return s end
        local function range(n) return function(_, i) if i < n then return i + 1 end end, nil, 0 end
        local function sum_range(n) local s = 0 for i in range(n) do s = s + i end return s end
        local V = {}
        V.__add = function(a, b) return setmetatable({x = a.x + b.x}, V) end
        local function three(i) return i, i + 1, i + 2 end
        local function addv(v, w, i)
          local ok, p, q, r = pcall(three, i)
          local u = v + w
          return u, p + q + r
        end
        local t = {a = 1, b = 2, c = 3}
        local arr = {4, 5, 6}
        local s = 0
        local v = setmetatable({x = 0}, V)
        local one = setmetatable({x = 1}, V)
        for i = 1, 3000 do
          local n
          s = s + sum_pairs(t) + sum_ipairs(arr) + sum_range(i % 5)
          v, n = addv(v, one, i)
          s = s + n
        end
        return s .. ' ' .. v.x",
    );
}

/// Tail calls in inlined bodies replace the body's frame, as the interpreter's
/// do: `f` is inlined in `g`'s place and `k` (an irreducible loop, never
/// compiled) called from `g2`'s header, so level 3 from either is the loop in
/// `run`, `g` and `g2` gone; `k`'s four results come back through the slots.
#[test]
fn tail_calls_in_bodies_replace_their_frames() {
    agrees(
        "local function f(x)
          local ok, e = pcall(error, 'm', 3)
          return e, x
        end
        local function g(x)
          return f(x + 1)
        end
        local function k(m)
          local i = 0
          if m > 3 then goto b end
          ::a:: i = i + 1
          ::b:: i = i + 2
          if i < 5 then goto a end
          local ok, e = pcall(error, 'n', 3)
          return e, m + i, i, m
        end
        local function g2(x)
          return k(x * 2)
        end
        local function run(n)
          local s, e1, e2 = 0
          for i = 1, n do
            local e, v = g(i)
            local d, p, q, r = g2(i)
            s = s + v + p + q + r
            e1, e2 = e, d
          end
          return s .. ' ' .. e1 .. ' ' .. e2
        end
        return run(3000)",
    );
}

/// A native an inlined body tail-calls runs above the body, as the
/// interpreter runs it from the tail-calling frame: its error names the
/// body's line, not that of the call the body was inlined at. A body made
/// where it is called is known by its prototype; a callee that varies is
/// told from a Lua function where it is called.
#[test]
fn natives_tail_called_from_bodies_name_the_body() {
    for (made, call) in [
        ("local function f(y) return string.rep(y) end", "f(x)"),
        ("", "local f = function(y) return string.rep(y) end f(x)"),
        (
            "local hs = {string.rep, string.len, string.upper}
            local n = 0
            local function f(y) n = n + 1 return hs[n % 3 + 1](y and nil) end",
            "f(x)",
        ),
    ] {
        agrees(&format!(
            "{made}
            local function g(x)
              {call}
            end
            local msg
            for i = 1, 3000 do
              local ok, e = pcall(g, 'a')
              msg = e
            end
            return msg"
        ));
    }
}

/// A metamethod inlined at `SUB` and at `GETFIELD` tail-calls a callee too
/// polymorphic to inline, which replaces the metamethod and so invalidates
/// the region: the lazy exit past the call resumes after the instruction
/// with the result in `R[A]`, which still held the instruction's operand.
#[test]
fn metamethod_tail_calls_leave_with_the_result() {
    for (set, use_) in [
        (
            "V.__sub = function(a, b) return mk(a.x - b.x) end",
            "local d = P[i].v - b.v",
        ),
        (
            "V.__index = function(t, k) return mk(t.x - 3) end",
            "local d = P[i].v; d = d.y",
        ),
    ] {
        agrees(&format!(
            "local V = {{}}
            local mk
            local function new(x) return setmetatable({{x = x}}, V) end
            local function new2(x) return setmetatable({{x = x + 0}}, V) end
            local function new3(x)
              if x == 1499 then {set} end
              return setmetatable({{x = x}}, V)
            end
            {set}
            local ks = {{new, new2, new3}}
            local P = {{}}
            for i = 1, 2000 do P[i] = {{v = new(i + 3)}} end
            local b = {{v = new(3)}}
            local s = 0
            for i = 1, 2000 do
              mk = ks[i % 3 + 1]
              {use_}
              s = s + d.x
            end
            return tostring(s)"
        ));
    }
}

/// A tail call in a region's own frame: `outer` (its region made through
/// `pcall`, which no caller inlines) tail-calls `inner`, which takes over its
/// frame. Level 3 from `inner` is the `pcall`, `outer` gone; at 2500 a float
/// exits inside `inner`, its frame's header must name it for the positions
/// after.
#[test]
fn tail_calls_take_over_the_region_frame() {
    agrees(
        "local xs = {}
        for i = 1, 3000 do xs[i] = i end
        xs[2500] = 0.5
        local function inner(i, y)
          local v = xs[i] + y
          local ok, e = pcall(error, 'm', 2)
          local ok3, e3 = pcall(error, 'n', 3)
          return v, e, e3
        end
        local function outer(i)
          return inner(i, 1)
        end
        local s, l1, l2, at = 0
        for i = 1, 3000 do
          local ok, v, e, e3 = pcall(outer, i)
          s = s + v
          if i == 2500 then at = e end
          l1, l2 = e, e3
        end
        return s .. ' ' .. l1 .. ' ' .. l2 .. ' ' .. at",
    );
}

/// A tail call of the region frame's replacement inlines in its place only a
/// constant callee: `mid` takes over `outer`'s frame, and `t.f`, a closure of a
/// prototype that made several, is called for real. No slot below the region
/// frame's window holds such a closure for exits (its header entry was register
/// `-4`). `mid2`'s constant `leaf` takes `outer2`'s frame in turn.
#[test]
fn replacements_inline_constant_tail_callees_only() {
    compiles_cleanly(
        "local xs = {}
        for i = 1, 3000 do xs[i] = i end
        xs[2500] = 0.5
        local function mk(k)
          return function(i)
            local v = xs[i] * k
            local ok, e = pcall(error, 'm', 2)
            return v, e
          end
        end
        local o, h3 = {f = mk(2)}, mk(3)
        local function mid(t, i) return t.f(i) end
        local function outer(i) return mid(o, i) end
        local leaf = mk(4)
        local function mid2(i) return leaf(i) end
        local function outer2(i) return mid2(i) end
        local s, at, at2 = 0
        for i = 1, 3000 do
          local ok, v, e = pcall(outer, i)
          local ok2, v2, e2 = pcall(outer2, i)
          s = s + v + v2
          if i == 2500 then at, at2 = e, e2 end
        end
        return s .. ' ' .. at .. ' ' .. at2 .. ' ' .. h3(1)",
        true,
    );
}

/// A function region's tail calls of itself are its loop, the frame reused as
/// the interpreter's `TAILCALL` reuses it: level 3 from `find` is the loop
/// calling it. `f1` and `f2` are closures of one prototype tail-calling each
/// other: only the running closure loops, so each call keeps its own `k`.
#[test]
fn tail_self_calls_loop() {
    agrees(
        "local function sum(n, acc)
          if n == 0 then return acc end
          return sum(n - 1, acc + n)
        end
        local function find(t, i, x)
          if t[i] == x then
            local ok, e = pcall(error, 'at', 3)
            return i, e
          end
          return find(t, i + 1, x)
        end
        local function mk(k)
          local f
          f = function(n, other, acc)
            if n == 0 then return acc end
            return other(n - 1, f, acc + k)
          end
          return f
        end
        local f1, f2 = mk(1), mk(1000)
        local t = {}
        for i = 1, 200 do t[i] = i end
        local s, where = 0
        for k = 1, 300 do
          local i, e = find(t, 1, k % 200 + 1)
          s = s + sum(500, 0) + i + f1(9, f2, 0) + f1(10, f1, 0)
          where = e
        end
        return s .. ' ' .. where",
    );
}

/// Counts up to `top` in inlined bodies: `pair` returns `1` (a register) and
/// what `table.unpack` left in its slots; `add3` is inlined with arguments up
/// to `top`, a register before them or not, nil past the count.
#[test]
fn bodies_take_values_up_to_top() {
    agrees(
        "local function pair(t) return 1, table.unpack(t) end
        local function add3(a, b, c) return a + b + (c or 100) end
        local t1, t2, t3 = {5}, {10, 20}, {10, 20, 30}
        local s = 0
        for i = 1, 3000 do
          local x, y, z, w = pair(i % 2 == 0 and t2 or t3)
          s = s + x + y + z + (w or 7)
          s = s + add3(table.unpack(i % 3 == 0 and t2 or t3))
          s = s + add3(i, table.unpack(i % 4 == 0 and t1 or t2))
        end
        return tostring(s)",
    );
}

/// Inlined calls wanting every result: `two` always gives two (the next call's
/// count fixed), `maybe` one or two (a count the next call, inlined or real,
/// takes as a value), `wrap` returns `maybe`'s. The callee of the last call
/// changes at 2500: that exit runs it in the interpreter, which takes `top` as
/// the inlined call left it.
#[test]
fn inlined_calls_leave_values_up_to_top() {
    agrees(
        "local function two(x) return x, x + 1 end
        local function maybe(x) if x % 2 == 0 then return x end return x, x * 2 end
        local function add3(a, b, c) return a + (b or 0) + (c or 100) end
        local function wrap(x) return 1, maybe(x) end
        local function count(...) return select('#', ...) end
        local function count2(...) return select('#', ...) * 1000 end
        local s = 0
        for i = 1, 3000 do
          s = s + add3(two(i))
          s = s + add3(maybe(i))
          s = s + count(maybe(i)) + count(two(i))
          local a, b, c = wrap(i)
          s = s + a + b + (c or 7)
          local f = i == 2500 and count2 or count
          s = s + f(two(i))
        end
        return tostring(s)",
    );
}

/// An exit right before the instruction that takes what an inlined call
/// left up to `top` (the callee guard of `fs[1]`, replaced at 2501) runs
/// it in the interpreter, which reads `top` and the slots: the inlined
/// call leaves them as a real one would. `maybe` gives one or two values,
/// so a `top` left from another run would be wrong.
#[test]
fn exits_before_takers_see_the_inlined_top() {
    agrees(
        "local function maybe(x) if x % 2 == 0 then return x end return x, x * 2 end
        local fs = {function(a, b, c) if b == nil then return 1 end return 2 end}
        local other = function(a, b, c) if b == nil then return 10 end return 20 end
        local s = 0
        for i = 1, 3000 do
          if i == 2501 then fs[1] = other end
          s = s + fs[1](maybe(i))
        end
        return tostring(s)",
    );
}

/// Recursion inlines one copy, through another function too: a float at depth
/// exits inside a copy, an error two levels up names the copy's caller, and
/// `even`/`odd` recurse through each other.
#[test]
fn recursion_inlines_one_copy() {
    compiles_cleanly(
        "local function fib(n)
          if n < 2 then
            if n == 0.5 then error('half', 3) end
            return n
          end
          return fib(n - 1) + fib(n - 2)
        end
        local odd
        local function even(n) if n == 0 then return true end return not odd(n - 1) end
        odd = function(n) if n == 0 then return false end return not even(n - 1) end
        local s, e = 0
        for i = 1, 300 do s = s + fib(i % 12) end
        s = s + fib(6.25)
        local ok, m = pcall(fib, 4.5)
        local t = 0
        for i = 1, 300 do if even(i % 20) then t = t + 1 end end
        return s .. ' ' .. m .. ' ' .. t",
        true,
    );
}

/// A call whose callee joins constant functions dispatches on them: `get` of
/// `A` or `B` by the receiver's shape, each body inlined in its arm. Float
/// fields exit inside each arm; then a `C` takes the real call for any other
/// callee.
#[test]
fn polymorphic_calls_dispatch() {
    compiles_cleanly(
        "local A = {} A.__index = A
        function A.new(v) return setmetatable({v = v}, A) end
        function A:get() return self.v + 1 end
        local B = {} B.__index = B
        function B.new(v) return setmetatable({v = v}, B) end
        function B:get() return self.v * 2 end
        local C = {} C.__index = C
        function C:get() return -self.v end
        local objs = {}
        for i = 1, 100 do objs[i] = (i % 2 == 0) and A.new(i) or B.new(i) end
        local function sum(n)
          local s = 0
          for i = 1, n do s = s + objs[(i % 100) + 1]:get() end
          return s
        end
        local t = 0
        for k = 1, 300 do t = t + sum(100) end
        objs[3] = A.new(0.5)
        objs[4] = B.new(0.25)
        t = t + sum(100)
        objs[5] = setmetatable({v = 7}, C)
        t = t + sum(100)
        return tostring(t)",
        true,
    );
}

/// A metamethod the operand's class gives as a constant inlines: `a + b`'s
/// `__add`, where `a.y` guarded `a`'s shape. At 2500 its i32 add overflows and
/// exits inside it (its frame's continuation stores the result to the `ADD`'s
/// register), and a table field raises inside it.
#[test]
fn class_metamethods_inline() {
    compiles_cleanly(
        "local V = {}
        V.__index = V
        local function vec(x, y) return setmetatable({x = x, y = y}, V) end
        V.__add = function(a, b) return vec(a.x + b.x, a.y + b.y) end
        local s = 0
        for i = 1, 3000 do
          local x = i % 3
          if i == 2500 then x = 2147483647 end
          local a, b = vec(x, i % 5), vec(1, 2)
          s = s + a.y + b.y
          local r = a + b
          s = s + r.x + r.y
        end
        local ok, e = pcall(function() local a = vec({}, 1) local t = a.y return a + vec(1, 2) end)
        return s .. ' ' .. tostring(ok) .. ' ' .. e",
        true,
    );
}

/// A region frame's tail call taking a call's results up to `top`: `f` and
/// `ack` loop, `f`'s arguments past `acc` from `m`'s zero, one or two results,
/// nil past them.
#[test]
fn tail_calls_up_to_top_loop() {
    compiles_cleanly(
        "local function m(n)
          if n % 7 == 3 then return end
          if n % 5 == 0 then return n, 'x' end
          return n
        end
        local function f(acc, n, tag)
          if n == nil then return acc .. '-' end
          if n == 0 then return acc .. (tag or '') end
          return f(acc + 1, m(n - 1))
        end
        local function ack(m, n)
          if m == 0 then return n + 1 end
          if n == 0 then return ack(m - 1, 1) end
          return ack(m - 1, ack(m, n - 1))
        end
        local out = {}
        for i = 1, 300 do out[#out + 1] = f(0, i % 40) end
        local s = 0
        for i = 1, 30 do s = s + ack(2, i % 9) end
        return table.concat(out, ' ', 280) .. ' ' .. s",
        true,
    );
}

/// Typed self-calls: `f`'s calls of itself enter past its entry guard and trust
/// its integer result. `f(31)` overflows an i32 add and exits, and the interpreter
/// returns a big integer to a typed call, which its landing tests.
#[test]
fn typed_self_calls() {
    compiles_cleanly(
        "local function f(n) if n == 0 then return 1 end local x = f(n - 1) return x + x end
        local function depth(n) if n == 0 then return 0 end return depth(n - 1) + 1 end
        local s = 0
        for i = 1, 300 do s = s + f(i % 20) + depth(i % 30) end
        return s .. ' ' .. f(40) .. ' ' .. f(62) .. ' ' .. depth(3000)",
        true,
    );
}

/// Known callees: `fs`'s two closures of one prototype take turns at a site,
/// inlined by the prototype with each one's `k` read from the closure called;
/// `add` and `bump` are closures the loop makes, inlined unguarded, `bump`
/// writing the loop's captured `count` through its cell.
#[test]
fn known_callees_inline() {
    agrees(
        "local function mk(k) return function(x) return x * k end end
        local fs = {mk(2), mk(3)}
        local s, count = 0, 0
        for i = 1, 3000 do
          s = s + fs[i % 2 + 1](i)
          local add = function(x) return x + i end
          local bump = function() count = count + 1 end
          s = s + add(1)
          bump()
        end
        return s .. ' ' .. count",
    );
}

/// A panic in a native that compiled code called, under a chain of fast
/// calls, unwinds through their frames (`unwind::Fde`) to the host.
#[test]
#[should_panic(expected = "native panic below compiled frames")]
fn panic_through_compiled_frames() {
    let mut lua = Lua::new();
    lua.load_all();
    lua.enter(|ctx| {
        let f = Function::new_native(ctx.mutation(), panicker as NativeFn, &[]);
        let key = Value::string(LuaString::new(ctx, b"maybe_panic"));
        ctx.globals().raw_set(ctx, key, Value::function(f));
    });
    let ex = start_on(
        &mut lua,
        "local function c(i) maybe_panic(i) return i end
         local function b(i) return c(i) + 1 end
         local function a(i) return b(i) + 1 end
         local s = 0
         for i = 1, 10000 do s = s + a(i) end
         return tostring(s)",
    );
    let _ = lua.finish(&ex);
}

/// `os.exit` in a level compiled code started ends the thread from its
/// bottom level instead of returning into that code.
#[test]
fn exit_from_an_upper_level() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = start_on(
        &mut lua,
        "local function g(i) if i == 5000 then os.exit(7) end end
         local co = coroutine.wrap(function()
           for i = 1, 10000 do pcall(g, i) end
         end)
         co()
         return 'after'",
    );
    assert!(matches!(lua.finish(&ex), Err(RuntimeError::Exit(7))));
}

/// A generic `for`'s closing value stays in its slot while the collector
/// runs with the loop's region parked at a call: the close it exits for
/// reads it there.
#[test]
fn closing_value_survives_collections() {
    agrees(
        "local closed = 0
        local function mk() return setmetatable({}, {__close = function() closed = closed + 1 end}) end
        local function iter(n)
          return function(_, i) i = i + 1 if i <= n then return i end end, nil, 0, mk()
        end
        local function churn(i) local t = {} for j = 1, 20 do t[j] = {i, j} end return #t end
        local s = 0
        for i in iter(30000) do s = s + churn(i) if i == 25000 then break end end
        return s .. ' ' .. closed",
    );
}

/// Cases an intrinsic's inline code does not take, once compiled: the builtin
/// runs from the region.
#[test]
fn intrinsic_fallbacks() {
    agrees(
        "local mt = {__pairs = function(t) return function(_, k) if not k then return 1, 'p' end end, t, nil end}
        local plain, special = {10, 20}, setmetatable({}, mt)
        local out, n = {}, 0
        for i = 1, 400 do
          local t = i > 300 and special or plain
          for k, v in pairs(t) do n = n + 1 end
          local ok, e = pcall(function() return assert(i < 350 or nil, 'falsy at ' .. i) end)
          if not ok then out[#out + 1] = e end
          local x = i > 380 and ('-' .. i) or -i
          n = n + math.abs(x)
        end
        return n .. ' ' .. #out .. ' ' .. out[1]",
    );
}

/// `math.min`, `math.max`, `table.insert` and `table.remove` as intrinsics:
/// mixed number kinds, NaN, and lists whose metatables the builtin must see.
#[test]
fn list_and_min_max_intrinsics() {
    agrees(
        r#"
        local out = {}
        local function p(...) out[#out + 1] = table.concat({...}, ",") end
        local nan = 0/0
        local vals = {1, 2, -3, 2.5, 1.0, math.huge, -0.0, 0, 2^53, math.maxinteger, math.mininteger}
        local s1, s2 = 0, 0
        for r = 1, 50 do
          for i = 1, #vals do for j = 1, #vals do
            local a, b = vals[i], vals[j]
            local x, y = math.max(a, b), math.min(a, b)
            if r == 50 then p(math.type(x), tostring(x), math.type(y), tostring(y)) end
          end end
          local ia, ib = r, 50 - r
          s1 = s1 + math.max(ia, ib) - math.min(ia, ib)
          local fa, fb = r * 0.5, 25.25
          s2 = s2 + math.max(fa, fb) + math.min(fa, fb)
        end
        p(tostring(math.max(nan, 1)), tostring(math.max(1, nan)), tostring(math.min(nan, 1)), tostring(math.min(1, nan)))
        p(s1, s2)
        -- insert / remove
        local t, sum = {}, 0
        for i = 1, 300 do table.insert(t, i) end
        for i = 1, 100 do sum = sum + table.remove(t) end
        p(#t, sum, tostring(table.remove({})), tostring(table.remove({[0] = 'z'})))
        local logged = {}
        local proxy = setmetatable({}, {__newindex = function(tt, k, v) logged[#logged + 1] = k rawset(tt, k, v) end})
        for i = 1, 300 do table.insert(proxy, i * 2) end
        p(#logged, #proxy)
        local lenned = setmetatable({}, {__len = function() return 7 end})
        for i = 1, 300 do table.insert(lenned, i) end
        p(tostring(rawget(lenned, 8)), tostring(rawget(lenned, 9)))
        local e = {}
        for i = 1, 300 do table.insert(e, nil) end
        p(#e)
        return table.concat(out, "\n")
"#,
    );
}

/// Callees the interpreter saw at a site: one function loaded from varying
/// tables is guarded and called as a constant, a site that changes its callee
/// after compiling leaves and recompiles, and a megamorphic one stays a call.
#[test]
fn callee_feedback() {
    agrees(
        "local function inc(x) return x + 1 end
        local objs, poly = {}, {}
        for i = 1, 10 do
          objs[i] = {f = math.abs, g = inc}
          poly[i] = {g = function(x) return x * i end}
        end
        local s = 0
        for i = 1, 3000 do
          local o = objs[i % 10 + 1]
          s = s + o.f(-i) + o.g(i) + poly[i % 10 + 1].g(2)
          if i == 2000 then
            for j = 1, 10 do objs[j].f, objs[j].g = math.floor, function(x) return x - 1 end end
          end
        end
        return tostring(s)",
    );
}

/// Calls with fewer arguments than the callee's parameters: a known callee's
/// caller stores the nils, an unknown one's entry fills them in.
#[test]
fn arity_shortfall() {
    agrees(
        "local function f(a, b, c) return a + (b == nil and 1 or b) + (c == nil and 10 or c) end
        local fs = {f, function(a, b) return a * 2 + (b or 3) end}
        local s = 0
        for i = 1, 3000 do
          s = s + f(i) + f(i, 2) + f(i, 2, 3)
          local g = fs[i % 2 + 1]
          s = s + g(i) + g(i, 1)
        end
        return tostring(s)",
    );
}

/// A call's results landed in its registers: all of them with `top` past them,
/// for up to three and for more, and a fixed count above two from a callee
/// returning fewer, compiled and native.
#[test]
fn result_landing() {
    agrees(
        "local function many(n, ...) if n == 0 then return end return n, many(n - 1) end
        local function pack(...) return select('#', ...), ... end
        local s, txt = 0, {}
        for i = 1, 3000 do
          local k = i % 6
          local n, a, b, c = pack(many(k))
          s = s + n + (a or 0) + (b or 0) + (c or 0)
          local x, y, z = many(k % 3)
          s = s + (x or 100) + (y or 200) + (z or 300)
          for w in string.gmatch('a b', '%a') do txt[#txt + 1] = w end
          local p, q, r = string.byte('xy', 1, -1)
          s = s + p + q + (r or 7)
        end
        return s .. ' ' .. #txt",
    );
}

/// A recursive call of the closure being compiled enters its region's known
/// entry by `bl`; another closure of the same prototype fails the identity
/// guard and recompiles with a guard on the prototype.
#[test]
fn self_calls() {
    agrees(
        "local function make(k)
          local f
          f = function(n) if n == 0 then return k end return f(n - 1) + 1 end
          return f
        end
        local a, b = make(0), make(100)
        local s = 0
        for i = 1, 2000 do s = s + a(20) end
        for i = 1, 2000 do s = s + a(i % 30) + b(10) end
        return tostring(s)",
    );
}

/// Callees known by their prototype: new closures of one prototype on every
/// call, inlined with their upvalues loaded from the closure and an exit inside
/// the inlined body; tail calls of a known callee that recompiles under its
/// callers; a site that turns polymorphic.
#[test]
fn prototype_callees() {
    agrees(
        "local function make(k)
          local function add(x, y) return x + y + k end
          return add
        end
        local s = 0
        for i = 1, 3000 do
          local add = make(i % 7)
          for j = 1, 3 do s = s + add(j, i) end
          if i == 2500 then s = s + add(0.5, 1) end
        end
        local function leaf(a, b) return a * 2 + (b or 0) end
        local function mid(a) return leaf(a) end
        local t = 0
        for i = 1, 4000 do t = t + mid(i) end
        for i = 1, 4000 do t = t + mid(i + 0.5) end
        local function f1(x) return x + 1 end
        local function f2(x) return x + 2 end
        local fs = {f1, f2}
        local u = 0
        for i = 1, 4000 do
          local f = i < 3000 and f1 or fs[i % 2 + 1]
          u = u + f(i)
        end
        return s .. ' ' .. t .. ' ' .. u",
    );
}

/// Methods found through `__index` inlined from the call sites' callee
/// feedback: three results, and an intrinsic whose builtin an inlined body
/// cannot call, which exits there instead.
#[test]
fn inlined_methods() {
    agrees(
        "local V = {}
        V.__index = V
        function V.new(x, y, z) return setmetatable({x = x, y = y, z = z}, V) end
        function V:parts() return self.x, self.y, self.z end
        function V:mag() return math.abs(self.x) + math.abs(self.y) end
        local vs = {}
        for i = 1, 50 do vs[i] = V.new(i, -i, i * 2) end
        local m = 0
        for k = 1, 200 do
          if k == 150 then vs[7].x = '5' end
          for i = 1, #vs do
            local a, b, c = vs[i]:parts()
            m = m + b + c + vs[i]:mag()
          end
        end
        return tostring(m)",
    );
}

/// `assert` is its arguments, a falsy first one an exit; `error` is an exit
/// (the interpreter's builtins raise, at the call's position), also in
/// inlined bodies and for `return error(...)`. A tail call of `assert` or
/// `select` returns all their results.
#[test]
fn assert_and_error_exit() {
    agrees(
        "local function check(t, i)
          assert(t[i] ~= nil, 'missing ' .. i)
          local a, b = assert(t[i], 'x')
          if t[i] < 0 then error('negative at ' .. i) end
          return a + #b
        end
        local t = {}
        for i = 1, 300 do t[i] = i end
        local s = 0
        for i = 1, 300 do s = s + check(t, i) end
        local msgs = {}
        local function run()
          local x = 0
          for i = 1, 300 do x = x + check(t, i) end
          return x
        end
        for k = 1, 3 do
          t[150] = nil
          local ok, e = pcall(run)
          msgs[#msgs + 1] = tostring(ok) .. ':' .. tostring(e)
          t[150] = -1
          ok, e = pcall(run)
          msgs[#msgs + 1] = tostring(ok) .. ':' .. tostring(e)
          t[150] = 150
        end
        local function thrower(n) if n > 250 then return error({code = n}) end return n end
        local u = 0
        for i = 1, 300 do
          local ok, e = pcall(thrower, i)
          u = u + (ok and e or e.code)
        end
        local function two(v, w)
          local p, q, r = assert(v, w)
          return (p or 0) + (q or 0) + (r and 1 or 0)
        end
        local w = 0
        for i = 1, 500 do w = w + two(i, i) end
        local function both(x) return assert(x, 'm') end
        local function rest(...) return select(2, ...) end
        for i = 1, 500 do
          local p, q = both(i)
          local r, t = rest(i, i, i)
          w = w + #q + r + t
        end
        return s .. ' ' .. table.concat(msgs, ' | ') .. ' ' .. u .. ' ' .. w",
    );
}

/// A register a call does not write keeps its value in its slot across the
/// call, but a captured one may change through its cell: `current` is not
/// reloaded from `last_id`'s slot, which held the same value before.
#[test]
fn captured_slots_change_across_calls() {
    agrees(
        "local List = {}
        List.__index = List
        function List.new() return setmetatable({n = 0}, List) end
        function List:add(x) self.n = self.n + 1; self[self.n] = x end
        function List:each(f) for i = 1, self.n do f(self[i]) end end
        local Finder = {}
        Finder.__index = Finder
        function Finder.new(n)
          local o = setmetatable({number = {}, last = {}, nodes = {}}, Finder)
          for i = 1, n do o.nodes[i] = {out = List.new(), id = i} end
          for i = 1, n - 1 do o.nodes[i].out:add(o.nodes[i + 1]) end
          for i = 1, n - 2, 3 do o.nodes[i].out:add(o.nodes[i + 2]) end
          return o
        end
        function Finder:dfs(node, current)
          self.number[node.id] = current
          local last_id = current
          node.out:each(function(target)
            if self.number[target.id] == nil then
              last_id = self:dfs(target, last_id + 1)
            end
          end)
          self.last[current] = last_id
          return last_id
        end
        local s = 0
        for k = 1, 300 do
          local f = Finder.new(30)
          f:dfs(f.nodes[1], 1)
          for i = 1, 30 do s = s + (f.last[i] or -1000) end
        end
        return tostring(s)",
    );
}

/// A class gains `__index` in a call from a compiled loop that folded its
/// absence: the frame leaves past the call, the next field read walks.
#[test]
fn class_change_in_a_call_deopts_the_caller() {
    agrees(
        "local mt = {}
        local o = setmetatable({x = 1}, mt)
        local go = false
        local function hook(i) if go and i == 150 then mt.__index = {y = 5} end end
        local function f()
          local s = 0
          for i = 1, 300 do
            hook(i)
            local y = o.y
            if y then s = s + y end
          end
          return s
        end
        for k = 1, 20 do assert(f() == 0) end
        go = true
        return f() .. ':' .. f()",
    );
}

/// The compiled frame changes a class itself, through the store helper and
/// `rawset`, whose stores it compiled while they changed nothing; each class
/// is changed first while the region depends on it.
#[test]
fn class_change_by_the_frame_deopts_it() {
    agrees(
        "local mt, mt2 = {}, {}
        local o, o2 = setmetatable({x = 1}, mt), setmetatable({}, mt2)
        local sel, news = {}, {}
        local count = 0
        local function nf(t, k, v) count = count + 1 end
        local function g()
          local s = 0
          for i = 1, 300 do
            mt.__index = sel[i]
            rawset(mt2, '__newindex', news[i])
            local y = o.y
            if y then s = s + y end
            o2.z = i
            o2.z = nil
          end
          return s
        end
        local r = {}
        for k = 1, 10 do r[#r + 1] = g() end
        for i = 150, 300 do sel[i] = {y = 5} end
        r[#r + 1] = g()
        for i = 1, 300 do sel[i] = nil end
        for k = 1, 3 do r[#r + 1] = g() end
        for i = 200, 300 do news[i] = nf end
        r[#r + 1] = g()
        return table.concat(r, ',') .. ':' .. count",
    );
}

/// A coroutine suspended in a compiled frame resumes after the class
/// changed: the patched code reaches its frame on its own stack.
#[test]
fn suspended_coroutine_frames_deopt() {
    agrees(
        "local mt = {}
        local o = setmetatable({x = 1}, mt)
        local co = coroutine.wrap(function()
          local s = 0
          for i = 1, 600 do
            coroutine.yield(i)
            local y = o.y
            if y then s = s + y end
          end
          return s
        end)
        for i = 1, 300 do co() end
        mt.__index = {y = 7}
        local r
        for i = 1, 301 do r = co() end
        return tostring(r)",
    );
}

/// Own-slot reads and stores that folded a class's `__index` and
/// `__newindex` absence, through a metamethod call's arm and function
/// entries after the region is gone.
#[test]
fn class_changes_reach_metamethod_arms_and_entries() {
    agrees(
        "local mt = {}
        local function new(x) return setmetatable({x = x, y = false}, mt) end
        local objs = {}
        for i = 1, 8 do objs[i] = new(i) end
        local function reader(o)
          local v = o.x
          o.y = nil
          local w = o.y
          o.y = false
          return v, w
        end
        local function run()
          local s, n = 0, 0
          for i = 1, 400 do
            local v, w = reader(objs[i % 8 + 1])
            s = s + v
            if w then n = n + 1 end
          end
          return s .. ':' .. n
        end
        local out = {}
        for k = 1, 10 do run() end
        out[#out + 1] = run()
        mt.__index = function(t, k) return k == 'y' and 1 or nil end
        out[#out + 1] = run()
        local log = 0
        mt.__newindex = function(t, k, v) log = log + 1; rawset(t, k, v) end
        out[#out + 1] = run() .. '/' .. log
        local lt = setmetatable({}, {__lt = function(a, b)
          mt.__index = nil
          return true
        end})
        local c = 0
        for i = 1, 400 do
          if lt < lt then c = c + 1 end
          local w = objs[1].zz
          if w then c = c + 1000 end
          mt.__index = function() return 1 end
        end
        out[#out + 1] = c
        mt.__index = nil
        mt.__newindex = nil
        out[#out + 1] = run()
        return table.concat(out, ' ')",
    );
}

/// Methods read through `__index` tables at any depth fold to constants;
/// replacing one, shadowing an inherited one, deleting one and re-parenting a
/// holder each reach the compiled readers, from outside, from a call and from
/// the loop's own stores.
#[test]
fn holder_changes_deopt_folded_method_reads() {
    agrees(
        "local Base = {}
        function Base:kind() return 1 end
        function Base:tag() return 'b' end
        local Mid = setmetatable({}, {__index = Base})
        local Derived = setmetatable({}, {__index = Mid})
        function Derived:extra() return 2 end
        local function new() return setmetatable({v = 3}, {__index = Derived}) end
        local o = new()
        local hook = function(i) end
        local function run()
          local s, t = 0, ''
          for i = 1, 300 do
            hook(i)
            s = s + o:kind() + o:extra() + (o.missing and 100 or 0)
            if i == 300 then t = o:tag() end
          end
          return s .. t
        end
        local r = {}
        for k = 1, 10 do r[#r + 1] = run() end
        Base.kind = function() return 5 end
        r[#r + 1] = run()
        Mid.kind = function() return 7 end
        r[#r + 1] = run()
        hook = function(i) if i == 150 then Derived.missing = true end end
        r[#r + 1] = run()
        hook = function(i) if i == 100 then Derived.missing = nil end end
        r[#r + 1] = run()
        hook = function(i) if i == 120 then setmetatable(Derived, {__index = Base}) end end
        r[#r + 1] = run()
        Mid.tag = function() return 'm' end
        hook = function(i) if i == 10 then setmetatable(Derived, {__index = Mid}) end end
        r[#r + 1] = run()
        Mid.kind = nil
        r[#r + 1] = run()
        local function self_edit()
          local s = 0
          for i = 1, 300 do
            s = s + o:extra()
            if i == 200 then Derived.extra = function() return 40 end end
            if i == 250 then rawset(Derived, 'extra', function() return 400 end) end
          end
          return s
        end
        for k = 1, 5 do r[#r + 1] = self_edit(); Derived.extra = function() return 2 end end
        return table.concat(r, ',')",
    );
}

/// A holder slot that keeps changing is read, not folded, and its stores
/// stay right once cached; a holder that grows past the shape limit into a
/// dictionary leaves its dependents.
#[test]
fn changing_and_growing_holders() {
    agrees(
        "local C = {count = 0}
        function C:get() return C.count end
        local mt = {__index = C}
        local o = setmetatable({}, mt)
        local function run()
          local s = 0
          for i = 1, 300 do
            C.count = C.count + 1
            s = s + o:get() + o.count
          end
          return s
        end
        local r = {}
        for k = 1, 10 do r[#r + 1] = run() end
        for i = 1, 600 do C['k' .. i] = i end
        r[#r + 1] = run()
        r[#r + 1] = o.k599
        return table.concat(r, ',')",
    );
}

/// An `__index` function past a holder is called with that holder, not the
/// receiver; one on the receiver's class with the receiver.
#[test]
fn index_functions_past_holders_get_the_holder() {
    agrees(
        "local B = {}
        setmetatable(B, {__index = function(t, k) return t == B and 1 or 2 end})
        local o = setmetatable({}, {__index = B})
        local p = setmetatable({}, {__index = function(t, k) return t == o and 10 or 20 end})
        local s = 0
        for i = 1, 400 do s = s + o.foo + p.bar end
        return tostring(s)",
    );
}

/// String methods fold through the strings' metatable and the `string`
/// table, which a replaced function reaches.
#[test]
fn string_methods_follow_the_string_table() {
    agrees(
        "local function run()
          local n = 0
          for i = 1, 300 do n = n + #(('ab'):upper()) end
          return n
        end
        local r = {}
        for k = 1, 10 do r[#r + 1] = run() end
        local up = string.upper
        string.upper = function(s) return up(s) .. '!' end
        r[#r + 1] = run()
        string.upper = up
        r[#r + 1] = run()
        return table.concat(r, ',')",
    );
}

/// An arithmetic metamethod of a table whose shape the code knows is
/// called as a constant; replacing it reaches the loop.
#[test]
fn known_arithmetic_metamethods_follow_the_class() {
    agrees(
        "local mt = {}
        mt.__add = function(a, b) return a.v + b end
        local o = setmetatable({v = 1}, mt)
        local function run()
          local s = 0
          for i = 1, 300 do
            s = s + o.v
            s = s + (o + i)
            if i == 200 then mt.__add = function(a, b) return 1000 end end
          end
          mt.__add = function(a, b) return a.v + b end
          return s
        end
        local r = {}
        for k = 1, 10 do r[#r + 1] = run() end
        return table.concat(r, ',')",
    );
}

/// Compiled loads take a slot's kind as their type; a store that widens it,
/// from outside, from a call or from the loop itself, reaches them.
#[test]
fn widened_slots_reach_typed_loads() {
    agrees(
        "local function new(x) return {x = x, y = 2} end
        local ps = {}
        for i = 1, 8 do ps[i] = new(i) end
        local hook = function(i) end
        local function run()
          local s = 0
          for i = 1, 300 do
            local p = ps[i % 8 + 1]
            hook(i)
            s = s + p.x * p.y
            if p.x == nil then s = s + 1000 end
          end
          return s
        end
        local r = {}
        for k = 1, 10 do r[#r + 1] = run() end
        ps[3].x = 1.5
        r[#r + 1] = run()
        hook = function(i) if i == 150 then ps[4].x = 'z' == 'z' and 2.25 end end
        r[#r + 1] = run()
        hook = function(i) if i == 100 then ps[5].x = 7 end end
        r[#r + 1] = run()
        local function self_edit()
          local s = 0
          for i = 1, 300 do
            local p = ps[i % 8 + 1]
            s = s + p.x
            if i == 200 then p.x = 0.5 end
          end
          return s
        end
        for k = 1, 5 do r[#r + 1] = self_edit() end
        return table.concat(r, ',')",
    );
}

/// A compiled store whose value does not fit the slot leaves before it,
/// and the interpreter's widens the slot for every table of the shapes
/// sharing it.
#[test]
fn stores_that_do_not_fit_widen_their_slot() {
    agrees(
        "local function new() return {n = 0, s = ''} end
        local ts = {new(), new(), new()}
        local function set(t, v, w) t.n = v; t.s = w end
        local function run(v, w)
          local acc = {}
          for i = 1, 300 do
            local t = ts[i % 3 + 1]
            set(t, i < 250 and i or v, w)
            acc[#acc + 1] = type(t.n) .. type(t.s)
          end
          return table.concat(acc, '', 295)
        end
        local r = {}
        for k = 1, 10 do r[#r + 1] = run(1, 'a') end
        r[#r + 1] = run(1.5, 'a')
        r[#r + 1] = run({}, nil)
        r[#r + 1] = run(nil, true)
        r[#r + 1] = run(1, 'a')
        return table.concat(r, ',')",
    );
}

/// A constructor's constant nil is in every table it makes, so the slot's
/// kind holds nil though the other stores to it are tables (quadtree's
/// `child1 = nil`).
#[test]
fn baked_nil_fields_stay_nil() {
    agrees(
        "local function cell() return {child = nil, color = 0.5} end
        local root = cell()
        local function split(q) q.child = cell() end
        local function depth(q)
          local d = 0
          while q.child ~= nil do q = q.child; d = d + 1 end
          return d
        end
        local r = {}
        for k = 1, 300 do
          local q = root
          for i = 1, k % 5 do
            if q.child == nil then split(q) end
            q = q.child
          end
          r[#r + 1] = depth(root) + depth(cell())
        end
        return table.concat(r, ',')",
    );
}

/// The collector clears a weak-valued table's slots to nil whatever their
/// kind: mid-loop, in an arm shared with a plain shape warmed first, after
/// a `__mode` set in place under code that loaded from the table, and in a
/// table that then moves to a strong shape sharing the slot. Typed loads
/// must agree with `rawget` (a helper, kinds aside) whenever collections
/// run, which differs between the interpreter and compiled code.
#[test]
fn weak_slots_load_as_nilable() {
    const PROBE: &str = "bad, nils = 0, 0
function probe(t, gc)
  for i = 1, 300 do
    -- Collections at its checks clear `weak` mid-loop under gc stress.
    local junk = {i}
    if gc and i == 150 then yielder() end
    local typed = t.obj == nil
    if typed ~= (rawget(t, 'obj') == nil) then bad = bad + 1 end
    if typed then nils = nils + 1 end
  end
end
";
    let runs = [
        "plain = setmetatable({obj = {}, n = 1}, {})
weak = setmetatable({obj = {}, n = 1}, {__mode = 'v'})
flip = setmetatable({obj = {}, n = 1}, {})
for k = 1, 10 do probe(plain); probe(flip) end
for k = 1, 10 do probe(k % 2 == 0 and plain or weak) end
weak2 = setmetatable({obj = {}, n = 1}, {__mode = 'v'})
probe(weak2, true)
getmetatable(flip).__mode = 'v'
--gc
for k = 1, 4 do probe(k % 2 == 0 and plain or weak) end
probe(flip)
return bad .. ' ' .. nils",
        "plain = setmetatable({obj = {}, n = 1}, {})
weak = setmetatable({obj = {}, n = 1}, {__mode = 'v'})
for k = 1, 10 do probe(plain) end
--gc
setmetatable(weak, getmetatable(plain))
for k = 1, 10 do probe(weak); probe(plain) end
return bad .. ' ' .. nils",
    ];
    for src in runs {
        for jit in [true, false] {
            let mut lua = tcvm_lua_yielding();
            lua.set_jit(jit);
            let out = crate::common::run_chunks(&mut lua, &format!("{PROBE}{src}"));
            let (bad, nils) = out.split_once(' ').unwrap();
            assert!(bad == "0" && nils != "0", "jit {jit}: {out}\n{src}");
        }
    }
}

fn tcvm_lua_yielding() -> Lua {
    crate::common::yielding_lua()
}

/// Writers past the cached stores widen slots too: `rawset`, a
/// metatable kept in a poly shape's slot, and a holder's own stores.
#[test]
fn every_writer_widens_slots() {
    agrees(
        "local ts = {}
        for i = 1, 6 do ts[i] = {x = i} end
        local mts = {{}, {}}
        local objs = {}
        for i = 1, 6 do objs[i] = setmetatable({v = i}, mts[i % 2 + 1]) end
        local function run()
          local s = 0
          for i = 1, 300 do
            local t, o = ts[i % 6 + 1], objs[i % 6 + 1]
            s = s + t.x + o.v
          end
          return s
        end
        local r = {}
        for k = 1, 10 do r[#r + 1] = run() end
        rawset(ts[2], 'x', 0.25)
        r[#r + 1] = run()
        rawset(objs[3], 'v', 0.5)
        r[#r + 1] = run()
        setmetatable(objs[4], nil)
        r[#r + 1] = run()
        return table.concat(r, ',')",
    );
}

/// Tables loaded from a slot that tracks their stable shape leave it every way
/// a table can: a new key, a metatable, becoming an `__index` holder; a
/// compiled loop over them is invalidated each time.
#[test]
fn tables_leaving_tracked_shapes_invalidate() {
    agrees(
        "local head
        for i = 1, 200 do head = {v = i, next = head} end
        local function sum(l) local s = 0 while l do s = s + l.v l = l.next end return s end
        local r = {}
        for k = 1, 20 do r[#r + 1] = sum(head) end
        head.next.next.w = 1
        r[#r + 1] = sum(head)
        setmetatable(head.next.next.next, {__index = {v = 0}})
        r[#r + 1] = sum(head)
        local n4 = head.next.next.next.next
        setmetatable({}, {__index = n4})
        r[#r + 1] = sum(head)
        n4.v = 0.5
        r[#r + 1] = sum(head)
        return table.concat(r, ',')",
    );
}

/// A slot tracking one shape takes a table of another, stored by the
/// interpreter, by compiled code, into itself, and by a transition.
#[test]
fn tracked_slots_take_other_shapes() {
    agrees(
        "local function mk(i) return {a = i, b = i * 2} end
        local holder = {p = mk(1)}
        local function get(h) return h.p.a + h.p.b end
        local function put(h, v) h.p = v end
        local r = {}
        for i = 1, 300 do put(holder, mk(i)); r[#r + 1] = get(holder) end
        put(holder, {b = 5, a = 7})
        r[#r + 1] = get(holder)
        holder.p = {a = 1, b = 2, c = 3}
        r[#r + 1] = get(holder)
        local s = {a = 1, b = 2}
        s.p = s
        s.p = s
        r[#r + 1] = s.p.p.a
        local t = {}
        t.self = t
        t.self = t
        r[#r + 1] = tostring(t.self == t)
        return table.concat(r, ',')",
    );
}

/// Shapes known before a call are known after it while stable; a callee
/// that moves the table, or another of its shape, invalidates the caller.
#[test]
fn stable_shapes_across_calls() {
    agrees(
        "local P = {}
        P.__index = P
        local function new(x) return setmetatable({x = x, y = x + 1}, P) end
        local go = false
        local other = new(0)
        local function g(o) if go then o.z = 1 end return 1 end
        local function h() if go then other.w = 2 end return 1 end
        local function m(o) local s = o.x + g(o) s = s + o.y + h() return s + o.x + (o.z or 0) end
        local r = {}
        local a = new(5)
        for i = 1, 3000 do r[#r + 1] = m(a) end
        go = true
        r[#r + 1] = m(new(7))
        r[#r + 1] = m(a)
        return table.concat(r, ',')",
    );
}

/// A nil own slot under `__index` walks the chain each time without
/// refilling its entry, and sees the chain and the slot change.
#[test]
fn nil_own_slots_walk_without_refills() {
    agrees(
        "local P = {v = 5}
        local mt = {__index = P}
        local objs = {}
        for i = 1, 4 do objs[i] = setmetatable({v = nil, w = i}, mt) end
        local function get(o) return o.v end
        local r = {}
        for i = 1, 400 do r[#r + 1] = get(objs[i % 4 + 1]) end
        P.v = 7
        r[#r + 1] = get(objs[1])
        mt.__index = function(t, k) return k end
        r[#r + 1] = get(objs[2])
        objs[3].v = 1
        r[#r + 1] = get(objs[3])
        return table.concat(r, ',')",
    );
}

/// binary-trees at depth 8: branch elimination clones `BottomUpTree`'s
/// constructor into both arms, and its repair joined the table and its
/// unboxed pointer, which the loop's next call kept live: a raw pointer
/// across a call, failing the verifier (now folded into an unboxing of the
/// joined table).
#[test]
fn joined_table_pointers_across_calls() {
    compiles_cleanly(
        "local function BottomUpTree(item, depth)
          local left, right = nil, nil
          if depth > 0 then
            local i = item + item
            depth = depth - 1
            left = BottomUpTree(i-1, depth)
            right = BottomUpTree(i, depth)
          end
          return { item = item, left = left, right = right }
        end
        local function ItemCheck(tree)
          if tree.left then
            return tree.item + ItemCheck(tree.left) - ItemCheck(tree.right)
          else
            return tree.item
          end
        end
        local out = {}
        local mindepth, maxdepth = 4, 8
        out[#out + 1] = ItemCheck(BottomUpTree(0, maxdepth + 1))
        local longlivedtree = BottomUpTree(0, maxdepth)
        for depth = mindepth, maxdepth, 2 do
          local iterations = 2 ^ (maxdepth - depth + mindepth)
          local check = 0
          for i = 1, iterations do
            check = check + ItemCheck(BottomUpTree(1, depth)) +
                    ItemCheck(BottomUpTree(-1, depth))
          end
          out[#out + 1] = iterations * 2 .. ' ' .. depth .. ' ' .. check
        end
        out[#out + 1] = ItemCheck(longlivedtree)
        return table.concat(out, ' | ')",
        true,
    );
}

/// A loaded value read after its slot is written again is not read from the
/// slot: an exit of a loop's first compiled iteration names a local's old
/// value, after a closure's capture had the new one stored.
#[test]
fn loaded_values_outlive_their_slots() {
    agrees(
        r##"
        local tbl = {}
        for i = 1, 400 do tbl[i] = i end
        tbl[300], tbl[301], tbl[302] = 0.5, 0.5, 0.5
        local a = 5
        local res = 0
        for i = 1, 400 do
            local t = a
            a = i
            local c = function() return a end
            local y = tbl[i] + 1
            if y > 100000 then res = 0 end
            res = res + t
        end
        return tostring(res)
"##,
    );
}

/// A compare of numbers of any kinds stays behind the test that found
/// numbers: in the loop's preheader it would read a nil as a big integer's
/// box. The first iteration skips the compare, so no copy of it runs first.
#[test]
fn number_compare_stays_behind_its_type_test() {
    let got = result(
        "local function f(a, b, n)
          local c = 0
          for i = 1, n do
            if i > 1 and a <= b then c = c + 1 end
          end
          return c
        end
        local t = 0
        for r = 1, 300 do
          t = t + f(1 << 40, 2.5, 3) + f(1.5, 1 << 40, 3) + f(1 << 40, 1 << 41, 3)
        end
        return t .. ' ' .. tostring(pcall(f, nil, 1 << 40, 3))",
        true,
    );
    assert_eq!(got, "1200 false");
}

/// An integer loop's iteration count is divided out behind the guard that
/// its step is not zero: x86-64's `div` traps. Here the zero step is never
/// reached, behind a branch on a value the outer loop does not change.
/// Inlining is off so that the calls with it enter `f`'s own region.
#[test]
fn loop_count_division_stays_behind_its_step_guard() {
    let mut lua = Lua::new();
    lua.set_jit_inline(false);
    lua.load_all();
    let ex = start_on(
        &mut lua,
        "local function f(a, b, s, go, n)
          local c = 0
          for i = 1, n do if go then for k = a, b, s do c = c + 1 end end end
          return c
        end
        local t = 0
        for r = 1, 300 do
          t = t + f(1, 1 << 40, 1 << 38, true, 50) + f((1 << 40) - 5, 1 << 40, 1, true, 50)
        end
        return t .. ' ' .. f(1, 1 << 40, 0, false, 100000) .. ' '
          .. f((1 << 40) - 5, 1 << 40, 0, false, 100000)",
    );
    lua.finish(&ex).unwrap();
    let got = lua.enter(|ctx| {
        let v: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        String::from_utf8_lossy(v.get_string().expect("a string result").as_bytes()).into_owned()
    });
    assert_eq!(got, "150000 0 0");
}

/// An unboxed constant a late fold left live across a call: the third
/// program of #340, whose region failed to compile at these counts.
#[test]
fn folded_constants_cross_calls_in_recompiles() {
    let src = r##"
    local function emit(s) io.write(s, "\n") end
    local function tick()
    end
    local function reset()
    V = {}
    V.si = {}
    V.sp = { 1 }
    V.ib = {  1 << 40 }
    for k, l in pairs(V) do n = #l end
    end
    do
    reset()
    local function f(a, b, c)
    local function h2()
    end
    for i7 = 2, 1, (-1) do
    end
    for i10 = 0, (#("")) do
    tick()
    local v12 = {z = ((c) or h2), y = a, (i10 - h2)}
    for i13 = 3, 3 do
    tick()
    end
    end
    end
    local function via_tail(a, b, c) return f(a, b, c) end
    local function run(LA, LB, LC, n)
    local co = (function()
    for i = pos + 1, n do
    pos = i
    local a, b, c = LA[(i - 1) % n + 1], LB[1], LC[1]
    local s = (call(a, b, c)) .. probes
    end
    end)
    while co() ~= "" do end
    end
    local function drive(LA, LB, LC, n)
    pos = 0
    while pos < n do
    local e = pcall(run, LA, LB, LC, n)
    end
    end
    call = via_tail
    drive(V.ib, V.si, V.sp, 60)
    end
    return "END"
"##;
    compiles_hot(src, false, 1, 3);
}
