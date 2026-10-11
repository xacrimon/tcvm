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
/// library's slot, each guarded; replacing a method, or `__index` itself,
/// after compiling (the design's `hijacked` example, 11.5).
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
