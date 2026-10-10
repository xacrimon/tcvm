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
