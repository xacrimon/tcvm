//! The inline CALL entries for `math.sqrt/sin/cos/abs/floor/ceil` on edge
//! values, argument counts and result counts. The snapshot is `lua` 5.5.1's
//! output for the same chunk. Each probe also runs as a tail call, which takes
//! the generic native path, and must agree.

use tcvm::env::LuaString;
use tcvm::{Executor, LoadError, Lua};

const CHUNK: &str = r##"
local function show(v)
  local t = math.type(v)
  if t == "float" and v ~= v then return "float:nan" end
  if t then return t .. ":" .. tostring(v) end
  if type(v) == "string" then return "string:\"" .. v .. "\"" end
  return type(v)
end
local vals = {0, 1, -1, 7, math.mininteger, math.maxinteger, 2147483647, 2147483648, -2147483648, -2147483649,
  0.0, -0.0, 0.5, -0.5, 3.7, -3.7, 2^31, -2^31, 2^53, 2^63, -2^63, 1e308, 1/0, -1/0, 0/0,
  "10", "3.7", " 0x10 ", "-2.5e1", "abc", true}
local out = {}
for _, name in ipairs({"sqrt", "sin", "cos", "abs", "floor", "ceil"}) do
  local f = math[name]
  for i = 1, #vals do
    local v = vals[i]
    local ok, r = pcall(function() local r = f(v) return r end)
    local tok, tr = pcall(function() return f(v) end)
    assert(tok == ok and (not ok or show(tr) == show(r)), name)
    out[#out + 1] = name .. "(" .. show(v) .. ")=" .. (ok and show(r) or "error")
  end
  -- no argument, nil, extra arguments, result count adjustment
  out[#out + 1] = name .. "()=" .. (pcall(function() local r = f() return r end) and "ok" or "error")
  out[#out + 1] = name .. "(nil)=" .. (pcall(function() local r = f(nil) return r end) and "ok" or "error")
  out[#out + 1] = name .. "(2,9,9)=" .. show(f(2, 9, 9))
  out[#out + 1] = name .. " nres=" .. select("#", f(2))
  local a, b, c = f(2)
  out[#out + 1] = name .. " 3res=" .. show(a) .. "," .. show(b) .. "," .. show(c)
  f(2)
end
return table.concat(out, "\n")
"##;

#[test]
fn math_fast_entries() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(CHUNK, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    let out = lua.enter(|ctx| {
        let s = ctx
            .fetch(&ex)
            .take_result::<LuaString>(ctx)
            .expect("result");
        String::from_utf8_lossy(s.as_bytes()).into_owned()
    });
    insta::assert_snapshot!(out);
}
