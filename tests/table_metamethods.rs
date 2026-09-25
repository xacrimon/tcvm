//! The table library honours `__index`, `__newindex` and `__len` (#185),
//! including on non-tables that carry them. Expected strings come from `lua`
//! 5.5.1 running the same chunk.

use tcvm::env::Value;
use tcvm::{Executor, LoadError, Lua, RuntimeError};

/// Kept on the chunk's first line so error positions are unaffected.
const PRELUDE: &str = "local function cat(...) local t = table.pack(...) \
    for i = 1, t.n do t[i] = tostring(t[i]) end return table.concat(t, ' ') end ";

/// `proxy(store)` logs each metamethod call; `flush()` returns and clears the log.
const PROXY: &str = r#"
local log = {}
local function proxy(store)
  return setmetatable({}, {
    __index = function(_, k) log[#log + 1] = 'r' .. k; return store[k] end,
    __newindex = function(_, k, v) log[#log + 1] = 'w' .. k .. '=' .. tostring(v); store[k] = v end,
    __len = function() log[#log + 1] = '#'; return #store end,
  })
end
local function flush() local s = table.concat(log, ' '); log = {}; return s end
"#;

fn run(src: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    let src = format!("{PRELUDE}{PROXY}{src}");
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(&src, Some("=c"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    match lua.finish(&ex) {
        Ok(()) => lua.enter(|ctx| {
            let v = ctx.fetch(&ex).take_result::<Value>(ctx).expect("result");
            let s = v.get_string().expect("string result");
            String::from_utf8_lossy(s.as_bytes()).into_owned()
        }),
        Err(RuntimeError::Lua(e)) => panic!(
            "{src:?} raised {:?}",
            lua.enter(|ctx| {
                ctx.fetch(&e)
                    .value()
                    .get_string()
                    .map(|s| String::from_utf8_lossy(s.as_bytes()).into_owned())
            })
        ),
        Err(e) => panic!("unexpected failure for {src:?}: {e:?}"),
    }
}

/// The repro from #185: proxies read, written and measured through metamethods.
#[test]
fn issue_185_repro() {
    let src = r#"
local out = {}
local p = setmetatable({}, {__index = function(_, k) return k * 10 end, __len = function() return 3 end})
out[#out + 1] = cat(table.concat(p, ','), table.unpack(p))
local store = {}
p = setmetatable({}, {__index = store, __newindex = function(_, k, v) rawset(store, k, v) end,
                      __len = function() return #store end})
table.insert(p, 'a'); table.insert(p, 'b')
out[#out + 1] = cat(#store, store[1], store[2], rawget(p, 1))
store = {'a', 'b', 'c'}
p = setmetatable({}, {__index = store, __newindex = store, __len = function() return #store end})
out[#out + 1] = cat(table.remove(p), #store, store[3])
local dst = {}
store = {'a', 'b'}
table.move(setmetatable({}, {__index = store}), 1, 2, 1, setmetatable({}, {__newindex = dst}))
out[#out + 1] = cat(dst[1], dst[2])
return table.concat(out, ' | ')
"#;
    assert_eq!(run(src), r#"10,20,30 10 20 30 | 2 a b nil | c 2 nil | a b"#);
}

/// Every `__index`/`__newindex`/`__len` call, in the reference's order.
#[test]
fn access_order() {
    let src = r#"
local s = {1, 2, 3, 4, 5}
local out = {}
table.insert(proxy(s), 2, 'x'); out[#out + 1] = flush()
table.insert(proxy(s), 'y'); out[#out + 1] = flush()
out[#out + 1] = cat(table.remove(proxy(s), 1), flush())
out[#out + 1] = cat(table.remove(proxy(s)), flush())
out[#out + 1] = cat(table.remove(proxy({})), flush())
out[#out + 1] = cat(table.concat(proxy({1, 2, 3}), '-', 2), flush())
out[#out + 1] = cat(select('#', table.unpack(proxy({1, 2, 3}), 2, 3)), flush())
table.move(proxy({1, 2, 3, 4}), 1, 4, 2, proxy({})); out[#out + 1] = flush()
local ps = proxy({1, 2, 3, 4})
table.move(ps, 1, 3, 2); out[#out + 1] = flush()
table.move(ps, 1, 3, 2, ps); out[#out + 1] = flush()
return table.concat(out, ' | ')
"#;
    assert_eq!(
        run(src),
        r#"# r5 w6=5 r4 w5=4 r3 w4=3 r2 w3=2 w2=x | # w7=y | 1 # r1 r2 w1=x r3 w2=2 r4 w3=3 r5 w4=4 r6 w5=5 r7 w6=y w7=nil | y # r6 w6=nil | nil # r0 w0=nil | 2-3 # r2 r3 | 2 # r2 r3 | r1 w2=1 r2 w3=2 r3 w4=3 r4 w5=4 | r3 w4=3 r2 w3=2 r1 w2=1 | r3 w4=2 r2 w3=1 r1 w2=1"#
    );
}

/// `move` asks `__eq` whether overlapping ranges share a table.
#[test]
fn move_consults_eq() {
    let src = r#"
local out = {}
local n = 0
local mt = {__eq = function() n = n + 1; return true end}
local a, b = setmetatable({1, 2, 3}, mt), setmetatable({}, mt)
table.move(a, 1, 3, 2, b)
out[#out + 1] = cat(n, b[1], b[2], b[3], b[4])
mt.__eq = function() n = n + 1; return false end
b = setmetatable({}, mt)
table.move(a, 1, 3, 2, b)
out[#out + 1] = cat(n, b[1], b[2], b[3], b[4])
table.move(a, 1, 3, 5, b)
out[#out + 1] = cat(n)
return table.concat(out, ' | ')
"#;
    assert_eq!(run(src), r#"1 nil 1 2 3 | 2 nil 1 2 3 | 2"#);
}

/// `__len` results convert like `luaL_len`.
#[test]
fn length_conversion() {
    let src = r#"
local function lenp(n) return setmetatable({}, {__index = function(_, k) return k end, __len = function() return n end}) end
return cat(table.concat(lenp(3.0), ','), table.concat(lenp(' 0x3 '), ','), select('#', table.unpack(lenp('2'))),
  pcall(table.concat, lenp(1.5)))
"#;
    assert_eq!(
        run(src),
        r#"1,2,3 1,2,3 2 false object length is not an integer"#
    );
}

/// `checktab` accepts non-tables with the needed metamethods.
#[test]
fn non_table_arguments() {
    let src = r#"
local out = {}
getmetatable(io.stdout).__len = function() return 0 end
out[#out + 1] = cat(pcall(table.concat, io.stdout))
out[#out + 1] = cat(table.concat('ab', ',', 1, 0), pcall(table.unpack, 'abc'))
out[#out + 1] = cat(pcall(table.concat, setmetatable({}, {__index = {}})))
return table.concat(out, ' | ')
"#;
    assert_eq!(run(src), r#"true  |  true nil nil nil | true "#);
}

/// Errors from metamethods, chains and arguments, and when they are raised.
#[test]
fn errors() {
    let src = r#"
local out = {}
out[#out + 1] = cat(pcall(table.concat, setmetatable({}, {__index = 5, __len = function() return 1 end})))
out[#out + 1] = cat(pcall(table.insert, setmetatable({}, {__newindex = 5, __len = function() return 0 end}), 1))
local loop = {}; loop.__newindex = loop; setmetatable(loop, loop)
out[#out + 1] = cat(pcall(table.insert, setmetatable({}, {__newindex = loop, __len = function() return 0 end}), 1))
out[#out + 1] = cat(pcall(table.concat, setmetatable({}, {__len = function() error('lenerr', 0) end}), {}))
out[#out + 1] = cat(pcall(table.insert, proxy({})), flush())
out[#out + 1] = cat(pcall(table.insert, setmetatable({}, {__len = function() return math.maxinteger end}), 1, 2))
out[#out + 1] = cat(pcall(table.unpack, setmetatable({}, {__index = function() error('boom', 0) end, __len = function() return 1 end})))
return table.concat(out, ' | ')
"#;
    assert_eq!(
        run(src),
        r#"false attempt to index a number value | false attempt to index a number value | false '__newindex' chain too long; possible loop | false lenerr | false # | true | false boom"#
    );
}

/// `sort` reads, writes and compares in `auxsort`'s exact order.
#[test]
fn sort_access_order() {
    let src = r#"
local out = {}
local store = {5, 4, 3}
table.sort(setmetatable({}, {__index = store, __newindex = store, __len = function() return #store end}))
out[#out + 1] = table.concat(store, ',')
for _, arr in ipairs({{3, 1, 2}, {2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5}, {'b', 'a', 'c', 'd'}}) do
  table.sort(proxy(arr))
  out[#out + 1] = flush() .. ' => ' .. table.concat(arr, ',')
end
local arr = {2, 7, 1, 8, 2, 8, 1, 8}
table.sort(proxy(arr), function(a, b) log[#log + 1] = 'c' .. a .. b; return a > b end)
out[#out + 1] = flush() .. ' => ' .. table.concat(arr, ',')
return table.concat(out, ' | ')
"#;
    assert_eq!(
        run(src),
        r#"3,4,5 | # r1 r3 w1=2 w3=3 r2 r1 w2=2 w1=1 => 1,2,3 | # r1 r16 r8 r1 r16 w8=5 w16=8 r8 r15 w8=4 w15=5 r2 r14 w2=0 w14=7 r3 r4 r13 r12 w4=5 w12=8 r5 r6 r11 w6=4 w11=8 r7 r8 r9 r10 r10 r9 w15=8 w10=5 r11 r16 r13 r11 r16 w13=8 w16=9 r13 r15 w13=8 w15=8 r12 r14 w12=7 w14=8 r13 r13 w13=8 w13=8 r14 r12 w15=8 w14=8 r15 r16 r11 r13 r12 r11 w12=8 w11=7 r1 r9 r5 r1 r9 r5 r8 w5=4 w8=2 r2 r3 r4 r7 w4=1 w7=5 r5 r6 r5 r4 w8=4 w5=2 r6 r9 w6=2 w9=4 r7 r6 r9 w7=4 w9=5 r7 r8 w7=4 w8=4 r7 r7 w7=4 w7=4 r8 r6 w8=4 w8=4 r6 r7 r1 r4 w1=1 w4=2 r2 r1 w2=1 w1=0 r2 r3 w2=1 w3=1 r2 r2 w2=1 w2=1 r3 r1 w3=1 w3=1 r1 r2 => 0,1,1,2,2,2,4,4,5,5,7,8,8,8,8,9 | # r1 r4 r2 r1 w2=b w1=a r2 r3 w2=c w3=b r2 r2 r1 w3=c w2=b r3 r4 => a,b,c,d | # r1 r8 c82 w1=8 w8=2 r4 r1 c88 r8 c28 r4 r7 w4=1 w7=8 r2 c78 r6 c88 w2=8 w6=7 r3 c18 r5 c82 r4 c81 r3 c81 r2 c88 w7=1 w3=8 r1 r2 c88 r4 r8 c21 w4=2 w8=1 r6 r4 c72 w6=2 w4=7 r6 r7 w6=1 w7=2 r5 c22 r6 c21 r5 c22 w5=2 w5=2 r6 c12 r4 c27 w7=1 w6=2 r7 r8 c11 r4 r5 c27 => 8,8,8,7,2,2,1,1"#
    );
}

/// Metamethods a comparator adds mid-sort take effect for the rest of it.
#[test]
fn sort_rechecks_metatable() {
    let src = r#"
-- A comparator that gives the array metamethods partway through: every later
-- read and write must go through them.
local arr = {5, 3, 8, 1, 9, 2, 7}
local calls = 0
table.sort(arr, function(a, b)
  calls = calls + 1
  if calls == 4 then
    local store = {}
    for i = 1, #arr do store[i] = rawget(arr, i); rawset(arr, i, nil) end
    setmetatable(arr, {
      __index = function(_, k) log[#log + 1] = 'r' .. k; return store[k] end,
      __newindex = function(_, k, v) log[#log + 1] = 'w' .. k; store[k] = v end,
    })
    rawset(arr, "store", store)
  end
  return a < b
end)
return flush() .. ' => ' .. table.concat(rawget(arr, 'store'), ',')
"#;
    assert_eq!(
        run(src),
        r#"r5 r4 w3 w4 r4 r3 w6 w4 r5 r7 w5 w7 r6 r5 r7 r1 r3 r2 r1 r3 w2 w3 => 1,2,3,5,7,8,9"#
    );
}

/// `sort` errors on proxies, after the reads that precede them.
#[test]
fn sort_errors() {
    let src = r#"
local out = {}
out[#out + 1] = cat(pcall(table.sort, proxy({3, 2, 1, 4}), function() return true end), flush())
out[#out + 1] = cat(pcall(table.sort, proxy({3, 2, 1}), 5), flush())
out[#out + 1] = cat(pcall(table.sort, setmetatable({}, {__index = {2, {}}, __newindex = {}, __len = function() return 2 end})))
return table.concat(out, ' | ')
"#;
    assert_eq!(
        run(src),
        r#"false # r1 r4 w1=4 w4=3 r2 r1 w2=4 w1=2 r2 r3 w2=1 w3=4 r2 r3 | false # | false attempt to compare table with number"#
    );
}

/// Yielding from a metamethod the table library calls is an intentional
/// divergence: Lua raises "attempt to yield across a C-call boundary".
#[test]
fn metamethods_may_yield() {
    let src = r#"
local co = coroutine.wrap(function()
  local p = setmetatable({}, {
    __index = function(_, k) coroutine.yield('y' .. k); return k end,
    __len = function() coroutine.yield('len'); return 2 end,
  })
  return table.concat(p, '+')
end)
return cat(co(), co(), co(), co())
"#;
    assert_eq!(run(src), "len y1 y2 1+2");
}
