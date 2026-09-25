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
