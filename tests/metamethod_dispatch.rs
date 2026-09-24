//! Metamethods on values other than tables: userdata use their own
//! metatable, every other type the one shared by its type. Expected strings
//! come from `lua` 5.5.1 running the same chunk.

use tcvm::env::Value;
use tcvm::{Executor, LoadError, Lua, RuntimeError};

/// Kept on the chunk's first line so error positions are unaffected.
const PRELUDE: &str = "local function cat(...) local t = table.pack(...) \
    for i = 1, t.n do t[i] = tostring(t[i]) end return table.concat(t, ' ') end ";

fn run(src: &str) -> Result<String, String> {
    let mut lua = Lua::new();
    lua.load_all();
    let src = format!("{PRELUDE}{src}");
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(&src, Some("=c"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let as_string = |v: Value<'_>| {
        let s = v.get_string().expect("string value");
        String::from_utf8_lossy(s.as_bytes()).into_owned()
    };
    match lua.finish(&ex) {
        Ok(()) => Ok(lua.enter(|ctx| {
            let v = ctx.fetch(&ex).take_result::<Value>(ctx).expect("result");
            as_string(v)
        })),
        Err(RuntimeError::Lua(e)) => Err(lua.enter(|ctx| as_string(ctx.fetch(&e).value()))),
        Err(e) => panic!("unexpected failure for {src:?}: {e:?}"),
    }
}

fn ok(src: &str) -> String {
    run(src).unwrap_or_else(|e| panic!("{src:?} raised {e:?}"))
}

fn err(src: &str) -> String {
    run(src).expect_err(src)
}

#[test]
fn userdata_operators() {
    assert_eq!(
        ok("local mt = getmetatable(io.stdout)
            mt.__add = function() return 'add' end
            mt.__call = function(self, x) return 'call ' .. x end
            mt.__concat = function() return 'cat' end
            mt.__unm = function() return 'unm' end
            mt.__band = function() return 'band' end
            mt.__bnot = function() return 'bnot' end
            local u = io.stdout
            return cat(u + 1, 1 + u, u(5), u .. 'x', 'x' .. u, -u, u & 1, ~u)"),
        "add add call 5 cat cat unm band bnot"
    );
    assert_eq!(
        ok("local mt = getmetatable(io.stdout)
            mt.__lt = function() return true end
            mt.__le = function() return false end
            mt.__eq = function() return true end
            local u, v = io.stdout, io.stderr
            return cat(u < v, u <= v, u == v, u ~= v, u < 1, 2 > u, u == 1)"),
        "true false true false true true false"
    );
}

#[test]
fn per_type_operators() {
    assert_eq!(
        ok("debug.setmetatable(true, {
                __add = function() return 'bool' end,
                __call = function(self, x) return 'called ' .. tostring(self) .. ' ' .. x end,
                __len = function() return 42 end,
            })
            return cat(true + 1, 1 + false, (true)(7), #false)"),
        "bool bool called true 7 42"
    );
    assert_eq!(
        ok("debug.setmetatable(0, {
                __call = function(n, x) return n * x end,
                __len = function(n) return -n end,
                __concat = function() return 'num-cat' end,
            })
            return cat((6)(7), #5, 1 .. {})"),
        "42 -5 num-cat"
    );
    assert_eq!(
        ok("debug.setmetatable(nil, {
                __unm = function() return 'neg-nil' end,
                __bnot = function() return 'bnot-nil' end,
                __lt = function() return true end,
            })
            local x
            return cat(-x, ~x, x < 1, 1 < x)"),
        "neg-nil bnot-nil true true"
    );
}

#[test]
fn string_type_metatable_rules() {
    // `#` and `==` on strings never consult the metatable.
    assert_eq!(
        ok("debug.setmetatable('', {
                __concat = function() return 'cc' end,
                __lt = function() return true end,
                __call = function(s, a) return s .. a end,
                __len = function() return 99 end,
                __eq = function() return true end,
            })
            return cat('a' .. {}, 'a' < {}, ('x')('y'), #'abc', 'a' == 'b')"),
        "cc true xy 3 false"
    );
}

#[test]
fn len_metamethod_gets_operand_twice() {
    assert_eq!(
        ok("local t = setmetatable({}, {__len = function(a, b)
                return select('#', a, b) .. tostring(rawequal(a, b))
            end})
            return #t"),
        "2true"
    );
}

#[test]
fn library_lookups_use_type_metatables() {
    assert_eq!(
        ok("debug.setmetatable(print, {__pairs = function(f)
                return function(_, k) if not k then return 1, 'one' end end, f, nil
            end})
            local out = {}
            for k, v in pairs(print) do out[#out + 1] = k .. '=' .. v end
            return table.concat(out, ',')"),
        "1=one"
    );
    assert_eq!(
        ok(
            "getmetatable(io.stdout).__lt = function(a, b) return tostring(a) < tostring(b) end
            local t = {io.stderr, io.stdout, io.stdin}
            table.sort(t)
            return cat(tostring(t[1]) < tostring(t[2]), tostring(t[2]) < tostring(t[3]))"
        ),
        "true true"
    );
}

#[test]
fn missing_metamethod_errors() {
    assert_eq!(
        err("return #(function() end)"),
        "c:1: attempt to get length of a function value"
    );
    assert_eq!(err("return (1)()"), "c:1: attempt to call a number value");
    assert_eq!(
        err("return true + 1"),
        "c:1: attempt to perform arithmetic on a boolean value"
    );
}
