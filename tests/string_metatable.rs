//! The string metatable: `__index = string` for method calls and the
//! arithmetic metamethods that coerce numeric strings (lstrlib.c). Expected
//! strings come from `lua` 5.5.1 running the same chunk.

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
fn methods_resolve_through_string_table() {
    assert_eq!(
        ok("local s = 'hello'
            return cat(('hi'):upper(), s:sub(2, 3), s:len(), s:rep(2, ','), #s)"),
        "HI el 5 hello,hello 5"
    );
    assert_eq!(
        ok("return cat(('x').len == string.len, ('x').nope, ('x')[1],
                getmetatable('').__index == string, rawequal(getmetatable('a'), getmetatable('b')))"),
        "true nil nil true true"
    );
    assert_eq!(
        ok("local out = {}
            for w in ('a b c'):gmatch('%a') do out[#out + 1] = w end
            return cat(('%d-%s'):format(5, 'x'), ('a,b'):gsub(',', ';'), table.concat(out))"),
        "5-x a;b abc"
    );
    assert_eq!(ok("string.extra = 42 return cat(('a').extra)"), "42");
}

#[test]
fn metatable_contents_and_replacement() {
    assert_eq!(
        ok("local ks = {}
            for k in pairs(getmetatable('')) do ks[#ks + 1] = k end
            table.sort(ks)
            return table.concat(ks, ' ')"),
        "__add __div __idiv __index __mod __mul __pow __sub __unm"
    );
    assert_eq!(
        ok(
            "getmetatable('').__index = function(s, k) return s .. ':' .. k end
            return ('x').foo"
        ),
        "x:foo"
    );
    assert_eq!(
        err("local function s() return 'x' end
             getmetatable('').__index = nil
             return s():upper()"),
        "c:3: attempt to index a string value"
    );
    assert_eq!(
        err("local function s() return 'x' end s().y = 1"),
        "c:1: attempt to index a string value"
    );
}

#[test]
fn arithmetic_coerces_numeric_strings() {
    assert_eq!(
        ok(
            "return cat('10' + 1, '3' * '4', -'2', '10' // 3, '0x10' + 0, '1e1' + 0,
                '7' % 4, '2' ^ 2, '9' / 3)"
        ),
        "11 12 -2 3 16 10.0 3 4.0 3.0"
    );
    assert_eq!(
        ok("return cat(' 0x10 ' + 0, '1e2' * 1, '  5  ' - 1, '5.' + 0, '.5' + 0, '0x1p4' + 0)"),
        "16 100.0 4 5.0 0.5 16.0"
    );
    assert_eq!(
        ok(
            "return cat(math.type('10' + 1), math.type('10' + 1.0), math.type('10.0' + 1),
                '9223372036854775807' + 1, '9223372036854775808' + 0)"
        ),
        "integer float float -9223372036854775808 9.2233720368547758e+18"
    );
    assert_eq!(
        ok("return cat(-'2', -'2.0', -'0', -'-0.0', math.type(-'0'), '7' % 0.0)"),
        "-2 -2.0 0 0.0 integer nan"
    );
    assert_eq!(
        ok(
            "return cat('2' ^ '0.5', '7' / '2', '-7' // '2', '-7' % '2', '7.5' % '-2', 'a' .. 1 + '2')"
        ),
        "1.4142135623730951 3.5 -4 1 -0.5 a3"
    );
}

#[test]
fn arithmetic_errors() {
    assert_eq!(
        err("return 'abc' + 1"),
        "c:1: attempt to add a 'string' with a 'number'"
    );
    assert_eq!(
        err("return 1 + 'abc'"),
        "c:1: attempt to add a 'number' with a 'string'"
    );
    assert_eq!(
        err("return -'abc'"),
        "c:1: attempt to unm a 'string' with a 'string'"
    );
    assert_eq!(
        err("return {} + '10'"),
        "c:1: attempt to add a 'table' with a 'string'"
    );
    assert_eq!(
        err("return '10' + true"),
        "c:1: attempt to add a 'string' with a 'boolean'"
    );
    // Raised inside the metamethod, so without a position.
    assert_eq!(err("return '7' % '0'"), "attempt to perform 'n%0'");
    assert_eq!(err("return '7' // 0"), "attempt to divide by zero");
    // Bitwise operators never coerce strings.
    assert_eq!(
        err("local function s() return '3' end return s() & 1"),
        "c:1: attempt to perform bitwise operation on a string value"
    );
    // Without the metamethod, strings don't coerce at all.
    assert_eq!(
        err("local function s() return '10' end
             getmetatable('').__add = nil
             return s() + 1"),
        "c:3: attempt to perform arithmetic on a string value"
    );
}

#[test]
fn falls_back_to_the_other_operands_metamethod() {
    assert_eq!(
        ok("return '1' + setmetatable({}, {__add = function(a, b) return 'second' end})"),
        "second"
    );
    assert_eq!(
        ok("return 'x' + setmetatable({}, {__add = function(a, b) return type(a) .. type(b) end})"),
        "stringtable"
    );
    assert_eq!(
        ok("return setmetatable({}, {__add = function() return 'tbl-first' end}) + 'x'"),
        "tbl-first"
    );
    assert_eq!(
        ok(
            "local t = setmetatable({}, {__add = function() return 'fwd', 'extra' end})
            return cat(pcall(getmetatable('').__add, 'x', t))"
        ),
        "true fwd"
    );
    assert_eq!(
        ok("local callable = setmetatable({}, {__call = function(self, a, b) return 'via-call' end})
            return 'x' + setmetatable({}, {__add = callable})"),
        "via-call"
    );
}

#[test]
fn direct_calls_follow_lstrlib_stack_quirks() {
    assert_eq!(
        ok("local mt = getmetatable('')
            return cat(mt.__add('1', 2), mt.__add('1'), mt.__unm('3'), mt.__unm('3', '4'))"),
        "3 2 -3 -4"
    );
}
