//! `luaL_tolstring` in the library: `tostring` and `string.format`'s `%s`
//! call `__tostring`, and fall back to `__name` for the `name: 0x...` form.
//! Expected strings come from `lua` 5.5.1 running the same chunk; pointers
//! are masked as `PTR`.

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
fn tostring_and_format_call_tostring() {
    assert_eq!(
        ok(
            "local t = setmetatable({}, {__tostring = function() return 'X' end})
            return cat(tostring(t), string.format('%s|%3s|%-4s|%.1s|', t, t, t, t))"
        ),
        "X X|  X|X   |X|"
    );
    assert_eq!(
        ok(
            "local t = setmetatable({}, {__tostring = function() return 1 end})
            return cat(tostring(t), type(tostring(t)))"
        ),
        "1 string"
    );
    assert_eq!(
        ok(
            "local callable = setmetatable({}, {__call = function(self, o) return 'callable' end})
            return tostring(setmetatable({}, {__tostring = callable}))"
        ),
        "callable"
    );
    assert_eq!(
        ok(
            "local t = setmetatable({}, {__tostring = function() return 'a\\0b' end})
            return cat(#tostring(t), #string.format('%s', t))"
        ),
        "3 3"
    );
}

#[test]
fn format_calls_tostring_in_order() {
    assert_eq!(
        ok("local log = {}
            local function mk(n)
                return setmetatable({}, {__tostring = function() log[#log + 1] = n return n end})
            end
            return cat(string.format('%s %d %s', mk('a'), 5, mk('b')), table.concat(log, ','))"),
        "a 5 b a,b"
    );
    assert_eq!(
        err(
            "local t = setmetatable({}, {__tostring = function() return 'X' end})
             local s = string.format('%s %d', t, 'x')"
        ),
        "c:2: bad argument #3 to 'format' (number expected, got string)"
    );
}

#[test]
fn per_type_tostring() {
    assert_eq!(
        ok(
            "debug.setmetatable(nil, {__tostring = function() return 'NIL!' end})
            return cat(tostring(nil), string.format('%s %q', nil, nil))"
        ),
        "NIL! NIL! NIL!"
    );
    assert_eq!(
        ok(
            "debug.setmetatable(true, {__tostring = function(b) return b and 'YES' or 'NO' end})
            return string.format('%q %q %5s', true, false, true)"
        ),
        "YES NO   YES"
    );
}

#[test]
fn name_and_default_forms() {
    assert_eq!(
        ok(
            "local function p(v) return (tostring(v):gsub('0x%x+', 'PTR')) end
            return cat(p(setmetatable({}, {__name = 'MyType'})), p(io.stdout),
                (string.format('%s', io.stdout):gsub('0x%x+', 'PTR')),
                p(setmetatable({}, {__name = 42})))"
        ),
        "MyType: PTR file (PTR) file (PTR) table: PTR"
    );
    // Native and Lua functions print alike.
    assert_eq!(
        ok(
            "local function p(v) return (tostring(v):gsub('0x%x+', 'PTR')) end
            local a = p(print) .. ' ' .. p(function() end) .. ' ' .. p(coroutine.create(print))
            debug.setmetatable(print, {__name = 'Fn'})
            return a .. ' ' .. p(print)"
        ),
        "function: PTR function: PTR thread: PTR Fn: PTR"
    );
}

#[test]
fn tostring_errors() {
    assert_eq!(
        err(
            "local t = setmetatable({}, {__tostring = function() return true end})
             local s = tostring(t)"
        ),
        "c:2: '__tostring' must return a string"
    );
    assert_eq!(
        ok(
            "return cat(pcall(tostring, setmetatable({}, {__tostring = function() return {} end})))"
        ),
        "false '__tostring' must return a string"
    );
    assert_eq!(
        err(
            "local t = setmetatable({}, {__tostring = function() error('boom', 0) end})
             local s = string.format('%s', t)"
        ),
        "boom"
    );
}
