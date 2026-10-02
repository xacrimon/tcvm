//! Opcode faults raise ordinary Lua errors: the reference message (ldebug.c's
//! `luaG_*error` family) prefixed with the faulting frame's `source:line:`,
//! routed through the normal unwinder so `coroutine.resume` can catch them.
//! Expected strings come from `lua` 5.5.1 via `load(src, "=c")` + `pcall`,
//! minus the `(local 'x')`-style variable attribution, which isn't
//! implemented yet.

use tcvm::{Executor, LoadError, Lua, RuntimeError};

fn raise_str(src: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=c"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    match lua.execute::<()>(&ex) {
        Err(RuntimeError::Lua(stashed)) => lua.enter(|ctx| {
            let v = ctx.fetch(&stashed).value();
            let s = v.get_string().expect("string error value");
            String::from_utf8_lossy(s.as_bytes()).into_owned()
        }),
        other => panic!("expected a Lua error for {src:?}, got {other:?}"),
    }
}

fn run_i64(src: &str) -> i64 {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=c"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
}

#[test]
fn index_and_call() {
    assert_eq!(
        raise_str("local x; return x.y"),
        "c:1: attempt to index a nil value"
    );
    assert_eq!(
        raise_str("local x = 5; return x.y"),
        "c:1: attempt to index a number value"
    );
    assert_eq!(
        raise_str("local x; x.y = 1"),
        "c:1: attempt to index a nil value"
    );
    assert_eq!(
        raise_str("local t = {} return t.a.b"),
        "c:1: attempt to index a nil value"
    );
    assert_eq!(
        raise_str("local x; return x()"),
        "c:1: attempt to call a nil value"
    );
    assert_eq!(
        raise_str("return (1)()"),
        "c:1: attempt to call a number value"
    );
    assert_eq!(
        raise_str("return undefinedfn()"),
        "c:1: attempt to call a nil value"
    );
    assert_eq!(
        raise_str("local t = {}; return t:nope()"),
        "c:1: attempt to call a nil value"
    );
    assert_eq!(
        raise_str("local t = setmetatable({}, {__call = 5}); return t()"),
        "c:1: attempt to call a number value"
    );
    // A non-function `__index`/`__newindex` is indexed, not called.
    assert_eq!(
        raise_str("local t = setmetatable({}, {__index = 5}); return t.x"),
        "c:1: attempt to index a number value"
    );
    assert_eq!(
        raise_str("local t = setmetatable({}, {__index = 5}); return t:m()"),
        "c:1: attempt to index a number value"
    );
    assert_eq!(
        raise_str("local t = setmetatable({}, {__newindex = 5}); t.x = 1"),
        "c:1: attempt to index a number value"
    );
    assert_eq!(
        raise_str("local t = setmetatable({}, {__add = 5}); return t + 1"),
        "c:1: attempt to call a number value"
    );
}

#[test]
fn arithmetic_bitwise_concat() {
    assert_eq!(
        raise_str("return 1 + {}"),
        "c:1: attempt to perform arithmetic on a table value"
    );
    assert_eq!(
        raise_str("return {} + 1"),
        "c:1: attempt to perform arithmetic on a table value"
    );
    assert_eq!(
        raise_str("return -{}"),
        "c:1: attempt to perform arithmetic on a table value"
    );
    assert_eq!(
        raise_str("return 1 | 1.5"),
        "c:1: number has no integer representation"
    );
    assert_eq!(
        raise_str("return 1.5 | 1"),
        "c:1: number has no integer representation"
    );
    assert_eq!(
        raise_str("return 'x' | 1"),
        "c:1: attempt to perform bitwise operation on a string value"
    );
    assert_eq!(
        raise_str("return ~{}"),
        "c:1: attempt to perform bitwise operation on a table value"
    );
    assert_eq!(
        raise_str("return 1 .. {}"),
        "c:1: attempt to concatenate a table value"
    );
    assert_eq!(
        raise_str("return {} .. 1"),
        "c:1: attempt to concatenate a table value"
    );
    assert_eq!(raise_str("return 1 // 0"), "c:1: attempt to divide by zero");
    assert_eq!(raise_str("return 1 % 0"), "c:1: attempt to perform 'n%0'");
}

#[test]
fn comparison_and_length() {
    assert_eq!(
        raise_str("return 1 < 'a'"),
        "c:1: attempt to compare number with string"
    );
    assert_eq!(
        raise_str("return {} < {}"),
        "c:1: attempt to compare two table values"
    );
    assert_eq!(
        raise_str("return {} <= 1"),
        "c:1: attempt to compare table with number"
    );
    assert_eq!(
        raise_str("return #5"),
        "c:1: attempt to get length of a number value"
    );
    assert_eq!(
        raise_str("return #nil"),
        "c:1: attempt to get length of a nil value"
    );
}

#[test]
fn metatable_name_is_used_as_type() {
    assert_eq!(
        raise_str("local t = setmetatable({}, {__name = 'MyType'}); return t + 1"),
        "c:1: attempt to perform arithmetic on a MyType value"
    );
}

#[test]
fn table_keys_and_for_loops() {
    assert_eq!(
        raise_str("local t = {} t[nil] = 1"),
        "c:1: table index is nil"
    );
    assert_eq!(
        raise_str("local t = {} t[0/0] = 1"),
        "c:1: table index is NaN"
    );
    assert_eq!(raise_str("rawset({}, nil, 1)"), "table index is nil");
    assert_eq!(
        raise_str("for i = 1, 10, 0 do end"),
        "c:1: 'for' step is zero"
    );
    assert_eq!(
        raise_str("for i = 1.0, 10, 0 do end"),
        "c:1: 'for' step is zero"
    );
    assert_eq!(
        raise_str("for i = 'a', 10 do end"),
        "c:1: bad 'for' initial value (number expected, got string)"
    );
    assert_eq!(
        raise_str("for i = 1, {} do end"),
        "c:1: bad 'for' limit (number expected, got table)"
    );
    // Numeric strings are coerced, and the loop then runs on floats.
    assert_eq!(
        run_i64(
            "local n = 0 for i = '1', 2 do n = n + (math.type(i) == 'float' and 1 or 0) end return n"
        ),
        2
    );
}

#[test]
fn faulting_line_is_reported() {
    assert_eq!(
        raise_str("local x\n\nlocal y = x\n  .z"),
        "c:4: attempt to index a nil value"
    );
    assert_eq!(
        raise_str("local function f()\n  return nil + 1\nend\nf()"),
        "c:2: attempt to perform arithmetic on a nil value"
    );
}

#[test]
fn opcode_errors_are_catchable() {
    assert_eq!(
        run_i64(
            "local co = coroutine.create(function()\n\
               local x\n\
               return x.y\n\
             end)\n\
             local ok, e = coroutine.resume(co)\n\
             return (not ok and e == 'c:3: attempt to index a nil value') and 1 or 0"
        ),
        1
    );
}
