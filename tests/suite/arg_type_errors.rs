//! "bad argument" type errors (`luaL_typeerror`): the offending value is named
//! by its metatable's `__name` when it has one, and a missing argument is "no
//! value". Expected strings come from `lua` 5.5.1 running the same chunk.

use crate::common::err;

#[test]
fn name_replaces_the_type() {
    let cases = [
        (
            "string.rep(N)",
            "bad argument #1 to 'rep' (string expected, got MyType)",
        ),
        (
            "string.rep('x', N)",
            "bad argument #2 to 'rep' (number expected, got MyType)",
        ),
        (
            "string.format(N)",
            "bad argument #1 to 'format' (string expected, got MyType)",
        ),
        (
            "string.format('%d', N)",
            "bad argument #2 to 'format' (number expected, got MyType)",
        ),
        (
            "math.floor(N)",
            "bad argument #1 to 'floor' (number expected, got MyType)",
        ),
        (
            "table.concat({}, N)",
            "bad argument #2 to 'concat' (string expected, got MyType)",
        ),
        (
            "table.insert(io.stdout, 1)",
            "bad argument #1 to 'insert' (table expected, got FILE*)",
        ),
        (
            "table.sort({1, 2}, N)",
            "bad argument #2 to 'sort' (function expected, got MyType)",
        ),
        (
            "coroutine.wrap(N)",
            "bad argument #1 to 'wrap' (function expected, got MyType)",
        ),
        (
            "coroutine.status(io.stdout)",
            "bad argument #1 to 'status' (thread expected, got FILE*)",
        ),
        (
            "io.open(N)",
            "bad argument #1 to 'open' (string expected, got MyType)",
        ),
        (
            "io.stdout.write(N)",
            "bad argument #1 to 'write' (FILE* expected, got MyType)",
        ),
        (
            "os.getenv(N)",
            "bad argument #1 to 'getenv' (string expected, got MyType)",
        ),
        (
            "utf8.len(N)",
            "bad argument #1 to 'len' (string expected, got MyType)",
        ),
        (
            "tonumber(N, 10)",
            "bad argument #1 to 'tonumber' (string expected, got MyType)",
        ),
        (
            "xpcall(print, N)",
            "bad argument #2 to 'xpcall' (function expected, got MyType)",
        ),
        (
            "warn(N)",
            "bad argument #1 to 'warn' (string expected, got MyType)",
        ),
    ];
    for (call, msg) in cases {
        assert_eq!(
            err(&format!(
                "N = setmetatable({{}}, {{__name = 'MyType'}})\nlocal r = {call}"
            )),
            format!("c:2: {msg}")
        );
    }
}

#[test]
fn every_type_error_says_what_it_got() {
    assert_eq!(
        err("local r = coroutine.create(1)"),
        "c:1: bad argument #1 to 'create' (function expected, got number)"
    );
    assert_eq!(
        err("local r = coroutine.resume(1)"),
        "c:1: bad argument #1 to 'resume' (thread expected, got number)"
    );
    assert_eq!(
        err("local r = coroutine.close(1)"),
        "c:1: bad argument #1 to 'close' (thread expected, got number)"
    );
    assert_eq!(
        err("local r = coroutine.isyieldable(1)"),
        "c:1: bad argument #1 to 'isyieldable' (thread expected, got number)"
    );
    assert_eq!(
        err("local r = rawget(1)"),
        "c:1: bad argument #1 to 'rawget' (table expected, got number)"
    );
    assert_eq!(
        err("local r = rawget()"),
        "c:1: bad argument #1 to 'rawget' (table expected, got no value)"
    );
    assert_eq!(
        err("local r = setmetatable(1)"),
        "c:1: bad argument #1 to 'setmetatable' (table expected, got number)"
    );
    assert_eq!(
        err("local r = setmetatable({}, 1)"),
        "c:1: bad argument #2 to 'setmetatable' (nil or table expected, got number)"
    );
    assert_eq!(
        err("local r = os.difftime()"),
        "c:1: bad argument #1 to 'difftime' (number expected, got no value)"
    );
    assert_eq!(
        err("local r = warn()"),
        "c:1: bad argument #1 to 'warn' (string expected, got no value)"
    );
}
