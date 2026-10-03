//! "bad argument" type errors (`luaL_typeerror`): the offending value is named
//! by its metatable's `__name` when it has one, and a missing argument is "no
//! value". Expected strings come from `lua` 5.5.1 running the same chunk.

use crate::common::{err, ok};

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

#[test]
fn missing_arguments_are_checked() {
    // `luaL_checkany`: a missing argument is an error, an explicit nil is not
    // (#198). Function names are short until #186.
    for (src, msg) in [
        ("rawset({})", "bad argument #2 to 'rawset' (value expected)"),
        ("rawget({})", "bad argument #2 to 'rawget' (value expected)"),
        (
            "rawset({}, 1)",
            "bad argument #3 to 'rawset' (value expected)",
        ),
        (
            "rawequal()",
            "bad argument #1 to 'rawequal' (value expected)",
        ),
        (
            "rawequal(1)",
            "bad argument #2 to 'rawequal' (value expected)",
        ),
        ("io.type()", "bad argument #1 to 'type' (value expected)"),
        ("math.type()", "bad argument #1 to 'type' (value expected)"),
        (
            "tonumber()",
            "bad argument #1 to 'tonumber' (value expected)",
        ),
        (
            "coroutine.isyieldable(nil)",
            "bad argument #1 to 'isyieldable' (thread expected, got nil)",
        ),
    ] {
        assert_eq!(err(&format!("local r = {src}")), format!("c:1: {msg}"));
    }
    assert_eq!(
        ok(
            "return cat(select('#', rawset({}, 1, nil)), rawequal(nil, nil), io.type(nil), \
            tonumber(nil), rawget({}, nil), math.type(nil), coroutine.isyieldable(), \
            coroutine.wrap(function() return coroutine.isyieldable() end)())"
        ),
        "1 true nil nil nil nil false true"
    );
}

#[test]
fn close_defaults_to_the_running_coroutine() {
    assert_eq!(err("coroutine.close()"), "c:1: cannot close main thread");
    assert_eq!(
        ok("local log = {} \
            local co = coroutine.create(function() \
              local t <close> = setmetatable({}, {__close = function() log[#log + 1] = 'closed' end}) \
              coroutine.close() \
              log[#log + 1] = 'not reached' \
            end) \
            local ok = coroutine.resume(co) \
            return cat(ok, coroutine.status(co), table.concat(log, ','))"),
        "true dead closed"
    );
}

#[test]
fn value_expected() {
    // Every `luaL_checkany`, and min/max's equivalent `luaL_argcheck`, goes
    // through `util::check_any`. Expected messages from lua 5.5.1.
    for (src, name) in [
        ("assert()", "assert"),
        ("getmetatable()", "getmetatable"),
        ("debug.getmetatable()", "getmetatable"),
        ("ipairs()", "ipairs"),
        ("pairs()", "pairs"),
        ("pcall()", "pcall"),
        ("tostring()", "tostring"),
        ("type()", "type"),
        ("math.min()", "min"),
        ("math.max()", "max"),
    ] {
        assert_eq!(
            err(&format!("local r = {src}")),
            format!("c:1: bad argument #1 to '{name}' (value expected)"),
            "{src}"
        );
    }
}
