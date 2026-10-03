//! How the CLI reports an uncaught error object, as `lua` 5.5.1's
//! `msghandler` does (minus the traceback and with a `tcvm:` prefix).

use std::process::Command;

fn stderr_of(name: &str, src: &str) -> String {
    let path = std::env::temp_dir().join(format!("tcvm_cli_{}_{name}.lua", std::process::id()));
    std::fs::write(&path, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_tcvm-cli"))
        .arg("-f")
        .arg(&path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(!out.status.success());
    String::from_utf8(out.stderr).unwrap()
}

#[test]
fn error_object_reported_through_tostring() {
    assert_eq!(
        stderr_of(
            "tostring",
            "error(setmetatable({}, {__tostring = function() return 'custom err' end}))\n"
        ),
        "tcvm: custom err\n"
    );
    assert_eq!(
        stderr_of(
            "type_mt",
            "debug.setmetatable(true, {__tostring = function() return 'bool err' end})\nerror(true)\n"
        ),
        "tcvm: bool err\n"
    );
    assert_eq!(
        stderr_of(
            "callable",
            "error(setmetatable({}, {__tostring = setmetatable({}, {__call = function() return 'callable err' end})}))\n"
        ),
        "tcvm: callable err\n"
    );
}

#[test]
fn error_object_without_string_tostring() {
    assert_eq!(
        stderr_of("plain_table", "error({})\n"),
        "tcvm: (error object is a table value)\n"
    );
    // Only a string result counts; a number falls back to the type.
    assert_eq!(
        stderr_of(
            "number_result",
            "error(setmetatable({}, {__tostring = function() return 42 end}))\n"
        ),
        "tcvm: (error object is a table value)\n"
    );
    // An error inside `__tostring` is reported instead.
    assert_eq!(
        stderr_of(
            "raising",
            "error(setmetatable({}, {__tostring = function() error('inner', 0) end}))\n"
        ),
        "tcvm: inner\n"
    );
    // So is the failure to call a non-callable one.
    assert_eq!(
        stderr_of(
            "not_callable",
            "error(setmetatable({}, {__tostring = 5}))\n"
        ),
        "tcvm: attempt to call a number value\n"
    );
}

#[test]
fn nil_error_object() {
    // The handler sees the nil before it becomes "<no error object>" (#197).
    let nil = "tcvm: (error object is a nil value)\n";
    assert_eq!(stderr_of("nil", "error(nil)\n"), nil);
    assert_eq!(stderr_of("none", "error()\n"), nil);
    assert_eq!(
        stderr_of(
            "nil_close",
            "local x <close> = setmetatable({}, {__close = function() end})\nerror(nil)\n"
        ),
        nil
    );
    // A nil caught on the way is substituted, and re-raised with a position.
    for (name, src) in [
        ("caught", "error(select(2, pcall(error)))\n"),
        ("wrap", "coroutine.wrap(function() error(nil) end)()\n"),
    ] {
        let err = stderr_of(name, src);
        assert!(
            err.ends_with(".lua:1: <no error object>\n"),
            "{name}: {err}"
        );
    }
}

#[test]
fn tostring_runs_before_close() {
    // `msghandler` runs before the chunk unwinds, so a `__close` that changes
    // the error object comes too late to affect the message.
    let src = "local e = setmetatable({msg = 'before'}, {__tostring = function(s) return s.msg end})\n\
               local x <close> = setmetatable({}, {__close = function() e.msg = 'after' end})\n\
               error(e)\n";
    assert_eq!(stderr_of("tostring_close", src), "tcvm: before\n");
}
