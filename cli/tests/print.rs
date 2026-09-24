//! `print` output through the CLI, byte for byte as `lua` 5.5.1 prints it.

use std::process::Command;

fn stdout_of(name: &str, src: &str) -> String {
    let path = std::env::temp_dir().join(format!("tcvm_cli_{}_{name}.lua", std::process::id()));
    std::fs::write(&path, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_tcvm-cli"))
        .arg("-f")
        .arg(&path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&path);
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn print_calls_tostring() {
    assert_eq!(
        stdout_of(
            "tostring",
            "local t = setmetatable({}, {__tostring = function() return 'X' end})\n\
             print(t, 1, nil, true, 2.5)\n"
        ),
        "X\t1\tnil\ttrue\t2.5\n"
    );
    assert_eq!(
        stdout_of(
            "nil_type",
            "debug.setmetatable(nil, {__tostring = function() return 'NIL!' end})\nprint(nil)\n"
        ),
        "NIL!\n"
    );
}

#[test]
fn print_writes_each_argument_as_converted() {
    // The `__tostring` of the second argument prints before its tab.
    assert_eq!(
        stdout_of(
            "order",
            "local t = setmetatable({}, {__tostring = function() print('inner') return 'T' end})\n\
             print('a', t, 'b')\n"
        ),
        "ainner\n\tT\tb\n"
    );
    // An error part-way leaves the earlier arguments written.
    assert_eq!(
        stdout_of(
            "error",
            "print(pcall(print, 'a', setmetatable({}, {__tostring = function() error('boom', 0) end}), 'c'))\n"
        ),
        "afalse\tboom\n"
    );
}
