//! Script arguments reach the main chunk both as the global `arg` and as
//! its varargs, as `lua` 5.5.1's `createargtable` and `pushargs` pass them.

use std::process::Command;

fn stdout_of(name: &str, src: &str, args: &[&str]) -> String {
    let path = std::env::temp_dir().join(format!("tcvm_cli_{}_{name}.lua", std::process::id()));
    std::fs::write(&path, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_tcvm-cli"))
        .arg("-f")
        .arg(&path)
        .args(args)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

const SRC: &str = "print(select('#', ...), ...)\nprint(#arg, arg[1], arg[2], arg[0] ~= nil)\n";

#[test]
fn args_are_chunk_varargs() {
    assert_eq!(
        stdout_of("varargs", SRC, &["x", "y"]),
        "2\tx\ty\n2\tx\ty\ttrue\n"
    );
}

#[test]
fn no_args_no_varargs() {
    assert_eq!(stdout_of("none", SRC, &[]), "0\n0\tnil\tnil\ttrue\n");
}
