//! `loadfile` and `dofile`. Expected strings come from `lua` 5.5.1 loading
//! the same files.

use std::path::{Path, PathBuf};

use crate::common::ok;

/// A fresh directory holding `files`, as `(name, contents)`.
fn dir_with(test: &str, files: &[(&str, &[u8])]) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("loadfile")
        .join(test);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (name, contents) in files {
        std::fs::write(dir.join(name), contents).unwrap();
    }
    dir
}

/// Run `src` with `dir` (ending in `/`) bound to the local `dir`.
fn run(dir: &Path, src: &str) -> String {
    ok(&format!(
        "local dir = {:?} {src}",
        format!("{}/", dir.display())
    ))
}

#[test]
fn skips_bom_and_comment_line() {
    let dir = dir_with(
        "skips",
        &[
            (
                "shebang.lua",
                b"#!/usr/bin/env lua\nlocal a = ...\nreturn a, error(\"e\")\n",
            ),
            ("bom.lua", b"\xEF\xBB\xBFreturn 'bom'\n"),
            ("only_comment.lua", b"# nothing else"),
        ],
    );
    // The skipped line still counts, so the error is on line 3.
    assert_eq!(
        run(
            &dir,
            r#"local ok, e = pcall(loadfile(dir .. "shebang.lua"), 7)
               return cat(ok, e:match(":(%d+): e$"))"#
        ),
        "false 3"
    );
    assert_eq!(
        run(&dir, r#"return cat(loadfile(dir .. "bom.lua")())"#),
        "bom"
    );
    assert_eq!(
        run(&dir, r#"return cat(loadfile(dir .. "only_comment.lua")())"#),
        ""
    );
}

#[test]
fn mode_and_env() {
    let dir = dir_with("mode_env", &[("env.lua", b"return x")]);
    assert_eq!(
        run(
            &dir,
            r#"return cat(loadfile(dir .. "env.lua", "t", {x = "E"})())"#
        ),
        "E"
    );
    assert_eq!(
        run(
            &dir,
            r#"return cat(pcall(loadfile, dir .. "env.lua", "b"))"#
        ),
        "false bad argument #2 to 'loadfile' (binary chunks are not supported)"
    );
    assert_eq!(
        run(
            &dir,
            r#"return cat(pcall(loadfile, dir .. "env.lua", "B"))"#
        ),
        "false bad argument #2 to 'loadfile' (invalid mode)"
    );
}

#[test]
fn file_errors() {
    let dir = dir_with("file_errors", &[]);
    let missing = format!("{}/missing.lua", dir.display());
    assert_eq!(
        run(&dir, r#"return cat(loadfile(dir .. "missing.lua"))"#),
        format!("nil cannot open {missing}: No such file or directory")
    );
    // A directory opens but can't be read; Lua reports no `strerror` then.
    assert_eq!(
        run(&dir, "return cat(loadfile(dir))"),
        format!("nil cannot read {}/", dir.display())
    );
    assert_eq!(
        run(&dir, "return cat(pcall(loadfile, {}))"),
        "false bad argument #1 to 'loadfile' (string expected, got table)"
    );
}

#[test]
fn dofile_runs_and_raises() {
    let dir = dir_with(
        "dofile",
        &[
            ("multi.lua", b"return 1, 2, ..."),
            ("err.lua", b"error('in file', 0)"),
        ],
    );
    assert_eq!(
        run(&dir, r#"return cat(dofile(dir .. "multi.lua"))"#),
        "1 2"
    );
    assert_eq!(
        run(&dir, r#"return cat(pcall(dofile, dir .. "err.lua"))"#),
        "false in file"
    );
    let missing = format!("{}/missing.lua", dir.display());
    assert_eq!(
        run(&dir, r#"return cat(pcall(dofile, dir .. "missing.lua"))"#),
        format!("false cannot open {missing}: No such file or directory")
    );
}
