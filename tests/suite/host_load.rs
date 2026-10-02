//! What a host sees from `Context::load` and `Context::load_file`.

use std::path::PathBuf;

use tcvm::{Executor, LoadError, Lua};

fn load_err(src: &str) -> String {
    let mut lua = Lua::new();
    lua.enter(|ctx| match ctx.load(src, Some("=c")) {
        Ok(_) => panic!("{src:?} loaded"),
        Err(e) => e.to_string(),
    })
}

#[test]
fn parse_error_is_the_rendered_report() {
    assert_eq!(
        load_err("x = = 1"),
        "Error: expected a statement\n   \
         ╭─[ c:1:5 ]\n   \
         │\n \
         1 │ x = = 1\n   \
         │     ┬  \n   \
         │     ╰── expected a statement but got \"=\"\n\
         ───╯"
    );
}

#[test]
fn compile_error_names_the_chunk() {
    assert_eq!(
        load_err("local x <const> = 1\nx = 2"),
        "c:2: attempt to assign to const variable 'x'"
    );
}

#[test]
fn load_file() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("host_load");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("script.lua"),
        b"#!/usr/bin/env tcvm\nreturn true\n",
    )
    .unwrap();
    std::fs::write(dir.join("latin1.lua"), b"return '\xFF'\n").unwrap();

    let mut lua = Lua::new();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load_file(&dir.join("script.lua"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    assert!(lua.execute::<bool>(&ex).expect("run"));

    let mut err = |name: &str| {
        lua.enter(|ctx| match ctx.load_file(&dir.join(name)) {
            Ok(_) => panic!("{name} loaded"),
            Err(e) => e.to_string(),
        })
    };
    // The chunk id keeps only the tail of a long path.
    let e = err("latin1.lua");
    assert!(e.ends_with("/latin1.lua: chunk is not valid UTF-8"), "{e}");
    assert_eq!(
        err("missing.lua"),
        format!(
            "cannot open {}/missing.lua: No such file or directory",
            dir.display()
        )
    );
}
