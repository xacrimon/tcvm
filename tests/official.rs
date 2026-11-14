//! The official Lua 5.5.1 test suite (`tests/lua-5.5.1-tests`), one test per
//! file, run as `all.lua` runs them for user tests (`_U`): `_soft`, `_port`
//! and `_nomsg` set, no `T`. Ignored by default; run with
//! `cargo test --test official -- --ignored`.
//!
//! Each file is expected to pass or to fail with an outcome starting with the
//! given prefix, so a fix that moves a file to its next failure shows up as a
//! mismatch, as does a regression.

use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use tcvm::{Executor, LoadError, Lua, RuntimeError};

const SUITE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lua-5.5.1-tests");

const TIMEOUT: Duration = Duration::from_secs(60);

/// What `all.lua` sets up before running a file.
const PRELUDE: &str = "_soft, _port, _nomsg = true, true, true \
    Message = function() end debug = nil";

/// Runs `file` and describes how it ended: `None` when it ran to the end,
/// else its error message, `load: <error>` or `panic: <message>`.
fn run(file: &'static str) -> Option<String> {
    // Files load their helpers and scratch files by relative path. Every test
    // sets the same directory, so running them in parallel is fine.
    std::env::set_current_dir(SUITE).unwrap();
    let (tx, rx) = mpsc::channel();
    // A thread so a hang can be reported; on timeout it is left running until
    // the test process exits.
    let worker = thread::spawn(move || {
        let outcome = run_file(file);
        let _ = tx.send(());
        outcome
    });
    match rx.recv_timeout(TIMEOUT) {
        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {}
        Err(mpsc::RecvTimeoutError::Timeout) => return Some("timeout".into()),
    }
    worker.join().unwrap_or_else(|payload| {
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        Some(format!("panic: {msg}"))
    })
}

fn run_file(file: &str) -> Option<String> {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua.enter(|ctx| -> Result<_, LoadError> {
        let prelude = ctx.load(PRELUDE, Some("=prelude"))?;
        let chunk = ctx.load_file(Path::new(file))?;
        Ok((
            ctx.stash(Executor::start(ctx, prelude, ())),
            ctx.stash(chunk),
        ))
    });
    let (prelude, chunk) = match ex {
        Ok(ex) => ex,
        Err(e) => return Some(format!("load: {e}")),
    };
    lua.finish(&prelude).expect("prelude");
    let ex = lua.enter(|ctx| ctx.stash(Executor::start(ctx, ctx.fetch(&chunk), ())));
    match lua.finish(&ex) {
        Ok(()) => None,
        Err(RuntimeError::Lua(e)) => Some(lua.enter(|ctx| {
            String::from_utf8_lossy(ctx.fetch(&e).message(ctx).as_bytes()).into_owned()
        })),
        Err(e) => Some(format!("{e}")),
    }
}

/// `prefix` is matched against [`run`]'s outcome; `issue` is what tracks it.
fn expect_fail(file: &'static str, prefix: &str, issue: &str) {
    match run(file) {
        None => panic!("{file} now passes (was failing with {prefix:?}, {issue})"),
        Some(out) => assert!(
            out.starts_with(prefix),
            "{file} expected to fail with {prefix:?} ({issue}), got:\n{out}"
        ),
    }
}

fn expect_pass(file: &'static str) {
    if let Some(out) = run(file) {
        panic!("{file} failed:\n{out}");
    }
}

macro_rules! official {
    ($($name:ident => $kind:ident $(($prefix:literal, $issue:literal))?,)*) => {
        $(
            #[test]
            #[ignore]
            fn $name() {
                official!(@$kind stringify!($name) $(, $prefix, $issue)?);
            }
        )*
    };
    (@pass $name:expr) => {
        expect_pass(concat!($name, ".lua"))
    };
    (@fail $name:expr, $prefix:literal, $issue:literal) => {
        expect_fail(concat!($name, ".lua"), $prefix, $issue)
    };
}

// all.lua is the driver, main.lua tests the standalone `lua` binary (and
// returns early under `_port`), heavy.lua exhausts memory on purpose, and
// bwcoercion.lua and tracegc.lua are modules the others require.
official! {
    api => pass,
    attrib => fail("panic: index out of bounds", "untriaged"),
    big => pass,
    bitwise => fail("bitwise.lua:6:", "#226"),
    calls => fail("calls.lua:8:", "#226"),
    closure => fail("closure.lua:239:", "#226"),
    code => pass,
    constructs => fail("constructs.lua:6:", "#226"),
    coroutine => fail("coroutine.lua:6:", "#226"),
    cstack => fail("cstack.lua:5:", "#226"),
    db => fail("load: db.lua:218:", "untriaged"),
    errors => fail("errors.lua:6:", "#226"),
    events => fail("events.lua:6:", "#226"),
    files => fail("files.lua:6:", "#226"),
    gc => fail("gc.lua:6:", "#226"),
    gengc => fail("gengc.lua:6:", "#226"),
    goto => fail("goto.lua:14:", "untriaged"),
    literals => fail("load: Error: syntax error", "untriaged"),
    locals => fail("locals.lua:8:", "#226"),
    math => fail("math.lua:6:", "#226"),
    memerr => pass,
    nextvar => fail("nextvar.lua:422:", "untriaged"),
    pm => fail("pm.lua:425:", "untriaged"),
    sort => fail("sort.lua:22:", "untriaged"),
    strings => fail("load: strings.lua: chunk is not valid UTF-8", "untriaged"),
    tpack => fail("tpack.lua:41:", "untriaged"),
    utf8 => fail("utf8.lua:10:", "#226"),
    vararg => fail("vararg.lua:128:", "untriaged"),
    verybig => pass,
}
