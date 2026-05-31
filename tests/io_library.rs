//! `io` library + userdata method-dispatch tests. Each Lua chunk performs
//! file I/O against a unique temp path and returns a boolean verdict; the
//! closed-file case asserts a *raised* error instead.
//!
//! Covers issue #92 (`io.write` returns the file handle so `:write` chains)
//! and the broader `io` subtask of #27 (file handles, read formats, seek,
//! `io.lines`, `io.type`, default input/output, `getmetatable` on userdata).

use tcvm::{Executor, LoadError, Lua, RuntimeError};

/// A temp path unique to this process + `name`, removed before the test so
/// each run starts clean.
fn tmp_path(name: &str) -> String {
    let mut p = std::env::temp_dir();
    p.push(format!("tcvm_io_{}_{}.txt", std::process::id(), name));
    let s = p.to_string_lossy().into_owned();
    let _ = std::fs::remove_file(&s);
    s
}

/// Run `src` and return its boolean result (or the runtime error).
fn run_bool(src: &str) -> Result<bool, RuntimeError> {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("io_test"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute::<bool>(&ex)
}

fn assert_ok(src: &str) {
    assert!(
        run_bool(src).expect("run should not error"),
        "chunk returned false: {src}"
    );
}

#[test]
fn io_write_returns_handle_and_chains() {
    // Issue #92: io.write returns the default-output handle (io.stdout), so
    // chaining works. The "" writes are no-ops on stdout.
    assert_ok("return io.write(\"\") == io.stdout");
    assert_ok("return type(io.write(\"\")) == \"userdata\"");
    assert_ok("io.write(\"\"):write(\"\"); return true");
}

#[test]
fn file_write_chains_and_returns_self() {
    let p = tmp_path("chain");
    assert_ok(&format!(
        "local f = io.open({p:?}, \"w\")\n\
         local same = f:write(\"a\"):write(\"b\", \"c\") == f\n\
         f:close()\n\
         return same"
    ));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn write_read_roundtrip_and_number_subtype() {
    let p = tmp_path("roundtrip");
    assert_ok(&format!(
        "local w = io.open({p:?}, \"w\")\n\
         w:write(\"hello\\n\", \"world\\n\")\n\
         w:write(42, \" \", 3.5, \"\\n\")\n\
         w:close()\n\
         local r = io.open({p:?}, \"r\")\n\
         local l1 = r:read(\"l\")            -- hello (no newline)\n\
         local L2 = r:read(\"L\")            -- world\\n (with newline)\n\
         local n1 = r:read(\"n\")            -- 42 integer\n\
         local n2 = r:read(\"n\")            -- 3.5 float\n\
         local rest = r:read(\"a\")          -- \"\\n\"\n\
         local eof = r:read(\"l\")           -- nil at EOF\n\
         r:close()\n\
         return l1 == \"hello\" and L2 == \"world\\n\"\n\
            and n1 == 42 and math.type(n1) == \"integer\"\n\
            and n2 == 3.5 and math.type(n2) == \"float\"\n\
            and rest == \"\\n\" and eof == nil"
    ));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn read_byte_count_and_zero_probe() {
    let p = tmp_path("bytes");
    assert_ok(&format!(
        "local w = io.open({p:?}, \"w\"); w:write(\"abcdef\"); w:close()\n\
         local r = io.open({p:?}, \"r\")\n\
         local five = r:read(5)             -- abcde\n\
         local probe = r:read(0)            -- \"\" (more to read)\n\
         local one = r:read(1)              -- f\n\
         local at_eof = r:read(0)           -- nil at EOF\n\
         r:close()\n\
         return five == \"abcde\" and probe == \"\" and one == \"f\" and at_eof == nil"
    ));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn multi_format_read_stops_at_eof() {
    let p = tmp_path("multi");
    assert_ok(&format!(
        "local w = io.open({p:?}, \"w\"); w:write(\"only\\n\"); w:close()\n\
         local r = io.open({p:?}, \"r\")\n\
         local a = r:read(\"l\")             -- only\n\
         local b, c = r:read(\"l\", \"l\")    -- nil (single value at EOF)\n\
         r:close()\n\
         return a == \"only\" and b == nil and c == nil"
    ));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn seek_positions() {
    let p = tmp_path("seek");
    assert_ok(&format!(
        "local w = io.open({p:?}, \"w\"); w:write(\"0123456789\"); w:close()\n\
         local s = io.open({p:?}, \"r\")\n\
         local five = s:read(5)             -- 01234\n\
         local cur = s:seek()               -- 5\n\
         local size = s:seek(\"end\")        -- 10\n\
         s:seek(\"set\", 2)\n\
         local from2 = s:read(3)            -- 234\n\
         s:close()\n\
         return five == \"01234\" and cur == 5 and size == 10 and from2 == \"234\""
    ));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn io_type_classifies_handles() {
    let p = tmp_path("type");
    assert_ok(&format!(
        "local f = io.open({p:?}, \"w\")\n\
         local t_open = io.type(f)\n\
         f:close()\n\
         local t_closed = io.type(f)\n\
         return t_open == \"file\" and t_closed == \"closed file\"\n\
            and io.type({{}}) == nil and io.type(io.stdout) == \"file\""
    ));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn io_lines_iterates_and_counts() {
    let p = tmp_path("lines");
    assert_ok(&format!(
        "local w = io.open({p:?}, \"w\"); w:write(\"a\\nb\\nc\\n\"); w:close()\n\
         local count, first = 0, nil\n\
         for line in io.lines({p:?}) do\n\
           count = count + 1\n\
           if count == 1 then first = line end\n\
         end\n\
         return count == 3 and first == \"a\""
    ));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn default_output_redirection() {
    let p = tmp_path("output");
    assert_ok(&format!(
        "io.output({p:?})\n\
         io.write(\"redirected\\n\")\n\
         io.close()                          -- flush+close the redirected file\n\
         io.output(io.stdout)                -- restore so later writes are safe\n\
         local r = io.open({p:?}, \"r\")\n\
         local content = r:read(\"a\")\n\
         r:close()\n\
         return content == \"redirected\\n\""
    ));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn getmetatable_on_file_handle() {
    // getmetatable works on userdata; the file metatable carries __name.
    assert_ok(
        "local mt = getmetatable(io.stdout)\n\
               return type(mt) == \"table\" and mt.__name == \"FILE*\"",
    );
}

#[test]
fn missing_field_is_nil_missing_method_would_call_nil() {
    // Indexing a userdata for an absent key resolves through __index to nil
    // (no error); the method-call error only happens at the call site.
    assert_ok("return io.stdout.nonexistent == nil");
}

#[test]
fn open_failure_returns_nil_msg_errno() {
    assert_ok(
        "local f, msg, code = io.open(\"/no/such/dir/nope\", \"r\")\n\
               return f == nil and type(msg) == \"string\" and type(code) == \"number\"",
    );
}

#[test]
fn closing_standard_file_is_rejected() {
    // Via io.close (free function): the method-call form io.stdout:close()
    // also returns (nil, msg), but a *pre-existing* SELF multi-return bug
    // (method calls truncate to one value) would drop the message — so we
    // assert the two-value shape through the free-function path.
    assert_ok(
        "local ok, msg = io.close(io.stdout)\n\
               return ok == nil and msg == \"cannot close standard file\"",
    );
}

#[test]
fn reading_closed_file_raises() {
    let p = tmp_path("closed");
    // No pcall yet (#27), so the raised error surfaces as a RuntimeError.
    let src = format!(
        "local f = io.open({p:?}, \"w\"); f:close()\n\
         return f:read(\"l\") == nil"
    );
    let res = run_bool(&src);
    assert!(
        res.is_err(),
        "reading a closed file must raise, got {res:?}"
    );
    let _ = std::fs::remove_file(&p);
}
