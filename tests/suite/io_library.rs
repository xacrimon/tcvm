//! `io` library + userdata method-dispatch tests. Each Lua chunk performs
//! file I/O against a unique temp path and returns a boolean verdict or
//! `assert`s each check; the closed-file case asserts a *raised* error
//! instead.
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

/// Run `src`, panicking with the message of any error it raises, so a failed
/// `assert(cond, "what")` in the chunk names its check.
fn assert_runs(src: &str) {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("io_test"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    match lua.execute::<()>(&ex) {
        Ok(()) => {}
        Err(RuntimeError::Lua(err)) => {
            let msg = lua.enter(|ctx| {
                String::from_utf8_lossy(ctx.fetch(&err).message(ctx).as_bytes()).into_owned()
            });
            panic!("{msg}");
        }
        Err(e) => panic!("{e}"),
    }
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
fn read_number_length_limit() {
    // liolib reads at most 200 characters of a numeral (#254); a longer one
    // fails and leaves the rest unread. Expected values from lua 5.5.1.
    let p = tmp_path("numlimit");
    assert_ok(&format!(
        "local function try(content)\n\
           local f = io.open({p:?}, \"w\") f:write(content) f:close()\n\
           f = io.open({p:?}, \"r\")\n\
           local n = f:read(\"n\") local rest = f:read(\"a\")\n\
           f:close()\n\
           return tostring(n) .. \" \" .. #rest\n\
         end\n\
         return try(\"1234\" .. (\"0\"):rep(1000) .. \"\\n\") == \"nil 805\"\n\
           and try((\"1\"):rep(200) .. \"x\") == \"1.1111111111111111e+199 1\"\n\
           and try((\"1\"):rep(201) .. \"x\") == \"nil 2\"\n\
           and try(\"-\" .. (\"1\"):rep(200) .. \" y\") == \"nil 3\"\n\
           and try(\"0x\" .. (\"f\"):rep(198) .. \"z\") == \"-1 1\"\n\
           and try(\"1.\" .. (\"5\"):rep(197) .. \"e5 z\") == \"nil 3\""
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
fn file_close_and_gc_metamethods() {
    // #251: a file is a to-be-closed value, and `io.lines(name)` hands the
    // file to the generic for as its closing value.
    let p = tmp_path("tbc");
    assert_runs(&format!(
        "local F\n\
         do local f <close> = assert(io.open({p:?}, \"w\")); F = f end\n\
         assert(io.type(F) == \"closed file\", \"<close> closes\")\n\
         local mt = getmetatable(F)\n\
         assert(mt.__gc == mt.__close, \"shared __gc/__close\")\n\
         assert(not pcall(mt.__gc), \"__gc checks its argument\")\n\
         assert(select('#', mt.__close(io.stdout)) == 0, \"__close returns nothing\")\n\
         assert(io.type(io.stdout) == \"file\", \"stdout stays open\")\n\
         local w = io.open({p:?}, \"w\"); w:write(\"a\\nb\\n\"); w:close()\n\
         assert(select('#', io.lines({p:?})) == 4, \"io.lines(name) returns 4\")\n\
         local it, s, c, h = io.lines({p:?})\n\
         assert(s == nil and c == nil, \"nil state and control\")\n\
         for _ in it, s, c, h do break end\n\
         assert(io.type(h) == \"closed file\", \"break closes\")\n\
         local it2, _, _, h2 = io.lines({p:?})\n\
         assert(not pcall(function()\n\
           for _ in it2, nil, nil, h2 do error(\"x\") end\n\
         end))\n\
         assert(io.type(h2) == \"closed file\", \"error closes\")\n\
         assert(select('#', io.lines()) == 1, \"io.lines() returns 1\")"
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

#[test]
fn closed_file_messages() {
    // #252: liolib's distinct messages for a finished lines iterator and a
    // closed default input/output.
    let p = tmp_path("closed_msgs");
    assert_runs(&format!(
        "local w = io.open({p:?}, \"w\"); w:write(\"a\\n\"); w:close()\n\
         local function err(f, ...) return select(2, pcall(f, ...)) end\n\
         local it = io.lines({p:?}); for _ in it do end\n\
         local done = err(it)\n\
         io.input({p:?}); io.close(io.input())\n\
         local inp, lines = err(io.read), err(io.lines)\n\
         io.input(io.stdin)\n\
         io.output({p:?}); io.close(io.output())\n\
         local out, flush = err(io.write, \"x\"), err(io.flush)\n\
         io.output(io.stdout)\n\
         assert(done == \"file is already closed\", done)\n\
         assert(inp == \"default input file is closed\", inp)\n\
         assert(lines == \"attempt to use a closed file\", lines)\n\
         assert(out == \"default output file is closed\", out)\n\
         assert(flush == \"default output file is closed\", flush)"
    ));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn lines_format_cap() {
    // #252: at most 250 formats; the 251st is blamed as argument #252.
    assert_runs(
        "local t = {} for i = 1, 251 do t[i] = \"l\" end\n\
         assert(pcall(io.stdin.lines, io.stdin, table.unpack(t, 1, 250)), \"250 fit\")\n\
         local ok, msg = pcall(io.stdin.lines, io.stdin, table.unpack(t))\n\
         assert(not ok, \"251 raise\")\n\
         assert(msg:find(\"#252\", 1, true), msg)\n\
         assert(msg:find(\"too many arguments\", 1, true), msg)",
    );
}
