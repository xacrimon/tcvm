//! The `io` library: file handles as userdata, the predefined
//! `stdin`/`stdout`/`stderr` streams, and the default input/output that
//! the free `io.read`/`io.write`/`io.lines` functions operate on.
//!
//! A file handle is a `Userdata` whose payload is a [`LuaFile`] and whose
//! metatable is the single shared file metatable (`__index` → the methods
//! table, plus `__name`/`__tostring`). Method dispatch (`f:write(...)`)
//! reaches the methods through that metatable's `__index`, which the VM
//! resolves for userdata receivers. The metatable, the methods table, and
//! the current default input/output handles live in an internal "io-state"
//! table captured as upvalue 0 by every `io` native.
//!
//! The OS file descriptor is owned by the `std::fs::File` inside the
//! handle; it is released either by an explicit `close`/`io.close` (which
//! drops the stream) or by the GC dropping the userdata (the collector
//! runs drop glue). Standard streams hold no owned descriptor, so dropping
//! a `stdin`/`stdout`/`stderr` handle never closes fd 0/1/2.

use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};

use crate::Context;
use crate::builtin::util;
use crate::dmm::Gc;
use crate::env::{
    Error, Function, LuaString, NativeContext, NativeFn, Stack, Table, Userdata, Value,
};
use crate::vm::sequence::CallbackAction;

// ---------------------------------------------------------------------------
// File handle payload (plain std types, lives behind `Box<dyn Any>`)
// ---------------------------------------------------------------------------

struct LuaFile {
    state: RefCell<FileState>,
}

enum FileState {
    Open {
        stream: Stream,
        readable: bool,
        writable: bool,
    },
    Closed,
}

/// The underlying byte stream. `Stdin`/`Stdout`/`Stderr` are markers — the
/// process-global handles are locked per call, never owned here, so the
/// real fds survive handle collection.
enum Stream {
    File(BufReader<File>),
    Stdin,
    Stdout,
    Stderr,
}

impl LuaFile {
    fn open(stream: Stream, readable: bool, writable: bool) -> Self {
        LuaFile {
            state: RefCell::new(FileState::Open {
                stream,
                readable,
                writable,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Read formats
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum ReadFmt {
    /// `"l"` — a line without its end-of-line, or `"L"` keeping it.
    Line { keep_eol: bool },
    /// `"a"` — the rest of the file (empty string at EOF, never fails).
    All,
    /// `"n"` — a numeral, preserving integer/float subtype.
    Number,
    /// numeric `n` — up to `n` bytes (`0` probes for EOF).
    Bytes(usize),
}

/// One format's outcome, as owned bytes/number so the generic reader stays
/// free of `'gc`. `Nil` is EOF / format failure — it stops a multi-format
/// read, matching PUC-Lua's break-on-first-failure.
enum ReadOne {
    Nil,
    Bytes(Vec<u8>),
    Int(i64),
    Float(f64),
}

// ---------------------------------------------------------------------------
// Library setup
// ---------------------------------------------------------------------------

pub fn load<'gc>(ctx: Context<'gc>) {
    // io-state: internal table never exposed to Lua. Captured as upvalue 0
    // by every io native; holds the file metatable and the default
    // input/output handles.
    let io_state = Table::new(ctx);
    let upv = || Box::new([Value::table(io_state)]) as Box<[Value<'gc>]>;
    let native = |f: NativeFn| Function::new_native(ctx.mutation(), f, upv());

    // Methods table (the metatable's `__index`).
    let methods = Table::new(ctx);
    let method_fns: &[(&str, NativeFn)] = &[
        ("close", lua_file_close),
        ("flush", lua_file_flush),
        ("lines", lua_file_lines),
        ("read", lua_file_read),
        ("seek", lua_file_seek),
        ("setvbuf", lua_file_setvbuf),
        ("write", lua_file_write),
    ];
    for &(name, f) in method_fns {
        methods.raw_set(
            ctx,
            str_val(ctx, name.as_bytes()),
            Value::function(native(f)),
        );
    }

    // Shared file metatable.
    let mt = Table::new(ctx);
    mt.raw_set(ctx, str_val(ctx, b"__index"), Value::table(methods));
    mt.raw_set(ctx, str_val(ctx, b"__name"), str_val(ctx, b"FILE*"));
    mt.raw_set(
        ctx,
        str_val(ctx, b"__tostring"),
        Value::function(native(lua_file_tostring)),
    );
    io_state.raw_set(ctx, str_val(ctx, b"mt"), Value::table(mt));

    // Predefined handles.
    let stdin = new_handle(ctx, mt, LuaFile::open(Stream::Stdin, true, false));
    let stdout = new_handle(ctx, mt, LuaFile::open(Stream::Stdout, false, true));
    let stderr = new_handle(ctx, mt, LuaFile::open(Stream::Stderr, false, true));
    io_state.raw_set(ctx, str_val(ctx, b"input"), Value::userdata(stdin));
    io_state.raw_set(ctx, str_val(ctx, b"output"), Value::userdata(stdout));

    // The public `io` table.
    let lib = Table::new(ctx);
    let free_fns: &[(&str, NativeFn)] = &[
        ("close", lua_close),
        ("flush", lua_flush),
        ("input", lua_input),
        ("lines", lua_lines),
        ("open", lua_open),
        ("output", lua_output),
        ("popen", lua_popen),
        ("read", lua_read),
        ("tmpfile", lua_tmpfile),
        ("type", lua_type),
        ("write", lua_write),
    ];
    for &(name, f) in free_fns {
        lib.raw_set(
            ctx,
            str_val(ctx, name.as_bytes()),
            Value::function(native(f)),
        );
    }
    lib.raw_set(ctx, str_val(ctx, b"stdin"), Value::userdata(stdin));
    lib.raw_set(ctx, str_val(ctx, b"stdout"), Value::userdata(stdout));
    lib.raw_set(ctx, str_val(ctx, b"stderr"), Value::userdata(stderr));

    ctx.globals()
        .raw_set(ctx, str_val(ctx, b"io"), Value::table(lib));
}

#[inline]
fn str_val<'gc>(ctx: Context<'gc>, s: &[u8]) -> Value<'gc> {
    Value::string(LuaString::new(ctx, s))
}

fn new_handle<'gc>(ctx: Context<'gc>, mt: Table<'gc>, file: LuaFile) -> Userdata<'gc> {
    let u = Userdata::new(ctx.mutation(), file, 0);
    u.set_metatable(ctx.mutation(), Some(mt));
    u
}

/// The io-state table (upvalue 0 of every io native).
#[inline]
fn io_state<'gc>(nctx: &NativeContext<'gc, '_>) -> Table<'gc> {
    nctx.upvalues[0]
        .get_table()
        .expect("io native upvalue 0 must be the io-state table")
}

#[inline]
fn state_get<'gc>(ctx: Context<'gc>, st: Table<'gc>, key: &[u8]) -> Value<'gc> {
    st.raw_get(str_val(ctx, key))
}

/// The shared file metatable held in io-state.
#[inline]
fn file_metatable<'gc>(nctx: &NativeContext<'gc, '_>) -> Table<'gc> {
    state_get(nctx.ctx, io_state(nctx), b"mt")
        .get_table()
        .expect("io-state must hold the file metatable")
}

/// `v` as a file handle iff it is userdata carrying the file metatable
/// (`luaL_testudata` by metatable identity).
fn as_file<'gc>(nctx: &NativeContext<'gc, '_>, v: Value<'gc>) -> Option<Userdata<'gc>> {
    let u = v.get_userdata()?;
    let umt = u.metatable()?;
    let fmt = file_metatable(nctx);
    (Gc::as_ptr(umt.inner()) == Gc::as_ptr(fmt.inner())).then_some(u)
}

fn check_file<'gc>(
    nctx: &NativeContext<'gc, '_>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<Userdata<'gc>, Error<'gc>> {
    as_file(nctx, v).ok_or_else(|| {
        Error::from_str(
            nctx.ctx,
            &format!(
                "bad argument #{n} to '{fname}' (FILE* expected, got {})",
                v.type_name()
            ),
        )
    })
}

fn closed_file_error<'gc>(ctx: Context<'gc>) -> Error<'gc> {
    Error::from_str(ctx, "attempt to use a closed file")
}

// ---------------------------------------------------------------------------
// Low-level I/O on a `FileState` (no `'gc`)
// ---------------------------------------------------------------------------

enum WriteOutcome {
    Ok,
    Closed,
    Io(std::io::Error),
}

fn write_bytes(fs: &mut FileState, buf: &[u8]) -> WriteOutcome {
    let (stream, writable) = match fs {
        FileState::Closed => return WriteOutcome::Closed,
        FileState::Open {
            stream, writable, ..
        } => (stream, *writable),
    };
    if !writable {
        // Mirror the OS EBADF a write to a non-writable handle would hit.
        return WriteOutcome::Io(std::io::Error::from_raw_os_error(9));
    }
    let res = match stream {
        Stream::File(br) => br.get_mut().write_all(buf),
        Stream::Stdout => std::io::stdout().write_all(buf),
        Stream::Stderr => std::io::stderr().write_all(buf),
        Stream::Stdin => Err(std::io::Error::from_raw_os_error(9)),
    };
    match res {
        Ok(()) => WriteOutcome::Ok,
        Err(e) => WriteOutcome::Io(e),
    }
}

fn flush_stream(fs: &mut FileState) -> WriteOutcome {
    let stream = match fs {
        FileState::Closed => return WriteOutcome::Closed,
        FileState::Open { stream, .. } => stream,
    };
    let res = match stream {
        Stream::File(br) => br.get_mut().flush(),
        Stream::Stdout => std::io::stdout().flush(),
        Stream::Stderr => std::io::stderr().flush(),
        Stream::Stdin => Ok(()),
    };
    match res {
        Ok(()) => WriteOutcome::Ok,
        Err(e) => WriteOutcome::Io(e),
    }
}

enum SeekOutcome {
    Pos(u64),
    Closed,
    Io(std::io::Error),
}

fn seek_stream(fs: &mut FileState, pos: SeekFrom) -> SeekOutcome {
    let stream = match fs {
        FileState::Closed => return SeekOutcome::Closed,
        FileState::Open { stream, .. } => stream,
    };
    match stream {
        Stream::File(br) => match br.seek(pos) {
            Ok(n) => SeekOutcome::Pos(n),
            Err(e) => SeekOutcome::Io(e),
        },
        // ESPIPE — standard streams aren't seekable in this model.
        _ => SeekOutcome::Io(std::io::Error::from_raw_os_error(29)),
    }
}

/// Read each format in order. EOF / failure on a format yields `Nil` and
/// stops (no further formats are read), matching PUC-Lua.
fn read_formats<R: BufRead>(r: &mut R, fmts: &[ReadFmt]) -> std::io::Result<Vec<ReadOne>> {
    let mut out = Vec::with_capacity(fmts.len());
    for &fmt in fmts {
        let one = read_one(r, fmt)?;
        let stop = matches!(one, ReadOne::Nil);
        out.push(one);
        if stop {
            break;
        }
    }
    Ok(out)
}

fn read_one<R: BufRead>(r: &mut R, fmt: ReadFmt) -> std::io::Result<ReadOne> {
    match fmt {
        ReadFmt::Line { keep_eol } => {
            let mut buf = Vec::new();
            let n = r.read_until(b'\n', &mut buf)?;
            if n == 0 {
                return Ok(ReadOne::Nil);
            }
            if !keep_eol && buf.last() == Some(&b'\n') {
                buf.pop();
            }
            Ok(ReadOne::Bytes(buf))
        }
        ReadFmt::All => {
            let mut buf = Vec::new();
            r.read_to_end(&mut buf)?;
            Ok(ReadOne::Bytes(buf))
        }
        ReadFmt::Bytes(0) => {
            // `read(0)`: "" if more input remains, nil at EOF.
            if r.fill_buf()?.is_empty() {
                Ok(ReadOne::Nil)
            } else {
                Ok(ReadOne::Bytes(Vec::new()))
            }
        }
        ReadFmt::Bytes(n) => {
            // Grow the buffer as bytes actually arrive rather than allocating
            // `n` up front: a caller-supplied `read(1<<60)` must surface as a
            // catchable error, not an eager multi-exabyte allocation abort.
            let mut buf = Vec::new();
            r.take(n as u64).read_to_end(&mut buf)?;
            if buf.is_empty() {
                Ok(ReadOne::Nil)
            } else {
                Ok(ReadOne::Bytes(buf))
            }
        }
        ReadFmt::Number => read_number(r),
    }
}

/// Read a numeral, preserving integer vs float subtype. Skips leading
/// whitespace, then consumes a maximal numeric token (decimal or `0x`
/// hex, with optional sign / fraction / exponent) via peek-and-consume,
/// and parses it with the shared `str_to_number`. A token that doesn't
/// parse, or no token at all, yields `Nil`.
fn read_number<R: BufRead>(r: &mut R) -> std::io::Result<ReadOne> {
    loop {
        // Copy the whitespace check out before `consume` so the `fill_buf`
        // borrow doesn't overlap the mutable `consume`.
        let is_ws = matches!(r.fill_buf()?.first(), Some(b) if b.is_ascii_whitespace());
        if is_ws {
            r.consume(1);
        } else {
            break;
        }
    }
    let mut tok: Vec<u8> = Vec::new();
    let mut seen_hex = false;
    loop {
        let b = match r.fill_buf()?.first() {
            Some(&b) => b,
            None => break,
        };
        let accept = if b.is_ascii_digit() || b == b'.' {
            true
        } else if b == b'+' || b == b'-' {
            // A sign starts the token or follows the exponent marker — which is
            // `p`/`P` in hex (where `e`/`E` are mantissa digits) and `e`/`E` in
            // decimal.
            let exp_mark = if seen_hex { (b'p', b'P') } else { (b'e', b'E') };
            tok.is_empty() || matches!(tok.last(), Some(&c) if c == exp_mark.0 || c == exp_mark.1)
        } else if !seen_hex && (b == b'x' || b == b'X') {
            matches!(tok.as_slice(), b"0" | b"-0" | b"+0")
        } else if seen_hex && b.is_ascii_hexdigit() {
            true
        } else if seen_hex && (b == b'p' || b == b'P') {
            true
        } else {
            !seen_hex && (b == b'e' || b == b'E')
        };
        if !accept {
            break;
        }
        if (b == b'x' || b == b'X') && !seen_hex {
            seen_hex = true;
        }
        tok.push(b);
        r.consume(1);
    }
    Ok(match util::str_to_number(&tok) {
        Some(v) if v.get_integer().is_some() => ReadOne::Int(v.get_integer().unwrap()),
        Some(v) => ReadOne::Float(v.get_float().unwrap()),
        None => ReadOne::Nil,
    })
}

// ---------------------------------------------------------------------------
// Mid-level helpers shared by free functions and methods
// ---------------------------------------------------------------------------

/// Serialize `vals` (strings and numbers only) and write them to `u`.
/// Returns the write outcome; a non-string/number arg is a (raised) Lua
/// error. `fname`/`first_arg` shape the bad-argument index.
fn do_write<'gc>(
    nctx: &NativeContext<'gc, '_>,
    u: Userdata<'gc>,
    vals: &[Value<'gc>],
    fname: &str,
    first_arg: usize,
) -> Result<WriteOutcome, Error<'gc>> {
    let mut buf = Vec::new();
    for (i, v) in vals.iter().enumerate() {
        if let Some(s) = v.get_string() {
            buf.extend_from_slice(s.as_bytes());
        } else if let Some(n) = v.get_integer() {
            util::push_int(&mut buf, n);
        } else if let Some(f) = v.get_float() {
            util::push_float(&mut buf, f);
        } else {
            return Err(Error::from_str(
                nctx.ctx,
                &format!(
                    "bad argument #{} to '{fname}' (string expected, got {})",
                    first_arg + i,
                    v.type_name()
                ),
            ));
        }
    }
    Ok(
        u.with_data::<LuaFile, _>(|lf| write_bytes(&mut lf.state.borrow_mut(), &buf))
            .expect("file handle must carry a LuaFile payload"),
    )
}

/// Read `fmts` from `u`, returning the resulting Lua values (with the
/// break-on-failure semantics of `read_formats`). A closed file is a
/// raised error; a read I/O error surfaces as a trailing `nil` value.
fn do_read<'gc>(
    ctx: Context<'gc>,
    u: Userdata<'gc>,
    fmts: &[ReadFmt],
) -> Result<Vec<Value<'gc>>, Error<'gc>> {
    enum Dispatch {
        Closed,
        NotReadable,
        Done(std::io::Result<Vec<ReadOne>>),
    }
    let dispatch = u
        .with_data::<LuaFile, _>(|lf| {
            let mut fs = lf.state.borrow_mut();
            match &mut *fs {
                FileState::Closed => Dispatch::Closed,
                FileState::Open {
                    readable: false, ..
                } => Dispatch::NotReadable,
                FileState::Open { stream, .. } => match stream {
                    Stream::File(br) => Dispatch::Done(read_formats(br, fmts)),
                    Stream::Stdin => {
                        let stdin = std::io::stdin();
                        Dispatch::Done(read_formats(&mut stdin.lock(), fmts))
                    }
                    Stream::Stdout | Stream::Stderr => Dispatch::NotReadable,
                },
            }
        })
        .expect("file handle must carry a LuaFile payload");

    let raw = match dispatch {
        Dispatch::Closed => return Err(closed_file_error(ctx)),
        // A non-readable stream reads as immediate EOF.
        Dispatch::NotReadable => return Ok(vec![Value::nil()]),
        // A genuine read error degrades to a fail value, matching Lua's read.
        Dispatch::Done(Err(_)) => return Ok(vec![Value::nil()]),
        Dispatch::Done(Ok(v)) => v,
    };
    Ok(raw
        .into_iter()
        .map(|r| match r {
            ReadOne::Nil => Value::nil(),
            ReadOne::Bytes(b) => Value::string(LuaString::new(ctx, &b)),
            ReadOne::Int(i) => Value::integer(i),
            ReadOne::Float(f) => Value::float(f),
        })
        .collect())
}

/// One read format from a Lua value: a string spec (`"l"`,`"L"`,`"n"`,`"a"`,
/// with an optional leading `*` for 5.1 compatibility) or a byte count.
fn parse_format<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<ReadFmt, Error<'gc>> {
    if let Some(i) = util::to_integer(v) {
        if i < 0 {
            return Err(Error::from_str(
                ctx,
                &format!("bad argument #{n} to '{fname}' (invalid format)"),
            ));
        }
        return Ok(ReadFmt::Bytes(i as usize));
    }
    if let Some(s) = v.get_string() {
        let b = s.as_bytes();
        let spec = b.strip_prefix(b"*").unwrap_or(b);
        match spec {
            b"l" => return Ok(ReadFmt::Line { keep_eol: false }),
            b"L" => return Ok(ReadFmt::Line { keep_eol: true }),
            b"a" => return Ok(ReadFmt::All),
            b"n" => return Ok(ReadFmt::Number),
            _ => {}
        }
    }
    Err(Error::from_str(
        ctx,
        &format!("bad argument #{n} to '{fname}' (invalid format)"),
    ))
}

/// Build the read-format list from a window of argument values, defaulting
/// to a single `"l"` when empty.
fn parse_formats<'gc>(
    ctx: Context<'gc>,
    args: &[Value<'gc>],
    fname: &str,
    first_arg: usize,
) -> Result<Vec<ReadFmt>, Error<'gc>> {
    if args.is_empty() {
        return Ok(vec![ReadFmt::Line { keep_eol: false }]);
    }
    args.iter()
        .enumerate()
        .map(|(i, v)| parse_format(ctx, *v, fname, first_arg + i))
        .collect()
}

/// Open `path` per a Lua mode string, returning a new file handle. Errors
/// are `std::io::Error` so callers choose between Lua's `(nil, msg, errno)`
/// return (`io.open`) and a raised error (`io.lines`/`io.input`).
fn open_file<'gc>(
    nctx: &NativeContext<'gc, '_>,
    path: &[u8],
    mode: &[u8],
) -> std::io::Result<Userdata<'gc>> {
    let base = mode.first().copied().unwrap_or(b'r');
    let plus = mode.contains(&b'+');
    let mut opts = OpenOptions::new();
    match base {
        b'r' => {
            opts.read(true).write(plus);
        }
        b'w' => {
            opts.write(true).create(true).truncate(true).read(plus);
        }
        b'a' => {
            opts.append(true).create(true).read(plus);
        }
        _ => return Err(std::io::Error::from_raw_os_error(22)), // EINVAL: bad mode
    }
    let p = std::path::Path::new(std::ffi::OsStr::new(
        std::str::from_utf8(path).unwrap_or(""),
    ));
    let file = opts.open(p)?;
    let readable = base == b'r' || plus;
    let writable = base != b'r' || plus;
    Ok(new_handle(
        nctx.ctx,
        file_metatable(nctx),
        LuaFile::open(Stream::File(BufReader::new(file)), readable, writable),
    ))
}

/// `(nil, "what: msg", errno)` from a failed `std::io::Error`.
fn io_fail<'gc>(ctx: Context<'gc>, fname: Option<&str>, e: &std::io::Error) -> [Value<'gc>; 3] {
    // `luaL_fileresult`: the message is bare `strerror(errno)`, prefixed with
    // the *filename* only (never the operation name) and only when one is
    // available. Rust's `Display` appends " (os error N)", which Lua omits, so
    // strip it back to the strerror text.
    let raw = e.to_string();
    let bare = match raw.find(" (os error ") {
        Some(cut) => &raw[..cut],
        None => &raw,
    };
    let text = match fname {
        Some(f) => format!("{f}: {bare}"),
        None => bare.to_string(),
    };
    [
        Value::nil(),
        Value::string(LuaString::new(ctx, text.as_bytes())),
        Value::integer(e.raw_os_error().unwrap_or(0) as i64),
    ]
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// `io.open(filename [, mode])`.
fn lua_open<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let name = stack
        .get(0)
        .get_string()
        .ok_or_else(|| Error::from_str(nctx.ctx, "bad argument #1 to 'open' (string expected)"))?;
    let mode_val = stack.get(1);
    let mode = if mode_val.is_nil() {
        b"r".as_slice()
    } else {
        mode_val.get_string().map(|s| s.as_bytes()).ok_or_else(|| {
            Error::from_str(nctx.ctx, "bad argument #2 to 'open' (string expected)")
        })?
    };
    match open_file(&nctx, name.as_bytes(), mode) {
        Ok(u) => stack.replace(&[Value::userdata(u)]),
        Err(e) => {
            let what = std::str::from_utf8(name.as_bytes()).unwrap_or("");
            stack.replace(&io_fail(nctx.ctx, Some(what), &e));
        }
    }
    Ok(CallbackAction::Return)
}

/// `io.write(...)` — write to the default output; return that handle so
/// `io.write("a"):write("b")` chains (issue #92).
fn lua_write<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let out_val = state_get(nctx.ctx, io_state(&nctx), b"output");
    let out = out_val
        .get_userdata()
        .expect("io-state output must be a file handle");
    let vals: Vec<Value<'gc>> = stack.as_slice().to_vec();
    match do_write(&nctx, out, &vals, "write", 1)? {
        WriteOutcome::Ok => stack.replace(&[out_val]),
        WriteOutcome::Closed => return Err(closed_file_error(nctx.ctx)),
        WriteOutcome::Io(e) => stack.replace(&io_fail(nctx.ctx, None, &e)),
    }
    Ok(CallbackAction::Return)
}

/// `io.read(...)` — read from the default input.
fn lua_read<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let inp = state_get(nctx.ctx, io_state(&nctx), b"input")
        .get_userdata()
        .expect("io-state input must be a file handle");
    let fmts = parse_formats(nctx.ctx, stack.as_slice(), "read", 1)?;
    let vals = do_read(nctx.ctx, inp, &fmts)?;
    stack.replace(&vals);
    Ok(CallbackAction::Return)
}

/// `io.close([file])` — close `file` or the default output.
fn lua_close<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let arg = stack.get(0);
    let file = if arg.is_nil() {
        state_get(nctx.ctx, io_state(&nctx), b"output")
            .get_userdata()
            .expect("io-state output must be a file handle")
    } else {
        check_file(&nctx, arg, "close", 1)?
    };
    close_handle(nctx.ctx, file, &mut stack)
}

/// `io.flush()` — flush the default output; return that handle.
fn lua_flush<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let out_val = state_get(nctx.ctx, io_state(&nctx), b"output");
    let out = out_val
        .get_userdata()
        .expect("io-state output must be a file handle");
    match out.with_data::<LuaFile, _>(|lf| flush_stream(&mut lf.state.borrow_mut())) {
        Some(WriteOutcome::Closed) => return Err(closed_file_error(nctx.ctx)),
        Some(WriteOutcome::Io(e)) => stack.replace(&io_fail(nctx.ctx, None, &e)),
        _ => stack.replace(&[out_val]),
    }
    Ok(CallbackAction::Return)
}

/// `io.input([file])` — get/set the default input. A string opens that
/// file in read mode (raising on failure, unlike `io.open`).
fn lua_input<'gc>(
    nctx: NativeContext<'gc, '_>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    default_file(nctx, stack, b"input", b"r", "input")
}

/// `io.output([file])` — get/set the default output (string opens in `w`).
fn lua_output<'gc>(
    nctx: NativeContext<'gc, '_>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    default_file(nctx, stack, b"output", b"w", "output")
}

fn default_file<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
    slot: &[u8],
    mode: &[u8],
    fname: &str,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let arg = stack.get(0);
    if !arg.is_nil() {
        let handle = if let Some(s) = arg.get_string() {
            open_file(&nctx, s.as_bytes(), mode).map_err(|e| {
                let what = std::str::from_utf8(s.as_bytes()).unwrap_or("");
                Error::from_str(nctx.ctx, &format!("{what}: {e}"))
            })?
        } else {
            check_file(&nctx, arg, fname, 1)?
        };
        io_state(&nctx).raw_set(nctx.ctx, str_val(nctx.ctx, slot), Value::userdata(handle));
    }
    let cur = state_get(nctx.ctx, io_state(&nctx), slot);
    stack.replace(&[cur]);
    Ok(CallbackAction::Return)
}

/// `io.lines([filename] [, formats...])`. With a filename, the file is
/// opened (raising on failure) and auto-closed at EOF; otherwise the
/// default input is used.
fn lua_lines<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let first = stack.get(0);
    let (handle, close_eof, fmt_args): (Value<'gc>, bool, &[Value<'gc>]) =
        if let Some(s) = first.get_string() {
            let u = open_file(&nctx, s.as_bytes(), b"r").map_err(|e| {
                let what = std::str::from_utf8(s.as_bytes()).unwrap_or("");
                Error::from_str(nctx.ctx, &format!("{what}: {e}"))
            })?;
            (Value::userdata(u), true, &stack.as_slice()[1..])
        } else {
            let inp = state_get(nctx.ctx, io_state(&nctx), b"input");
            (inp, false, stack.as_slice())
        };
    // Validate formats up front (Lua reports lines-format errors eagerly).
    parse_formats(nctx.ctx, fmt_args, "lines", 2)?;
    let iter = make_lines_iter(nctx.ctx, handle, close_eof, fmt_args);
    stack.replace(&[Value::function(iter)]);
    Ok(CallbackAction::Return)
}

/// `io.type(v)` — `"file"` / `"closed file"` / `nil`.
fn lua_type<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let result = match as_file(&nctx, stack.get(0)) {
        Some(u) => {
            let open = u
                .with_data::<LuaFile, _>(|lf| matches!(*lf.state.borrow(), FileState::Open { .. }))
                .unwrap_or(false);
            str_val(nctx.ctx, if open { b"file" } else { b"closed file" })
        }
        None => Value::nil(),
    };
    stack.replace(&[result]);
    Ok(CallbackAction::Return)
}

/// `io.tmpfile()` — a fresh temporary file open for update, removed from
/// the directory immediately (its descriptor stays valid until close/GC).
fn lua_tmpfile<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("tcvm_tmp_{}_{n}", std::process::id()));
    let opened = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path);
    match opened {
        Ok(file) => {
            let _ = std::fs::remove_file(&path); // unlink; fd remains valid
            let u = new_handle(
                nctx.ctx,
                file_metatable(&nctx),
                LuaFile::open(Stream::File(BufReader::new(file)), true, true),
            );
            stack.replace(&[Value::userdata(u)]);
        }
        Err(e) => stack.replace(&io_fail(nctx.ctx, None, &e)),
    }
    Ok(CallbackAction::Return)
}

/// `io.popen` — subprocess plumbing deferred to #27.
fn lua_popen<'gc>(
    _nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!("io.popen needs subprocess plumbing (#27)")
}

// ---------------------------------------------------------------------------
// File methods (self = arg 0)
// ---------------------------------------------------------------------------

/// `file:write(...)` — write the args, return `self` (chaining).
fn lua_file_write<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let self_val = stack.get(0);
    let u = check_file(&nctx, self_val, "write", 1)?;
    let vals: Vec<Value<'gc>> = stack.as_slice()[1..].to_vec();
    match do_write(&nctx, u, &vals, "write", 2)? {
        WriteOutcome::Ok => stack.replace(&[self_val]),
        WriteOutcome::Closed => return Err(closed_file_error(nctx.ctx)),
        WriteOutcome::Io(e) => stack.replace(&io_fail(nctx.ctx, None, &e)),
    }
    Ok(CallbackAction::Return)
}

/// `file:read(...)`.
fn lua_file_read<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let u = check_file(&nctx, stack.get(0), "read", 1)?;
    let fmts = parse_formats(nctx.ctx, &stack.as_slice()[1..], "read", 2)?;
    let vals = do_read(nctx.ctx, u, &fmts)?;
    stack.replace(&vals);
    Ok(CallbackAction::Return)
}

/// `file:lines(...)` — like `io.lines` but never auto-closes at EOF.
fn lua_file_lines<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let self_val = stack.get(0);
    check_file(&nctx, self_val, "lines", 1)?;
    let fmt_args = &stack.as_slice()[1..];
    parse_formats(nctx.ctx, fmt_args, "lines", 2)?;
    let iter = make_lines_iter(nctx.ctx, self_val, false, fmt_args);
    stack.replace(&[Value::function(iter)]);
    Ok(CallbackAction::Return)
}

/// `file:seek([whence [, offset]])` — `set`/`cur`/`end`, default `("cur",0)`.
fn lua_file_seek<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let u = check_file(&nctx, stack.get(0), "seek", 1)?;
    let whence_val = stack.get(1);
    let whence = if whence_val.is_nil() {
        b"cur".as_slice()
    } else {
        whence_val
            .get_string()
            .map(|s| s.as_bytes())
            .ok_or_else(|| {
                Error::from_str(nctx.ctx, "bad argument #2 to 'seek' (string expected)")
            })?
    };
    let offset = {
        let o = stack.get(2);
        if o.is_nil() {
            0
        } else {
            util::check_integer(nctx.ctx, o, "seek", 3)?
        }
    };
    let pos = match whence {
        // A negative absolute offset is invalid; surface the OS EINVAL the
        // underlying lseek would return rather than clamping to the start.
        b"set" if offset < 0 => None,
        b"set" => Some(SeekFrom::Start(offset as u64)),
        b"cur" => Some(SeekFrom::Current(offset)),
        b"end" => Some(SeekFrom::End(offset)),
        _ => {
            return Err(Error::from_str(
                nctx.ctx,
                "bad argument #2 to 'seek' (invalid option)",
            ));
        }
    };
    let outcome = match pos {
        Some(pos) => u
            .with_data::<LuaFile, _>(|lf| seek_stream(&mut lf.state.borrow_mut(), pos))
            .expect("file handle must carry a LuaFile payload"),
        None => SeekOutcome::Io(std::io::Error::from_raw_os_error(22)), // EINVAL
    };
    match outcome {
        SeekOutcome::Pos(n) => stack.replace(&[Value::integer(n as i64)]),
        SeekOutcome::Closed => return Err(closed_file_error(nctx.ctx)),
        SeekOutcome::Io(e) => stack.replace(&io_fail(nctx.ctx, None, &e)),
    }
    Ok(CallbackAction::Return)
}

/// `file:flush()` — flush, return `self`.
fn lua_file_flush<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let self_val = stack.get(0);
    let u = check_file(&nctx, self_val, "flush", 1)?;
    match u
        .with_data::<LuaFile, _>(|lf| flush_stream(&mut lf.state.borrow_mut()))
        .expect("file handle must carry a LuaFile payload")
    {
        WriteOutcome::Ok => stack.replace(&[self_val]),
        WriteOutcome::Closed => return Err(closed_file_error(nctx.ctx)),
        WriteOutcome::Io(e) => stack.replace(&io_fail(nctx.ctx, None, &e)),
    }
    Ok(CallbackAction::Return)
}

/// `file:close()`.
fn lua_file_close<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let u = check_file(&nctx, stack.get(0), "close", 1)?;
    close_handle(nctx.ctx, u, &mut stack)
}

/// `file:setvbuf(mode [, size])` — accepted, no-op (we don't expose buffer
/// control); returns `self`.
fn lua_file_setvbuf<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let self_val = stack.get(0);
    check_file(&nctx, self_val, "setvbuf", 1)?;
    stack.replace(&[self_val]);
    Ok(CallbackAction::Return)
}

/// `__tostring` — `"file (0x..)"` / `"file (closed)"`. Set on the metatable
/// for forward-compat; `tostring`/`print` don't dispatch `__tostring` yet (#27).
fn lua_file_tostring<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let u = check_file(&nctx, stack.get(0), "tostring", 1)?;
    let open = u
        .with_data::<LuaFile, _>(|lf| matches!(*lf.state.borrow(), FileState::Open { .. }))
        .unwrap_or(false);
    let s = if open {
        format!("file ({:p})", Gc::as_ptr(u.inner()))
    } else {
        "file (closed)".to_string()
    };
    stack.replace(&[str_val(nctx.ctx, s.as_bytes())]);
    Ok(CallbackAction::Return)
}

// ---------------------------------------------------------------------------
// Shared close + lines iterator
// ---------------------------------------------------------------------------

/// Close `u`'s stream. Standard streams can't be closed (Lua returns
/// `(nil, "cannot close standard file")`); a regular file transitions to
/// `Closed`, dropping its `File` (closing the fd) and returns `true`;
/// an already-closed file is a raised error.
fn close_handle<'gc>(
    ctx: Context<'gc>,
    u: Userdata<'gc>,
    stack: &mut Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    enum Outcome {
        Ok,
        Standard,
        AlreadyClosed,
    }
    let outcome = u
        .with_data::<LuaFile, _>(|lf| {
            let mut fs = lf.state.borrow_mut();
            match &*fs {
                FileState::Closed => Outcome::AlreadyClosed,
                FileState::Open {
                    stream: Stream::Stdin | Stream::Stdout | Stream::Stderr,
                    ..
                } => Outcome::Standard,
                FileState::Open { .. } => {
                    *fs = FileState::Closed;
                    Outcome::Ok
                }
            }
        })
        .expect("file handle must carry a LuaFile payload");
    match outcome {
        Outcome::Ok => stack.replace(&[Value::boolean(true)]),
        Outcome::Standard => {
            stack.replace(&[Value::nil(), str_val(ctx, b"cannot close standard file")])
        }
        Outcome::AlreadyClosed => return Err(closed_file_error(ctx)),
    }
    Ok(CallbackAction::Return)
}

/// Build the iterator closure for `io.lines`/`file:lines`. Upvalues:
/// `[handle, close_at_eof: bool, format-values...]`.
fn make_lines_iter<'gc>(
    ctx: Context<'gc>,
    handle: Value<'gc>,
    close_eof: bool,
    fmt_args: &[Value<'gc>],
) -> Function<'gc> {
    let mut upv: Vec<Value<'gc>> = Vec::with_capacity(2 + fmt_args.len());
    upv.push(handle);
    upv.push(Value::boolean(close_eof));
    upv.extend_from_slice(fmt_args);
    Function::new_native(ctx.mutation(), lines_iter, upv.into_boxed_slice())
}

/// The per-iteration body of a lines iterator. Reads one record; at EOF it
/// returns nothing (loop stops) and, for `io.lines(filename)`, closes the
/// auto-opened file.
fn lines_iter<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let handle = nctx.upvalues[0];
    let close_eof = nctx.upvalues[1].get_boolean().unwrap_or(false);
    let u = handle
        .get_userdata()
        .expect("lines iterator upvalue 0 must be a file handle");
    let fmt_vals = &nctx.upvalues[2..];
    let fmts: Vec<ReadFmt> = if fmt_vals.is_empty() {
        vec![ReadFmt::Line { keep_eol: false }]
    } else {
        fmt_vals
            .iter()
            .enumerate()
            .map(|(i, v)| parse_format(nctx.ctx, *v, "lines", i + 1))
            .collect::<Result<_, _>>()?
    };

    let vals = do_read(nctx.ctx, u, &fmts)?;
    // EOF iff the (first) record came back nil.
    if vals.first().map(|v| v.is_nil()).unwrap_or(true) {
        if close_eof {
            // Best-effort close of the auto-opened file; ignore the result.
            let _ = u.with_data::<LuaFile, _>(|lf| {
                let mut fs = lf.state.borrow_mut();
                if matches!(*fs, FileState::Open { .. }) {
                    *fs = FileState::Closed;
                }
            });
        }
        stack.replace(&[]);
    } else {
        stack.replace(&vals);
    }
    Ok(CallbackAction::Return)
}
