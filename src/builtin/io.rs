//! The `io` library: file handles as userdata, the predefined
//! `stdin`/`stdout`/`stderr` streams, and the default input/output that
//! the free `io.read`/`io.write`/`io.lines` functions operate on.
//!
//! A file handle is a `Userdata` whose payload is a [`LuaFile`] and whose
//! metatable is the single shared file metatable (`__index` → the methods
//! table, plus `__name`/`__tostring`/`__gc`/`__close`). Method dispatch
//! (`f:write(...)`) reaches the methods through that metatable's `__index`,
//! which the VM resolves for userdata receivers. The metatable, the methods
//! table, and the current default input/output handles live in an internal
//! "io-state" table captured as upvalue 0 by every `io` native.
//!
//! The OS file descriptor is owned by the `std::fs::File` inside the
//! handle; it is released either by an explicit `close`/`io.close` (which
//! drops the stream) or by the GC dropping the userdata (the collector
//! runs drop glue). Standard streams hold no owned descriptor, so dropping
//! a `stdin`/`stdout`/`stderr` handle never closes fd 0/1/2.

use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::os::unix::ffi::OsStrExt;

use crate::Context;
use crate::builtin::util;
use crate::dmm::Gc;
use crate::env::{
    Error, Function, LuaString, NativeClosure, NativeFn, Stack, Table, Userdata, Value,
};
use crate::lua::bare_io_msg;
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
    File(FileStream),
    Stdin,
    Stdout,
    Stderr,
}

/// A regular file. Reads go through `reader`'s read-ahead; writes collect in
/// `wbuf` per `mode` and reach the file before any read, seek or close, and
/// on drop, which covers collection and the host dropping the state.
struct FileStream {
    reader: BufReader<File>,
    wbuf: Vec<u8>,
    mode: BufMode,
}

/// `setvbuf`'s modes; the sizes are the buffer's capacity.
#[derive(Clone, Copy)]
enum BufMode {
    No,
    Line(usize),
    Full(usize),
}

/// A new file's buffer, like stdio's default full buffering.
const DEFAULT_BUF_SIZE: usize = 8192;

/// `setvbuf`'s default size (liolib's `LUAL_BUFFERSIZE`).
const LUAL_BUFFERSIZE: usize = 16 * size_of::<usize>() * size_of::<f64>();

impl FileStream {
    fn new(file: File) -> Self {
        FileStream {
            reader: BufReader::new(file),
            wbuf: Vec::new(),
            mode: BufMode::Full(DEFAULT_BUF_SIZE),
        }
    }

    fn flush_buf(&mut self) -> std::io::Result<()> {
        if self.wbuf.is_empty() {
            return Ok(());
        }
        let (res, _) = count_write(self.reader.get_mut(), &self.wbuf);
        // Dropped even on failure, so one error isn't reported by every
        // later operation.
        self.wbuf.clear();
        res
    }

    /// Buffer `buf`, or write it through once it no longer fits. The count is
    /// the bytes of `buf` accepted before an error.
    fn write(&mut self, buf: &[u8]) -> (std::io::Result<()>, u64) {
        let (cap, line) = match self.mode {
            BufMode::No => (0, false),
            BufMode::Line(cap) => (cap, true),
            BufMode::Full(cap) => (cap, false),
        };
        // Line mode sends everything through the last newline and holds back
        // the rest, as macOS stdio does.
        let head_len = match line {
            true => buf.iter().rposition(|&b| b == b'\n').map_or(0, |nl| nl + 1),
            false => 0,
        };
        let (head, tail) = buf.split_at(head_len);
        if self.wbuf.len() + buf.len() > cap
            && let Err(e) = self.flush_buf()
        {
            return (Err(e), 0);
        }
        if !head.is_empty() {
            if self.wbuf.is_empty() {
                let (res, n) = count_write(self.reader.get_mut(), head);
                if res.is_err() {
                    return (res, n);
                }
            } else {
                // Not flushed above, so `buf` fits: one write for both.
                self.wbuf.extend_from_slice(head);
                if let Err(e) = self.flush_buf() {
                    return (Err(e), 0);
                }
            }
        }
        if self.wbuf.len() + tail.len() > cap {
            let (res, n) = count_write(self.reader.get_mut(), tail);
            return (res, head_len as u64 + n);
        }
        self.wbuf.extend_from_slice(tail);
        (Ok(()), buf.len() as u64)
    }
}

impl Drop for FileStream {
    fn drop(&mut self) {
        let _ = self.flush_buf();
    }
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
    let upv = [Value::table(io_state)];
    let native = |f: NativeFn| Function::new_native(ctx.mutation(), f, &upv);

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
    // One function for both, as in liolib, so `mt.__gc == mt.__close`.
    let gc = Value::function(native(lua_file_gc));
    mt.raw_set(ctx, str_val(ctx, b"__gc"), gc);
    mt.raw_set(ctx, str_val(ctx, b"__close"), gc);
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
    util::set_not_implemented(ctx, lib, "io", &["popen"]);
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
fn io_state<'gc>(closure: &NativeClosure<'gc>) -> Table<'gc> {
    closure.upvalues[0]
        .get_table()
        .expect("io native upvalue 0 must be the io-state table")
}

#[inline]
fn state_get<'gc>(ctx: Context<'gc>, st: Table<'gc>, key: &[u8]) -> Value<'gc> {
    st.raw_get(str_val(ctx, key))
}

/// The shared file metatable held in io-state.
#[inline]
fn file_metatable<'gc>(ctx: Context<'gc>, closure: &NativeClosure<'gc>) -> Table<'gc> {
    state_get(ctx, io_state(closure), b"mt")
        .get_table()
        .expect("io-state must hold the file metatable")
}

/// `v` as a file handle iff it is userdata carrying the file metatable
/// (`luaL_testudata` by metatable identity).
fn as_file<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    v: Value<'gc>,
) -> Option<Userdata<'gc>> {
    let u = v.get_userdata()?;
    let umt = u.metatable()?;
    let fmt = file_metatable(ctx, closure);
    (Gc::as_ptr(umt.inner()) == Gc::as_ptr(fmt.inner())).then_some(u)
}

fn check_file<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<Userdata<'gc>, Error<'gc>> {
    as_file(ctx, closure, v).ok_or_else(|| util::type_error(ctx, fname, n, "FILE*", Some(v)))
}

fn closed_file_error<'gc>(ctx: Context<'gc>) -> Error<'gc> {
    Error::from_str(ctx, "attempt to use a closed file")
}

/// Run `f` on file handle `u`'s state.
fn with_state<R>(u: Userdata<'_>, f: impl FnOnce(&mut FileState) -> R) -> R {
    u.with_data::<LuaFile, _>(|lf| f(&mut lf.state.borrow_mut()))
        .expect("file handle must carry a LuaFile payload")
}

fn is_closed(u: Userdata<'_>) -> bool {
    with_state(u, |fs| matches!(fs, FileState::Closed))
}

/// The default `"input"`/`"output"` handle (`getiofile`), which the free
/// read/write/flush refuse to use once closed.
fn io_file<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    slot: &str,
) -> Result<Userdata<'gc>, Error<'gc>> {
    let u = state_get(ctx, io_state(closure), slot.as_bytes())
        .get_userdata()
        .expect("io-state default file must be a file handle");
    if is_closed(u) {
        return Err(Error::from_str(
            ctx,
            &format!("default {slot} file is closed"),
        ));
    }
    Ok(u)
}

// ---------------------------------------------------------------------------
// Low-level I/O on a `FileState` (no `'gc`)
// ---------------------------------------------------------------------------

enum WriteOutcome {
    Ok,
    Closed,
    /// I/O error plus the number of bytes successfully written before it — Lua's
    /// failed `file:write` returns `(nil, msg, errno, bytes_written)`.
    Io(std::io::Error, u64),
}

/// `write_all`, but reporting the byte count reached so a failure can surface it
/// (the count is 0 for an immediate `EBADF`, the common forced-failure case).
fn count_write<W: Write>(w: &mut W, buf: &[u8]) -> (std::io::Result<()>, u64) {
    let mut written = 0u64;
    while (written as usize) < buf.len() {
        match w.write(&buf[written as usize..]) {
            Ok(0) => {
                return (
                    Err(std::io::Error::from(std::io::ErrorKind::WriteZero)),
                    written,
                );
            }
            Ok(n) => written += n as u64,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return (Err(e), written),
        }
    }
    (Ok(()), written)
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
        return WriteOutcome::Io(std::io::Error::from_raw_os_error(9), 0);
    }
    let (res, written) = match stream {
        Stream::File(file) => file.write(buf),
        Stream::Stdout => count_write(&mut std::io::stdout(), buf),
        Stream::Stderr => count_write(&mut std::io::stderr(), buf),
        Stream::Stdin => (Err(std::io::Error::from_raw_os_error(9)), 0),
    };
    match res {
        Ok(()) => WriteOutcome::Ok,
        Err(e) => WriteOutcome::Io(e, written),
    }
}

fn flush_stream(fs: &mut FileState) -> WriteOutcome {
    let stream = match fs {
        FileState::Closed => return WriteOutcome::Closed,
        FileState::Open { stream, .. } => stream,
    };
    let res = match stream {
        Stream::File(file) => file.flush_buf(),
        Stream::Stdout => std::io::stdout().flush(),
        Stream::Stderr => std::io::stderr().flush(),
        Stream::Stdin => Ok(()),
    };
    match res {
        Ok(()) => WriteOutcome::Ok,
        Err(e) => WriteOutcome::Io(e, 0),
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
        Stream::File(file) => match file.flush_buf().and_then(|()| file.reader.seek(pos)) {
            Ok(n) => SeekOutcome::Pos(n),
            Err(e) => SeekOutcome::Io(e),
        },
        // ESPIPE — standard streams aren't seekable in this model.
        _ => SeekOutcome::Io(std::io::Error::from_raw_os_error(29)),
    }
}

/// One format from `fs`. A stream without read access fails with the EBADF
/// its fd would give.
fn read_stream(fs: &mut FileState, fmt: ReadFmt) -> std::io::Result<ReadOne> {
    let ebadf = || Err(std::io::Error::from_raw_os_error(9));
    match fs {
        FileState::Closed => unreachable!("do_read rejects closed files"),
        FileState::Open {
            readable: false, ..
        } => ebadf(),
        FileState::Open { stream, .. } => match stream {
            Stream::File(file) => {
                file.flush_buf()?;
                read_one(&mut file.reader, fmt)
            }
            Stream::Stdin => read_one(&mut std::io::stdin().lock(), fmt),
            Stream::Stdout | Stream::Stderr => ebadf(),
        },
    }
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

/// liolib's numeral buffer size: `read("n")` fails on a longer numeral.
const L_MAXLENNUM: usize = 200;

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
    // Mirror PUC-Lua's `read_number`: at most one decimal point and at most one
    // exponent marker, exponent only after a digit. Without these "already
    // seen" guards the flat accept-loop would swallow `1.2.3` or `1e2e3` whole
    // and then fail to parse, eating the rest of the stream.
    let mut tok: Vec<u8> = Vec::new();
    let mut seen_hex = false;
    let mut seen_dot = false;
    let mut seen_exp = false;
    let mut digits = 0usize;
    while let Some(&b) = r.fill_buf()?.first() {
        let mut is_digit = false;
        let accept = if b == b'+' || b == b'-' {
            // A sign starts the token or follows the exponent marker (`p`/`P` in
            // hex, `e`/`E` in decimal).
            let exp_mark = if seen_hex { (b'p', b'P') } else { (b'e', b'E') };
            tok.is_empty() || matches!(tok.last(), Some(&c) if c == exp_mark.0 || c == exp_mark.1)
        } else if b == b'.' {
            !seen_dot && !seen_exp
        } else if !seen_hex && (b == b'x' || b == b'X') {
            matches!(tok.as_slice(), b"0" | b"-0" | b"+0")
        } else if seen_hex {
            if seen_exp {
                // After `p`/`P` the exponent is decimal, so `a`..`f` stop here.
                is_digit = b.is_ascii_digit();
                is_digit
            } else if b.is_ascii_hexdigit() {
                is_digit = true;
                true
            } else {
                (b == b'p' || b == b'P') && digits > 0
            }
        } else if b.is_ascii_digit() {
            is_digit = true;
            true
        } else {
            (b == b'e' || b == b'E') && !seen_exp && digits > 0
        };
        if !accept {
            break;
        }
        // Fails with the overflowing character and the rest left unread.
        if tok.len() >= L_MAXLENNUM {
            return Ok(ReadOne::Nil);
        }
        if b == b'.' {
            seen_dot = true;
        } else if (b == b'x' || b == b'X') && !seen_hex {
            seen_hex = true;
            // The leading '0' of the "0x" prefix is not a mantissa digit; the
            // exponent gate counts only digits after the prefix (Lua).
            digits = 0;
        } else if (seen_hex && (b == b'p' || b == b'P')) || (!seen_hex && (b == b'e' || b == b'E'))
        {
            seen_exp = true;
        }
        if is_digit {
            digits += 1;
        }
        tok.push(b);
        r.consume(1);
    }
    Ok(match util::str_to_number(&tok) {
        Some(util::Number::Int(i)) => ReadOne::Int(i),
        Some(util::Number::Float(f)) => ReadOne::Float(f),
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
    ctx: Context<'gc>,
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
            return Err(util::type_error(
                ctx,
                fname,
                first_arg + i,
                "string",
                Some(*v),
            ));
        }
    }
    Ok(with_state(u, |fs| write_bytes(fs, &buf)))
}

/// Read `fmt_vals` from `u` (`g_read`), no formats meaning one line. Each
/// format is parsed only when its turn comes, so those after a failed one
/// are never checked. A closed file is a raised error; an I/O error replaces
/// the results with `(nil, msg, errno)`.
fn do_read<'gc>(
    ctx: Context<'gc>,
    u: Userdata<'gc>,
    fmt_vals: &[Value<'gc>],
    fname: &str,
    first_arg: usize,
) -> Result<Vec<Value<'gc>>, Error<'gc>> {
    if is_closed(u) {
        return Err(closed_file_error(ctx));
    }
    let n = fmt_vals.len().max(1);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let fmt = match fmt_vals.get(i) {
            Some(&v) => parse_format(ctx, v, fname, first_arg + i)?,
            None => ReadFmt::Line { keep_eol: false },
        };
        let one = with_state(u, |fs| read_stream(fs, fmt));
        let v = match one {
            Err(e) => return Ok(io_fail(ctx, None, &e).to_vec()),
            Ok(ReadOne::Nil) => {
                out.push(Value::nil());
                break;
            }
            Ok(ReadOne::Bytes(b)) => Value::string(LuaString::new(ctx, &b)),
            Ok(ReadOne::Int(i)) => Value::integer(ctx.mutation(), i),
            Ok(ReadOne::Float(f)) => Value::float(f),
        };
        out.push(v);
    }
    Ok(out)
}

/// One read format: a byte count, or a string whose first letter, after an
/// optional `*`, picks the format.
fn parse_format<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<ReadFmt, Error<'gc>> {
    let invalid = || Err(util::arg_error(ctx, fname, n, "invalid format"));
    if v.get_integer().is_some() || v.get_float().is_some() {
        // A negative count is invalid, where Lua's size_t cast fails it as
        // "resulting string too large".
        return match usize::try_from(util::check_integer(ctx, v, fname, n)?) {
            Ok(count) => Ok(ReadFmt::Bytes(count)),
            Err(_) => invalid(),
        };
    }
    let s = util::check_string(ctx, v, fname, n)?;
    let b = s.as_bytes();
    match b.strip_prefix(b"*").unwrap_or(b).first() {
        Some(b'l') => Ok(ReadFmt::Line { keep_eol: false }),
        Some(b'L') => Ok(ReadFmt::Line { keep_eol: true }),
        Some(b'a') => Ok(ReadFmt::All),
        Some(b'n') => Ok(ReadFmt::Number),
        _ => invalid(),
    }
}

/// Open `path` per a Lua mode string, returning a new file handle. Errors
/// are `std::io::Error` so callers choose between Lua's `(nil, msg, errno)`
/// return (`io.open`) and a raised error (`io.lines`/`io.input`).
fn open_file<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
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
    // The OS takes raw bytes; wrap them in an `OsStr` without UTF-8 validation
    // (a lossy `from_utf8` would silently target the wrong path on non-UTF-8
    // names) — matching C's `fopen` on the raw `const char*`.
    let p = std::path::Path::new(std::ffi::OsStr::from_bytes(path));
    let file = opts.open(p)?;
    let readable = base == b'r' || plus;
    let writable = base != b'r' || plus;
    Ok(new_handle(
        ctx,
        file_metatable(ctx, closure),
        LuaFile::open(Stream::File(FileStream::new(file)), readable, writable),
    ))
}

/// The l_checkmode regex `[rwa]\+?b*` — the only mode strings Lua's `io.open`
/// accepts. An invalid mode is a raised argument error, not a file result.
fn check_mode(mode: &[u8]) -> bool {
    let Some((&first, rest)) = mode.split_first() else {
        return false;
    };
    if !matches!(first, b'r' | b'w' | b'a') {
        return false;
    }
    let rest = match rest.split_first() {
        Some((&b'+', tail)) => tail,
        _ => rest,
    };
    rest.iter().all(|&c| c == b'b')
}

/// `(nil, "what: msg", errno)` from a failed `std::io::Error`.
fn io_fail<'gc>(ctx: Context<'gc>, fname: Option<&str>, e: &std::io::Error) -> [Value<'gc>; 3] {
    // `luaL_fileresult`: the message is bare `strerror(errno)`, prefixed with
    // the *filename* only (never the operation name) and only when one is
    // available.
    let bare = bare_io_msg(e);
    let text = match fname {
        Some(f) => format!("{f}: {bare}"),
        None => bare,
    };
    [
        Value::nil(),
        Value::string(LuaString::new(ctx, text.as_bytes())),
        Value::integer(ctx.mutation(), e.raw_os_error().unwrap_or(0) as i64),
    ]
}

/// A failed `write`'s return tuple: `io_fail` plus the bytes-written counter
/// Lua appends as a 4th value (`g_write`).
fn write_fail<'gc>(ctx: Context<'gc>, e: &std::io::Error, written: u64) -> [Value<'gc>; 4] {
    let [a, b, c] = io_fail(ctx, None, e);
    [a, b, c, Value::integer(ctx.mutation(), written as i64)]
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// `io.open(filename [, mode])`.
fn lua_open<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let name = match stack.arg(0) {
        Some(v) => util::check_string(ctx, v, "open", 1)?,
        None => return Err(util::type_error(ctx, "open", 1, "string", None)),
    };
    let mode = util::opt_string(ctx, stack.get(1), "open", 2)?;
    let mode = mode.map_or(&b"r"[..], |m| m.as_bytes());
    // An invalid mode is a raised argument error, not a `(nil, msg, errno)`
    // return (Lua's `luaL_argcheck(l_checkmode(...))`).
    if !check_mode(mode) {
        return Err(util::arg_error(ctx, "open", 2, "invalid mode"));
    }
    match open_file(ctx, closure, name.as_bytes(), mode) {
        Ok(u) => stack.replace(&[Value::userdata(u)]),
        Err(e) => {
            let what = String::from_utf8_lossy(name.as_bytes());
            stack.replace(&io_fail(ctx, Some(&what), &e));
        }
    }
    Ok(CallbackAction::Return)
}

/// `io.write(...)` — write to the default output; return that handle so
/// `io.write("a"):write("b")` chains (issue #92).
fn lua_write<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let out = io_file(ctx, closure, "output")?;
    let vals: Vec<Value<'gc>> = stack.as_slice().to_vec();
    match do_write(ctx, out, &vals, "write", 1)? {
        WriteOutcome::Ok => stack.replace(&[Value::userdata(out)]),
        WriteOutcome::Closed => return Err(closed_file_error(ctx)),
        WriteOutcome::Io(e, written) => stack.replace(&write_fail(ctx, &e, written)),
    }
    Ok(CallbackAction::Return)
}

/// `io.read(...)` — read from the default input.
fn lua_read<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let inp = io_file(ctx, closure, "input")?;
    let vals = do_read(ctx, inp, stack.as_slice(), "read", 1)?;
    stack.replace(&vals);
    Ok(CallbackAction::Return)
}

/// `io.close([file])` — close `file` or the default output.
fn lua_close<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let arg = stack.get(0);
    let file = if arg.is_nil() {
        state_get(ctx, io_state(closure), b"output")
            .get_userdata()
            .expect("io-state output must be a file handle")
    } else {
        check_file(ctx, closure, arg, "close", 1)?
    };
    close_handle(ctx, file, &mut stack)
}

/// `io.flush()` — flush the default output; return that handle.
fn lua_flush<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let out = io_file(ctx, closure, "output")?;
    match with_state(out, flush_stream) {
        WriteOutcome::Closed => return Err(closed_file_error(ctx)),
        WriteOutcome::Io(e, _) => stack.replace(&io_fail(ctx, None, &e)),
        WriteOutcome::Ok => stack.replace(&[Value::userdata(out)]),
    }
    Ok(CallbackAction::Return)
}

/// `io.input([file])` — get/set the default input. A string opens that
/// file in read mode (raising on failure, unlike `io.open`).
fn lua_input<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    default_file(ctx, closure, stack, b"input", b"r", "input")
}

/// `io.output([file])` — get/set the default output (string opens in `w`).
fn lua_output<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    default_file(ctx, closure, stack, b"output", b"w", "output")
}

fn default_file<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    slot: &[u8],
    mode: &[u8],
    fname: &str,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let arg = stack.get(0);
    if !arg.is_nil() {
        let handle = if let Some(s) = util::to_lstring(ctx, arg) {
            open_file(ctx, closure, s.as_bytes(), mode).map_err(|e| {
                let what = String::from_utf8_lossy(s.as_bytes());
                Error::from_str(
                    ctx,
                    &format!("cannot open file '{what}' ({})", bare_io_msg(&e)),
                )
            })?
        } else {
            check_file(ctx, closure, arg, fname, 1)?
        };
        io_state(closure).raw_set(ctx, str_val(ctx, slot), Value::userdata(handle));
    }
    let cur = state_get(ctx, io_state(closure), slot);
    stack.ret1(cur);
    Ok(CallbackAction::Return)
}

/// `io.lines([filename] [, formats...])`. With a filename, the file is
/// opened (raising on failure), auto-closed at EOF, and also returned as the
/// generic-for closing value so `break` or an error closes it too; otherwise
/// the default input is used.
fn lua_lines<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let fmt_args = stack.as_slice().get(1..).unwrap_or_default();
    let (handle, close_eof) = if stack.get(0).is_nil() {
        let inp = state_get(ctx, io_state(closure), b"input");
        if is_closed(
            inp.get_userdata()
                .expect("io-state input must be a file handle"),
        ) {
            return Err(closed_file_error(ctx));
        }
        (inp, false)
    } else {
        let s = util::check_string(ctx, stack.get(0), "lines", 1)?;
        let u = open_file(ctx, closure, s.as_bytes(), b"r").map_err(|e| {
            let what = String::from_utf8_lossy(s.as_bytes());
            Error::from_str(
                ctx,
                &format!("cannot open file '{what}' ({})", bare_io_msg(&e)),
            )
        })?;
        (Value::userdata(u), true)
    };
    let iter = Value::function(make_lines_iter(ctx, handle, close_eof, fmt_args)?);
    if close_eof {
        stack.replace(&[iter, Value::nil(), Value::nil(), handle]);
    } else {
        stack.ret1(iter);
    }
    Ok(CallbackAction::Return)
}

/// `io.type(v)` — `"file"` / `"closed file"` / `nil`.
fn lua_type<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    util::check_any(ctx, &stack, "type", 1)?;
    let result = match as_file(ctx, closure, stack.get(0)) {
        Some(u) if is_closed(u) => str_val(ctx, b"closed file"),
        Some(_) => str_val(ctx, b"file"),
        None => Value::nil(),
    };
    stack.ret1(result);
    Ok(CallbackAction::Return)
}

/// `io.tmpfile()` — a fresh temporary file open for update, removed from
/// the directory immediately (its descriptor stays valid until close/GC).
fn lua_tmpfile<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
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
                ctx,
                file_metatable(ctx, closure),
                LuaFile::open(Stream::File(FileStream::new(file)), true, true),
            );
            stack.ret1(Value::userdata(u));
        }
        Err(e) => stack.replace(&io_fail(ctx, None, &e)),
    }
    Ok(CallbackAction::Return)
}

// ---------------------------------------------------------------------------
// File methods (self = arg 0)
// ---------------------------------------------------------------------------

/// `file:write(...)` — write the args, return `self` (chaining).
fn lua_file_write<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let self_val = stack.get(0);
    let u = check_file(ctx, closure, self_val, "write", 1)?;
    let vals: Vec<Value<'gc>> = stack.as_slice()[1..].to_vec();
    match do_write(ctx, u, &vals, "write", 2)? {
        WriteOutcome::Ok => stack.replace(&[self_val]),
        WriteOutcome::Closed => return Err(closed_file_error(ctx)),
        WriteOutcome::Io(e, written) => stack.replace(&write_fail(ctx, &e, written)),
    }
    Ok(CallbackAction::Return)
}

/// `file:read(...)`.
fn lua_file_read<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let u = check_file(ctx, closure, stack.get(0), "read", 1)?;
    let vals = do_read(ctx, u, &stack.as_slice()[1..], "read", 2)?;
    stack.replace(&vals);
    Ok(CallbackAction::Return)
}

/// `file:lines(...)` — like `io.lines` but never auto-closes at EOF.
fn lua_file_lines<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let self_val = stack.get(0);
    check_file(ctx, closure, self_val, "lines", 1)?;
    let fmt_args = &stack.as_slice()[1..];
    let iter = make_lines_iter(ctx, self_val, false, fmt_args)?;
    stack.ret1(Value::function(iter));
    Ok(CallbackAction::Return)
}

/// `file:seek([whence [, offset]])` — `set`/`cur`/`end`, default `("cur",0)`.
fn lua_file_seek<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let u = check_file(ctx, closure, stack.get(0), "seek", 1)?;
    let whence = util::opt_string(ctx, stack.get(1), "seek", 2)?;
    let whence = whence.map_or(&b"cur"[..], |w| w.as_bytes());
    let offset = {
        let o = stack.get(2);
        if o.is_nil() {
            0
        } else {
            util::check_integer(ctx, o, "seek", 3)?
        }
    };
    let pos = match whence {
        // A negative absolute offset is invalid; surface the OS EINVAL the
        // underlying lseek would return rather than clamping to the start.
        b"set" if offset < 0 => None,
        b"set" => Some(SeekFrom::Start(offset as u64)),
        b"cur" => Some(SeekFrom::Current(offset)),
        b"end" => Some(SeekFrom::End(offset)),
        other => {
            return Err(util::arg_error(
                ctx,
                "seek",
                2,
                &format!("invalid option '{}'", String::from_utf8_lossy(other)),
            ));
        }
    };
    let outcome = match pos {
        Some(pos) => with_state(u, |fs| seek_stream(fs, pos)),
        None => SeekOutcome::Io(std::io::Error::from_raw_os_error(22)), // EINVAL
    };
    match outcome {
        SeekOutcome::Pos(n) => stack.replace(&[Value::integer(ctx.mutation(), n as i64)]),
        SeekOutcome::Closed => return Err(closed_file_error(ctx)),
        SeekOutcome::Io(e) => stack.replace(&io_fail(ctx, None, &e)),
    }
    Ok(CallbackAction::Return)
}

/// `file:flush()` — flush, return `self`.
fn lua_file_flush<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let self_val = stack.get(0);
    let u = check_file(ctx, closure, self_val, "flush", 1)?;
    match with_state(u, flush_stream) {
        WriteOutcome::Ok => stack.replace(&[self_val]),
        WriteOutcome::Closed => return Err(closed_file_error(ctx)),
        WriteOutcome::Io(e, _) => stack.replace(&io_fail(ctx, None, &e)),
    }
    Ok(CallbackAction::Return)
}

/// `file:close()`.
fn lua_file_close<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let u = check_file(ctx, closure, stack.get(0), "close", 1)?;
    close_handle(ctx, u, &mut stack)
}

/// `file:setvbuf(mode [, size])`, flushing what the old mode buffered. A
/// non-positive size falls back to the default; standard streams keep their
/// own buffering.
fn lua_file_setvbuf<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let u = check_file(ctx, closure, stack.get(0), "setvbuf", 1)?;
    if is_closed(u) {
        return Err(closed_file_error(ctx));
    }
    let mode = match stack.arg(1) {
        Some(v) => util::check_string(ctx, v, "setvbuf", 2)?,
        None => return Err(util::type_error(ctx, "setvbuf", 2, "string", None)),
    };
    let size = match stack.get(2) {
        v if v.is_nil() => LUAL_BUFFERSIZE,
        v => usize::try_from(util::check_integer(ctx, v, "setvbuf", 3)?)
            .ok()
            .filter(|&n| n > 0)
            .unwrap_or(LUAL_BUFFERSIZE),
    };
    let mode = match mode.as_bytes() {
        b"no" => BufMode::No,
        b"full" => BufMode::Full(size),
        b"line" => BufMode::Line(size),
        other => {
            return Err(util::arg_error(
                ctx,
                "setvbuf",
                2,
                &format!("invalid option '{}'", String::from_utf8_lossy(other)),
            ));
        }
    };
    let res = with_state(u, |fs| match fs {
        FileState::Open {
            stream: Stream::File(file),
            ..
        } => {
            file.mode = mode;
            file.flush_buf()
        }
        _ => Ok(()),
    });
    match res {
        Ok(()) => stack.ret1(Value::boolean(true)),
        Err(e) => stack.replace(&io_fail(ctx, None, &e)),
    }
    Ok(CallbackAction::Return)
}

/// `__gc`/`__close` — close the file unless it is a standard stream or
/// already closed, ignoring the outcome. The collector never calls this (no
/// Lua finalizers yet); dropping the userdata closes the fd instead.
fn lua_file_gc<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let u = match stack.arg(0) {
        Some(v) => check_file(ctx, closure, v, "__gc", 1)?,
        None => return Err(util::type_error(ctx, "__gc", 1, "FILE*", None)),
    };
    close_stream(u);
    stack.replace(&[]);
    Ok(CallbackAction::Return)
}

/// `__tostring` — `"file (0x..)"` / `"file (closed)"`. Set on the metatable
/// for forward-compat; `tostring`/`print` don't dispatch `__tostring` yet (#27).
fn lua_file_tostring<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let u = check_file(ctx, closure, stack.get(0), "tostring", 1)?;
    let s = if is_closed(u) {
        "file (closed)".to_string()
    } else {
        format!("file ({:p})", Gc::as_ptr(u.inner()))
    };
    stack.replace(&[str_val(ctx, s.as_bytes())]);
    Ok(CallbackAction::Return)
}

// ---------------------------------------------------------------------------
// Shared close + lines iterator
// ---------------------------------------------------------------------------

enum CloseOutcome {
    /// Closed, with the result of flushing what was still buffered.
    Closed(std::io::Result<()>),
    Standard,
    AlreadyClosed,
}

/// Flush a regular file and move it to `Closed`, dropping its `File` (closing
/// the fd) whether or not the flush failed. Standard streams stay open.
fn close_stream(u: Userdata<'_>) -> CloseOutcome {
    with_state(u, |fs| match fs {
        FileState::Closed => CloseOutcome::AlreadyClosed,
        FileState::Open {
            stream: Stream::Stdin | Stream::Stdout | Stream::Stderr,
            ..
        } => CloseOutcome::Standard,
        FileState::Open { .. } => {
            let FileState::Open {
                stream: Stream::File(mut file),
                ..
            } = std::mem::replace(fs, FileState::Closed)
            else {
                unreachable!()
            };
            CloseOutcome::Closed(file.flush_buf())
        }
    })
}

/// `close`/`io.close`: `true`, a failed final flush's `(nil, msg, errno)`,
/// `(nil, "cannot close standard file")`, or a raised error for an
/// already-closed file.
fn close_handle<'gc>(
    ctx: Context<'gc>,
    u: Userdata<'gc>,
    stack: &mut Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    match close_stream(u) {
        CloseOutcome::Closed(Ok(())) => stack.replace(&[Value::boolean(true)]),
        CloseOutcome::Closed(Err(e)) => stack.replace(&io_fail(ctx, None, &e)),
        CloseOutcome::Standard => {
            stack.replace(&[Value::nil(), str_val(ctx, b"cannot close standard file")])
        }
        CloseOutcome::AlreadyClosed => return Err(closed_file_error(ctx)),
    }
    Ok(CallbackAction::Return)
}

/// liolib's cap on the formats a lines iterator keeps.
const MAXARGLINE: usize = 250;

/// Build the iterator closure for `io.lines`/`file:lines`. Upvalues:
/// `[handle, close_at_eof: bool, format-values...]`.
fn make_lines_iter<'gc>(
    ctx: Context<'gc>,
    handle: Value<'gc>,
    close_eof: bool,
    fmt_args: &[Value<'gc>],
) -> Result<Function<'gc>, Error<'gc>> {
    // Lua blames the argument past the cap whichever form was called.
    if fmt_args.len() > MAXARGLINE {
        return Err(util::arg_error(
            ctx,
            "lines",
            MAXARGLINE + 2,
            "too many arguments",
        ));
    }
    let mut upv: Vec<Value<'gc>> = Vec::with_capacity(2 + fmt_args.len());
    upv.push(handle);
    upv.push(Value::boolean(close_eof));
    upv.extend_from_slice(fmt_args);
    Ok(Function::new_native(ctx.mutation(), lines_iter, &upv))
}

/// The per-iteration body of a lines iterator. Reads one record; at EOF it
/// returns nothing (loop stops) and, for `io.lines(filename)`, closes the
/// auto-opened file.
fn lines_iter<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let handle = closure.upvalues[0];
    let close_eof = closure.upvalues[1].get_boolean().unwrap_or(false);
    let u = handle
        .get_userdata()
        .expect("lines iterator upvalue 0 must be a file handle");
    if is_closed(u) {
        return Err(Error::from_str(ctx, "file is already closed"));
    }
    // Lua blames the formats from argument #2, as if called with the file.
    let vals = do_read(ctx, u, &closure.upvalues[2..], "for iterator", 2)?;
    if vals[0].is_nil() {
        // A fail carrying a message is an I/O error, raised rather than
        // ending the loop.
        if let Some(&msg) = vals.get(1) {
            return Err(Error::new(ctx, msg));
        }
        if close_eof {
            close_stream(u);
        }
        stack.replace(&[]);
    } else {
        stack.replace(&vals);
    }
    Ok(CallbackAction::Return)
}
