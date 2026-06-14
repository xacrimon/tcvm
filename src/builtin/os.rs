use std::os::unix::ffi::OsStrExt;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::Context;
use crate::builtin::util;
use crate::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Table, Value};
use crate::vm::sequence::CallbackAction;

/// `luaL_fileresult`-style outcome: `true` on success, `(nil, "msg", errno?)`
/// on failure. The message is bare `strerror(errno)` (Rust's `Display` appends
/// " (os error N)", which Lua omits), prefixed with the filename only when one
/// is given — `os.remove` passes the path, `os.rename` passes `None` (C calls
/// `luaL_fileresult` with a NULL filename there, so no prefix).
fn file_result<'gc>(
    nctx: &NativeContext<'gc, '_>,
    stack: &mut Stack<'gc, '_>,
    res: std::io::Result<()>,
    fname: Option<&str>,
) {
    match res {
        Ok(()) => stack.replace(&[Value::boolean(true)]),
        Err(e) => {
            let raw = e.to_string();
            let bare = match raw.find(" (os error ") {
                Some(cut) => &raw[..cut],
                None => &raw,
            };
            let text = match fname {
                Some(f) => format!("{f}: {bare}"),
                None => bare.to_string(),
            };
            let errno = e.raw_os_error().unwrap_or(0);
            stack.replace(&[
                Value::nil(),
                Value::string(LuaString::new(nctx.ctx, text.as_bytes())),
                Value::integer(errno as i64),
            ]);
        }
    }
}

fn check_str_arg<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<LuaString<'gc>, Error<'gc>> {
    v.get_string().ok_or_else(|| {
        Error::from_str(
            ctx,
            &format!(
                "bad argument #{n} to '{fname}' (string expected, got {})",
                v.type_name()
            ),
        )
    })
}

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("clock", lua_clock),
        ("date", lua_date),
        ("difftime", lua_difftime),
        ("execute", lua_execute),
        ("exit", lua_exit),
        ("getenv", lua_getenv),
        ("remove", lua_remove),
        ("rename", lua_rename),
        ("setlocale", lua_setlocale),
        ("time", lua_time),
        ("tmpname", lua_tmpname),
    ];

    let lib = Table::new(ctx);
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, Box::new([]));
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(handler));
    }

    let lib_name = Value::string(LuaString::new(ctx, b"os"));
    ctx.globals().raw_set(ctx, lib_name, Value::table(lib));
}

/// `clock()` — seconds of program runtime as a float. Approximated by
/// wall-clock elapsed since first call; TODO(#27): use real per-process CPU
/// time (no portable `std` API today).
fn lua_clock<'gc>(
    _nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    static START: OnceLock<Instant> = OnceLock::new();
    let start = START.get_or_init(Instant::now);
    stack.replace(&[Value::float(start.elapsed().as_secs_f64())]);
    Ok(CallbackAction::Return)
}

fn lua_date<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

/// `difftime(t2, t1)` — `t2 - t1` in seconds. Lua 5.5 requires both arguments
/// (5.3/5.4 defaulted `t1` to 0; that changed).
fn lua_difftime<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    if stack.is_empty() {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'difftime' (number expected, got no value)",
        ));
    }
    let t2 = util::check_number(nctx.ctx, stack.get(0), "difftime", 1)?;
    if stack.len() < 2 {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #2 to 'difftime' (number expected, got no value)",
        ));
    }
    let t1 = util::check_number(nctx.ctx, stack.get(1), "difftime", 2)?;
    stack.replace(&[Value::float(t2 - t1)]);
    Ok(CallbackAction::Return)
}

fn lua_execute<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

/// `exit([code [, close]])` — terminate the process. `code` may be a boolean
/// (`true`→0, `false`→1), an integer status, or nil (0). The `close` flag is
/// ignored (we always run normal process teardown).
fn lua_exit<'gc>(
    nctx: NativeContext<'gc, '_>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let arg = stack.get(0);
    // Boolean: true→0, false→1. Otherwise (and for nil/none) an integer status,
    // via `luaL_optinteger` — so non-integers raise rather than silently exit 0.
    let code = if let Some(b) = arg.get_boolean() {
        if b { 0 } else { 1 }
    } else if arg.is_nil() {
        0
    } else {
        util::check_integer(nctx.ctx, arg, "exit", 1)? as i32
    };
    use std::io::Write;
    let _ = std::io::stdout().flush();
    std::process::exit(code);
}

/// `getenv(name)` — the value of environment variable `name`, or nil.
fn lua_getenv<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let name = check_str_arg(nctx.ctx, stack.get(0), "getenv", 1)?;
    // Look up by raw bytes (env vars/values needn't be UTF-8), matching C.
    let val = std::env::var_os(std::ffi::OsStr::from_bytes(name.as_bytes()));
    let result = match val {
        Some(v) => Value::string(LuaString::new(nctx.ctx, v.as_os_str().as_bytes())),
        None => Value::nil(),
    };
    stack.replace(&[result]);
    Ok(CallbackAction::Return)
}

/// `remove(filename)` — delete a file (or empty directory). Returns `true`, or
/// `(nil, msg, errno)` on failure.
fn lua_remove<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let name = check_str_arg(nctx.ctx, stack.get(0), "remove", 1)?;
    let path = std::path::Path::new(std::ffi::OsStr::from_bytes(name.as_bytes()));
    // C `remove` deletes files and empty directories; try the file path first.
    let res = std::fs::remove_file(path).or_else(|_| std::fs::remove_dir(path));
    // `os.remove` passes the filename to `luaL_fileresult`, so it prefixes.
    let what = String::from_utf8_lossy(name.as_bytes());
    file_result(&nctx, &mut stack, res, Some(&what));
    Ok(CallbackAction::Return)
}

/// `rename(from, to)` — rename/move a file. Returns `true`, or
/// `(nil, msg, errno)` on failure.
fn lua_rename<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let from = check_str_arg(nctx.ctx, stack.get(0), "rename", 1)?;
    let to = check_str_arg(nctx.ctx, stack.get(1), "rename", 2)?;
    let from_p = std::path::Path::new(std::ffi::OsStr::from_bytes(from.as_bytes()));
    let to_p = std::path::Path::new(std::ffi::OsStr::from_bytes(to.as_bytes()));
    let res = std::fs::rename(from_p, to_p);
    // Unlike `remove`, C's `os_rename` passes NULL to `luaL_fileresult`, so the
    // error message carries no filename prefix.
    file_result(&nctx, &mut stack, res, None);
    Ok(CallbackAction::Return)
}

fn lua_setlocale<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

/// `time([table])` — with no argument, the current Unix time as an integer.
/// The table form (build a time from broken-down fields) needs timezone/DST
/// handling and is deferred; see TODO(#27).
fn lua_time<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let arg = stack.get(0);
    if arg.is_nil() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        stack.replace(&[Value::integer(now)]);
        Ok(CallbackAction::Return)
    } else if arg.get_table().is_some() {
        Err(Error::from_str(
            nctx.ctx,
            "os.time with a table argument is not yet supported (needs timezone handling)",
        ))
    } else {
        Err(Error::from_str(
            nctx.ctx,
            &format!(
                "bad argument #1 to 'time' (table expected, got {})",
                arg.type_name()
            ),
        ))
    }
}

/// `tmpname()` — a path usable as a temporary file name (not created here).
fn lua_tmpname<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("lua_{}_{n}", std::process::id()));
    let s = LuaString::new(nctx.ctx, path.to_string_lossy().as_bytes());
    stack.replace(&[Value::string(s)]);
    Ok(CallbackAction::Return)
}
