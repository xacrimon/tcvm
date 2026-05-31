use crate::Context;
use crate::builtin::util;
use crate::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Table, Value};
use crate::vm::sequence::CallbackAction;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// `luaL_fileresult`-style outcome: `true` on success, `(nil, "msg", errno?)`
/// on failure.
fn file_result<'gc>(
    nctx: &NativeContext<'gc, '_>,
    stack: &mut Stack<'gc, '_>,
    res: std::io::Result<()>,
    what: &str,
) {
    match res {
        Ok(()) => stack.replace(&[Value::boolean(true)]),
        Err(e) => {
            let msg = LuaString::new(nctx.ctx, format!("{what}: {e}").as_bytes());
            let errno = e.raw_os_error().unwrap_or(0);
            stack.replace(&[
                Value::nil(),
                Value::string(msg),
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

/// `difftime(t2 [, t1])` — `t2 - t1` in seconds (`t1` defaults to 0).
fn lua_difftime<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let t2 = util::check_number(nctx.ctx, stack.get(0), "difftime", 1)?;
    let t1_arg = stack.get(1);
    let t1 = if t1_arg.is_nil() {
        0.0
    } else {
        util::check_number(nctx.ctx, t1_arg, "difftime", 2)?
    };
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
    _nctx: NativeContext<'gc, '_>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let arg = stack.get(0);
    let code = if let Some(i) = arg.get_integer() {
        i as i32
    } else if arg.is_nil() {
        0 // os.exit() / os.exit(nil) / os.exit(true) -> success
    } else if arg.is_falsy() {
        1 // os.exit(false) -> failure
    } else {
        0
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
    let val = std::str::from_utf8(name.as_bytes())
        .ok()
        .and_then(|k| std::env::var(k).ok());
    let result = match val {
        Some(v) => Value::string(LuaString::new(nctx.ctx, v.as_bytes())),
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
    let path = std::path::Path::new(std::ffi::OsStr::new(
        std::str::from_utf8(name.as_bytes()).unwrap_or(""),
    ));
    // C `remove` deletes files and empty directories; try the file path first.
    let res = std::fs::remove_file(path).or_else(|_| std::fs::remove_dir(path));
    let what = std::str::from_utf8(name.as_bytes()).unwrap_or("");
    file_result(&nctx, &mut stack, res, what);
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
    let from_s = std::str::from_utf8(from.as_bytes()).unwrap_or("");
    let to_s = std::str::from_utf8(to.as_bytes()).unwrap_or("");
    let res = std::fs::rename(from_s, to_s);
    file_result(&nctx, &mut stack, res, from_s);
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
