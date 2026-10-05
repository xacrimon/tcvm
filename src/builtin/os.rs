use std::os::unix::ffi::OsStrExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use libc::c_int;

use crate::Context;
use crate::builtin::util;
use crate::env::{Error, Function, LuaString, NativeClosure, NativeFn, Stack, Table, Value};
use crate::vm::async_sequence::{SequenceReturn, async_sequence};
use crate::vm::sequence::CallbackAction;

/// `luaL_fileresult`-style outcome: `true` on success, `(nil, "msg", errno?)`
/// on failure. The message is bare `strerror(errno)` (Rust's `Display` appends
/// " (os error N)", which Lua omits), prefixed with the filename only when one
/// is given — `os.remove` passes the path, `os.rename` passes `None` (C calls
/// `luaL_fileresult` with a NULL filename there, so no prefix).
fn file_result<'gc>(
    ctx: Context<'gc>,
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
                Value::string(LuaString::new(ctx, text.as_bytes())),
                Value::integer(ctx.mutation(), errno as i64),
            ]);
        }
    }
}

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("clock", lua_clock),
        ("date", lua_date),
        ("difftime", lua_difftime),
        ("exit", lua_exit),
        ("getenv", lua_getenv),
        ("remove", lua_remove),
        ("rename", lua_rename),
        ("tmpname", lua_tmpname),
    ];

    let lib = Table::new(ctx);
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, &[]);
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(handler));
    }
    let time = Function::new_action(ctx.mutation(), lua_time, &[]);
    lib.raw_set(
        ctx,
        Value::string(LuaString::new(ctx, b"time")),
        Value::function(time),
    );
    util::set_not_implemented(ctx, lib, "os", &["execute", "setlocale"]);

    let lib_name = Value::string(LuaString::new(ctx, b"os"));
    ctx.globals().raw_set(ctx, lib_name, Value::table(lib));
}

/// `clock()` — process CPU time in seconds, straight from C `clock()` like
/// `os_clock` in `loslib.c`.
fn lua_clock<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    // The `libc` crate binds neither `clock` nor `CLOCKS_PER_SEC` on unix
    // targets; XSI fixes the latter at one million.
    unsafe extern "C" {
        safe fn clock() -> libc::clock_t;
    }
    const CLOCKS_PER_SEC: f64 = 1_000_000.0;
    stack.ret1(Value::float(clock() as f64 / CLOCKS_PER_SEC));
    Ok(())
}

/// `date([format [, time]])` — `time` (default: now) broken down in local
/// time, or UTC when `format` starts with `!`; as a table for `*t`, else
/// through `strftime` one validated conversion at a time, as `os_date` does.
fn lua_date<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    // `SIZETIMEFMT`: longer expansions of a single conversion come out empty.
    const MAX_ITEM: usize = 250;

    let fmt = util::opt_string(ctx, stack.get(0), "date", 1)?.map_or(&b"%c"[..], |s| s.as_bytes());
    let t: libc::time_t = if stack.get(1).is_nil() {
        now()
    } else {
        util::check_integer(ctx, stack.get(1), "date", 2)?
    };
    let (utc, mut s) = match fmt.strip_prefix(b"!") {
        Some(rest) => (true, rest),
        None => (false, fmt),
    };
    // SAFETY: an all-zero `tm` is valid (null `tm_zone`); both calls only
    // write through `tm`.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let stm = unsafe {
        if utc {
            libc::gmtime_r(&t, &mut tm)
        } else {
            libc::localtime_r(&t, &mut tm)
        }
    };
    if stm.is_null() {
        return Err(Error::from_str(
            ctx,
            "date result cannot be represented in this installation",
        ));
    }

    // `strcmp`, so the format ends at its first NUL.
    if until_nul(s) == b"*t" {
        let fields = Table::new(ctx);
        let (ints, isdst) = tm_fields(&tm);
        for (key, v) in ints {
            let key = Value::string(LuaString::new(ctx, key.as_bytes()));
            fields.raw_set(ctx, key, Value::integer(ctx.mutation(), v));
        }
        if let Some(b) = isdst {
            let key = Value::string(LuaString::new(ctx, b"isdst"));
            fields.raw_set(ctx, key, Value::boolean(b));
        }
        stack.ret1(Value::table(fields));
        return Ok(());
    }

    let mut out = Vec::with_capacity(s.len());
    let mut buf = [0u8; MAX_ITEM];
    while let Some((&c, rest)) = s.split_first() {
        s = rest;
        if c != b'%' {
            out.push(c);
            continue;
        }
        let Some(n) = conversion_len(s) else {
            return Err(util::arg_error(
                ctx,
                "date",
                1,
                &format!(
                    "invalid conversion specifier '%{}'",
                    String::from_utf8_lossy(until_nul(s))
                ),
            ));
        };
        let mut cc = [b'%', 0, 0, 0];
        cc[1..=n].copy_from_slice(&s[..n]);
        s = &s[n..];
        // SAFETY: `cc` is NUL-terminated and `strftime` writes at most
        // `buf.len()` bytes.
        let len =
            unsafe { libc::strftime(buf.as_mut_ptr().cast(), buf.len(), cc.as_ptr().cast(), &tm) };
        out.extend_from_slice(&buf[..len]);
    }
    stack.ret1(Value::string(LuaString::new(ctx, &out)));
    Ok(())
}

fn until_nul(s: &[u8]) -> &[u8] {
    s.split(|&b| b == 0).next().unwrap_or(s)
}

/// Length of the valid `strftime` conversion that `conv` starts with: the C99
/// set of `LUA_STRFTIMEOPTIONS`.
fn conversion_len(conv: &[u8]) -> Option<usize> {
    const ONE: &[u8] = b"aAbBcCdDeFgGhHIjmMnprRStTuUVwWxXyYzZ%";
    const TWO: &[u8] = b"EcECExEXEyEYOdOeOHOIOmOMOSOuOUOVOwOWOy";
    match conv {
        [c, ..] if ONE.contains(c) => Some(1),
        [a, b, ..] if TWO.chunks(2).any(|o| o == [*a, *b]) => Some(2),
        _ => None,
    }
}

/// `setallfields`: a `tm`'s date-table fields, plus `isdst` unless unknown.
fn tm_fields(tm: &libc::tm) -> ([(&'static str, i64); 8], Option<bool>) {
    let ints = [
        ("year", tm.tm_year as i64 + 1900),
        ("month", tm.tm_mon as i64 + 1),
        ("day", tm.tm_mday as i64),
        ("hour", tm.tm_hour as i64),
        ("min", tm.tm_min as i64),
        ("sec", tm.tm_sec as i64),
        ("yday", tm.tm_yday as i64 + 1),
        ("wday", tm.tm_wday as i64 + 1),
    ];
    (ints, (tm.tm_isdst >= 0).then_some(tm.tm_isdst != 0))
}

/// `difftime(t2, t1)` — `t2 - t1` in seconds. Lua 5.5 requires both arguments
/// (5.3/5.4 defaulted `t1` to 0; that changed).
fn lua_difftime<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    if stack.is_empty() {
        return Err(util::type_error(ctx, "difftime", 1, "number", None));
    }
    let t2 = util::check_number(ctx, stack.get(0), "difftime", 1)?;
    if stack.len() < 2 {
        return Err(util::type_error(ctx, "difftime", 2, "number", None));
    }
    let t1 = util::check_number(ctx, stack.get(1), "difftime", 2)?;
    stack.ret1(Value::float(t2 - t1));
    Ok(())
}

/// `exit([code [, close]])` — stop the executor and hand `code` to the host
/// (`RuntimeError::Exit`). `code` may be a boolean (`true`→0, `false`→1), an
/// integer status, or nil (0). `close` is ignored: pending `__close`s never
/// run, and the host drops the state either way.
fn lua_exit<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let arg = stack.get(0);
    // Boolean: true→0, false→1. Otherwise (and for nil/none) an integer status,
    // via `luaL_optinteger` — so non-integers raise rather than silently exit 0.
    let code = if let Some(b) = arg.get_boolean() {
        if b { 0 } else { 1 }
    } else if arg.is_nil() {
        0
    } else {
        util::check_integer(ctx, arg, "exit", 1)? as i32
    };
    Err(Error::exit_process(ctx, code))
}

/// `getenv(name)` — the value of environment variable `name`, or nil.
fn lua_getenv<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let name = util::check_string(ctx, stack.get(0), "getenv", 1)?;
    // Look up by raw bytes (env vars/values needn't be UTF-8), matching C.
    let val = std::env::var_os(std::ffi::OsStr::from_bytes(name.as_bytes()));
    let result = match val {
        Some(v) => Value::string(LuaString::new(ctx, v.as_os_str().as_bytes())),
        None => Value::nil(),
    };
    stack.ret1(result);
    Ok(())
}

/// `remove(filename)` — delete a file (or empty directory). Returns `true`, or
/// `(nil, msg, errno)` on failure.
fn lua_remove<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let name = util::check_string(ctx, stack.get(0), "remove", 1)?;
    let path = std::path::Path::new(std::ffi::OsStr::from_bytes(name.as_bytes()));
    // C `remove` deletes files and empty directories; try the file path first.
    let res = std::fs::remove_file(path).or_else(|_| std::fs::remove_dir(path));
    // `os.remove` passes the filename to `luaL_fileresult`, so it prefixes.
    let what = String::from_utf8_lossy(name.as_bytes());
    file_result(ctx, &mut stack, res, Some(&what));
    Ok(())
}

/// `rename(from, to)` — rename/move a file. Returns `true`, or
/// `(nil, msg, errno)` on failure.
fn lua_rename<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let from = util::check_string(ctx, stack.get(0), "rename", 1)?;
    let to = util::check_string(ctx, stack.get(1), "rename", 2)?;
    let from_p = std::path::Path::new(std::ffi::OsStr::from_bytes(from.as_bytes()));
    let to_p = std::path::Path::new(std::ffi::OsStr::from_bytes(to.as_bytes()));
    let res = std::fs::rename(from_p, to_p);
    // Unlike `remove`, C's `os_rename` passes NULL to `luaL_fileresult`, so the
    // error message carries no filename prefix.
    file_result(ctx, &mut stack, res, None);
    Ok(())
}

/// `time([table])` — the current time, or the time the date table names
/// (local time, normalized by `mktime`, which also writes the normalized
/// fields back into the table). Table accesses go through metamethods.
fn lua_time<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let arg = stack.get(0);
    if arg.is_nil() {
        stack.ret1(Value::integer(ctx.mutation(), now()));
        return Ok(CallbackAction::Return);
    }
    if arg.get_table().is_none() {
        return Err(util::type_error(ctx, "time", 1, "table", Some(arg)));
    }
    let seq = async_sequence(ctx.mutation(), |_locals, mut seq| async move {
        let mut ints = [0; 6];
        for (slot, &(k, d, delta)) in ints.iter_mut().zip(&DATE_FIELDS) {
            util::getfield(&mut seq, 0, k.as_bytes()).await?;
            *slot = seq.try_enter(|ctx, _locals, _exec, mut stack| {
                date_field(ctx, stack.pop(), k, d, delta)
            })?;
        }
        util::getfield(&mut seq, 0, b"isdst").await?;
        let isdst = seq.enter(|_ctx, _locals, _exec, mut stack| bool_field(stack.pop()));
        let (time, tm) = make_time(ints, isdst);
        let (ints, isdst) = tm_fields(&tm);
        for (k, v) in ints {
            seq.enter(|ctx, _locals, _exec, mut stack| {
                stack.push(Value::integer(ctx.mutation(), v))
            });
            util::setfield(&mut seq, 0, k.as_bytes()).await?;
        }
        if let Some(b) = isdst {
            seq.enter(|_ctx, _locals, _exec, mut stack| stack.push(Value::boolean(b)));
            util::setfield(&mut seq, 0, b"isdst").await?;
        }
        seq.try_enter(|ctx, _locals, _exec, mut stack| {
            stack.ret1(time_result(ctx, time)?);
            Ok(())
        })?;
        Ok(SequenceReturn::Return)
    });
    Ok(CallbackAction::sequence(seq))
}

/// `os_time`'s `getfield` calls, in order: key, default (`None`: required),
/// and the offset between the Lua field and the `tm` member.
const DATE_FIELDS: [(&str, Option<c_int>, c_int); 6] = [
    ("year", None, 1900),
    ("month", None, 1),
    ("day", None, 0),
    ("hour", Some(12), 0),
    ("min", Some(0), 0),
    ("sec", Some(0), 0),
];

/// `getfield`: date-table field `key`, holding `v`, as a `tm` member.
fn date_field<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    key: &str,
    default: Option<c_int>,
    delta: c_int,
) -> Result<c_int, Error<'gc>> {
    let msg = match util::to_integer(v) {
        Some(n) => match n.checked_sub(delta.into()).map(c_int::try_from) {
            Some(Ok(n)) => return Ok(n),
            _ => "is out-of-bound",
        },
        None if !v.is_nil() => "is not an integer",
        None => match default {
            Some(d) => return Ok(d),
            None => "missing in date table",
        },
    };
    Err(Error::from_str(ctx, &format!("field '{key}' {msg}")))
}

/// `getboolfield`: `-1` (let `mktime` decide) for nil, else truthiness.
fn bool_field(v: Value<'_>) -> c_int {
    if v.is_nil() {
        -1
    } else {
        !v.is_falsy() as c_int
    }
}

fn make_time(ints: [c_int; 6], isdst: c_int) -> (libc::time_t, libc::tm) {
    // SAFETY: an all-zero `tm` is valid (null `tm_zone`).
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    [
        tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec,
    ] = ints;
    tm.tm_isdst = isdst;
    // SAFETY: `mktime` only reads and normalizes `tm`.
    let time = unsafe { libc::mktime(&mut tm) };
    (time, tm)
}

fn time_result<'gc>(ctx: Context<'gc>, time: libc::time_t) -> Result<Value<'gc>, Error<'gc>> {
    if time == -1 {
        return Err(Error::from_str(
            ctx,
            "time result cannot be represented in this installation",
        ));
    }
    Ok(Value::integer(ctx.mutation(), time))
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// `tmpname()` — a path usable as a temporary file name (not created here).
fn lua_tmpname<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("lua_{}_{n}", std::process::id()));
    let s = LuaString::new(ctx, path.to_string_lossy().as_bytes());
    stack.ret1(Value::string(s));
    Ok(())
}
