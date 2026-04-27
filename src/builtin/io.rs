use crate::Context;
use crate::env::{Function, LuaString, NativeContext, NativeError, NativeFn, Stack, Table, Value};

// See #27: predefined handles — stdin, stdout, stderr

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
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

    let lib = Table::new(ctx.mutation());
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, Box::new([]));
        let key = Value::String(LuaString::new(ctx.mutation(), name.as_bytes()));
        lib.raw_set(ctx.mutation(), key, Value::Function(handler));
    }

    let lib_name = Value::String(LuaString::new(ctx.mutation(), b"io"));
    ctx.globals()
        .raw_set(ctx.mutation(), lib_name, Value::Table(lib));
}

fn lua_close<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_flush<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_input<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_lines<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_open<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_output<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_popen<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_read<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_tmpfile<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_type<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_write<'gc>(
    _ctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    use std::io::Write;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for v in stack.as_slice() {
        let res = match v {
            Value::String(s) => out.write_all(s.as_bytes()),
            Value::Integer(i) => write!(out, "{i}"),
            Value::Float(f) => {
                // Lua's default number-to-string is "%.14g"; Rust's `{}` for
                // f64 is close enough for typical values and doesn't add a
                // trailing ".0" when the value rounds to an integer in our
                // current usage. The full "%.14g" goes through string.format.
                if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 {
                    write!(out, "{}", *f as i64)
                } else {
                    write!(out, "{}", f)
                }
            }
            other => {
                return Err(NativeError::new(format!(
                    "bad argument to 'write' (string expected, got {})",
                    other.type_name()
                )));
            }
        };
        if let Err(e) = res {
            return Err(NativeError::new(format!("io.write: {e}")));
        }
    }
    // TODO: return the file handle once io userdata exists.
    stack.replace(&[]);
    Ok(())
}

// See #27: file-handle methods — registered on the file-userdata metatable once
// userdata plumbing exists.

#[allow(dead_code)]
fn lua_file_close<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

#[allow(dead_code)]
fn lua_file_flush<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

#[allow(dead_code)]
fn lua_file_lines<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

#[allow(dead_code)]
fn lua_file_read<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

#[allow(dead_code)]
fn lua_file_seek<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

#[allow(dead_code)]
fn lua_file_setvbuf<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

#[allow(dead_code)]
fn lua_file_write<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}
