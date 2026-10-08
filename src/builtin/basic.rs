use std::io::Write;

use crate::Context;
use crate::LoadError;
use crate::builtin::util;
use crate::env::function::NativeKind;
use crate::env::{
    Error, Function, LuaString, MetamethodBits, NativeClosure, NativeFn, Stack, Value,
};
use crate::vm::debug::where_prefix;
use crate::vm::native::{ContFn, NativeOut, OnOk, Protect, cont};

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("collectgarbage", lua_collectgarbage),
        ("error", lua_error),
        ("getmetatable", lua_getmetatable),
        ("loadfile", lua_loadfile),
        ("rawequal", lua_rawequal),
        ("rawget", lua_rawget),
        ("rawlen", lua_rawlen),
        ("rawset", lua_rawset),
        ("select", lua_select),
        ("tonumber", lua_tonumber),
        ("type", lua_type),
        ("warn", lua_warn),
    ];
    let actions: &[(&str, ContFn)] = &[
        ("dofile", lua_dofile),
        ("load", lua_load),
        ("print", lua_print),
        ("tostring", lua_tostring),
    ];

    let set = |name: &str, f: Function<'gc>| {
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        ctx.globals().raw_set(ctx, key, Value::function(f));
    };
    for &(name, handler) in fns {
        set(name, Function::new_native(ctx.mutation(), handler, &[]));
    }
    for &(name, handler) in actions {
        set(name, Function::new_cont(ctx.mutation(), handler, &[]));
    }
    set(
        "assert",
        Function::new_native_with_entry(
            ctx.mutation(),
            NativeKind::Plain(lua_assert),
            &[],
            crate::vm::ff::ff_assert,
        ),
    );
    set(
        "setmetatable",
        Function::new_native_with_entry(
            ctx.mutation(),
            NativeKind::Plain(lua_setmetatable),
            &[],
            crate::vm::ff::ff_setmetatable,
        ),
    );
    set(
        "pcall",
        Function::new_native_with_entry(
            ctx.mutation(),
            NativeKind::Cont(lua_pcall),
            &[],
            crate::vm::native::ff_pcall,
        ),
    );
    set(
        "xpcall",
        Function::new_native_with_entry(
            ctx.mutation(),
            NativeKind::Cont(lua_xpcall),
            &[],
            crate::vm::native::ff_xpcall,
        ),
    );
    // `pairs` hands back the same `next` the global holds, so `pairs(t) == next`.
    let next = ctx.next_fn();
    set("next", next);
    set(
        "pairs",
        Function::new_native_with_entry(
            ctx.mutation(),
            NativeKind::Cont(lua_pairs),
            &[Value::function(next)],
            crate::vm::native::ff_pairs,
        ),
    );
    let ipairs_iter = ctx.ipairs_iter();
    set(
        "ipairs",
        Function::new_native_with_entry(
            ctx.mutation(),
            NativeKind::Plain(lua_ipairs),
            &[Value::function(ipairs_iter)],
            crate::vm::native::ff_ipairs,
        ),
    );

    let globals = ctx.globals();
    let s = |bytes: &[u8]| Value::string(LuaString::new(ctx, bytes));
    globals.raw_set(ctx, s(b"_G"), Value::table(globals));
    // The language version, as LuaJIT reports "Lua 5.1".
    globals.raw_set(ctx, s(b"_VERSION"), s(b"Lua 5.5"));
}

/// `assert(v [, message, ...])` — if `v` is truthy, return all arguments
/// unchanged; otherwise raise `message` (default `"assertion failed!"`) as
/// `error(message)` would, i.e. with the caller's position when it's a string.
fn lua_assert<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    util::check_any(ctx, &stack, "assert", 1)?;
    if !stack.get(0).is_falsy() {
        // Leaving the window untouched returns all arguments.
        return Ok(());
    }
    if stack.len() >= 2 {
        Err(Error::new(ctx, stack.get(1)).with_level(1))
    } else {
        Err(Error::from_str(ctx, "assertion failed!"))
    }
}

/// `collectgarbage([opt [, arg]])` — light stand-in until the GC exposes the
/// control/introspection API this needs. Recognized options return
/// plausible values; collection requests are accepted as no-ops.
fn lua_collectgarbage<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let opt = stack.get(0);
    let opt = opt.get_string().map_or(&b"collect"[..], |s| s.as_bytes());
    match opt {
        // Lua returns a single value: total memory in use, in Kbytes. (The
        // absolute figure differs from PUC-Lua — different allocator — but the
        // shape/units match; full GC accounting tracked in #62.)
        b"count" => {
            let kb = ctx.mutation().metrics().total_allocation() as f64 / 1024.0;
            stack.ret1(Value::float(kb));
        }
        b"isrunning" => stack.replace(&[Value::boolean(true)]),
        b"step" => stack.replace(&[Value::boolean(false)]),
        _ => stack.replace(&[Value::integer(ctx.mutation(), 0)]),
    }
    Ok(())
}

/// `dofile([filename])` — run a file's chunk (stdin without a filename) and
/// return its results. A load error is raised.
fn lua_dofile<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let fname = util::opt_string(ctx, stack.get(0), "dofile", 1)?;
    let path = fname.map(|f| f.as_bytes());
    match ctx.load_file_with(path, Value::table(ctx.globals())) {
        Ok(f) => {
            stack.stage(0, Value::function(f), &[]);
            NativeOut::call_then(0, cont::DOFILE, Protect::No, OnOk::Return)
        }
        Err(e) => NativeOut::error(Error::new(ctx, load_error_value(ctx, &e))),
    }
}

/// `error(message [, level])`. The position prefix for `level >= 1` is
/// applied when the error is raised (`ThreadState::raise`); a negative level
/// names no frame, like any level past the bottom of the stack.
fn lua_error<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let level = if stack.len() >= 2 && !stack.get(1).is_nil() {
        let l = util::check_integer(ctx, stack.get(1), "error", 2)?;
        usize::try_from(l).unwrap_or(0)
    } else {
        1
    };
    Err(Error::new(ctx, stack.get(0)).with_level(level))
}

fn lua_getmetatable<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    util::check_any(ctx, &stack, "getmetatable", 1)?;
    // A `__metatable` field shadows the real metatable (protection).
    let result = match ctx.metatable_of(stack.get(0)) {
        Some(mt) => {
            let prot = mt.raw_get(Value::string(ctx.symbols().metatable));
            if prot.is_nil() {
                Value::table(mt)
            } else {
                prot
            }
        }
        None => Value::nil(),
    };
    stack.ret1(result);
    Ok(())
}

/// `ipairs(t)` — returns `(iterator, t, 0)`, the iterator from upvalue 0 so
/// every call hands back the same function.
fn lua_ipairs<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    util::check_any(ctx, &stack, "ipairs", 1)?;
    let t = stack.get(0);
    stack.replace(&[closure.upvalues()[0], t, Value::integer(ctx.mutation(), 0)]);
    Ok(())
}

/// Iterator body for `ipairs`: `(t, i) -> (i + 1, t[i + 1])`, or a lone `nil`
/// once that is nil. Indexing goes through `__index`.
pub(crate) fn ipairs_aux<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let i = util::check_integer(ctx, stack.get(1), "for iterator", 2)?.wrapping_add(1);
    if let Some(t) = stack.get(0).get_table() {
        let v = t.raw_get(Value::integer(ctx.mutation(), i));
        if !v.is_nil() {
            stack.replace(&[Value::integer(ctx.mutation(), i), v]);
            return NativeOut::RETURN;
        }
        if !t.shape().has_mm(MetamethodBits::INDEX) {
            stack.ret1(Value::nil());
            return NativeOut::RETURN;
        }
    }
    ipairs_meta(ctx, &mut stack, i)
}

/// `ipairs_aux` for a value whose `[i]` may run `__index`.
#[cold]
#[inline(never)]
fn ipairs_meta<'gc>(ctx: Context<'gc>, stack: &mut Stack<'gc, '_>, i: i64) -> NativeOut {
    stack.spawn_action(ctx, move |cx| async move {
        util::geti(&cx, 0, i).await?;
        cx.enter(|ctx, mut stack| {
            let v = stack.get(stack.len() - 1);
            if v.is_nil() {
                stack.ret1(v);
            } else {
                stack.replace(&[Value::integer(ctx.mutation(), i), v]);
            }
        });
        Ok(())
    })
}

/// `load(chunk [, chunkname [, mode [, env]]])` — compile a string chunk, or
/// the pieces a reader function returns, into a function; `(nil, message)`
/// when that fails.
fn lua_load<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let chunk = util::to_lstring(ctx, stack.get(0));
    check_mode(ctx, stack.get(2), "load", 3)?;
    if let Some(s) = chunk {
        let name = util::opt_string(ctx, stack.get(1), "load", 2)?.unwrap_or(s);
        let env = env_arg(ctx, &stack, 3);
        let loaded = ctx.load_bytes(s.as_bytes(), name, env);
        push_loaded(ctx, &mut stack, loaded);
        return NativeOut::RETURN;
    }
    let name = util::opt_string(ctx, stack.get(1), "load", 2)?
        .unwrap_or_else(|| LuaString::new(ctx, b"=(load)"));
    let Some(reader) = stack.get(0).get_function() else {
        return NativeOut::error(util::type_error(ctx, "load", 1, "function", stack.arg(0)));
    };
    if stack.len() < 2 {
        stack.push(Value::nil());
    }
    stack.as_mut_slice()[..2].copy_from_slice(&[Value::function(reader), Value::string(name)]);
    load_reader(ctx, &mut stack)
}

/// `load` from the reader function at window slot 0, called until it
/// returns nil or an empty string, the chunk's name at slot 1. An error the
/// reader raises becomes `load`'s `(nil, message)`. Unlike Lua, which parses
/// as it reads, the whole chunk is read first.
fn load_reader<'gc>(ctx: Context<'gc>, stack: &mut Stack<'gc, '_>) -> NativeOut {
    stack.spawn_action(ctx, |cx| async move {
        let mut source = Vec::new();
        // Each call at the top of the window: the reader, then its results.
        let at = cx.enter(|_, stack| stack.len());
        loop {
            cx.enter(|_, mut stack| {
                stack.truncate(at);
                stack.push(stack.get(0));
            });
            if let Err(e) = cx.pcall(at).await {
                cx.enter(|_, mut stack| {
                    let e = stack.get_local(e.value());
                    stack.replace(&[Value::nil(), e]);
                });
                return Ok(());
            }
            // `Some(done)`, or `None` for a piece that isn't a string.
            let piece = cx.enter(|ctx, stack| {
                let v = stack.get(at);
                if v.is_nil() {
                    return Some(true);
                }
                let s = util::to_lstring(ctx, v)?;
                source.extend_from_slice(s.as_bytes());
                Some(s.is_empty())
            });
            match piece {
                Some(false) => {}
                Some(true) => break,
                None => {
                    cx.enter(|ctx, mut stack| {
                        // `luaL_error`, located at `load`'s caller, below
                        // its frame.
                        let mut msg = where_prefix(stack.thread_mut(), 1);
                        msg.extend_from_slice(b"reader function must return a string");
                        let msg = LuaString::new(ctx, &msg);
                        stack.replace(&[Value::nil(), Value::string(msg)]);
                    });
                    return Ok(());
                }
            }
        }
        cx.enter(|ctx, mut stack| {
            stack.truncate(at);
            let name = stack.get(1).get_string().expect("a string name");
            let env = env_arg(ctx, &stack, 3);
            let loaded = ctx.load_bytes(&source, name, env);
            push_loaded(ctx, &mut stack, loaded);
        });
        Ok(())
    })
}

/// `load_aux`'s `env`: the argument at slot `i` when present, even nil.
fn env_arg<'gc>(ctx: Context<'gc>, stack: &Stack<'gc, '_>, i: usize) -> Value<'gc> {
    stack.arg(i).unwrap_or(Value::table(ctx.globals()))
}

fn push_loaded<'gc>(
    ctx: Context<'gc>,
    stack: &mut Stack<'gc, '_>,
    loaded: Result<Function<'gc>, LoadError>,
) {
    match loaded {
        Ok(f) => stack.ret1(Value::function(f)),
        Err(e) => stack.replace(&[Value::nil(), load_error_value(ctx, &e)]),
    }
}

/// A load error as the Lua string `load` returns and `dofile` raises.
fn load_error_value<'gc>(ctx: Context<'gc>, e: &LoadError) -> Value<'gc> {
    Value::string(LuaString::new(ctx, e.to_string().as_bytes()))
}

/// `getMode` for text-only loading: `B` (fixed buffers) is only for the C API,
/// and a mode without `t` admits nothing tcvm can load.
fn check_mode<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    n: usize,
) -> Result<(), Error<'gc>> {
    let Some(mode) = util::opt_string(ctx, v, fname, n)? else {
        return Ok(());
    };
    let msg = if mode.as_bytes().contains(&b'B') {
        "invalid mode"
    } else if !mode.as_bytes().contains(&b't') {
        "binary chunks are not supported"
    } else {
        return Ok(());
    };
    Err(util::arg_error(ctx, fname, n, msg))
}

/// `loadfile([filename [, mode [, env]]])` — `load` for a file's contents
/// (stdin without a filename).
fn lua_loadfile<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let fname = util::opt_string(ctx, stack.get(0), "loadfile", 1)?;
    check_mode(ctx, stack.get(1), "loadfile", 2)?;
    let env = env_arg(ctx, &stack, 2);
    let path = fname.map(|f| f.as_bytes());
    let loaded = ctx.load_file_with(path, env);
    push_loaded(ctx, &mut stack, loaded);
    Ok(())
}

/// `next(t [, k])` — `(k', t[k'])` for the entry after `k` in traversal
/// order, or a lone `nil` at the end.
pub(crate) fn lua_next<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let Some(t) = stack.get(0).get_table() else {
        return Err(util::type_error(ctx, "next", 1, "table", stack.arg(0)));
    };
    match t.next(ctx.mutation(), stack.get(1)) {
        Ok(Some((k, v))) => stack.replace(&[k, v]),
        Ok(None) => stack.replace(&[Value::nil()]),
        Err(_) => return Err(Error::from_str(ctx, "invalid key to 'next'")),
    }
    Ok(())
}

/// `pairs(t)` — `(next, t, nil, nil)` with `next` from upvalue 0, or the
/// four values `__pairs(t)` returns when the metatable defines it. Like the
/// reference, the argument is only checked for presence; a non-table
/// without `__pairs` fails later in `next`.
fn lua_pairs<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    util::check_any(ctx, &stack, "pairs", 1)?;
    let t = stack.get(0);
    let mm = ctx.mm_of(t, MetamethodBits::PAIRS);
    if mm.is_nil() {
        stack.replace(&[closure.upvalues()[0], t, Value::nil(), Value::nil()]);
        return NativeOut::RETURN;
    }
    stack.stage(0, mm, &[t]);
    NativeOut::call_then(0, cont::PAIRS, Protect::No, OnOk::Cont)
}

/// `__pairs` returned: its first four results, as `lua_call(L, 1, 4)`.
pub(crate) fn pairs_cont<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    _status: Result<(), Error<'gc>>,
) -> NativeOut {
    stack.truncate(4);
    while stack.len() < 4 {
        stack.push(Value::nil());
    }
    NativeOut::RETURN
}

/// The continuation of a call whose results are the native's.
pub(crate) fn return_cont<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
    _status: Result<(), Error<'gc>>,
) -> NativeOut {
    NativeOut::RETURN
}

/// `pcall(f, ...)`: a protected call of `f`, whose results come back as
/// `(true, ...)` and a caught error as `(false, err)`. The callee and its
/// arguments only need the header's slots opened; a non-callable `f` raises
/// inside the protected call, so it comes back as `(false, msg)` like the
/// reference.
fn lua_pcall<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    util::check_any(ctx, &stack, "pcall", 1)?;
    stack.open_hidden(0);
    NativeOut::call_then(0, cont::PCALL, Protect::Errors, OnOk::ReturnTrue)
}

pub(crate) fn pcall_cont<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> NativeOut {
    match status {
        Ok(()) => stack.insert(0, Value::boolean(true)),
        Err(err) => stack.replace(&[Value::boolean(false), err.value()]),
    }
    NativeOut::RETURN
}

/// `print(...)` — write each argument's `tostring` form to stdout, separated
/// by tabs and followed by a newline.
fn lua_print<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let n = stack.len();
    let has_tostring = |i| !ctx.mm_of(stack.get(i), MetamethodBits::TOSTRING).is_nil();
    if !(0..n).any(has_tostring) {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        for i in 0..n {
            if i > 0 {
                let _ = out.write_all(b"\t");
            }
            let s = util::basic_tostring(ctx, stack.get(i));
            let _ = out.write_all(s.as_bytes());
        }
        let _ = out.write_all(b"\n");
        stack.replace(&[]);
        return NativeOut::RETURN;
    }
    // Some `__tostring` must run; like Lua, write each argument as soon as it
    // is converted.
    stack.spawn_action(ctx, move |cx| async move {
        for i in 0..n {
            let bytes = util::tolstring(&cx, i, n).await?;
            let mut out = std::io::stdout().lock();
            if i > 0 {
                let _ = out.write_all(b"\t");
            }
            let _ = out.write_all(&bytes);
        }
        let _ = std::io::stdout().write_all(b"\n");
        cx.enter(|_, mut stack| stack.replace(&[]));
        Ok(())
    })
}

/// `rawequal(a, b)` — primitive equality, bypassing `__eq`.
fn lua_rawequal<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    util::check_any(ctx, &stack, "rawequal", 1)?;
    util::check_any(ctx, &stack, "rawequal", 2)?;
    let eq = util::raw_eq(stack.get(0), stack.get(1));
    stack.ret1(Value::boolean(eq));
    Ok(())
}

fn lua_rawget<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let t_arg = stack.get(0);
    let key = stack.get(1);
    let Some(t) = t_arg.get_table() else {
        return Err(util::type_error(ctx, "rawget", 1, "table", stack.arg(0)));
    };
    util::check_any(ctx, &stack, "rawget", 2)?;
    let v = t.raw_get(key);
    stack.ret1(v);
    Ok(())
}

/// `rawlen(v)` — length of a table (border) or string (byte count), bypassing
/// `__len`.
fn lua_rawlen<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let v = stack.get(0);
    let len = if let Some(s) = v.get_string() {
        s.len() as i64
    } else if let Some(t) = v.get_table() {
        t.raw_len() as i64
    } else {
        return Err(util::type_error(
            ctx,
            "rawlen",
            1,
            "table or string",
            stack.arg(0),
        ));
    };
    stack.ret1(Value::integer(ctx.mutation(), len));
    Ok(())
}

fn lua_rawset<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let t_arg = stack.get(0);
    let key = stack.get(1);
    let value = stack.get(2);
    let Some(t) = t_arg.get_table() else {
        return Err(util::type_error(ctx, "rawset", 1, "table", stack.arg(0)));
    };
    util::check_any(ctx, &stack, "rawset", 2)?;
    util::check_any(ctx, &stack, "rawset", 3)?;
    // `luaH_set` raises from inside the C function, so unlike argument
    // errors these carry no position.
    if key.is_nil() {
        return Err(Error::from_str(ctx, "table index is nil").with_level(0));
    }
    if key.get_float().is_some_and(f64::is_nan) {
        return Err(Error::from_str(ctx, "table index is NaN").with_level(0));
    }
    t.raw_set_keyed(ctx, key, value);
    stack.ret1(Value::table(t));
    Ok(())
}

/// `select('#', ...)` returns the count of extra arguments; `select(n, ...)`
/// returns the arguments from position `n` onward (negative `n` counts from
/// the end).
fn lua_select<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let m = stack.len().saturating_sub(1); // count of arguments after the selector
    let sel = stack.get(0);
    if let Some(s) = sel.get_string()
        && s.as_bytes() == b"#"
    {
        stack.ret1(Value::integer(ctx.mutation(), m as i64));
        return Ok(());
    }
    let i = util::check_integer(ctx, sel, "select", 1)?;
    let pos = if i < 0 { m as i64 + i + 1 } else { i };
    if pos < 1 {
        return Err(util::arg_error(ctx, "select", 1, "index out of range"));
    }
    let mut out = Vec::new();
    let mut k = pos as usize;
    while k <= m {
        out.push(stack.get(k));
        k += 1;
    }
    stack.replace(&out);
    Ok(())
}

/// `setmetatable(t, mt)` — attach `mt` (a table or nil) as `t`'s
/// metatable, returning `t`. Drives a shape transition along the
/// `set_metatable` edge so future accesses observe the new identity.
fn lua_setmetatable<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let Some(t) = stack.get(0).get_table() else {
        return Err(util::type_error(
            ctx,
            "setmetatable",
            1,
            "table",
            stack.arg(0),
        ));
    };
    let mt = match stack.arg(1) {
        Some(v) if v.is_nil() => None,
        Some(v) if let Some(mt) = v.get_table() => Some(mt),
        got => {
            return Err(util::type_error(
                ctx,
                "setmetatable",
                2,
                "nil or table",
                got,
            ));
        }
    };
    // If the existing metatable carries a `__metatable` field, the
    // metatable is locked: refuse the change. Matches Lua 5.5 reference
    // behavior (`luaL_error("cannot change a protected metatable")`).
    if let Some(existing) = t.metatable()
        && !existing
            .raw_get(Value::string(ctx.symbols().metatable))
            .is_nil()
    {
        return Err(Error::from_str(ctx, "cannot change a protected metatable"));
    }
    t.set_metatable(ctx, mt);
    stack.ret1(Value::table(t));
    Ok(())
}

fn lua_tonumber<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let v = stack.get(0);
    let base_arg = stack.get(1);
    let result = if !base_arg.is_nil() {
        // 2-arg form `tonumber(s, base)`. Lua's argument order: the base must be
        // an integer (#2), then `s` must be a string (#1), and only then is the
        // base range validated (#2) — so e.g. `tonumber(nil, 99)` complains
        // about #1, not the out-of-range base.
        let base = util::check_integer(ctx, base_arg, "tonumber", 2)?;
        let s = v
            .get_string()
            .ok_or_else(|| util::type_error(ctx, "tonumber", 1, "string", Some(v)))?;
        if !(2..=36).contains(&base) {
            return Err(util::arg_error(ctx, "tonumber", 2, "base out of range"));
        }
        util::str_to_int_base(s.as_bytes(), base as u32)
            .map_or(Value::nil(), |i| Value::integer(ctx.mutation(), i))
    } else if v.get_integer().is_some() || v.get_float().is_some() {
        v
    } else if let Some(s) = v.get_string() {
        util::str_to_number(s.as_bytes()).map_or(Value::nil(), |n| n.into_value(ctx.mutation()))
    } else {
        util::check_any(ctx, &stack, "tonumber", 1)?;
        Value::nil()
    };
    stack.ret1(result);
    Ok(())
}

/// `tostring(v)` — `luaL_tolstring`: `__tostring`'s result, else the default
/// representation (with a string `__name` in place of the type).
fn lua_tostring<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    util::check_any(ctx, &stack, "tostring", 1)?;
    let v = stack.get(0);
    let mm = ctx.mm_of(v, MetamethodBits::TOSTRING);
    if mm.is_nil() {
        stack.ret1(Value::string(util::basic_tostring(ctx, v)));
        return NativeOut::RETURN;
    }
    stack.stage(0, mm, &[v]);
    NativeOut::call_then(0, cont::TOSTRING, Protect::No, OnOk::Cont)
}

/// `__tostring` returned: its first result, checked and converted.
pub(crate) fn tostring_cont<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    _status: Result<(), Error<'gc>>,
) -> NativeOut {
    let s = util::tostring_result(ctx, stack.get(0))?;
    stack.ret1(Value::string(s));
    NativeOut::RETURN
}

/// `type(v)` — the type name of `v` as a string.
fn lua_type<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    util::check_any(ctx, &stack, "type", 1)?;
    let name = stack.get(0).type_name();
    stack.replace(&[Value::string(LuaString::new(ctx, name.as_bytes()))]);
    Ok(())
}

/// `warn(msg, ...)` — only checks its arguments (strings or numbers, as
/// `luaL_checkstring`); there is no warning system yet (#227).
fn lua_warn<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    if stack.is_empty() {
        return Err(util::type_error(ctx, "warn", 1, "string", None));
    }
    for i in 0..stack.len() {
        util::check_string(ctx, stack.get(i), "warn", i + 1)?;
    }
    stack.clear();
    Ok(())
}

/// `xpcall(f, msgh, ...)`: like `pcall`, but the executor calls `msgh` with
/// the error value before unwinding (see `Catch::Here`).
fn lua_xpcall<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    if stack.get(1).get_function().is_none() {
        return NativeOut::error(util::type_error(ctx, "xpcall", 2, "function", stack.arg(1)));
    }
    // The handler goes first, where the unwinder finds it; the callee and
    // its arguments follow in frame layout.
    stack.as_mut_slice().swap(0, 1);
    stack.open_hidden(1);
    NativeOut::call_then(1, cont::XPCALL, Protect::Handler, OnOk::ReturnTrue)
}

pub(crate) fn xpcall_cont<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> NativeOut {
    match status {
        // The results follow the handler's slot.
        Ok(()) => stack.as_mut_slice()[0] = Value::boolean(true),
        Err(err) => stack.replace(&[Value::boolean(false), err.value()]),
    }
    NativeOut::RETURN
}
