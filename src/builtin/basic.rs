use std::io::Write;
use std::pin::Pin;

use crate::Context;
use crate::LoadError;
use crate::builtin::util;
use crate::dmm::{Collect, Trace};
use crate::env::{
    Error, Function, LuaString, MetamethodBits, NativeClosure, NativeFn, Stack, Value,
};
use crate::vm::async_sequence::{SequenceReturn, async_sequence};
use crate::vm::debug::where_prefix;
use crate::vm::interp;
use crate::vm::sequence::{
    BoxSequence, CallbackAction, Catch, Execution, Sequence, SequencePoll, seq_trace_pointers,
};

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("assert", lua_assert),
        ("collectgarbage", lua_collectgarbage),
        ("dofile", lua_dofile),
        ("error", lua_error),
        ("getmetatable", lua_getmetatable),
        ("load", lua_load),
        ("loadfile", lua_loadfile),
        ("pcall", lua_pcall),
        ("print", lua_print),
        ("rawequal", lua_rawequal),
        ("rawget", lua_rawget),
        ("rawlen", lua_rawlen),
        ("rawset", lua_rawset),
        ("select", lua_select),
        ("setmetatable", lua_setmetatable),
        ("tonumber", lua_tonumber),
        ("tostring", lua_tostring),
        ("type", lua_type),
        ("warn", lua_warn),
        ("xpcall", lua_xpcall),
    ];

    let set = |name: &str, f: Function<'gc>| {
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        ctx.globals().raw_set(ctx, key, Value::function(f));
    };
    for &(name, handler) in fns {
        set(name, Function::new_native(ctx.mutation(), handler, &[]));
    }
    // `pairs` hands back the same `next` the global holds, so `pairs(t) == next`.
    let next = ctx.next_fn();
    set("next", next);
    set(
        "pairs",
        Function::new_native_with_entry(
            ctx.mutation(),
            lua_pairs,
            &[Value::function(next)],
            interp::ff_pairs,
        ),
    );
    let ipairs_iter = ctx.ipairs_iter();
    set(
        "ipairs",
        Function::new_native_with_entry(
            ctx.mutation(),
            lua_ipairs,
            &[Value::function(ipairs_iter)],
            interp::ff_ipairs,
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
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    util::check_any(ctx, &stack, "assert", 1)?;
    if !stack.get(0).is_falsy() {
        // Leaving the window untouched returns all arguments.
        return Ok(CallbackAction::Return);
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
) -> Result<CallbackAction<'gc>, Error<'gc>> {
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
    Ok(CallbackAction::Return)
}

/// `dofile([filename])` — run a file's chunk (stdin without a filename) and
/// return its results. A load error is raised.
fn lua_dofile<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let fname = util::opt_string(ctx, stack.get(0), "dofile", 1)?;
    let path = fname.map(|f| f.as_bytes());
    match ctx.load_file_with(path, Value::table(ctx.globals())) {
        Ok(f) => {
            stack.replace(&[Value::function(f)]);
            Ok(CallbackAction::call(None))
        }
        Err(e) => Err(Error::new(ctx, load_error_value(ctx, &e))),
    }
}

/// `error(message [, level])`. The position prefix for `level >= 1` is
/// applied when the error is raised (`ThreadState::raise`); a negative level
/// names no frame, like any level past the bottom of the stack.
fn lua_error<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
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
) -> Result<CallbackAction<'gc>, Error<'gc>> {
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
    Ok(CallbackAction::Return)
}

/// `ipairs(t)` — returns `(iterator, t, 0)`, the iterator from upvalue 0 so
/// every call hands back the same function.
fn lua_ipairs<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    util::check_any(ctx, &stack, "ipairs", 1)?;
    let t = stack.get(0);
    stack.replace(&[closure.upvalues[0], t, Value::integer(ctx.mutation(), 0)]);
    Ok(CallbackAction::Return)
}

/// Iterator body for `ipairs`: `(t, i) -> (i + 1, t[i + 1])`, or a lone `nil`
/// once that is nil. Indexing goes through `__index`.
pub(crate) fn ipairs_aux<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let i = util::check_integer(ctx, stack.get(1), "for iterator", 2)?.wrapping_add(1);
    if let Some(t) = stack.get(0).get_table() {
        let v = t.raw_get(Value::integer(ctx.mutation(), i));
        if !v.is_nil() {
            stack.replace(&[Value::integer(ctx.mutation(), i), v]);
            return Ok(CallbackAction::Return);
        }
        if !t.shape().has_mm(MetamethodBits::INDEX) {
            stack.ret1(Value::nil());
            return Ok(CallbackAction::Return);
        }
    }
    Ok(ipairs_meta(ctx, i))
}

/// `ipairs_aux` for a value whose `[i]` may run `__index`.
#[cold]
#[inline(never)]
fn ipairs_meta<'gc>(ctx: Context<'gc>, i: i64) -> CallbackAction<'gc> {
    let seq = async_sequence(ctx.mutation(), move |_locals, mut seq| async move {
        util::geti(&mut seq, 0, i).await?;
        seq.enter(|ctx, _locals, _exec, mut stack| {
            let v = stack.get(stack.len() - 1);
            if v.is_nil() {
                stack.ret1(v);
            } else {
                stack.replace(&[Value::integer(ctx.mutation(), i), v]);
            }
        });
        Ok(SequenceReturn::Return)
    });
    CallbackAction::sequence(seq)
}

/// `load(chunk [, chunkname [, mode [, env]]])` — compile a string chunk, or
/// the pieces a reader function returns, into a function; `(nil, message)`
/// when that fails.
fn lua_load<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let chunk = util::to_lstring(ctx, stack.get(0));
    check_mode(ctx, stack.get(2), "load", 3)?;
    if let Some(s) = chunk {
        let name = util::opt_string(ctx, stack.get(1), "load", 2)?.unwrap_or(s);
        let env = env_arg(ctx, &stack, 3);
        let loaded = ctx.load_bytes(s.as_bytes(), name, env);
        push_loaded(ctx, &mut stack, loaded);
        return Ok(CallbackAction::Return);
    }
    let name = util::opt_string(ctx, stack.get(1), "load", 2)?
        .unwrap_or_else(|| LuaString::new(ctx, b"=(load)"));
    let Some(reader) = stack.get(0).get_function() else {
        return Err(util::type_error(ctx, "load", 1, "function", stack.arg(0)));
    };
    Ok(load_reader(ctx, reader, name))
}

/// `load` from a reader function, called until it returns nil or an empty
/// string. An error the reader raises becomes `load`'s `(nil, message)`.
/// Unlike Lua, which parses as it reads, the whole chunk is read first.
fn load_reader<'gc>(
    ctx: Context<'gc>,
    reader: Function<'gc>,
    name: LuaString<'gc>,
) -> CallbackAction<'gc> {
    let seq = async_sequence(ctx.mutation(), |locals, mut seq| {
        let mc = ctx.mutation();
        let reader = locals.stash(mc, reader);
        let name = locals.stash(mc, Value::string(name));
        async move {
            let mut source = Vec::new();
            loop {
                let bottom = seq.enter(|_ctx, _locals, _exec, stack| stack.len());
                if let Err(e) = seq.call(&reader, bottom).await {
                    seq.enter(|ctx, locals, _exec, mut stack| {
                        let e = locals.fetch(ctx.mutation(), &e);
                        stack.replace(&[Value::nil(), e.value()]);
                    });
                    return Ok(SequenceReturn::Return);
                }
                // `Some(done)`, or `None` for a piece that isn't a string.
                let piece = seq.enter(|ctx, _locals, _exec, mut stack| {
                    let v = stack.get(bottom);
                    stack.truncate(bottom);
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
                        seq.enter(|ctx, _locals, _exec, mut stack| {
                            // `luaL_error`, located at `load`'s caller.
                            let mut msg = where_prefix(stack.thread_mut(), 1);
                            msg.extend_from_slice(b"reader function must return a string");
                            let msg = LuaString::new(ctx, &msg);
                            stack.replace(&[Value::nil(), Value::string(msg)]);
                        });
                        return Ok(SequenceReturn::Return);
                    }
                }
            }
            seq.enter(|ctx, locals, _exec, mut stack| {
                let mc = ctx.mutation();
                let name = locals
                    .fetch(mc, &name)
                    .get_string()
                    .expect("stashed a string");
                let env = env_arg(ctx, &stack, 3);
                let loaded = ctx.load_bytes(&source, name, env);
                push_loaded(ctx, &mut stack, loaded);
            });
            Ok(SequenceReturn::Return)
        }
    });
    CallbackAction::sequence(seq)
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
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let fname = util::opt_string(ctx, stack.get(0), "loadfile", 1)?;
    check_mode(ctx, stack.get(1), "loadfile", 2)?;
    let env = env_arg(ctx, &stack, 2);
    let path = fname.map(|f| f.as_bytes());
    let loaded = ctx.load_file_with(path, env);
    push_loaded(ctx, &mut stack, loaded);
    Ok(CallbackAction::Return)
}

/// `next(t [, k])` — `(k', t[k'])` for the entry after `k` in traversal
/// order, or a lone `nil` at the end.
pub(crate) fn lua_next<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let Some(t) = stack.get(0).get_table() else {
        return Err(util::type_error(ctx, "next", 1, "table", stack.arg(0)));
    };
    match t.next(ctx.mutation(), stack.get(1)) {
        Ok(Some((k, v))) => stack.replace(&[k, v]),
        Ok(None) => stack.replace(&[Value::nil()]),
        Err(_) => return Err(Error::from_str(ctx, "invalid key to 'next'")),
    }
    Ok(CallbackAction::Return)
}

/// `pairs(t)` — `(next, t, nil, nil)` with `next` from upvalue 0, or the
/// four values `__pairs(t)` returns when the metatable defines it. Like the
/// reference, the argument is only checked for presence; a non-table
/// without `__pairs` fails later in `next`.
fn lua_pairs<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    util::check_any(ctx, &stack, "pairs", 1)?;
    let t = stack.get(0);
    let mm = ctx.mm_of(t, MetamethodBits::PAIRS);
    if mm.is_nil() {
        stack.replace(&[closure.upvalues[0], t, Value::nil(), Value::nil()]);
        return Ok(CallbackAction::Return);
    }
    stack.replace(&[mm, t]);
    let then = BoxSequence::new(ctx.mutation(), util::AdjustResults(4));
    Ok(CallbackAction::call(Some(then)))
}

/// `pcall(f, ...)`: run `f` under a [`ProtectedCall`] completion that turns
/// its results into `(true, ...)` and a caught error into `(false, err)`.
/// The callee and its arguments are already in `Call` layout; a
/// non-callable `f` is raised by the executor inside the protected call,
/// so it comes back as `(false, msg)` like the reference.
fn lua_pcall<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    util::check_any(ctx, &stack, "pcall", 1)?;
    let then = BoxSequence::new(ctx.mutation(), ProtectedCall { handler: None });
    Ok(CallbackAction::call(Some(then)))
}

/// Completion sequence for `pcall` and `xpcall`: the call's results come back
/// prefixed with `true`; an error that unwinds to it becomes `(false, err)`,
/// after the `xpcall` `handler` (if any) has run.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) struct ProtectedCall<'gc> {
    pub(crate) handler: Option<Function<'gc>>,
}

impl<'gc> Sequence<'gc> for ProtectedCall<'gc> {
    fn trace_pointers(&self, cc: &mut dyn Trace<'gc>) {
        seq_trace_pointers!(self, cc);
    }

    fn poll(
        self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        stack.insert(0, Value::boolean(true));
        Ok(SequencePoll::Return)
    }

    fn error(
        self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc>,
        err: Error<'gc>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        stack.replace(&[Value::boolean(false), err.value()]);
        Ok(SequencePoll::Return)
    }

    fn catch(&self) -> Catch<'gc> {
        Catch::Here(self.handler)
    }
}

/// `print(...)` — write each argument's `tostring` form to stdout, separated
/// by tabs and followed by a newline.
fn lua_print<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
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
        return Ok(CallbackAction::Return);
    }
    // Some `__tostring` must run; like Lua, write each argument as soon as it
    // is converted.
    let seq = async_sequence(ctx.mutation(), move |_locals, seq| async move {
        let mut seq = seq;
        for i in 0..n {
            let bytes = util::tolstring(&mut seq, i, n).await?;
            let mut out = std::io::stdout().lock();
            if i > 0 {
                let _ = out.write_all(b"\t");
            }
            let _ = out.write_all(&bytes);
        }
        let _ = std::io::stdout().write_all(b"\n");
        seq.enter(|_ctx, _locals, _exec, mut stack| stack.replace(&[]));
        Ok(SequenceReturn::Return)
    });
    Ok(CallbackAction::sequence(seq))
}

/// `rawequal(a, b)` — primitive equality, bypassing `__eq`.
fn lua_rawequal<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    util::check_any(ctx, &stack, "rawequal", 1)?;
    util::check_any(ctx, &stack, "rawequal", 2)?;
    let eq = util::raw_eq(stack.get(0), stack.get(1));
    stack.ret1(Value::boolean(eq));
    Ok(CallbackAction::Return)
}

fn lua_rawget<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let t_arg = stack.get(0);
    let key = stack.get(1);
    let Some(t) = t_arg.get_table() else {
        return Err(util::type_error(ctx, "rawget", 1, "table", stack.arg(0)));
    };
    util::check_any(ctx, &stack, "rawget", 2)?;
    let v = t.raw_get(key);
    stack.ret1(v);
    Ok(CallbackAction::Return)
}

/// `rawlen(v)` — length of a table (border) or string (byte count), bypassing
/// `__len`.
fn lua_rawlen<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
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
    Ok(CallbackAction::Return)
}

fn lua_rawset<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
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
    Ok(CallbackAction::Return)
}

/// `select('#', ...)` returns the count of extra arguments; `select(n, ...)`
/// returns the arguments from position `n` onward (negative `n` counts from
/// the end).
fn lua_select<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let m = stack.len().saturating_sub(1); // count of arguments after the selector
    let sel = stack.get(0);
    if let Some(s) = sel.get_string()
        && s.as_bytes() == b"#"
    {
        stack.ret1(Value::integer(ctx.mutation(), m as i64));
        return Ok(CallbackAction::Return);
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
    Ok(CallbackAction::Return)
}

/// `setmetatable(t, mt)` — attach `mt` (a table or nil) as `t`'s
/// metatable, returning `t`. Drives a shape transition along the
/// `set_metatable` edge so future accesses observe the new identity.
fn lua_setmetatable<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
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
    Ok(CallbackAction::Return)
}

fn lua_tonumber<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
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
    Ok(CallbackAction::Return)
}

/// `tostring(v)` — `luaL_tolstring`: `__tostring`'s result, else the default
/// representation (with a string `__name` in place of the type).
fn lua_tostring<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    util::check_any(ctx, &stack, "tostring", 1)?;
    let v = stack.get(0);
    let mm = ctx.mm_of(v, MetamethodBits::TOSTRING);
    if mm.is_nil() {
        stack.ret1(Value::string(util::basic_tostring(ctx, v)));
        return Ok(CallbackAction::Return);
    }
    stack.replace(&[mm, v]);
    let then = BoxSequence::new(ctx.mutation(), util::ToStringResult);
    Ok(CallbackAction::call(Some(then)))
}

/// `type(v)` — the type name of `v` as a string.
fn lua_type<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    util::check_any(ctx, &stack, "type", 1)?;
    let name = stack.get(0).type_name();
    stack.replace(&[Value::string(LuaString::new(ctx, name.as_bytes()))]);
    Ok(CallbackAction::Return)
}

/// `warn(msg, ...)` — only checks its arguments (strings or numbers, as
/// `luaL_checkstring`); there is no warning system yet (#227).
fn lua_warn<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    if stack.is_empty() {
        return Err(util::type_error(ctx, "warn", 1, "string", None));
    }
    for i in 0..stack.len() {
        util::check_string(ctx, stack.get(i), "warn", i + 1)?;
    }
    stack.clear();
    Ok(CallbackAction::Return)
}

/// `xpcall(f, msgh, ...)`: like `pcall`, but the executor calls `msgh` with
/// the error value before unwinding (see `Catch::Here`).
fn lua_xpcall<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let Some(handler) = stack.get(1).get_function() else {
        return Err(util::type_error(ctx, "xpcall", 2, "function", stack.arg(1)));
    };
    // Drop the handler slot so the callee and its args sit in `Call` layout.
    stack.remove(1);
    let then = BoxSequence::new(
        ctx.mutation(),
        ProtectedCall {
            handler: Some(handler),
        },
    );
    Ok(CallbackAction::call(Some(then)))
}
