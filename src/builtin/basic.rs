use std::io::Write;
use std::pin::Pin;

use crate::Context;
use crate::builtin::util;
use crate::dmm::{Collect, Trace};
use crate::env::{Error, Function, LuaString, NativeClosure, NativeFn, Stack, Value};
use crate::lua::StashedValue;
use crate::vm::async_sequence::{SequenceReturn, async_sequence};
use crate::vm::sequence::{
    BoxSequence, CallbackAction, Catch, Execution, Sequence, SequencePoll, seq_trace_pointers,
};

// TODO(#27): _G, _VERSION

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("assert", lua_assert),
        ("collectgarbage", lua_collectgarbage),
        ("dofile", lua_dofile),
        ("error", lua_error),
        ("getmetatable", lua_getmetatable),
        ("ipairs", lua_ipairs),
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
        set(
            name,
            Function::new_native(ctx.mutation(), handler, Box::new([])),
        );
    }
    // `pairs` hands back the same `next` the global holds, so `pairs(t) == next`.
    let next = Function::new_native(ctx.mutation(), lua_next, Box::new([]));
    set("next", next);
    set(
        "pairs",
        Function::new_native(ctx.mutation(), lua_pairs, Box::new([Value::function(next)])),
    );
}

/// `assert(v [, message, ...])` — if `v` is truthy, return all arguments
/// unchanged; otherwise raise `message` (default `"assertion failed!"`) as
/// `error(message)` would, i.e. with the caller's position when it's a string.
fn lua_assert<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    if stack.is_empty() {
        return Err(Error::from_str(
            ctx,
            "bad argument #1 to 'assert' (value expected)",
        ));
    }
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

fn lua_dofile<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
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
    if stack.is_empty() {
        return Err(Error::from_str(
            ctx,
            "bad argument #1 to 'getmetatable' (value expected)",
        ));
    }
    // A `__metatable` field shadows the real metatable (protection).
    let result = match ctx.metatable_of(stack.get(0)) {
        Some(mt) => {
            let prot = mt.raw_get(Value::string(LuaString::new(ctx, b"__metatable")));
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

/// `ipairs(t)` — returns `(iterator, t, 0)`. The iterator yields `1,t[1]`,
/// `2,t[2]`, … stopping at the first absent index.
///
/// Indexing is raw (no `__index`); Lua 5.3+ routes `ipairs` through
/// metamethod-aware `geti`, but that requires invoking `__index`, which can
/// re-enter Lua. TODO(#27): honor `__index` once native→Lua calls are wired.
fn lua_ipairs<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    if stack.is_empty() {
        return Err(Error::from_str(
            ctx,
            "bad argument #1 to 'ipairs' (value expected)",
        ));
    }
    let t = stack.get(0);
    let iter = Function::new_native(ctx.mutation(), ipairs_aux, Box::new([]));
    stack.replace(&[Value::function(iter), t, Value::integer(ctx.mutation(), 0)]);
    Ok(CallbackAction::Return)
}

/// Stateless iterator body for `ipairs`: `(t, i) -> (i+1, t[i+1])`, or a lone
/// `nil` once the array part ends.
fn ipairs_aux<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let t = stack.get(0).get_table().ok_or_else(|| {
        Error::from_str(ctx, "bad argument #1 to 'ipairs iterator' (table expected)")
    })?;
    let i = stack.get(1).get_integer().unwrap_or(0) + 1;
    let v = t.raw_get(Value::integer(ctx.mutation(), i));
    if v.is_nil() {
        stack.ret1(Value::nil());
    } else {
        stack.replace(&[Value::integer(ctx.mutation(), i), v]);
    }
    Ok(CallbackAction::Return)
}

fn lua_load<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_loadfile<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

/// `next(t [, k])` — `(k', t[k'])` for the entry after `k` in traversal
/// order, or a lone `nil` at the end.
fn lua_next<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let Some(t) = stack.get(0).get_table() else {
        let got = (!stack.is_empty()).then(|| stack.get(0));
        return Err(util::type_error(ctx, "next", 1, "table", got));
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
    if stack.is_empty() {
        return Err(Error::from_str(
            ctx,
            "bad argument #1 to 'pairs' (value expected)",
        ));
    }
    let t = stack.get(0);
    let mm = ctx.metamethod_of(t, LuaString::new(ctx, b"__pairs"));
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
    if stack.is_empty() {
        return Err(Error::from_str(
            ctx,
            "bad argument #1 to 'pcall' (value expected)",
        ));
    }
    let then = BoxSequence::new(ctx.mutation(), ProtectedCall { handler: None });
    Ok(CallbackAction::call(Some(then)))
}

/// Completion sequence for `pcall`, `xpcall`, and `coroutine.resume`: the
/// call's results come back prefixed with `true`; an error that unwinds to
/// it becomes `(false, err)`, after the `xpcall` `handler` (if any) has run.
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
    let has_tostring = |i| {
        !ctx.metamethod_of(stack.get(i), ctx.symbols().mm_tostring)
            .is_nil()
    };
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
    enum Piece {
        Text(Vec<u8>),
        ToString(StashedValue, StashedValue),
    }
    // Some `__tostring` must run; like Lua, write each argument as soon as it
    // is converted.
    let seq = async_sequence(ctx.mutation(), move |_locals, seq| async move {
        let mut seq = seq;
        for i in 0..n {
            let piece = seq.enter(|ctx, locals, _exec, stack| {
                let v = stack.get(i);
                let mm = ctx.metamethod_of(v, ctx.symbols().mm_tostring);
                if mm.is_nil() {
                    Piece::Text(util::basic_tostring(ctx, v).as_bytes().to_vec())
                } else {
                    let mc = ctx.mutation();
                    Piece::ToString(locals.stash(mc, mm), locals.stash(mc, v))
                }
            });
            let bytes = match piece {
                Piece::Text(bytes) => bytes,
                Piece::ToString(mm, v) => util::call_tostring(&mut seq, &mm, &v, n).await?,
            };
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
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
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
        let got = (!stack.is_empty()).then_some(t_arg);
        return Err(util::type_error(ctx, "rawget", 1, "table", got));
    };
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
        let got = (!stack.is_empty()).then_some(v);
        return Err(util::type_error(ctx, "rawlen", 1, "table or string", got));
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
        let got = (!stack.is_empty()).then_some(t_arg);
        return Err(util::type_error(ctx, "rawset", 1, "table", got));
    };
    // `luaH_set` raises from inside the C function, so unlike argument
    // errors these carry no position.
    if key.is_nil() {
        return Err(Error::from_str(ctx, "table index is nil").with_level(0));
    }
    if key.get_float().is_some_and(f64::is_nan) {
        return Err(Error::from_str(ctx, "table index is NaN").with_level(0));
    }
    t.raw_set(ctx, key, value);
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
        return Err(Error::from_str(
            ctx,
            "bad argument #1 to 'select' (index out of range)",
        ));
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
    let t_arg = stack.get(0);
    let mt_arg = stack.get(1);
    let Some(t) = t_arg.get_table() else {
        let got = (!stack.is_empty()).then_some(t_arg);
        return Err(util::type_error(ctx, "setmetatable", 1, "table", got));
    };
    let mt = match mt_arg.get_table() {
        Some(mt) => Some(mt),
        None if mt_arg.is_nil() && stack.len() >= 2 => None,
        None => {
            let got = (stack.len() >= 2).then_some(mt_arg);
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
    if let Some(existing) = t.metatable() {
        let lock_key = LuaString::new(ctx, b"__metatable");
        let lock_val = existing.raw_get(Value::string(lock_key));
        if !lock_val.is_nil() {
            return Err(Error::from_str(ctx, "cannot change a protected metatable"));
        }
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
            return Err(Error::from_str(
                ctx,
                "bad argument #2 to 'tonumber' (base out of range)",
            ));
        }
        util::str_to_int_base(s.as_bytes(), base as u32)
            .map_or(Value::nil(), |i| Value::integer(ctx.mutation(), i))
    } else if v.get_integer().is_some() || v.get_float().is_some() {
        v
    } else if let Some(s) = v.get_string() {
        util::str_to_number(s.as_bytes()).map_or(Value::nil(), |n| n.into_value(ctx.mutation()))
    } else {
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
    if stack.is_empty() {
        return Err(Error::from_str(
            ctx,
            "bad argument #1 to 'tostring' (value expected)",
        ));
    }
    let v = stack.get(0);
    let mm = ctx.metamethod_of(v, ctx.symbols().mm_tostring);
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
    if stack.is_empty() {
        return Err(Error::from_str(
            ctx,
            "bad argument #1 to 'type' (value expected)",
        ));
    }
    let name = stack.get(0).type_name();
    stack.replace(&[Value::string(LuaString::new(ctx, name.as_bytes()))]);
    Ok(CallbackAction::Return)
}

/// `warn(msg, ...)` — Lua's warning system defaults to off and we do not yet
/// track the on/off toggle, so this validates the arguments (as Lua does, via
/// `luaL_checkstring`, which accepts strings *and* numbers) and otherwise does
/// nothing. TODO(#27): emit to stderr and honor `@on`/`@off` control messages
/// once warning state lives in `State`.
fn lua_warn<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    if stack.is_empty() {
        return Err(util::type_error(ctx, "warn", 1, "string", None));
    }
    for i in 0..stack.len() {
        let v = stack.get(i);
        // `luaL_checkstring` coerces numbers to their string form, so integers
        // and floats are accepted; only truly non-coercible types error.
        if v.get_string().is_none() && v.get_integer().is_none() && v.get_float().is_none() {
            return Err(util::type_error(ctx, "warn", i + 1, "string", Some(v)));
        }
    }
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
        let got = (stack.len() >= 2).then(|| stack.get(1));
        return Err(util::type_error(ctx, "xpcall", 2, "function", got));
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
