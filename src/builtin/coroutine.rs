use crate::Context;
use crate::builtin::util;
use crate::env::function::NativeKind;
use crate::env::thread::ThreadStatus;
use crate::env::{
    Error, Function, LuaString, NativeClosure, NativeFn, Stack, Table, Thread, Value,
};
use crate::vm::native::{Execution, NativeOut, OnOk, cont};
use crate::vm::{close, coro};

pub fn load<'gc>(ctx: Context<'gc>) {
    let lib = Table::new(ctx);
    let set = |name: &str, f: Function<'gc>| {
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(f));
    };
    let fns: &[(&str, NativeFn)] = &[
        ("create", lua_create),
        ("isyieldable", lua_isyieldable),
        ("running", lua_running),
        ("status", lua_status),
        ("wrap", lua_wrap),
    ];
    for &(name, handler) in fns {
        set(name, Function::new_native(ctx.mutation(), handler, &[]));
    }
    let mc = ctx.mutation();
    set("close", Function::new_cont(mc, lua_close, &[]));
    let resume = NativeKind::Cont(lua_resume);
    set(
        "resume",
        Function::new_native_with_entry(mc, resume, &[], coro::ff_resume),
    );
    let yield_ = NativeKind::Cont(lua_yield);
    set(
        "yield",
        Function::new_native_with_entry(mc, yield_, &[], coro::ff_yield),
    );

    let lib_name = Value::string(LuaString::new(ctx, b"coroutine"));
    ctx.globals().raw_set(ctx, lib_name, Value::table(lib));
}

/// `coroutine.create(f)` — allocate a fresh `Thread`, prime it with a
/// `ExecKind::Start(f)`, return it.
fn lua_create<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let f = stack
        .get(0)
        .get_function()
        .ok_or_else(|| util::type_error(ctx, "create", 1, "function", stack.arg(0)))?;
    let thread = Thread::new(ctx.mutation());
    thread.borrow_mut(ctx.mutation()).seed(Value::function(f));
    stack.ret1(Value::thread(thread));
    Ok(())
}

/// `coroutine.resume(co, ...)` — switch to `co`, passing the rest as args.
/// On `co` yielding/returning, [`resume_cont`] makes the values `(true, ...)`;
/// on error, `(false, msg)`. If `co` isn't resumable (dead, currently
/// running, on the resume stack as a parent, or the main thread) we return
/// `(false, msg)` directly per the manual.
fn lua_resume<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let co = stack
        .get(0)
        .get_thread()
        .ok_or_else(|| util::type_error(ctx, "resume", 1, "thread", stack.arg(0)))?;
    if let Some(msg) = unresumable_reason(stack.exec(), co, stack.len() - 1) {
        let m = Value::string(LuaString::new(ctx, msg.as_bytes()));
        stack.replace(&[Value::boolean(false), m]);
        return NativeOut::RETURN;
    }
    NativeOut::resume(0, cont::RESUME, OnOk::ReturnTrue)
}

/// `auxresume`'s ending: `(true, ...)`, or `(false, msg)` for an error or
/// values with no room for the leading `true`.
pub(crate) fn resume_cont<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> NativeOut {
    match status {
        Ok(()) if stack.check_stack(1) => stack.insert(0, Value::boolean(true)),
        Ok(()) => {
            let m = Value::string(LuaString::new(ctx, b"too many results to resume"));
            stack.replace(&[Value::boolean(false), m]);
        }
        Err(err) => stack.replace(&[Value::boolean(false), err.value()]),
    }
    NativeOut::RETURN
}

/// `None` if `co` can be resumed with `nargs` arguments, else the Lua-spec
/// error message that `(false, msg)` should carry. Covers main thread, dead,
/// any non-suspended status (which subsumes `running` and `normal`), and
/// arguments that don't fit on `co`'s stack (`auxresume`).
///
/// Pointer-eq checks against `current_thread` come first because the
/// running thread's `RefLock` is already mutably borrowed by the
/// interpreter — calling `co.peer_status()` on it would re-borrow and panic.
fn unresumable_reason<'gc>(
    exec: Execution<'gc>,
    co: Thread<'gc>,
    nargs: usize,
) -> Option<&'static str> {
    if co.ptr_eq(exec.current_thread()) {
        return Some("cannot resume non-suspended coroutine");
    }
    match co.peer_status() {
        ThreadStatus::Suspended => {
            let ts = co.borrow();
            // The args land where `co` yielded, or above the header of the
            // function in slot 0 on a first resume.
            let bottom = ts.yield_bottom.unwrap_or(4);
            (nargs > ts.stack_limit.saturating_sub(bottom))
                .then_some("too many arguments to resume")
        }
        ThreadStatus::Result { .. } | ThreadStatus::Stopped => Some("cannot resume dead coroutine"),
        ThreadStatus::Normal => Some("cannot resume non-suspended coroutine"),
    }
}

/// `coroutine.yield(...)` — yield values to the resumer; on resumption,
/// the resume-args become the return values of `yield`.
fn lua_yield<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> NativeOut {
    NativeOut::YIELD
}

/// `coroutine.status(co)` — return one of `"suspended" | "normal" |
/// "running" | "dead"`. The currently-running thread is detected by
/// pointer-comparing `co` against `Execution::current_thread`.
fn lua_status<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let co = stack
        .get(0)
        .get_thread()
        .ok_or_else(|| util::type_error(ctx, "status", 1, "thread", stack.arg(0)))?;
    let s: &[u8] = if co.ptr_eq(stack.exec().current_thread()) {
        b"running"
    } else {
        match co.peer_status() {
            ThreadStatus::Stopped | ThreadStatus::Result { .. } => b"dead",
            ThreadStatus::Suspended => b"suspended",
            ThreadStatus::Normal => b"normal",
        }
    };
    let v = Value::string(LuaString::new(ctx, s));
    stack.ret1(v);
    Ok(())
}

/// `coroutine.running()` — `(currently_running_thread, is_main_thread)`.
fn lua_running<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let cur = stack.exec().current_thread();
    let is_main = stack.exec().is_main();
    stack.replace(&[Value::thread(cur), Value::boolean(is_main)]);
    Ok(())
}

/// `getoptco`: argument 1, or the running coroutine only when it's absent (a
/// nil is a bad argument).
fn opt_co<'gc>(
    ctx: Context<'gc>,
    stack: &Stack<'gc, '_>,
    fname: &str,
) -> Result<Thread<'gc>, Error<'gc>> {
    match stack.arg(0) {
        None => Ok(stack.exec().current_thread()),
        Some(v) => v
            .get_thread()
            .ok_or_else(|| util::type_error(ctx, fname, 1, "thread", Some(v))),
    }
}

/// `coroutine.isyieldable([co])` — true iff `co` (defaults to running) is
/// not the main thread, nor closing its variables for `coroutine.close`.
fn lua_isyieldable<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let co = opt_co(ctx, &stack, "isyieldable")?;
    // The running thread's lock is held by the interpreter.
    let yieldable = if co.ptr_eq(stack.exec().current_thread()) {
        let ts = stack.thread_mut();
        !ts.main && !ts.no_yield
    } else {
        let ts = co.borrow();
        !ts.main && !ts.no_yield
    };
    stack.ret1(Value::boolean(yieldable));
    Ok(())
}

/// `coroutine.wrap(f)` — return a callable that calls `coroutine.resume`
/// on a freshly-created thread; errors propagate (rather than being
/// caught as in `resume`).
fn lua_wrap<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let f = stack
        .get(0)
        .get_function()
        .ok_or_else(|| util::type_error(ctx, "wrap", 1, "function", stack.arg(0)))?;
    let thread = Thread::new(ctx.mutation());
    thread.borrow_mut(ctx.mutation()).seed(Value::function(f));
    let wrapper = Function::new_native_with_entry(
        ctx.mutation(),
        NativeKind::Cont(wrap_callback),
        &[Value::thread(thread)],
        coro::ff_wrap,
    );
    stack.ret1(Value::function(wrapper));
    Ok(())
}

/// `coroutine.close([co])` — close a suspended or dead coroutine's pending
/// to-be-closed variables on `co` itself, then `true`, or `false` and the
/// error it died with or a `__close` raised. The running coroutine (also
/// the default) closes its variables and ends, without returning.
fn lua_close<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let co = opt_co(ctx, &stack, "close")?;
    // Pointer-eq against current first to avoid re-borrowing the running
    // thread's RefLock (mut-borrowed by the interpreter).
    if co.ptr_eq(stack.exec().current_thread()) {
        if stack.exec().is_main() {
            return NativeOut::error(Error::from_str(ctx, "cannot close main thread"));
        }
        return close::close_running(ctx, stack);
    }
    match co.peer_status() {
        ThreadStatus::Suspended | ThreadStatus::Stopped | ThreadStatus::Result { .. } => {
            let mut ts = co.borrow_mut(ctx.mutation());
            if close::seed_thread_close(ctx, &mut ts) {
                // The coroutine's own close runner returns the results.
                stack.replace(&[Value::thread(co)]);
                return NativeOut::resume(0, cont::CORO_CLOSE, OnOk::Return);
            }
            // Surfaced once, so a second close is `true`, as in Lua.
            match ts.death_error.take() {
                Some(err) => stack.replace(&[Value::boolean(false), err]),
                None => stack.replace(&[Value::boolean(true)]),
            }
            NativeOut::RETURN
        }
        ThreadStatus::Normal => {
            NativeOut::error(Error::from_str(ctx, "cannot close a normal coroutine"))
        }
    }
}

/// A close runner that ended without returning (a process exit).
pub(crate) fn close_cont<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> NativeOut {
    if let Err(err) = status {
        stack.replace(&[Value::boolean(false), err.value()]);
    }
    NativeOut::RETURN
}

/// Body of the closure returned by `coroutine.wrap`. Upvalue 0 carries the
/// thread; we resume it and unwrap the success-prefix from the resume
/// protocol (errors rethrow rather than getting wrapped, matching Lua).
fn wrap_callback<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let co = closure.upvalues()[0]
        .get_thread()
        .expect("wrap_callback upvalue 0 must be a thread");
    // Gate the resume like `lua_resume` does; without this, resuming a dead
    // (or otherwise non-suspended) thread reaches `schedule_thread_resume`
    // and aborts the whole executor with `BadMode`. `wrap` re-raises errors
    // rather than wrapping them, so we throw the reason directly.
    if let Some(msg) = unresumable_reason(stack.exec(), co, stack.len()) {
        return NativeOut::error(Error::from_str(ctx, msg));
    }
    stack.insert(0, Value::thread(co));
    NativeOut::resume(0, cont::WRAP, OnOk::Return)
}

/// `auxwrap`'s ending: the coroutine's values verbatim; an error rethrown
/// once the dead coroutine has closed its variables (`lua_closethread`).
pub(crate) fn wrap_cont<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> NativeOut {
    let err = match status {
        Ok(()) if stack.check_stack(1) => return NativeOut::RETURN,
        Ok(()) => return NativeOut::error(Error::from_str(ctx, "too many results to resume")),
        Err(err) => err,
    };
    let co = closure.upvalues()[0].get_thread().expect("wrap's thread");
    if co.borrow().tbc_list.is_empty() {
        // `auxwrap` re-raises a string error with the wrap caller's position
        // prepended on top of the coroutine's own.
        return NativeOut::error(err.with_level(1));
    }
    close::seed_thread_close(ctx, &mut co.borrow_mut(ctx.mutation()));
    stack.replace(&[Value::thread(co)]);
    NativeOut::resume(0, cont::WRAP_CLOSE, OnOk::Cont)
}

/// [`wrap_cont`] once the coroutine closed its variables: rethrow its
/// result's error, or an error a `__close` raised.
pub(crate) fn wrap_close_cont<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> NativeOut {
    match status {
        Ok(()) => NativeOut::error(Error::new(ctx, stack.get(1)).with_level(1)),
        Err(err) => NativeOut::error(err.with_level(1)),
    }
}
