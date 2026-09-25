use std::pin::Pin;

use crate::Context;
use crate::builtin::basic::ProtectedCall;
use crate::builtin::util;
use crate::dmm::{Collect, Trace};
use crate::env::thread::{ExecKind, ThreadStatus};
use crate::env::{
    Error, Function, LuaString, NativeClosure, NativeFn, Stack, Table, Thread, Value,
};
use crate::vm::close;
use crate::vm::sequence::{
    BoxSequence, CallbackAction, Catch, Execution, Sequence, SequencePoll, seq_trace_pointers,
};

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("close", lua_close),
        ("create", lua_create),
        ("isyieldable", lua_isyieldable),
        ("resume", lua_resume),
        ("running", lua_running),
        ("status", lua_status),
        ("wrap", lua_wrap),
        ("yield", lua_yield),
    ];

    let lib = Table::new(ctx);
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, Box::new([]));
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(handler));
    }

    let lib_name = Value::string(LuaString::new(ctx, b"coroutine"));
    ctx.globals().raw_set(ctx, lib_name, Value::table(lib));
}

/// `coroutine.create(f)` — allocate a fresh `Thread`, prime it with a
/// `ExecKind::Start(f)`, return it.
fn lua_create<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let f = stack
        .get(0)
        .get_function()
        .ok_or_else(|| util::type_error(ctx, "create", 1, "function", stack.arg(0)))?;
    let thread = Thread::new(ctx.mutation());
    {
        let mc = ctx.mutation();
        let mut ts = thread.borrow_mut(mc);
        ts.push_exec(ExecKind::Start(f.into()));
        ts.status = ThreadStatus::Suspended;
    }
    stack.ret1(Value::thread(thread));
    Ok(CallbackAction::Return)
}

/// `coroutine.resume(co, ...)` — switch to `co`, passing the rest as args.
/// On `co` yielding/returning, the [`ProtectedCall`] wraps the values as
/// `(true, ...)`; on error, it produces `(false, msg)`. If `co` isn't
/// resumable (dead, currently running, on the resume stack as a parent, or
/// the main thread) we return `(false, msg)` directly per the manual
/// instead of routing through the executor.
fn lua_resume<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let co = stack
        .get(0)
        .get_thread()
        .ok_or_else(|| util::type_error(ctx, "resume", 1, "thread", stack.arg(0)))?;
    if let Some(msg) = unresumable_reason(ctx, stack.exec(), co) {
        let m = Value::string(LuaString::new(ctx, msg.as_bytes()));
        stack.replace(&[Value::boolean(false), m]);
        return Ok(CallbackAction::Return);
    }
    // Drop the thread-handle slot so the resume args start at index 0.
    stack.remove(0);
    let then = BoxSequence::new(ctx.mutation(), ProtectedCall { handler: None });
    Ok(CallbackAction::resume(co, Some(then)))
}

/// `None` if `co` can be resumed, else the Lua-spec error message that
/// `(false, msg)` should carry. Covers main thread, dead, and any
/// non-suspended status (which subsumes `running` and `normal`).
///
/// Pointer-eq checks against `current_thread` come first because the
/// running thread's `RefLock` is already mutably borrowed by the
/// interpreter — calling `co.status()` on it would re-borrow and panic.
fn unresumable_reason<'gc>(
    ctx: Context<'gc>,
    exec: Execution<'gc>,
    co: Thread<'gc>,
) -> Option<&'static str> {
    if co.ptr_eq(ctx.main_thread()) || co.ptr_eq(exec.current_thread()) {
        return Some("cannot resume non-suspended coroutine");
    }
    match co.status() {
        ThreadStatus::Suspended => None,
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
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Ok(CallbackAction::yield_(None))
}

/// `coroutine.status(co)` — return one of `"suspended" | "normal" |
/// "running" | "dead"`. The currently-running thread is detected by
/// pointer-comparing `co` against `Execution::current_thread`.
fn lua_status<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let co = stack
        .get(0)
        .get_thread()
        .ok_or_else(|| util::type_error(ctx, "status", 1, "thread", stack.arg(0)))?;
    let s: &[u8] = if co.ptr_eq(stack.exec().current_thread()) {
        b"running"
    } else {
        match co.status() {
            ThreadStatus::Stopped | ThreadStatus::Result { .. } => b"dead",
            ThreadStatus::Suspended => b"suspended",
            ThreadStatus::Normal => b"normal",
        }
    };
    let v = Value::string(LuaString::new(ctx, s));
    stack.ret1(v);
    Ok(CallbackAction::Return)
}

/// `coroutine.running()` — `(currently_running_thread, is_main_thread)`.
fn lua_running<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let cur = stack.exec().current_thread();
    let is_main = stack.exec().is_main(ctx);
    stack.replace(&[Value::thread(cur), Value::boolean(is_main)]);
    Ok(CallbackAction::Return)
}

/// `coroutine.isyieldable([co])` — true iff `co` (defaults to running) is
/// not the main thread, nor closing its variables for `coroutine.close`.
fn lua_isyieldable<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let arg = stack.get(0);
    let yieldable = if arg.is_nil() {
        !stack.exec().is_main(ctx) && !stack.thread_mut().no_yield
    } else {
        let target = arg
            .get_thread()
            .ok_or_else(|| util::type_error(ctx, "isyieldable", 1, "thread", Some(arg)))?;
        // The running thread's lock is held by the interpreter.
        let no_yield = if target.ptr_eq(stack.exec().current_thread()) {
            stack.thread_mut().no_yield
        } else {
            target.borrow().no_yield
        };
        !target.ptr_eq(ctx.main_thread()) && !no_yield
    };
    stack.ret1(Value::boolean(yieldable));
    Ok(CallbackAction::Return)
}

/// `coroutine.wrap(f)` — return a callable that calls `coroutine.resume`
/// on a freshly-created thread; errors propagate (rather than being
/// caught as in `resume`).
fn lua_wrap<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let f = stack
        .get(0)
        .get_function()
        .ok_or_else(|| util::type_error(ctx, "wrap", 1, "function", stack.arg(0)))?;
    let thread = Thread::new(ctx.mutation());
    {
        let mc = ctx.mutation();
        let mut ts = thread.borrow_mut(mc);
        ts.push_exec(ExecKind::Start(f.into()));
        ts.status = ThreadStatus::Suspended;
    }
    let upvalues: Box<[Value<'gc>]> = Box::new([Value::thread(thread)]);
    let wrapper = Function::new_native(ctx.mutation(), wrap_callback as NativeFn, upvalues);
    stack.ret1(Value::function(wrapper));
    Ok(CallbackAction::Return)
}

/// `coroutine.close(co)` — close a suspended or dead coroutine's pending
/// to-be-closed variables on `co` itself, then `true`, or `false` and the
/// error it died with or a `__close` raised. The running coroutine closes
/// its variables and ends, without returning.
fn lua_close<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let co = stack
        .get(0)
        .get_thread()
        .ok_or_else(|| util::type_error(ctx, "close", 1, "thread", stack.arg(0)))?;
    // Pointer-eq against current first to avoid re-borrowing the running
    // thread's RefLock (mut-borrowed by the interpreter).
    if co.ptr_eq(stack.exec().current_thread()) {
        if stack.exec().is_main(ctx) {
            return Err(plain_error(ctx, "cannot close main thread"));
        }
        return Ok(close::close_running(ctx));
    }
    match co.status() {
        ThreadStatus::Suspended | ThreadStatus::Stopped | ThreadStatus::Result { .. } => {
            let mut ts = co.borrow_mut(ctx.mutation());
            if close::seed_thread_close(ctx, &mut ts) {
                stack.clear();
                return Ok(CallbackAction::resume(co, None));
            }
            // Surfaced once, so a second close is `true`, as in Lua.
            match ts.death_error.take() {
                Some(err) => stack.replace(&[Value::boolean(false), err]),
                None => stack.replace(&[Value::boolean(true)]),
            }
            Ok(CallbackAction::Return)
        }
        ThreadStatus::Normal => Err(plain_error(ctx, "cannot close a normal coroutine")),
    }
}

/// An error without a position, as `luaL_error` raises from a C function.
fn plain_error<'gc>(ctx: Context<'gc>, msg: &str) -> Error<'gc> {
    Error::new(ctx, Value::string(LuaString::new(ctx, msg.as_bytes())))
}

/// Body of the closure returned by `coroutine.wrap`. Upvalue 0 carries the
/// thread; we resume it and unwrap the success-prefix from the resume
/// protocol (errors rethrow rather than getting wrapped, matching Lua).
fn wrap_callback<'gc>(
    ctx: Context<'gc>,
    closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let co = closure.upvalues[0]
        .get_thread()
        .expect("wrap_callback upvalue 0 must be a thread");
    // Gate the resume like `lua_resume` does; without this, resuming a dead
    // (or otherwise non-suspended) thread reaches `schedule_thread_resume`
    // and aborts the whole executor with `BadMode`. `wrap` re-raises errors
    // rather than wrapping them, so we throw the reason directly.
    if let Some(msg) = unresumable_reason(ctx, stack.exec(), co) {
        return Err(Error::from_str(ctx, msg));
    }
    let then = BoxSequence::new(ctx.mutation(), UnwrapResumeSequence { co, closing: false });
    Ok(CallbackAction::resume(co, Some(then)))
}

// ---------------------------------------------------------------------------
// Sequences
// ---------------------------------------------------------------------------

/// `coroutine.wrap`'s follow-up sequence: returns the inner thread's
/// values verbatim on success, rethrows on error once the dead thread has
/// closed its variables (`auxwrap`'s `lua_closethread`).
#[derive(Collect)]
#[collect(internal, no_drop)]
struct UnwrapResumeSequence<'gc> {
    co: Thread<'gc>,
    /// The thread is closing its variables; its result is `false, err`.
    closing: bool,
}

impl<'gc> Sequence<'gc> for UnwrapResumeSequence<'gc> {
    fn trace_pointers(&self, cc: &mut dyn Trace<'gc>) {
        seq_trace_pointers!(self, cc);
    }

    fn poll(
        self: Pin<&mut Self>,
        ctx: Context<'gc>,
        _exec: Execution<'gc>,
        stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        if self.closing {
            return Err(Error::new(ctx, stack.get(1)).with_level(1));
        }
        // Pass through whatever the inner left on the stack.
        Ok(SequencePoll::Return)
    }

    fn error(
        mut self: Pin<&mut Self>,
        ctx: Context<'gc>,
        _exec: Execution<'gc>,
        err: Error<'gc>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        let co = self.co;
        if !self.closing && !co.borrow().tbc_list.is_empty() {
            close::seed_thread_close(ctx, &mut co.borrow_mut(ctx.mutation()));
            self.closing = true;
            stack.clear();
            return Ok(SequencePoll::Resume {
                thread: co,
                bottom: 0,
            });
        }
        // `auxwrap` re-raises a string error with the wrap caller's position
        // prepended on top of the coroutine's own.
        Err(err.with_level(1))
    }

    /// The rethrow is a fresh raise on the resumer (`auxwrap`'s
    /// `lua_error`), so an enclosing `xpcall` handler must see the
    /// re-prefixed message, not the coroutine's original.
    fn catch(&self) -> Catch<'gc> {
        Catch::Here(None)
    }
}
