use std::pin::Pin;

use crate::Context;
use crate::dmm::{Collect, Trace};
use crate::env::thread::{Frame, ThreadStatus};
use crate::env::{
    Error, Function, LuaString, NativeContext, NativeFn, Stack, Table, Thread, Value,
};
use crate::vm::sequence::{BoxSequence, CallbackAction, Execution, Sequence, SequencePoll};

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
/// `Frame::Start(f)`, return it.
fn lua_create<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let f = stack.get(0).get_function().ok_or_else(|| {
        Error::from_str(nctx.ctx, "bad argument #1 to 'create' (function expected)")
    })?;
    let thread = Thread::new(nctx.ctx.mutation());
    {
        let mc = nctx.ctx.mutation();
        let mut ts = thread.borrow_mut(mc);
        ts.frames.push(Frame::Start(f));
        ts.status = ThreadStatus::Suspended;
    }
    stack.replace(&[Value::thread(thread)]);
    Ok(CallbackAction::Return)
}

/// `coroutine.resume(co, ...)` — switch to `co`, passing the rest as args.
/// On `co` yielding/returning, the [`PCallSequence`] wraps the values as
/// `(true, ...)`; on error, it produces `(false, msg)`. If `co` isn't
/// resumable (dead, currently running, on the resume stack as a parent, or
/// the main thread) we return `(false, msg)` directly per the manual
/// instead of routing through the executor.
fn lua_resume<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let co = stack.get(0).get_thread().ok_or_else(|| {
        Error::from_str(nctx.ctx, "bad argument #1 to 'resume' (coroutine expected)")
    })?;
    if let Some(msg) = unresumable_reason(co, &nctx) {
        let m = Value::string(LuaString::new(nctx.ctx, msg.as_bytes()));
        stack.replace(&[Value::boolean(false), m]);
        return Ok(CallbackAction::Return);
    }
    // Drop the thread-handle slot — args to pass start at index 1.
    let args: Vec<Value<'gc>> = stack.as_slice()[1..].to_vec();
    stack.replace(&args);
    let then = BoxSequence::new(nctx.ctx.mutation(), PCallSequence);
    Ok(CallbackAction::Resume {
        thread: co,
        then: Some(then),
    })
}

/// `None` if `co` can be resumed, else the Lua-spec error message that
/// `(false, msg)` should carry. Covers main thread, dead, and any
/// non-suspended status (which subsumes `running` and `normal`).
///
/// Pointer-eq checks against `current_thread` come first because the
/// running thread's `RefLock` is already mutably borrowed by the
/// interpreter — calling `co.status()` on it would re-borrow and panic.
fn unresumable_reason<'gc>(co: Thread<'gc>, nctx: &NativeContext<'gc, '_>) -> Option<&'static str> {
    if co.ptr_eq(nctx.ctx.main_thread()) || co.ptr_eq(nctx.exec.current_thread()) {
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
    _nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Ok(CallbackAction::Yield { then: None })
}

/// `coroutine.status(co)` — return one of `"suspended" | "normal" |
/// "running" | "dead"`. The currently-running thread is detected by
/// pointer-comparing `co` against `Execution::current_thread`.
fn lua_status<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let co = stack.get(0).get_thread().ok_or_else(|| {
        Error::from_str(nctx.ctx, "bad argument #1 to 'status' (coroutine expected)")
    })?;
    let s: &[u8] = if co.ptr_eq(nctx.exec.current_thread()) {
        b"running"
    } else {
        match co.status() {
            ThreadStatus::Stopped | ThreadStatus::Result { .. } => b"dead",
            ThreadStatus::Suspended => b"suspended",
            ThreadStatus::Normal => b"normal",
        }
    };
    let v = Value::string(LuaString::new(nctx.ctx, s));
    stack.replace(&[v]);
    Ok(CallbackAction::Return)
}

/// `coroutine.running()` — `(currently_running_thread, is_main_thread)`.
fn lua_running<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let cur = nctx.exec.current_thread();
    let is_main = nctx.exec.is_main(nctx.ctx);
    stack.replace(&[Value::thread(cur), Value::boolean(is_main)]);
    Ok(CallbackAction::Return)
}

/// `coroutine.isyieldable([co])` — true iff `co` (defaults to running) is
/// not the main thread. (TCVM doesn't yet model non-yieldable C frames;
/// the main-thread test is the only blocker.)
fn lua_isyieldable<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let arg = stack.get(0);
    let yieldable = if arg.is_nil() {
        !nctx.exec.is_main(nctx.ctx)
    } else {
        let target = arg.get_thread().ok_or_else(|| {
            Error::from_str(
                nctx.ctx,
                "bad argument #1 to 'isyieldable' (coroutine expected)",
            )
        })?;
        !target.ptr_eq(nctx.ctx.main_thread())
    };
    stack.replace(&[Value::boolean(yieldable)]);
    Ok(CallbackAction::Return)
}

/// `coroutine.wrap(f)` — return a callable that calls `coroutine.resume`
/// on a freshly-created thread; errors propagate (rather than being
/// caught as in `resume`).
fn lua_wrap<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let f = stack.get(0).get_function().ok_or_else(|| {
        Error::from_str(nctx.ctx, "bad argument #1 to 'wrap' (function expected)")
    })?;
    let thread = Thread::new(nctx.ctx.mutation());
    {
        let mc = nctx.ctx.mutation();
        let mut ts = thread.borrow_mut(mc);
        ts.frames.push(Frame::Start(f));
        ts.status = ThreadStatus::Suspended;
    }
    let upvalues: Box<[Value<'gc>]> = Box::new([Value::thread(thread)]);
    let wrapper = Function::new_native(nctx.ctx.mutation(), wrap_callback as NativeFn, upvalues);
    stack.replace(&[Value::function(wrapper)]);
    Ok(CallbackAction::Return)
}

/// `coroutine.close(co)` — clear the thread's stack/frames, set status
/// to `Stopped`, return `true`. Per the manual, valid only for dead /
/// suspended / running coroutines; for any other status we return
/// `(nil, "cannot close a non-suspended coroutine")` instead of
/// corrupting executor invariants.
///
/// The running-self case has special "does not return" semantics in the
/// reference, which depend on `__close` machinery we haven't built yet —
/// we reject it for now via the same path. Does NOT yet invoke `__close`
/// metamethods on TBC variables; that ships with the broader TBC work.
fn lua_close<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let co = stack.get(0).get_thread().ok_or_else(|| {
        Error::from_str(nctx.ctx, "bad argument #1 to 'close' (coroutine expected)")
    })?;
    // Pointer-eq against current first to avoid re-borrowing the running
    // thread's RefLock (mut-borrowed by the interpreter).
    if co.ptr_eq(nctx.exec.current_thread()) {
        let m = Value::string(LuaString::new(
            nctx.ctx,
            b"cannot close a non-suspended coroutine",
        ));
        stack.replace(&[Value::nil(), m]);
        return Ok(CallbackAction::Return);
    }
    match co.status() {
        ThreadStatus::Suspended | ThreadStatus::Stopped | ThreadStatus::Result { .. } => {
            let mc = nctx.ctx.mutation();
            let mut ts = co.borrow_mut(mc);
            ts.stack.clear();
            ts.frames.clear();
            ts.open_upvalues.clear();
            ts.tbc_slots.clear();
            ts.pending_action = None;
            ts.yield_bottom = None;
            ts.status = ThreadStatus::Stopped;
            stack.replace(&[Value::boolean(true)]);
            Ok(CallbackAction::Return)
        }
        ThreadStatus::Normal => {
            let m = Value::string(LuaString::new(
                nctx.ctx,
                b"cannot close a non-suspended coroutine",
            ));
            stack.replace(&[Value::nil(), m]);
            Ok(CallbackAction::Return)
        }
    }
}

/// Body of the closure returned by `coroutine.wrap`. Upvalue 0 carries the
/// thread; we resume it and unwrap the success-prefix from the resume
/// protocol (errors rethrow rather than getting wrapped, matching Lua).
fn wrap_callback<'gc>(
    nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let co = nctx.upvalues[0]
        .get_thread()
        .expect("wrap_callback upvalue 0 must be a thread");
    let then = BoxSequence::new(nctx.ctx.mutation(), UnwrapResumeSequence);
    Ok(CallbackAction::Resume {
        thread: co,
        then: Some(then),
    })
}

// ---------------------------------------------------------------------------
// Sequences
// ---------------------------------------------------------------------------

/// Wraps a coroutine resume: prepends `true` to the inner thread's
/// returned/yielded values; on a thrown error, returns `(false, msg)`.
struct PCallSequence;

unsafe impl<'gc> Collect<'gc> for PCallSequence {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> Sequence<'gc> for PCallSequence {
    fn trace_pointers(&self, _cc: &mut dyn Trace<'gc>) {}

    fn poll(
        self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        // Inner thread completed/yielded; values are at stack[..]. Prepend
        // `true` and return.
        let mut vals: Vec<Value<'gc>> = Vec::with_capacity(stack.len() + 1);
        vals.push(Value::boolean(true));
        vals.extend_from_slice(stack.as_slice());
        stack.replace(&vals);
        Ok(SequencePoll::Return)
    }

    fn error(
        self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        err: Error<'gc>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        stack.replace(&[Value::boolean(false), err.value()]);
        Ok(SequencePoll::Return)
    }
}

/// `coroutine.wrap`'s follow-up sequence: returns the inner thread's
/// values verbatim on success, rethrows on error.
struct UnwrapResumeSequence;

unsafe impl<'gc> Collect<'gc> for UnwrapResumeSequence {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> Sequence<'gc> for UnwrapResumeSequence {
    fn trace_pointers(&self, _cc: &mut dyn Trace<'gc>) {}

    fn poll(
        self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        _stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        // Pass through whatever the inner left on the stack.
        Ok(SequencePoll::Return)
    }
}
