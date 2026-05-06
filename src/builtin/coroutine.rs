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
/// `(true, ...)`; on error, it produces `(false, msg)`.
fn lua_resume<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let co = stack.get(0).get_thread().ok_or_else(|| {
        Error::from_str(nctx.ctx, "bad argument #1 to 'resume' (coroutine expected)")
    })?;
    // Drop the thread-handle slot — args to pass start at index 1.
    let args: Vec<Value<'gc>> = stack.as_slice()[1..].to_vec();
    stack.replace(&args);
    let then = BoxSequence::new(nctx.ctx.mutation(), PCallSequence);
    Ok(CallbackAction::Resume {
        thread: co,
        then: Some(then),
    })
}

/// `coroutine.yield(...)` — yield values to the resumer; on resumption,
/// the resume-args become the return values of `yield`.
fn lua_yield<'gc>(
    _nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Ok(CallbackAction::Yield {
        to_thread: None,
        then: None,
    })
}

/// `coroutine.status(co)` — return one of `"suspended" | "normal" |
/// "running" | "dead"`. Phase 7: cross-checks against the executor's
/// thread stack via `Execution` arrives in P8; for now we report the
/// thread's local status.
fn lua_status<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let co = stack.get(0).get_thread().ok_or_else(|| {
        Error::from_str(nctx.ctx, "bad argument #1 to 'status' (coroutine expected)")
    })?;
    let s: &[u8] = match co.status() {
        ThreadStatus::Stopped | ThreadStatus::Result { .. } => b"dead",
        ThreadStatus::Suspended => b"suspended",
        ThreadStatus::Normal => b"normal",
    };
    let v = Value::string(LuaString::new(nctx.ctx, s));
    stack.replace(&[v]);
    Ok(CallbackAction::Return)
}

/// `coroutine.running()` — `(currently_running_thread, is_main_thread)`.
/// Phase 7 stub: returns `(main, true)` until the executor handle exposes
/// its thread stack via `Execution` (P8). Most code uses the boolean.
fn lua_running<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let main = nctx.ctx.main_thread();
    stack.replace(&[Value::thread(main), Value::boolean(true)]);
    Ok(CallbackAction::Return)
}

/// `coroutine.isyieldable([co])` — true iff `co` (defaults to running) is
/// not the main thread. Phase 7 stub: returns `false` (main is the only
/// thing currently exposed via `Execution`).
fn lua_isyieldable<'gc>(
    _nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    stack.replace(&[Value::boolean(false)]);
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

/// `coroutine.close(co)` — Phase 7 simplified: clear the thread's stack/
/// frames, set status to `Stopped`, return `true`. Full Lua-5.4 `__close`
/// invocation is out of scope (tracks alongside the TBC `__close` work).
fn lua_close<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let co = stack.get(0).get_thread().ok_or_else(|| {
        Error::from_str(nctx.ctx, "bad argument #1 to 'close' (coroutine expected)")
    })?;
    {
        let mc = nctx.ctx.mutation();
        let mut ts = co.borrow_mut(mc);
        ts.stack.clear();
        ts.frames.clear();
        ts.open_upvalues.clear();
        ts.tbc_slots.clear();
        ts.pending_action = None;
        ts.status = ThreadStatus::Stopped;
    }
    stack.replace(&[Value::boolean(true)]);
    Ok(CallbackAction::Return)
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
