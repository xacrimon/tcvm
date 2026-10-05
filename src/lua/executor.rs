use crate::dmm::{Collect, Gc, RefLock};
use crate::env::error::Exit;
use crate::env::thread::{
    CallSite, ExecKind, LuaFrame, PendingAction, PendingKind, PendingRet, ThreadState, ThreadStatus,
};
use crate::env::{Error, LuaString, Thread, Value};
use crate::lua::RuntimeError;
use crate::lua::context::Context;
use crate::lua::convert::{FromMultiValue, IntoMultiValue};
use crate::vm;
use crate::vm::interp::CallTarget;
use crate::vm::interp::ret_exit;
use crate::vm::native::CallbackAction;

#[derive(Clone, Copy, PartialEq, Eq, Collect)]
#[collect(internal, require_static)]
pub enum ExecutorMode {
    /// Thread has no seeded call — cannot step.
    Stopped,
    /// Thread is running and can be stepped.
    Normal,
    /// Thread has returned; results are available on its stack.
    Result,
    /// Top thread yielded; values were drained on the last `step`. Feed
    /// resume args back via [`Executor::resume`] (or [`crate::Lua::resume`])
    /// to flip the executor back to `Normal` and continue.
    Yielded,
}

/// Outcome of a single `Executor::step` invocation.
pub enum StepResult<'gc> {
    /// Top thread reached terminal `Result` state. Caller may `take_result`.
    Done,
    /// Top thread yielded these values to the host. Feed resume args via
    /// [`Executor::resume`] (mode → `Normal`) and call `step` again.
    Yielded(Vec<Value<'gc>>),
    /// The host gets to run before stepping on: the collector is owed work,
    /// or an async native waits on the host. Mode stays `Normal`; call `step`
    /// again to keep going, once woken (see [`Executor::step_waker`]).
    Pending,
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) struct ExecutorInner<'gc> {
    /// The "main" thread for this executor — the entry point seeded by
    /// `start`. Always equals `thread_stack[0]`.
    pub(crate) thread: Thread<'gc>,
    /// Stack of currently-active threads. The top is the thread the driver
    /// is pumping; lower entries are `WaitThread`-suspended resumers.
    /// Coroutine `resume` pushes onto this stack.
    pub(crate) thread_stack: Vec<Thread<'gc>>,
    pub(crate) mode: ExecutorMode,
}

#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Executor<'gc>(Gc<'gc, RefLock<ExecutorInner<'gc>>>);

impl<'gc> Executor<'gc> {
    pub(crate) fn inner(self) -> Gc<'gc, RefLock<ExecutorInner<'gc>>> {
        self.0
    }

    pub(crate) fn from_inner(g: Gc<'gc, RefLock<ExecutorInner<'gc>>>) -> Self {
        Executor(g)
    }

    pub fn mode(self) -> ExecutorMode {
        self.0.borrow().mode
    }

    /// Seed a fresh main thread with `function(args...)` and return a
    /// Normal-mode executor. A non-function is called through its `__call`
    /// chain.
    ///
    /// Lua and native entry share the same shape: args at `stack[0..]`
    /// and a `ExecKind::Start(function)` on top. The driver's `ExecKind::Start`
    /// handler builds the call frame on first dispatch.
    pub fn start<A: IntoMultiValue<'gc>>(
        ctx: Context<'gc>,
        function: impl Into<Value<'gc>>,
        args: A,
    ) -> Self {
        let thread = Thread::new(ctx.mutation());
        {
            let mc = ctx.mutation();
            let mut buf: Vec<Value<'gc>> = Vec::new();
            args.push_into(mc, &mut buf);

            let mut ts = thread.borrow_mut(mc);
            ts.main = true;
            ts.set_window(0, buf);
            ts.push_exec(ExecKind::Start(function.into()));
            ts.status = ThreadStatus::Suspended;
        }

        Executor(Gc::new(
            ctx.mutation(),
            RefLock::new(ExecutorInner {
                thread,
                thread_stack: vec![thread],
                mode: ExecutorMode::Normal,
            }),
        ))
    }

    /// Drive the executor by pumping the top frame of the top thread until
    /// the executor reaches a terminal state.
    ///
    /// Returns:
    /// - [`StepResult::Done`] — the main thread completed; call `take_result`.
    /// - [`StepResult::Yielded(values)`] — the main thread yielded to the
    ///   host. Feed args back via [`Executor::resume`] then `step` again.
    /// - [`StepResult::Pending`] — the collector is owed work, or an async
    ///   native waits on the host. The waker of [`step_waker`](Self::step_waker)
    ///   is woken once stepping can go on, right away for the collector.
    ///   Leave the current [`Lua::enter`](crate::Lua::enter),
    ///   which collects on its way out, and call `step` again in a new one.
    ///   Stepping on inside the same `enter` still makes progress, but
    ///   nothing is collected until the host leaves it.
    ///
    /// The hot path (Lua-only execution, sync natives) makes a single call
    /// into `run_thread` and exits. Coroutines the executor resumes itself
    /// loop here until something terminal happens or values cross the host
    /// boundary.
    pub fn step(self, ctx: Context<'gc>) -> Result<StepResult<'gc>, RuntimeError> {
        self.step_waker(ctx, std::task::Waker::noop())
    }

    /// [`step`](Self::step), with the waker an async native's future that
    /// waits on the host gets: `Pending` then means stepping again once it
    /// is woken.
    pub fn step_waker(
        self,
        ctx: Context<'gc>,
        waker: &std::task::Waker,
    ) -> Result<StepResult<'gc>, RuntimeError> {
        let prev = ctx.set_waker(waker);
        let r = self.step_inner(ctx);
        ctx.set_waker(prev);
        r
    }

    fn step_inner(self, ctx: Context<'gc>) -> Result<StepResult<'gc>, RuntimeError> {
        {
            let inner = self.0.borrow();
            if inner.mode != ExecutorMode::Normal {
                return Err(RuntimeError::BadMode);
            }
        }

        let mc = ctx.mutation();

        loop {
            // Snapshot the top thread under a short borrow so we can release
            // the executor lock before re-borrowing the thread itself.
            let top = {
                let inner = self.0.borrow();
                *inner
                    .thread_stack
                    .last()
                    .expect("executor thread_stack is empty")
            };

            // (1) Inner-thread propagation. If the top thread terminated
            // (Result) or yielded (Suspended) and we're not the bottom of
            // the executor stack, hand its values off to the resumer's
            // `ExecKind::WaitThread` and continue.
            let stack_len = self.0.borrow().thread_stack.len();
            if stack_len > 1 {
                match top.status() {
                    ThreadStatus::Result {
                        bottom: result_bottom,
                    } => {
                        propagate_inner_to_resumer(self, ctx, top, result_bottom, false)?;
                        continue;
                    }
                    ThreadStatus::Suspended => {
                        // Inner yielded. The yielded values live at
                        // stack[bottom..] where `bottom` is the inner's
                        // yield_bottom (the Yield path stashed it).
                        let bottom = {
                            let ts = top.borrow();
                            ts.yield_bottom.map(|y| y.bottom).unwrap_or(0)
                        };
                        propagate_inner_to_resumer(self, ctx, top, bottom, true)?;
                        continue;
                    }
                    _ => {}
                }
            }

            // (2) Terminal check on the main thread.
            match top.status() {
                ThreadStatus::Result { .. } => {
                    let mut inner = self.0.borrow_mut(mc);
                    inner.mode = ExecutorMode::Result;
                    return Ok(StepResult::Done);
                }
                ThreadStatus::Suspended if stack_len == 1 => {
                    // Top thread (== main) is suspended at the bottom of
                    // the executor stack. yield_bottom == Some means it
                    // yielded to the host; yield_bottom == None means
                    // it's freshly seeded (ExecKind::Start on top), which
                    // the dispatch step below handles. Don't conflate.
                    let values: Option<Vec<Value<'gc>>> = {
                        let ts = top.borrow();
                        ts.yield_bottom.map(|y| ts.window(y.bottom).to_vec())
                    };
                    if let Some(values) = values {
                        let mut inner = self.0.borrow_mut(mc);
                        inner.mode = ExecutorMode::Yielded;
                        return Ok(StepResult::Yielded(values));
                    }
                }
                ThreadStatus::Stopped => return Err(RuntimeError::BadMode),
                _ => {}
            }

            // (3) Drain a pending native-callback action from the top thread.
            let pending = {
                let mut ts = top.borrow_mut(mc);
                ts.pending_action.take()
            };
            if let Some(p) = pending {
                apply_pending_action(self, ctx, top, p)?;
                continue;
            }

            // (4) Inspect and pump the top frame.
            #[derive(Clone, Copy)]
            enum FrameKind {
                Lua,
                Start,
                WaitThread,
                Error,
            }
            let kind = {
                let ts = top.borrow();
                match ts.top_exec() {
                    _ if ts.pending_ret.is_some() => FrameKind::Lua,
                    None if ts.top_is_lua() => FrameKind::Lua,
                    Some(ExecKind::Start(_)) => FrameKind::Start,
                    Some(ExecKind::WaitThread { .. }) => FrameKind::WaitThread,
                    Some(ExecKind::Error(_)) => FrameKind::Error,
                    None => unreachable!(
                        "active thread with empty frames violates the executor invariant"
                    ),
                }
            };

            match kind {
                FrameKind::Lua => {
                    let (exit, current) = vm::interp::run_thread(ctx, top);
                    if !current.ptr_eq(top) {
                        follow_switches(self, ctx, current);
                    }
                    match exit {
                        vm::interp::Exit::Gc => {
                            ctx.mutation().metrics().defer_gc_check();
                            // Ready again as soon as the host has collected.
                            ctx.waker().wake_by_ref();
                            return Ok(StepResult::Pending);
                        }
                        vm::interp::Exit::Pending => return Ok(StepResult::Pending),
                        vm::interp::Exit::End => {}
                    }
                }
                FrameKind::Start => {
                    // First-resume / first-dispatch. Pop ExecKind::Start(f),
                    // insert the function before the args, and delegate
                    // to schedule_call_at — which builds a LuaFrame for a
                    // Lua callee or invokes a native one inline.
                    let mut ts = top.borrow_mut(mc);
                    let f = match ts.pop_exec() {
                        Some(ExecKind::Start(f)) => f,
                        _ => unreachable!(),
                    };
                    ts.insert_at(0, f);
                    schedule_call_at(&mut ts, ctx, 0, ret_exit)?;
                    if ts.frames_empty() && ts.pending_action.is_none() {
                        // Native entry returned `Return` synchronously;
                        // results sit at stack[0..] and the thread is
                        // done.
                        ts.status = ThreadStatus::Result { bottom: 0 };
                    } else {
                        ts.status = ThreadStatus::Normal;
                    }
                }
                FrameKind::WaitThread => {
                    unreachable!("WaitThread on top of the *active* thread is invariant-violating");
                }
                FrameKind::Error => {
                    unwind_error(self, ctx, top)?;
                }
            }
        }
    }

    /// Re-arm a `Yielded` executor with fresh resume arguments. The
    /// previously-yielded values on the top thread's stack are replaced
    /// by `args`, executor mode flips back to `Normal`, and the next
    /// `step` call resumes the program.
    pub fn resume<A: IntoMultiValue<'gc>>(
        self,
        ctx: Context<'gc>,
        args: A,
    ) -> Result<(), RuntimeError> {
        let mc = ctx.mutation();
        {
            let inner = self.0.borrow();
            if inner.mode != ExecutorMode::Yielded {
                return Err(RuntimeError::BadMode);
            }
        }
        let top = {
            let inner = self.0.borrow();
            *inner
                .thread_stack
                .last()
                .expect("Yielded executor must have a top thread")
        };
        // Recover where the yielded values live and the original CALL
        // landing slot from the yield_bottom stash installed by the
        // Yield path.
        let cs = {
            let mut ts = top.borrow_mut(mc);
            ts.yield_bottom
                .take()
                .expect("Yielded mode must have stashed yield_bottom")
        };
        // Materialize the resume-args into a temporary vec and place
        // them at stack[bottom..], replacing the previously-yielded
        // values.
        let mut buf: Vec<Value<'gc>> = Vec::new();
        args.push_into(mc, &mut buf);
        {
            let mut ts = top.borrow_mut(mc);
            ts.set_window(cs.bottom, buf);
            land_call_results(&mut ts, cs);
            // `land_call_results` may have *terminated* the thread: a tail-called
            // native suspends with the calling Lua frame already popped, so the
            // resume that lands its results empties the frame stack and sets
            // `Result`. Clobbering that with `Normal` would send the driver back
            // around the loop with no frame to pump. Same guard as
            // `propagate_inner_to_resumer`.
            if !matches!(ts.status, ThreadStatus::Result { .. }) {
                ts.status = ThreadStatus::Normal;
            }
        }
        {
            let mut inner = self.0.borrow_mut(mc);
            inner.mode = ExecutorMode::Normal;
        }
        Ok(())
    }

    /// Extract typed results from the finished thread's stack and reset the
    /// executor to `Stopped`.
    pub fn take_result<R: FromMultiValue<'gc>>(self, ctx: Context<'gc>) -> Result<R, RuntimeError> {
        let (thread, mode) = {
            let inner = self.0.borrow();
            (inner.thread, inner.mode)
        };
        if mode != ExecutorMode::Result {
            return Err(RuntimeError::BadMode);
        }

        // Results are the window `stack[bottom..top]` recorded by the
        // `Result` status; the vec itself may run past `top` (grow-not-shrink).
        let result = {
            let ts = thread.borrow();
            let ThreadStatus::Result { bottom } = ts.status else {
                return Err(RuntimeError::BadMode);
            };
            R::from_multi_value(ts.window(bottom)).map_err(RuntimeError::from)
        };

        // Clear the thread so it can be reused.
        {
            let mc = ctx.mutation();
            let mut ts = thread.borrow_mut(mc);
            ts.reset();
            ts.status = ThreadStatus::Stopped;
        }
        {
            let mut inner = self.0.borrow_mut(ctx.mutation());
            inner.mode = ExecutorMode::Stopped;
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Driver helpers
// ---------------------------------------------------------------------------

/// Make the yield or resume a native asked for that dispatch left to the
/// executor (see [`PendingAction`]).
fn apply_pending_action<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    top: Thread<'gc>,
    p: PendingAction<'gc>,
) -> Result<(), RuntimeError> {
    let PendingAction { kind, call_site } = p;
    match kind {
        PendingKind::Yield => {
            let mut ts = top.borrow_mut(ctx.mutation());
            if ts.no_yield {
                ts.raise(ctx, yield_across_close(ctx));
                return Ok(());
            }
            ts.yield_bottom = Some(call_site);
            ts.status = ThreadStatus::Suspended;
        }
        PendingKind::Resume(target) => {
            schedule_thread_resume(exec, ctx, top, target, call_site.bottom, call_site)?;
        }
    }
    Ok(())
}

/// Rebuild the thread stack after dispatch switched coroutines and ended on
/// `current`: the chain of resumers below it.
fn follow_switches<'gc>(exec: Executor<'gc>, ctx: Context<'gc>, current: Thread<'gc>) {
    let mut chain = vec![current];
    let mut t = current;
    while let Some(r) = t.borrow().resumer {
        chain.push(r);
        t = r;
    }
    chain.reverse();
    exec.0.borrow_mut(ctx.mutation()).thread_stack = chain;
}

/// Run natives on `thread` from the executor, as dispatch would, leaving
/// what is left to run for the next `run_thread`.
fn run_natives<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    thread: Thread<'gc>,
    state: vm::interp::NativeState<'gc>,
) {
    // SAFETY: the executor runs one thread at a time and holds no borrow of
    // it, as for `run_thread`.
    let mut ts = unsafe { thread.state_mut(ctx.mutation()) };
    let step = vm::interp::run_natives(ctx, &mut ts, true, state);
    if let vm::interp::NativeStep::Return {
        ret,
        func_slot,
        values,
    } = step
    {
        ts.pending_ret = Some(PendingRet {
            ret,
            func_slot,
            values,
        });
    }
    let current = ts.handle();
    if !current.ptr_eq(thread) {
        follow_switches(exec, ctx, current);
    }
}

/// Hand off control from `resumer` to `target`.
///
/// Pushes a `ExecKind::WaitThread { wt }` onto the resumer (this is what
/// `propagate_inner_to_resumer` will pop when the target eventually
/// yields/returns), drains the resume-args from
/// `resumer.stack[args_abs_bottom..]`, transitions the target into
/// `Normal` (handling first-resume and mid-resume cases), and pushes the
/// target onto the executor's thread stack.
fn schedule_thread_resume<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    resumer: Thread<'gc>,
    target: Thread<'gc>,
    args_abs_bottom: usize,
    wt: CallSite,
) -> Result<(), RuntimeError> {
    let mc = ctx.mutation();
    // Raised on the resumer, leaving the target untouched (`lua_resume`'s
    // `resume_error`).
    if exec.0.borrow().thread_stack.len() >= vm::interp::MAX_RESUME_DEPTH as usize {
        let msg = Value::string(LuaString::new(ctx, b"C stack overflow"));
        resumer.borrow_mut(mc).raise(ctx, Error::new(ctx, msg));
        return Ok(());
    }
    {
        let mut rs = resumer.borrow_mut(mc);
        rs.push_exec(ExecKind::WaitThread { call_site: wt });
        rs.status = ThreadStatus::Normal;
    }
    let args: Vec<Value<'gc>> = {
        let mut rs = resumer.borrow_mut(mc);
        rs.take_window(args_abs_bottom)
    };
    let depth = resumer.borrow().resume_depth + 1;
    {
        let mut ts = target.borrow_mut(mc);
        ts.resumer = Some(resumer);
        ts.resume_depth = depth;
        if matches!(ts.status, ThreadStatus::Suspended)
            && matches!(ts.top_exec(), Some(ExecKind::Start(_)))
        {
            // First-resume: stash args at the bottom of the stack; the
            // `ExecKind::Start` handler sets up the call frame on next pump.
            ts.discard_above(0);
            ts.set_window(0, args);
            ts.status = ThreadStatus::Normal;
        } else if matches!(ts.status, ThreadStatus::Suspended) {
            // Mid-resume: target previously yielded. Place the resume-args
            // where it stashed its landing site and deliver them.
            let y = match ts.yield_bottom.take() {
                Some(y) => y,
                None => return Err(RuntimeError::BadMode),
            };
            ts.set_window(y.bottom, args);
            land_call_results(&mut ts, y);
            ts.status = ThreadStatus::Normal;
        } else {
            return Err(RuntimeError::BadMode);
        }
    }
    exec.0.borrow_mut(mc).thread_stack.push(target);
    Ok(())
}

/// Call the value at `stack[slot]` with the args above it (through any
/// `__call` chain), finished by `ret`. For Lua: push a `LuaFrame` with
/// `base = slot+1`. For Native: invoke synchronously and deliver its results,
/// or stash a pending action. A non-callable value (or too long a `__call`
/// chain) raises at level 0, as the raiser is a native (`luaG_callerror` adds
/// no position for a C `ci`).
fn schedule_call_at<'gc>(
    ts: &mut crate::env::thread::ThreadState<'gc>,
    ctx: Context<'gc>,
    slot: usize,
    ret: vm::interp::Handler,
) -> Result<(), RuntimeError> {
    let target = match vm::interp::resolve_call_chain(ctx, ts, slot, 0) {
        Ok((target, _)) => target,
        Err(e) => {
            let msg = vm::debug::op_error_message(ctx, ts, e);
            ts.raise(
                ctx,
                Error::new(ctx, Value::string(LuaString::new(ctx, msg.as_bytes()))),
            );
            return Ok(());
        }
    };
    if let CallTarget::Lua(closure) = target {
        let base = slot + 1;
        let caller_provided = ts.top.saturating_sub(base);
        let num_params = closure.num_params as usize;
        let num_extras = if closure.is_vararg {
            caller_provided.saturating_sub(num_params) as u16
        } else {
            0
        };
        if !ts.ensure_frame_slots(base + closure.max_stack_size as usize) {
            let err = vm::debug::stack_overflow(ctx, ts);
            ts.raise(ctx, err);
            return Ok(());
        }
        // Nil-fill fixed params the caller didn't supply.
        for i in caller_provided..num_params {
            ts.stack[base + i] = Value::nil();
        }
        ts.push_lua(LuaFrame {
            closure,
            pc: closure.code,
            ret,
            base: base as u32,
            num_extras,
            flags: 0,
        });
        Ok(())
    } else {
        // Native target. Drive synchronously; if it returns Return we land
        // values at [slot..]; otherwise stash a new pending_action.
        let CallTarget::Native(nc) = target else {
            unreachable!()
        };
        let args_base = slot + 1;
        let argc = ts.top - args_base;
        let action = match vm::interp::invoke_native(ctx, ts, nc, args_base, argc) {
            Ok(a) => a,
            Err(e) => {
                // Unwound from the frames as they are, like any error.
                ts.raise(ctx, e);
                return Ok(());
            }
        };
        match action {
            CallbackAction::Return => {
                deliver(
                    ts,
                    CallSite {
                        bottom: args_base,
                        func_idx: slot,
                        ret,
                    },
                );
                Ok(())
            }
            action => {
                let f = ts.stack[slot].get_function().expect("resolved native");
                let state = vm::interp::NativeState::Acted {
                    r: Ok(action),
                    framed: false,
                    f,
                    base: args_base,
                    ret,
                };
                let mut tsr = &mut *ts;
                match vm::interp::run_natives(ctx, &mut tsr, false, state) {
                    vm::interp::NativeStep::Return {
                        ret,
                        func_slot,
                        values,
                    } => deliver(
                        ts,
                        CallSite {
                            bottom: values,
                            func_idx: func_slot,
                            ret,
                        },
                    ),
                    vm::interp::NativeStep::EnterLua
                    | vm::interp::NativeStep::Exit
                    | vm::interp::NativeStep::Pending => {}
                }
                Ok(())
            }
        }
    }
}

/// A yield while `coroutine.close`/`wrap` closes the thread's variables,
/// which the reference runs on the C stack.
fn yield_across_close(ctx: Context<'_>) -> Error<'_> {
    let msg = LuaString::new(ctx, b"attempt to yield across a C-call boundary");
    Error::new(ctx, Value::string(msg))
}

/// After an inner thread terminated or yielded, transfer its result-bottom
/// values to the resumer's `ExecKind::WaitThread`, pop both, and let the
/// resumer continue.
fn propagate_inner_to_resumer<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    inner: Thread<'gc>,
    inner_bottom: usize,
    inner_yielded: bool,
) -> Result<(), RuntimeError> {
    let mc = ctx.mutation();
    let values: Vec<Value<'gc>> = {
        let ts = inner.borrow();
        ts.window(inner_bottom).to_vec()
    };
    // Pop the inner thread.
    exec.0.borrow_mut(mc).thread_stack.pop();
    // If the inner is just yielded (not terminated), keep its state for
    // future resumes; its `yield_bottom` already points at where its
    // resume-args should land. After Result, clear the inner's stack
    // so subsequent fetches see an empty thread.
    if !inner_yielded {
        inner.borrow_mut(mc).discard_above(0);
    }
    inner.borrow_mut(mc).resumer = None;
    let resumer = *exec.0.borrow().thread_stack.last().unwrap();
    let mut rs = resumer.borrow_mut(mc);
    let wt = match rs.pop_exec() {
        Some(ExecKind::WaitThread { call_site }) => call_site,
        // Resumed in dispatch: the resumer's native frame takes the values
        // through its continuation.
        Some(_) => unreachable!("propagate_inner_to_resumer: resumer top is not waiting"),
        None => {
            let bottom = rs.live_top() + 1;
            CallSite {
                bottom,
                func_idx: bottom - 1,
                ret: vm::interp::ret_native,
            }
        }
    };
    rs.set_window(wt.bottom, values);
    land_call_results(&mut rs, wt);
    // `land_call_results` may have terminated the resumer (frames went
    // empty → status Result). Only revert to Normal if it didn't.
    if !matches!(rs.status, ThreadStatus::Result { .. }) {
        rs.status = ThreadStatus::Normal;
    }
    Ok(())
}

/// Hand the values at `stack[cs.bottom..top]` to the call site: an executor
/// one takes them at its function slot (terminating the thread if nothing is
/// left to run), any other one through its continuation on the next
/// `run_thread`.
fn land_call_results<'gc>(ts: &mut ThreadState<'gc>, cs: CallSite) {
    deliver(ts, cs);
}

fn deliver<'gc>(ts: &mut ThreadState<'gc>, cs: CallSite) {
    let CallSite {
        bottom,
        func_idx,
        ret,
    } = cs;
    if !std::ptr::fn_addr_eq(ret, ret_exit as vm::interp::Handler) {
        ts.pending_ret = Some(PendingRet {
            ret,
            func_slot: func_idx,
            values: bottom,
        });
        return;
    }
    // The producer published its value count through `top`.
    let retc = ts.top - bottom;
    debug_assert!(func_idx <= bottom);
    ts.stack.copy_within(bottom..bottom + retc, func_idx);
    ts.set_top(func_idx + retc);
    if ts.frames_empty() {
        ts.status = ThreadStatus::Result { bottom: func_idx };
    }
}

/// Unwind the error on top of `top` (see `vm::unwind`). Once nothing on the
/// thread catches it, it ends the executor (`os.exit`), kills an inner
/// coroutine and goes on in its resumer, or surfaces to the host.
fn unwind_error<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    top: Thread<'gc>,
) -> Result<(), RuntimeError> {
    let mc = ctx.mutation();
    let Some(ExecKind::Error(err)) = top.borrow_mut(mc).pop_exec() else {
        unreachable!()
    };
    if let Exit::Process(code) = err.exit_kind() {
        // No `__close` runs, now or on a later `coroutine.close`. Upvalues are
        // closed first so closures that outlive the reset keep their values.
        let mut inner = exec.0.borrow_mut(mc);
        for t in &inner.thread_stack {
            let mut ts = t.borrow_mut(mc);
            vm::interp::close_upvalues(mc, &mut ts, 0);
            ts.reset();
            ts.status = ThreadStatus::Stopped;
        }
        inner.mode = ExecutorMode::Stopped;
        return Err(RuntimeError::Exit(code));
    }
    if !top.borrow().frames_empty() {
        run_natives(exec, ctx, top, vm::interp::NativeState::Raise(err));
        return Ok(());
    }
    if err.exit_kind() == Exit::Clean {
        // A coroutine closed itself: it returns nothing.
        let mut ts = top.borrow_mut(mc);
        ts.discard_above(0);
        ts.status = ThreadStatus::Result { bottom: 0 };
        return Ok(());
    }
    let exit = err.exit_kind() != Exit::No;
    let stack_len = exec.0.borrow().thread_stack.len();
    if stack_len > 1 {
        // Error terminates this coroutine: no result values, so `Stopped`
        // (not `Result`) is its terminal/dead marker. Every coroutine the
        // error unwinds through re-enters here on a later pump and is marked
        // dead too, matching Lua's "kill the whole unwound chain". Stash the
        // killing error so `coroutine.close` can surface it as `(false, err)`.
        {
            let mut ts = top.borrow_mut(mc);
            ts.status = ThreadStatus::Stopped;
            ts.death_error = Some(err.value());
            ts.resumer = None;
        }
        exec.0.borrow_mut(mc).thread_stack.pop();
        let resumer = *exec.0.borrow().thread_stack.last().unwrap();
        // Already located on the inner thread; don't re-raise on the resumer.
        // An exit ends only the coroutine that closed itself.
        let err = if exit {
            Error::new(ctx, err.value())
        } else {
            err
        };
        let mut rs = resumer.borrow_mut(mc);
        rs.status = ThreadStatus::Normal;
        match rs.top_exec() {
            Some(ExecKind::WaitThread { .. }) => {
                rs.pop_exec();
                rs.push_exec(ExecKind::Error(err));
            }
            Some(_) => unreachable!("inner-thread error: resumer top isn't waiting"),
            // Resumed in dispatch: the resumer's native frame catches it.
            None => {
                let nf = rs.top_lua().expect("resumer without a frame");
                let slot = nf.base() + nf.num_extras as usize;
                rs.set_top_unchecked(slot);
                drop(rs);
                run_natives(
                    exec,
                    ctx,
                    resumer,
                    vm::interp::NativeState::Resume(Err(err)),
                );
            }
        }
        return Ok(());
    }

    // Bottom of the thread stack — surface to host.
    Err(RuntimeError::Lua(crate::lua::Stashable::stash(
        err,
        mc,
        ctx.roots(),
    )))
}
