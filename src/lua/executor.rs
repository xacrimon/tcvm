use crate::dmm::{Collect, Gc, RefLock};
use crate::env::function::Function;
use crate::env::thread::{Frame, LuaFrame, PendingAction, ThreadStatus, YieldBottom};
use crate::env::{Thread, Value};
use crate::lua::RuntimeError;
use crate::lua::context::Context;
use crate::lua::convert::{FromMultiValue, IntoMultiValue};
use crate::vm;
use crate::vm::sequence::CallbackAction;

#[derive(Clone, Copy, PartialEq, Eq, Collect)]
#[collect(internal, require_static)]
pub enum ExecutorMode {
    /// Thread has no seeded call — cannot step.
    Stopped,
    /// Thread is running and can be stepped.
    Normal,
    /// Thread has returned; results are available on its stack.
    Result,
    /// Top thread yielded; values were drained on the last `step`. The
    /// host-side resume API to continue from here isn't implemented yet.
    Yielded,
}

/// Outcome of a single `Executor::step` invocation.
pub enum StepResult<'gc> {
    /// Top thread reached terminal `Result` state. Caller may `take_result`.
    Done,
    /// Top thread yielded these values to the host. The host-side
    /// resume API isn't implemented yet.
    Yielded(Vec<Value<'gc>>),
    /// Reserved for future fuel-based time-slicing — caller should call
    /// `step` again to continue. Not currently produced by the executor.
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

    /// Seed `main_thread` with `function(args...)` and return a Normal-mode
    /// executor. Any previous state on the main thread is cleared.
    ///
    /// Lua and native entry share the same shape: args at `stack[0..]`
    /// and a `Frame::Start(function)` on top. The driver's `Frame::Start`
    /// handler builds the call frame on first dispatch.
    pub fn start<A: IntoMultiValue<'gc>>(
        ctx: Context<'gc>,
        function: Function<'gc>,
        args: A,
    ) -> Self {
        let thread = ctx.main_thread();
        {
            let mc = ctx.mutation();
            let mut ts = thread.borrow_mut(mc);
            ts.stack.clear();
            ts.frames.clear();
            ts.open_upvalues.clear();
            ts.tbc_slots.clear();

            args.push_into(&mut ts.stack);
            ts.frames.push(Frame::Start(function));
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
    ///   host. The host-side resume API isn't implemented yet.
    /// - [`StepResult::Pending`] — reserved for fuel-based slicing.
    ///
    /// The hot path (Lua-only execution, sync natives) makes a single call
    /// into `run_thread` and exits. Coroutine resume / sequence pump cycles
    /// loop here until something terminal happens or values cross the host
    /// boundary. There is no fuel limit yet.
    pub fn step(self, ctx: Context<'gc>) -> Result<StepResult<'gc>, RuntimeError> {
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
            // `Frame::WaitThread` and continue.
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
                    // Main thread is suspended at the bottom of the
                    // executor stack. yield_bottom == Some means it
                    // yielded to the host; yield_bottom == None means
                    // it's freshly seeded (Frame::Start on top), which
                    // the dispatch step below handles. Don't conflate.
                    let values: Option<Vec<Value<'gc>>> = {
                        let ts = top.borrow();
                        ts.yield_bottom.map(|y| ts.stack[y.bottom..].to_vec())
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
                Sequence,
                Start,
                WaitThread,
                Error,
            }
            let kind = {
                let ts = top.borrow();
                match ts.frames.last() {
                    Some(Frame::Lua(_)) => FrameKind::Lua,
                    Some(Frame::Sequence { .. }) => FrameKind::Sequence,
                    Some(Frame::Start(_)) => FrameKind::Start,
                    Some(Frame::WaitThread { .. }) => FrameKind::WaitThread,
                    Some(Frame::Error(_)) => FrameKind::Error,
                    None => unreachable!(
                        "active thread with empty frames violates the executor invariant"
                    ),
                }
            };

            match kind {
                FrameKind::Lua => {
                    vm::interp::run_thread(ctx, top)
                        .map_err(|e| RuntimeError::Opcode { pc: e.pc })?;
                }
                FrameKind::Sequence => {
                    pump_sequence(self, ctx, top)?;
                }
                FrameKind::Start => {
                    // First-resume / first-dispatch. Pop Frame::Start(f),
                    // insert the function before the args, and delegate
                    // to schedule_call_at — which builds a LuaFrame for a
                    // Lua callee or invokes a native one inline.
                    let mut ts = top.borrow_mut(mc);
                    let f = match ts.frames.pop() {
                        Some(Frame::Start(f)) => f,
                        _ => unreachable!(),
                    };
                    ts.stack.insert(0, Value::function(f));
                    schedule_call_at(&mut ts, ctx, 0, f, 0)?;
                    if ts.frames.is_empty() && ts.pending_action.is_none() {
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
        let (bottom, func_idx, returns) = {
            let mut ts = top.borrow_mut(mc);
            let y = ts
                .yield_bottom
                .take()
                .expect("Yielded mode must have stashed yield_bottom");
            (y.bottom, y.func_idx, y.returns)
        };
        // Materialize the resume-args into a temporary vec and place
        // them at stack[bottom..], replacing the previously-yielded
        // values.
        let mut buf: Vec<Value<'gc>> = Vec::new();
        args.push_into(&mut buf);
        {
            let mut ts = top.borrow_mut(mc);
            ts.stack.truncate(bottom);
            ts.stack.extend(buf);
            // Branch on top frame: a Sequence consumes values from
            // stack[seq.bottom..] on its next poll, so we leave them at
            // `bottom`. A Lua frame on top means the yield came from a
            // native CALL inline — land the args at func_idx via the
            // standard call-return convention so dispatch resumes
            // correctly.
            if !matches!(ts.frames.last(), Some(Frame::Sequence { .. })) {
                land_call_results(&mut ts, bottom, func_idx, returns);
            }
            ts.status = ThreadStatus::Normal;
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

        let values: Vec<Value<'gc>> = {
            let ts = thread.borrow();
            ts.stack.clone()
        };
        let result = R::from_multi_value(&values).map_err(RuntimeError::from);

        // Clear the thread so it can be reused.
        {
            let mc = ctx.mutation();
            let mut ts = thread.borrow_mut(mc);
            ts.stack.clear();
            ts.frames.clear();
            ts.open_upvalues.clear();
            ts.tbc_slots.clear();
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

/// Translate a [`PendingAction`] (deposited by `op_call`/`op_tailcall` after
/// a non-`Return` `CallbackAction`) into frame-stack operations.
fn apply_pending_action<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    top: Thread<'gc>,
    p: PendingAction<'gc>,
) -> Result<(), RuntimeError> {
    let mc = ctx.mutation();
    let PendingAction {
        action,
        bottom,
        func_idx,
        returns,
    } = p;
    match action {
        CallbackAction::Return => {
            // op_call/tailcall handle Return inline. With yield-bottom
            // factored out into its own field, no sentinel use remains.
            unreachable!("apply_pending_action: Return is handled inline");
        }
        CallbackAction::Sequence(seq) => {
            let mut ts = top.borrow_mut(mc);
            ts.frames.push(Frame::Sequence {
                seq,
                bottom,
                func_idx,
                returns,
                pending_error: None,
            });
        }
        CallbackAction::Call { function, then } => {
            let mut ts = top.borrow_mut(mc);
            // If `then` provided, the sequence is the call's "completion
            // handler"; it inherits the caller's expected_returns. The
            // sequence sees the called function's results at stack[bottom..].
            if let Some(seq) = then {
                ts.frames.push(Frame::Sequence {
                    seq,
                    bottom,
                    func_idx,
                    returns,
                    pending_error: None,
                });
            }
            // Now schedule the call. For a Lua target, push a LuaFrame.
            // For a Native target, push a Frame::Start hand-off (driver
            // pumps it next iteration).
            // The function and args layout: stack[bottom] becomes the
            // function slot, args at [bottom+1..]. But the callback that
            // pushed `Call` already left its desired args at stack[bottom..]
            // (per piccolo convention), with no function slot in front. We
            // need to insert the function in front.
            ts.stack.insert(bottom, Value::function(function));
            // Now stack[bottom] = function, stack[bottom+1..] = args.
            schedule_call_at(&mut ts, ctx, bottom, function, returns)?;
        }
        CallbackAction::Yield { then } => {
            let mut ts = top.borrow_mut(mc);
            if let Some(seq) = then {
                ts.frames.push(Frame::Sequence {
                    seq,
                    bottom,
                    func_idx,
                    returns,
                    pending_error: None,
                });
            }
            // Yielded values live at stack[bottom..]. Mark thread
            // suspended and stash where they are so the next pump can
            // find them (propagation to resumer, host-side resume, etc.).
            ts.yield_bottom = Some(YieldBottom {
                bottom,
                func_idx,
                returns,
            });
            ts.status = ThreadStatus::Suspended;
        }
        CallbackAction::Resume {
            thread: target,
            then,
        } => {
            // Optional `then` fires when target yields/returns; install on
            // the resumer first so it's seen *after* the WaitThread frame.
            if let Some(seq) = then {
                top.borrow_mut(mc).frames.push(Frame::Sequence {
                    seq,
                    bottom,
                    func_idx,
                    returns,
                    pending_error: None,
                });
            }
            schedule_thread_resume(exec, ctx, top, target, bottom, bottom, func_idx, returns)?;
        }
    }
    Ok(())
}

/// Hand off control from `resumer` to `target`.
///
/// Pushes a `Frame::WaitThread { wt_bottom, wt_func_idx, wt_returns }`
/// onto the resumer (this is what `propagate_inner_to_resumer` will pop
/// when the target eventually yields/returns), drains the resume-args
/// from `resumer.stack[args_abs_bottom..]`, transitions the target into
/// `Normal` (handling first-resume and mid-resume cases), and pushes the
/// target onto the executor's thread stack.
///
/// Callers that want a follow-up `Sequence` on the resumer for the
/// returned values must push it BEFORE calling this helper (so it sits
/// beneath the WaitThread frame in stack order).
fn schedule_thread_resume<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    resumer: Thread<'gc>,
    target: Thread<'gc>,
    args_abs_bottom: usize,
    wt_bottom: usize,
    wt_func_idx: usize,
    wt_returns: u8,
) -> Result<(), RuntimeError> {
    let mc = ctx.mutation();
    {
        let mut rs = resumer.borrow_mut(mc);
        rs.frames.push(Frame::WaitThread {
            bottom: wt_bottom,
            func_idx: wt_func_idx,
            returns: wt_returns,
        });
        rs.status = ThreadStatus::Normal;
    }
    let args: Vec<Value<'gc>> = {
        let mut rs = resumer.borrow_mut(mc);
        rs.stack.drain(args_abs_bottom..).collect()
    };
    {
        let mut ts = target.borrow_mut(mc);
        if matches!(ts.status, ThreadStatus::Suspended)
            && matches!(ts.frames.last(), Some(Frame::Start(_)))
        {
            // First-resume: stash args at the bottom of the stack; the
            // `Frame::Start` handler sets up the call frame on next pump.
            ts.stack.clear();
            ts.stack.extend(args);
            ts.status = ThreadStatus::Normal;
        } else if matches!(ts.status, ThreadStatus::Suspended) {
            // Mid-resume: target previously yielded. Drain yield_bottom
            // to recover where the yielded native CALL landed, place the
            // resume-args, then either land via call-return convention
            // (Lua frame on top) or leave at bottom (Sequence on top —
            // it reads stack[seq.bottom..] directly on next poll).
            let y = match ts.yield_bottom.take() {
                Some(y) => y,
                None => return Err(RuntimeError::BadMode),
            };
            ts.stack.truncate(y.bottom);
            ts.stack.extend(args);
            if !matches!(ts.frames.last(), Some(Frame::Sequence { .. })) {
                land_call_results(&mut ts, y.bottom, y.func_idx, y.returns);
            }
            ts.status = ThreadStatus::Normal;
        } else {
            return Err(RuntimeError::BadMode);
        }
    }
    exec.0.borrow_mut(mc).thread_stack.push(target);
    Ok(())
}

/// Push a Lua/Native call frame for `function` whose function-slot is at
/// `slot` (so args live at `slot+1..`). For Lua: a `LuaFrame` with `base =
/// slot+1`. For Native: invoke synchronously and either land Return values
/// at `slot..` or stash a pending action.
fn schedule_call_at<'gc>(
    ts: &mut crate::env::thread::ThreadState<'gc>,
    ctx: Context<'gc>,
    slot: usize,
    function: Function<'gc>,
    caller_returns: u8,
) -> Result<(), RuntimeError> {
    if let Some(closure) = function.as_lua() {
        let base = slot + 1;
        let needed = base + closure.proto.max_stack_size as usize;
        if ts.stack.len() < needed {
            ts.stack.resize(needed, Value::nil());
        }
        ts.push_lua(LuaFrame {
            closure,
            base,
            pc: 0,
            num_results: caller_returns,
            continuation: None,
        });
        Ok(())
    } else {
        // Native target. Drive synchronously; if it returns Return we land
        // values at [slot..]; otherwise stash a new pending_action.
        let nc = function
            .as_native()
            .expect("function is neither Lua nor Native");
        let args_base = slot + 1;
        let argc = ts.stack.len() - args_base;
        let action = vm::interp::invoke_native(ctx, ts, nc, args_base, argc).map_err(|e| {
            RuntimeError::Lua(crate::lua::Stashable::stash(e, ctx.mutation(), ctx.roots()))
        })?;
        match action {
            CallbackAction::Return => {
                // Move stack[args_base..] down to stack[slot..]. (Slot is
                // where the function used to sit; the function itself is
                // stored back in [slot] before the call by the caller.)
                let retc = ts.stack.len() - args_base;
                for i in 0..retc {
                    ts.stack[slot + i] = ts.stack[args_base + i];
                }
                ts.stack.truncate(slot + retc);
                Ok(())
            }
            other => {
                ts.pending_action = Some(PendingAction {
                    action: other,
                    bottom: args_base,
                    func_idx: slot,
                    returns: caller_returns,
                });
                Ok(())
            }
        }
    }
}

/// Pump a `Frame::Sequence` on top of `top`. Pops the frame, invokes
/// `seq.poll()` (or `seq.error()` if `pending_error.is_some()`), and
/// translates the [`SequencePoll`] back to frame ops.
fn pump_sequence<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    top: Thread<'gc>,
) -> Result<(), RuntimeError> {
    use crate::vm::sequence::{Execution, SequencePoll};
    let mc = ctx.mutation();
    // Pop the sequence frame and call poll/error.
    let (mut seq, bottom, func_idx, returns, pending_error) = {
        let mut ts = top.borrow_mut(mc);
        match ts.frames.pop() {
            Some(Frame::Sequence {
                seq,
                bottom,
                func_idx,
                returns,
                pending_error,
            }) => (seq, bottom, func_idx, returns, pending_error),
            _ => unreachable!("pump_sequence: top wasn't Frame::Sequence"),
        }
    };
    let poll_result = {
        let mut ts = top.borrow_mut(mc);
        let stack_view = crate::env::function::Stack::new(&mut ts.stack, bottom);
        let exec = Execution::new(top);
        if let Some(err) = pending_error {
            seq.error(ctx, exec, err, stack_view)
        } else {
            seq.poll(ctx, exec, stack_view)
        }
    };
    match poll_result {
        Ok(SequencePoll::Pending) => {
            top.borrow_mut(mc).frames.push(Frame::Sequence {
                seq,
                bottom,
                func_idx,
                returns,
                pending_error: None,
            });
        }
        Ok(SequencePoll::Return) => {
            // Sequence finished. Land values at the original CALL's expected
            // window: stack[func_idx..func_idx+wanted].
            let mut ts = top.borrow_mut(mc);
            land_call_results(&mut ts, bottom, func_idx, returns);
        }
        Ok(SequencePoll::Call {
            function,
            bottom: rel,
        }) => {
            let abs_bottom = bottom + rel;
            let mut ts = top.borrow_mut(mc);
            // Re-push self to be re-polled with results at stack[bottom..].
            ts.frames.push(Frame::Sequence {
                seq,
                bottom,
                func_idx,
                returns,
                pending_error: None,
            });
            // Schedule the call: insert function at abs_bottom, args after.
            ts.stack.insert(abs_bottom, Value::function(function));
            schedule_call_at(&mut ts, ctx, abs_bottom, function, 0)?;
        }
        Ok(SequencePoll::TailCall(function)) => {
            // Sequence is done; the call's results must land at the
            // original CALL site `func_idx`. Args sit at stack[bottom..],
            // which is adjacent to `func_idx` after a normal CALL but not
            // after a TAILCALL→native→sequence chain (where bottom lives
            // inside the popped tail-callee's window). Compact the args
            // down to func_idx+1, then place the function at func_idx.
            let mut ts = top.borrow_mut(mc);
            let argc = ts.stack.len() - bottom;
            let new_args_base = func_idx + 1;
            if new_args_base < bottom {
                for i in 0..argc {
                    ts.stack[new_args_base + i] = ts.stack[bottom + i];
                }
                ts.stack.truncate(new_args_base + argc);
            }
            ts.stack[func_idx] = Value::function(function);
            schedule_call_at(&mut ts, ctx, func_idx, function, returns)?;
        }
        Ok(SequencePoll::Yield { bottom: rel }) => {
            let abs_bottom = bottom + rel;
            let mut ts = top.borrow_mut(mc);
            // Re-push self to be re-polled with resume-args at stack[bottom..].
            ts.frames.push(Frame::Sequence {
                seq,
                bottom,
                func_idx,
                returns,
                pending_error: None,
            });
            ts.yield_bottom = Some(YieldBottom {
                bottom: abs_bottom,
                func_idx,
                returns,
            });
            ts.status = ThreadStatus::Suspended;
        }
        Ok(SequencePoll::TailYield) => {
            let mut ts = top.borrow_mut(mc);
            ts.yield_bottom = Some(YieldBottom {
                bottom,
                func_idx,
                returns,
            });
            ts.status = ThreadStatus::Suspended;
        }
        Ok(SequencePoll::Resume {
            thread: target,
            bottom: rel,
        }) => {
            // Re-push self so propagate_inner_to_resumer finds the
            // sequence beneath the WaitThread and lands target's
            // eventual values at seq.bottom for the next poll.
            let abs_bottom = bottom + rel;
            top.borrow_mut(mc).frames.push(Frame::Sequence {
                seq,
                bottom,
                func_idx,
                returns,
                pending_error: None,
            });
            schedule_thread_resume(exec, ctx, top, target, abs_bottom, bottom, func_idx, returns)?;
        }
        Ok(SequencePoll::TailResume(target)) => {
            // Sequence is consumed; target's eventual values go straight
            // to the original CALL site via the WaitThread frame's
            // (bottom, func_idx, returns) when no Sequence is beneath.
            schedule_thread_resume(exec, ctx, top, target, bottom, bottom, func_idx, returns)?;
        }
        Err(err) => {
            top.borrow_mut(mc).frames.push(Frame::Error(err));
        }
    }
    Ok(())
}

/// After an inner thread terminated or yielded, transfer its result-bottom
/// values to the resumer's `Frame::WaitThread`, pop both, and let the
/// resumer continue (the next driver pump finds either a `Frame::Sequence`
/// or a `Frame::Lua` ready to resume).
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
        ts.stack[inner_bottom..].to_vec()
    };
    // Pop the inner thread.
    exec.0.borrow_mut(mc).thread_stack.pop();
    // If the inner is just yielded (not terminated), keep its state for
    // future resumes; the inner's pending_action retained the bottom.
    if !inner_yielded {
        // After Result, clear the inner's stack so subsequent fetches see
        // an empty thread.
        let mut ts = inner.borrow_mut(mc);
        ts.stack.clear();
    }
    let resumer = *exec.0.borrow().thread_stack.last().unwrap();
    let mut rs = resumer.borrow_mut(mc);
    let (wt_bottom, wt_func_idx, wt_returns) = match rs.frames.pop() {
        Some(Frame::WaitThread {
            bottom,
            func_idx,
            returns,
        }) => (bottom, func_idx, returns),
        _ => unreachable!("propagate_inner_to_resumer: resumer top is not WaitThread"),
    };
    // Land values at resumer.stack[wt_bottom..] for the next sequence /
    // call frame to consume. If a `then` sequence sits underneath, it'll
    // pick them up at its own `bottom == wt_bottom`. If not, do the
    // standard CALL-landing right now.
    rs.stack.truncate(wt_bottom);
    rs.stack.extend(values);
    let next_is_sequence = matches!(rs.frames.last(), Some(Frame::Sequence { .. }));
    if !next_is_sequence {
        // No follow-up sequence: deliver results directly to the original
        // Lua CALL window.
        land_call_results(&mut rs, wt_bottom, wt_func_idx, wt_returns);
    }
    // `land_call_results` may have terminated the resumer (frames went
    // empty → status Result). Only revert to Normal if it didn't.
    if !matches!(rs.status, ThreadStatus::Result { .. }) {
        rs.status = ThreadStatus::Normal;
    }
    Ok(())
}

/// Move values at `stack[bottom..]` into the original CALL's expected
/// landing slot per the standard Lua convention. If `frames` is empty
/// after the move (e.g., a tailcalled native suspended and the calling
/// Lua frame was already popped at TAILCALL time), terminate the thread
/// with `ThreadStatus::Result { bottom: func_idx }`.
fn land_call_results<'gc>(
    ts: &mut crate::env::thread::ThreadState<'gc>,
    bottom: usize,
    func_idx: usize,
    returns: u8,
) {
    let retc = ts.stack.len() - bottom;
    let wanted = if returns == 0 {
        retc
    } else {
        returns as usize - 1
    };
    let to_copy = retc.min(wanted);
    for i in 0..to_copy {
        ts.stack[func_idx + i] = ts.stack[bottom + i];
    }
    for i in to_copy..wanted {
        ts.stack[func_idx + i] = Value::nil();
    }
    if returns == 0 {
        ts.stack.truncate(func_idx + retc);
    } else {
        // Keep stack >= func_idx + wanted; restore caller's max_stack_size
        // window if a Lua frame is on top.
        if let Some(frame) = ts.top_lua() {
            let needed = frame.base + frame.closure.proto.max_stack_size as usize;
            if ts.stack.len() < needed {
                ts.stack.resize(needed, Value::nil());
            }
        }
    }
    if ts.frames.is_empty() {
        ts.status = ThreadStatus::Result { bottom: func_idx };
    }
}

/// Walk a thread's frame stack popping Lua/Wait frames (closing upvalues
/// at each `bottom`) until a `Sequence` frame can catch the error.
///
/// On no-catcher: if the thread isn't the bottom of the executor's
/// thread stack, route the error to the resumer's `Frame::WaitThread`
/// and pop the inner thread. This lets a coroutine error propagate to
/// the resumer's `PCallSequence::error`. If the thread *is* the bottom,
/// surface as `RuntimeError::Lua` to the host.
fn unwind_error<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    top: Thread<'gc>,
) -> Result<(), RuntimeError> {
    let mc = ctx.mutation();
    let err = {
        let mut ts = top.borrow_mut(mc);
        let err = match ts.frames.pop() {
            Some(Frame::Error(e)) => e,
            _ => unreachable!(),
        };
        loop {
            match ts.frames.last() {
                Some(Frame::Lua(lf)) => {
                    let base = lf.base;
                    ts.frames.pop();
                    close_upvalues_at(&mut ts, ctx, base);
                    ts.stack.truncate(base);
                }
                Some(Frame::Sequence { .. }) => {
                    if let Some(Frame::Sequence { pending_error, .. }) = ts.frames.last_mut() {
                        *pending_error = Some(err);
                    }
                    return Ok(());
                }
                Some(Frame::WaitThread { .. }) => {
                    ts.frames.pop();
                }
                Some(Frame::Start(_)) | Some(Frame::Error(_)) => {
                    // Frame::Start is only on a freshly-created thread
                    // that hasn't run yet, so it can't have errored.
                    // Frame::Error is removed by the next driver pump
                    // (which enters this function), so two can't coexist.
                    unreachable!("Frame::Start / Frame::Error mid-unwind violates the executor invariant");
                }
                None => break err,
            }
        }
    };

    // No catcher on this thread. If we're an inner coroutine, propagate to
    // the resumer's WaitThread → its next Sequence can catch.
    let stack_len = exec.0.borrow().thread_stack.len();
    if stack_len > 1 {
        exec.0.borrow_mut(mc).thread_stack.pop();
        let resumer = *exec.0.borrow().thread_stack.last().unwrap();
        let mut rs = resumer.borrow_mut(mc);
        // Pop the WaitThread.
        match rs.frames.pop() {
            Some(Frame::WaitThread { .. }) => {}
            _ => unreachable!("inner-thread error: resumer top isn't WaitThread"),
        }
        rs.frames.push(Frame::Error(err));
        rs.status = ThreadStatus::Normal;
        return Ok(());
    }

    // Bottom of the thread stack — surface to host.
    Err(RuntimeError::Lua(crate::lua::Stashable::stash(
        err,
        mc,
        ctx.roots(),
    )))
}

/// Close open upvalues and TBC variables whose stack indices are >=
/// `start_idx`. Mirrors `interp::close_upvalues` but lives in executor.rs
/// to avoid an interp-private dependency in the unwinder.
fn close_upvalues_at<'gc>(
    ts: &mut crate::env::thread::ThreadState<'gc>,
    ctx: Context<'gc>,
    start_idx: usize,
) {
    use crate::env::function::UpvalueState;
    let mc = ctx.mutation();
    ts.open_upvalues.retain(|uv| {
        let should_close = {
            let b = uv.borrow();
            match &*b {
                UpvalueState::Open { index, .. } => *index >= start_idx,
                UpvalueState::Closed(_) => false,
            }
        };
        if should_close {
            let val = ts.stack[{
                let b = uv.borrow();
                match &*b {
                    UpvalueState::Open { index, .. } => *index,
                    _ => unreachable!(),
                }
            }];
            *uv.borrow_mut(mc) = UpvalueState::Closed(val);
            false
        } else {
            true
        }
    });
    ts.tbc_slots.retain(|&i| i < start_idx);
}
