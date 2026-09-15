use std::pin::Pin;

use crate::dmm::{Collect, Gc, RefLock, Trace};
use crate::env::function::Function;
use crate::env::thread::{CallSite, Frame, LuaFrame, PendingAction, ThreadState, ThreadStatus};
use crate::env::{Error, LuaString, Stack, Thread, Value};
use crate::lua::RuntimeError;
use crate::lua::context::Context;
use crate::lua::convert::{FromMultiValue, IntoMultiValue};
use crate::vm;
use crate::vm::interp::{CallTarget, OpError};
use crate::vm::interp::{Continuation, ContinuationPayload};
use crate::vm::sequence::{
    BoxSequence, CallbackAction, Catch, Execution, Sequence, SequencePoll, seq_trace_pointers,
};

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
    /// A `Sequence` returned `SequencePoll::Pending`, asking the host to
    /// interleave other work / consult fuel. Mode stays `Normal`; call
    /// `step` again to keep going.
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
            let mut buf: Vec<Value<'gc>> = Vec::new();
            args.push_into(mc, &mut buf);

            let mut ts = thread.borrow_mut(mc);
            ts.discard_above(0);
            ts.frames.clear();
            ts.open_upvalues.clear();
            ts.tbc_slots.clear();

            ts.set_window(0, buf);
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
    ///   host. Feed args back via [`Executor::resume`] then `step` again.
    /// - [`StepResult::Pending`] — a `Sequence` returned `Pending`; the
    ///   driver yielded so the host can interleave other work. Call
    ///   `step` again to continue. (No fuel limit is enforced yet.)
    ///
    /// The hot path (Lua-only execution, sync natives) makes a single call
    /// into `run_thread` and exits. Coroutine resume / sequence pump cycles
    /// loop here until something terminal happens or values cross the host
    /// boundary.
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
                    // Top thread (== main) is suspended at the bottom of
                    // the executor stack. yield_bottom == Some means it
                    // yielded to the host; yield_bottom == None means
                    // it's freshly seeded (Frame::Start on top), which
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
                FrameKind::Lua => vm::interp::run_thread(ctx, top),
                FrameKind::Sequence => {
                    if matches!(pump_sequence(self, ctx, top)?, PumpOutcome::Pending) {
                        // Sequence asked for cooperative re-poll. Mode stays
                        // Normal so a follow-up `step` resumes the driver.
                        return Ok(StepResult::Pending);
                    }
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
                    ts.insert_at(0, Value::function(f));
                    schedule_call_at(&mut ts, ctx, 0, 0)?;
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
            ts.discard_above(0);
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
    let PendingAction { action, call_site } = p;
    match action {
        CallbackAction::Return => {
            // op_call/tailcall handle Return inline. With yield_bottom
            // factored out into its own field, no sentinel use remains.
            unreachable!("apply_pending_action: Return is handled inline");
        }
        CallbackAction::Sequence(seq) => {
            let mut ts = top.borrow_mut(mc);
            ts.frames.push(Frame::Sequence {
                seq,
                call_site,
                pending_error: None,
            });
        }
        CallbackAction::Call { then } => {
            let mut ts = top.borrow_mut(mc);
            // If `then` provided, the sequence is the call's "completion
            // handler"; it inherits the caller's expected_returns. The
            // sequence sees the called function's results at stack[bottom..].
            //
            // We push `then` BEFORE scheduling the call so that if the
            // callee can't be resolved or errors immediately, the
            // Frame::Error lands above this sequence and the unwinder
            // routes the error to it.
            let slot = match then {
                Some(seq) => {
                    ts.frames.push(Frame::Sequence {
                        seq,
                        call_site,
                        pending_error: None,
                    });
                    call_site.bottom
                }
                // No completion: the callee returns straight to the CALL
                // site, so it must sit where the CALL put its function.
                None => {
                    let (bottom, top) = (call_site.bottom, ts.top);
                    ts.stack.copy_within(bottom..top, call_site.func_idx);
                    ts.set_top(call_site.func_idx + (top - bottom));
                    call_site.func_idx
                }
            };
            schedule_call_at(&mut ts, ctx, slot, call_site.returns)?;
        }
        CallbackAction::Yield { then } => {
            let mut ts = top.borrow_mut(mc);
            // With a follow-up sequence the resume args are its input and
            // stay at `bottom`; without one they are the call's results.
            let landing = if then.is_some() {
                in_place(call_site.bottom)
            } else {
                call_site
            };
            if let Some(seq) = then {
                ts.frames.push(Frame::Sequence {
                    seq,
                    call_site,
                    pending_error: None,
                });
            }
            ts.yield_bottom = Some(landing);
            ts.status = ThreadStatus::Suspended;
        }
        CallbackAction::Resume {
            thread: target,
            then,
        } => {
            // Optional `then` fires when target yields/returns; install on
            // the resumer first so it's seen *after* the WaitThread frame.
            let landing = if then.is_some() {
                in_place(call_site.bottom)
            } else {
                call_site
            };
            if let Some(seq) = then {
                top.borrow_mut(mc).frames.push(Frame::Sequence {
                    seq,
                    call_site,
                    pending_error: None,
                });
            }
            schedule_thread_resume(exec, ctx, top, target, call_site.bottom, landing)?;
        }
    }
    Ok(())
}

/// Hand off control from `resumer` to `target`.
///
/// Pushes a `Frame::WaitThread { wt }` onto the resumer (this is what
/// `propagate_inner_to_resumer` will pop when the target eventually
/// yields/returns), drains the resume-args from
/// `resumer.stack[args_abs_bottom..]`, transitions the target into
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
    wt: CallSite,
) -> Result<(), RuntimeError> {
    let mc = ctx.mutation();
    {
        let mut rs = resumer.borrow_mut(mc);
        rs.frames.push(Frame::WaitThread { call_site: wt });
        rs.status = ThreadStatus::Normal;
    }
    let args: Vec<Value<'gc>> = {
        let mut rs = resumer.borrow_mut(mc);
        rs.take_window(args_abs_bottom)
    };
    {
        let mut ts = target.borrow_mut(mc);
        if matches!(ts.status, ThreadStatus::Suspended)
            && matches!(ts.frames.last(), Some(Frame::Start(_)))
        {
            // First-resume: stash args at the bottom of the stack; the
            // `Frame::Start` handler sets up the call frame on next pump.
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
/// `__call` chain). For Lua: push a `LuaFrame` with `base = slot+1`. For
/// Native: invoke synchronously and either land Return values at `slot..`
/// or stash a pending action. A non-callable value raises "attempt to
/// call" at level 0, as the raiser is a native (`luaG_callerror` adds no
/// position for a C `ci`).
fn schedule_call_at<'gc>(
    ts: &mut crate::env::thread::ThreadState<'gc>,
    ctx: Context<'gc>,
    slot: usize,
    caller_returns: u8,
) -> Result<(), RuntimeError> {
    let Some((target, _)) = vm::interp::resolve_call_chain(ctx, ts, slot, 0) else {
        let msg = vm::debug::op_error_message(ctx, ts, OpError::Call(ts.stack[slot]));
        ts.raise(
            ctx,
            Error::new(Value::string(LuaString::new(ctx, msg.as_bytes()))),
        );
        return Ok(());
    };
    if let CallTarget::Lua(closure) = target {
        let base = slot + 1;
        let caller_provided = ts.top.saturating_sub(base);
        let num_params = closure.proto.num_params as usize;
        let num_extras = if closure.proto.is_vararg {
            caller_provided.saturating_sub(num_params) as u32
        } else {
            0
        };
        ts.ensure_slots(base + closure.proto.max_stack_size as usize);
        // Nil-fill fixed params the caller didn't supply.
        for i in caller_provided..num_params {
            ts.stack[base + i] = Value::nil();
        }
        ts.push_lua(LuaFrame {
            closure,
            base,
            pc: closure.proto.code.as_ptr(),
            num_results: caller_returns,
            num_extras,
            continuation: None,
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
                // Mirror op_call's native-error path: push Frame::Error so the
                // executor unwinder can find the nearest Sequence catcher
                // (e.g. a PCallSequence wrapping coroutine.resume). Returning
                // Err here would short-circuit past any catcher pushed by
                // apply_pending_action / pump_sequence before this call.
                ts.raise(ctx, e);
                return Ok(());
            }
        };
        match action {
            CallbackAction::Return => {
                // Move stack[args_base..top] down to stack[slot..]. (Slot is
                // where the function used to sit; the function itself is
                // stored back in [slot] before the call by the caller.)
                let retc = ts.top - args_base;
                ts.stack.copy_within(args_base..args_base + retc, slot);
                // Drops the function slot and the one stale donor copy the
                // shift-by-one leaves behind, nil-filling both.
                ts.set_top(slot + retc);
                Ok(())
            }
            other => {
                ts.pending_action = Some(PendingAction {
                    action: other,
                    call_site: CallSite {
                        bottom: args_base,
                        func_idx: slot,
                        returns: caller_returns,
                        cont: None,
                    },
                });
                Ok(())
            }
        }
    }
}

/// What to do after pumping the top frame.
#[derive(Clone, Copy)]
enum PumpOutcome {
    /// Default — continue the driver loop.
    Continue,
    /// Sequence returned `Pending`; surface to the host so it can
    /// interleave other work / consult fuel.
    Pending,
}

/// Pump a `Frame::Sequence` on top of `top`. Pops the frame, invokes
/// `seq.poll()` (or `seq.error()` if `pending_error.is_some()`), and
/// translates the [`SequencePoll`] back to frame ops.
fn pump_sequence<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    top: Thread<'gc>,
) -> Result<PumpOutcome, RuntimeError> {
    use crate::vm::sequence::{Execution, SequencePoll};
    let mc = ctx.mutation();
    // Pop the sequence frame and call poll/error.
    let (mut seq, call_site, pending_error) = {
        let mut ts = top.borrow_mut(mc);
        match ts.frames.pop() {
            Some(Frame::Sequence {
                seq,
                call_site,
                pending_error,
            }) => (seq, call_site, pending_error),
            _ => unreachable!("pump_sequence: top wasn't Frame::Sequence"),
        }
    };
    // The sequence's input values are the window `stack[call_site.bottom..top]`,
    // already published by whoever produced them (the suspending native, a
    // landed call, a resume). Its mutators write `top` back through the view.
    let poll_result = {
        let mut ts = top.borrow_mut(mc);
        // Split disjoint field borrows through a single deref of the RefMut
        // (the compiler can't split borrows across `RefMut`'s `Deref`).
        let ts: &mut crate::env::thread::ThreadState<'gc> = &mut ts;
        let stack_view =
            crate::env::function::Stack::new(&mut ts.stack, &mut ts.top, call_site.bottom);
        let exec = Execution::new(top, &ts.frames);
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
                call_site,
                pending_error: None,
            });
            return Ok(PumpOutcome::Pending);
        }
        Ok(SequencePoll::Return) => {
            // Sequence finished. Land values at the original CALL's expected
            // window.
            let mut ts = top.borrow_mut(mc);
            land_call_results(&mut ts, call_site);
        }
        Ok(SequencePoll::Call {
            function,
            bottom: rel,
        }) => {
            let abs_bottom = call_site.bottom + rel;
            let mut ts = top.borrow_mut(mc);
            // Re-push self to be re-polled with results at stack[bottom..].
            ts.frames.push(Frame::Sequence {
                seq,
                call_site,
                pending_error: None,
            });
            // Schedule the call: insert function at abs_bottom, args after.
            ts.insert_at(abs_bottom, Value::function(function));
            schedule_call_at(&mut ts, ctx, abs_bottom, 0)?;
        }
        Ok(SequencePoll::TailCall(function)) => {
            // Sequence is done; the call's results must land at the
            // original CALL site `call_site.func_idx`. Args sit at
            // stack[bottom..], adjacent to func_idx after a normal CALL
            // but not after a TAILCALL→native→sequence chain (where
            // bottom lives inside the popped tail-callee's window).
            // Compact down to func_idx+1, then place the function.
            let mut ts = top.borrow_mut(mc);
            // The sequence left its tail-call args at `stack[bottom..top]`.
            let argc = ts.top - call_site.bottom;
            let new_args_base = call_site.func_idx + 1;
            if new_args_base < call_site.bottom {
                ts.stack
                    .copy_within(call_site.bottom..call_site.bottom + argc, new_args_base);
                // Nils the stale copies the down-shift left above the args.
                ts.set_top(new_args_base + argc);
            }
            ts.stack[call_site.func_idx] = Value::function(function);
            schedule_call_at(&mut ts, ctx, call_site.func_idx, call_site.returns)?;
        }
        Ok(SequencePoll::Yield { bottom: rel }) => {
            let abs_bottom = call_site.bottom + rel;
            let mut ts = top.borrow_mut(mc);
            // Re-push self to be re-polled with resume-args at stack[bottom..].
            ts.frames.push(Frame::Sequence {
                seq,
                call_site,
                pending_error: None,
            });
            ts.yield_bottom = Some(in_place(abs_bottom));
            ts.status = ThreadStatus::Suspended;
        }
        Ok(SequencePoll::TailYield) => {
            let mut ts = top.borrow_mut(mc);
            ts.yield_bottom = Some(call_site);
            ts.status = ThreadStatus::Suspended;
        }
        Ok(SequencePoll::Resume {
            thread: target,
            bottom: rel,
        }) => {
            // Re-push self so propagate_inner_to_resumer finds the
            // sequence beneath the WaitThread and lands target's
            // eventual values at seq.bottom for the next poll.
            let abs_bottom = call_site.bottom + rel;
            top.borrow_mut(mc).frames.push(Frame::Sequence {
                seq,
                call_site,
                pending_error: None,
            });
            schedule_thread_resume(exec, ctx, top, target, abs_bottom, in_place(abs_bottom))?;
        }
        Ok(SequencePoll::TailResume(target)) => {
            // Sequence is consumed; target's eventual values go straight
            // to the original CALL site via the WaitThread call_site
            // when no Sequence is beneath.
            schedule_thread_resume(exec, ctx, top, target, call_site.bottom, call_site)?;
        }
        Err(err) => {
            top.borrow_mut(mc).raise(ctx, err);
        }
    }
    Ok(PumpOutcome::Continue)
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
    let resumer = *exec.0.borrow().thread_stack.last().unwrap();
    let mut rs = resumer.borrow_mut(mc);
    let wt = match rs.frames.pop() {
        Some(Frame::WaitThread { call_site }) => call_site,
        _ => unreachable!("propagate_inner_to_resumer: resumer top is not WaitThread"),
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

/// A landing site that leaves values at `bottom`: what a sequence records
/// when it suspends on its own behalf and will read them back there.
fn in_place(bottom: usize) -> CallSite {
    CallSite {
        bottom,
        func_idx: bottom,
        returns: 0,
        cont: None,
    }
}

/// Move values at `stack[bottom..]` into the original CALL's expected
/// landing slot per the standard Lua convention. If `frames` is empty
/// after the move (e.g., a tailcalled native suspended and the calling
/// Lua frame was already popped at TAILCALL time), terminate the thread
/// with `ThreadStatus::Result { bottom: func_idx }`.
fn land_call_results<'gc>(ts: &mut crate::env::thread::ThreadState<'gc>, cs: CallSite) {
    // Continuation-backed native metamethod/iterator: apply the parked
    // continuation against the caller frame rather than the plain landing.
    if let Some(cont) = cs.cont {
        apply_native_continuation(ts, cs.bottom, cont);
        return;
    }
    let CallSite {
        bottom,
        func_idx,
        returns,
        cont: _,
    } = cs;
    // The producer published its value count through `top`.
    let retc = ts.top - bottom;
    let wanted = if returns == 0 {
        retc
    } else {
        returns as usize - 1
    };
    // Cover both the write range and — if a Lua caller is on top — its full
    // register window, which the interpreter will address off `base` through a
    // raw pointer the moment we hand control back.
    let needed = match ts.top_lua() {
        Some(frame) => {
            (func_idx + wanted).max(frame.base + frame.closure.proto.max_stack_size as usize)
        }
        None => func_idx + wanted,
    };
    ts.ensure_slots(needed);
    let to_copy = retc.min(wanted);
    for i in 0..to_copy {
        ts.stack[func_idx + i] = ts.stack[bottom + i];
    }
    for i in to_copy..wanted {
        ts.stack[func_idx + i] = Value::nil();
    }
    // Publish the logical top: MULTRET delivers all `retc`, a fixed-results
    // call exactly `wanted`. This also nils the stale copies the down-shift
    // left above the results — safe because `func_idx` is where the CALL put
    // the function, and everything at or above a call's function slot is free
    // scratch in the caller's register allocation.
    ts.set_top(func_idx + if returns == 0 { retc } else { wanted });
    if ts.frames.is_empty() {
        ts.status = ThreadStatus::Result { bottom: func_idx };
    }
}

/// Apply a [`Continuation`]'s payload after a *suspended native* metamethod /
/// iterator finally produces its results at `stack[bottom..]`. This is the
/// executor-side twin of the interpreter's `apply_cont_payload!`: the native
/// has no Lua frame to carry the continuation, so the executor replays the
/// payload directly against the caller frame (which is on top — no frame was
/// pushed for the native) and lets the next pump resume it.
///
/// `StoreResult`/`TForCall` write into the caller's register window;
/// `CondJump` bumps the caller frame's resume `pc` (choosing whether to skip
/// the comparison's following `JMP`); `IgnoreResult` discards. Results were
/// staged in scratch above the window, so the window is always in bounds.
fn apply_native_continuation<'gc>(
    ts: &mut crate::env::thread::ThreadState<'gc>,
    bottom: usize,
    cont: Continuation,
) {
    // Count via the logical top (set by the producing landing site).
    let retc = ts.top - bottom;
    let result0 = if retc > 0 {
        ts.stack[bottom]
    } else {
        Value::nil()
    };
    let base = ts
        .top_lua()
        .expect("native continuation must resume into a caller Lua frame")
        .base;

    match cont.payload {
        ContinuationPayload::StoreResult { dst } => {
            ts.stack[base + dst as usize] = result0;
        }
        ContinuationPayload::IgnoreResult => {}
        ContinuationPayload::CondJump { offset, inverted } => {
            let truthy = !result0.is_falsy();
            if truthy != inverted {
                let frame = ts.top_lua_mut().unwrap();
                frame.pc = unsafe { frame.pc.offset(offset as isize) };
            }
        }
        ContinuationPayload::TForCall { base: reg, count } => {
            let dst = base + reg as usize + 3;
            let to_copy = retc.min(count as usize);
            for i in 0..to_copy {
                ts.stack[dst + i] = ts.stack[bottom + i];
            }
            for i in to_copy..count as usize {
                ts.stack[dst + i] = Value::nil();
            }
        }
    }

    // The payload is applied, so the staging window (`meta_fn` + args + results,
    // all parked at `caller_top..`) is spent: drop it, which nils it. Doubles as
    // the guarantee that the caller's register window is physically covered
    // before the interpreter re-derives its raw register pointer off `base`.
    let caller_top = {
        let frame = ts.top_lua().unwrap();
        frame.base + frame.closure.proto.max_stack_size as usize
    };
    ts.set_top(caller_top);
}

/// Call an `xpcall` message handler with `err` above the failing frames
/// (`luaG_errormsg`). A `HandlerSequence` frame collects its result and
/// re-raises it, marked handled, so the unwind then proceeds to the catcher
/// without consulting the handler again; an error inside the handler is
/// handed back to the handler.
fn run_message_handler<'gc>(
    ts: &mut ThreadState<'gc>,
    ctx: Context<'gc>,
    handler: Function<'gc>,
    err: Error<'gc>,
) -> Result<(), RuntimeError> {
    // Stage above every live register (`top` alone can sit inside the
    // innermost window when the catcher's callee failed at once).
    let slot = ts.live_top();
    ts.ensure_slots(slot + 2);
    ts.stack[slot] = Value::function(handler);
    ts.stack[slot + 1] = err.value();
    ts.set_top(slot + 2);
    ts.frames.push(Frame::Sequence {
        seq: BoxSequence::new(ctx.mutation(), HandlerSequence { handler, depth: 0 }),
        call_site: CallSite {
            bottom: slot,
            func_idx: slot,
            returns: 0,
            cont: None,
        },
        pending_error: None,
    });
    schedule_call_at(ts, ctx, slot, 0)
}

/// Completion of an `xpcall` message handler: its first result becomes the
/// error value. An error inside the handler calls the handler again with
/// it (manual §2.3), on top of the still-intact failing frames, until the
/// loop is cut with "error in error handling" like the reference's C-stack
/// limit does. The handler may yield: the reference forbids that only
/// because it runs the handler on the C stack, and we keep every call
/// resumable, as LuaJIT does.
#[derive(Collect)]
#[collect(internal, no_drop)]
struct HandlerSequence<'gc> {
    handler: Function<'gc>,
    depth: u32,
}

/// `LUAI_MAXCCALLS`: nested handler invocations before giving up.
const MAX_HANDLER_DEPTH: u32 = 200;

impl<'gc> Sequence<'gc> for HandlerSequence<'gc> {
    fn trace_pointers(&self, cc: &mut dyn Trace<'gc>) {
        seq_trace_pointers!(self, cc);
    }

    fn poll(
        self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        Err(Error::new(stack.get(0)).mark_handled())
    }

    fn error(
        mut self: Pin<&mut Self>,
        ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        err: Error<'gc>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        if self.depth == MAX_HANDLER_DEPTH {
            let msg = LuaString::new(ctx, b"error in error handling");
            return Err(Error::new(Value::string(msg)).mark_handled());
        }
        self.depth += 1;
        stack.replace(&[err.value()]);
        Ok(SequencePoll::Call {
            function: self.handler,
            bottom: 0,
        })
    }

    fn catch(&self) -> Catch<'gc> {
        Catch::Here(None)
    }
}

/// Walk a thread's frame stack popping Lua/Wait/`Catch::Pass` frames
/// (closing upvalues at each Lua `bottom`) until a `Catch::Here` sequence
/// takes the error.
///
/// If the nearest catcher has a message handler (`xpcall`), the handler
/// runs first, on top of the still-intact failing frames; see
/// `run_message_handler`.
///
/// On no-catcher: if the thread isn't the bottom of the executor's
/// thread stack, route the error to the resumer's `Frame::WaitThread`
/// and pop the inner thread. This lets a coroutine error propagate to
/// the resumer's `ProtectedCall::error`. If the thread *is* the bottom,
/// surface as `RuntimeError::Lua` to the host.
fn unwind_error<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    top: Thread<'gc>,
) -> Result<(), RuntimeError> {
    let mc = ctx.mutation();
    let mut ts = top.borrow_mut(mc);
    let Some(Frame::Error(err)) = ts.frames.pop() else {
        unreachable!()
    };
    if !err.is_handled() {
        // Only the nearest catch point's handler applies (`L->errfunc`):
        // a plain `pcall` in between shadows an outer `xpcall`, while a
        // `Catch::Pass` sequence is looked through.
        let handler = ts
            .frames
            .iter()
            .rev()
            .find_map(|f| match f {
                Frame::Sequence { seq, .. } => match seq.catch() {
                    Catch::Pass => None,
                    Catch::Here(handler) => Some(handler),
                },
                _ => None,
            })
            .flatten();
        if let Some(handler) = handler {
            return run_message_handler(&mut ts, ctx, handler, err);
        }
    }
    // `luaD_seterrorobj`: only once the error is being caught (a handler
    // still sees the raw nil).
    let err = if err.value().is_nil() {
        err.with_value(Value::string(LuaString::new(ctx, b"<no error object>")))
    } else {
        err
    };
    loop {
        match ts.frames.last_mut() {
            Some(Frame::Lua(lf)) => {
                let base = lf.base;
                ts.frames.pop();
                vm::interp::close_upvalues(mc, &mut ts, base);
                vm::interp::close_tbc_vars(mc, &mut ts, base);
                // The frame and everything above it is dead, so this is one
                // of the few places a shrink is legal.
                ts.discard_above(base);
            }
            Some(Frame::Sequence {
                seq, pending_error, ..
            }) => {
                if matches!(seq.catch(), Catch::Pass) {
                    ts.frames.pop();
                    continue;
                }
                *pending_error = Some(err);
                return Ok(());
            }
            Some(Frame::WaitThread { .. }) => {
                ts.frames.pop();
            }
            Some(Frame::Start(_) | Frame::Error(_)) => {
                // Frame::Start is only on a freshly-created thread that
                // hasn't run yet, so it can't have errored. Frame::Error is
                // removed by the next driver pump (which enters this
                // function), so two can't coexist.
                unreachable!(
                    "Frame::Start / Frame::Error mid-unwind violates the executor invariant"
                );
            }
            None => break,
        }
    }
    drop(ts);

    // No catcher on this thread. If we're an inner coroutine, propagate to
    // the resumer's WaitThread → its next Sequence can catch.
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
        }
        exec.0.borrow_mut(mc).thread_stack.pop();
        let resumer = *exec.0.borrow().thread_stack.last().unwrap();
        let mut rs = resumer.borrow_mut(mc);
        // Pop the WaitThread.
        match rs.frames.pop() {
            Some(Frame::WaitThread { .. }) => {}
            _ => unreachable!("inner-thread error: resumer top isn't WaitThread"),
        }
        // Already located on the inner thread; don't re-raise on the resumer.
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
